# Licensed under a 3-clause BSD style license - see LICENSE.rst
# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
# cython: freethreading_compatible=True
"""
Cython kernel that builds the deblending watershed markers from a
level-quantized source cutout in a single component-tree pass.

The multithreshold marker construction is defined level by level:
at each threshold level, the connected components (with fewer than
``n_pixels`` pixels removed) of the pixels above the threshold are the
candidate sources, and a marker is replaced by its components at a
higher level whenever it contains at least two of them. Computing this
directly requires one full labeling pass per level.

This kernel instead builds the quantized component tree once, by
adding pixels in decreasing level order to a union-find structure and
snapshotting the components at each populated level (levels between
populated values have identical components and are provably no-ops in
the per-level construction, so they need no snapshots). The marker set
is then derived by descending the tree: starting from the components
of the lowest level that has at least two components with ``n_pixels``
or more pixels, each marker is replaced by its sufficiently large
components at the first higher level that contains at least two of them.
The resulting markers are identical to the per-level construction,
including the raster-scan ordering of the marker labels.

The component-tree core runs without the GIL and uses no global mutable
state, so this module is safe to use from multiple threads, including on
free-threaded Python builds.
"""

import numpy as np

from libc.math cimport isnan, pow, sinh
from libc.stdlib cimport free, malloc, realloc

__all__ = ['DEBLEND_FLAG_NMARKERS', 'DEBLEND_FLAG_NONPOSMIN',
           'deblend_markers_chunk', 'make_deblend_markers']

cdef enum:
    # Per-source mode-fallback flag bits reported by the chunk driver
    FLAG_NONPOSMIN = 1
    FLAG_NMARKERS = 2
    # Threshold-mode codes accepted by the chunk driver
    MODE_LINEAR = 0
    MODE_EXPONENTIAL = 1
    MODE_SINH = 2

DEBLEND_FLAG_NONPOSMIN = FLAG_NONPOSMIN
DEBLEND_FLAG_NMARKERS = FLAG_NMARKERS

ctypedef fused data_t:
    float
    double

ctypedef fused segm_t:
    int
    long long


cdef struct _Nodes:
    # Growable per-node storage for the component tree
    int* level
    int* repr_pix
    int* child
    int* sib
    unsigned char* kept
    Py_ssize_t n
    Py_ssize_t cap


cdef inline bint _nodes_grow(_Nodes* nodes) noexcept nogil:
    """
    Double the node storage capacity; return False on failure.
    """
    cdef Py_ssize_t cap = nodes.cap * 2
    cdef int* level = <int*>realloc(nodes.level, cap * sizeof(int))
    if level == NULL:
        return False
    nodes.level = level
    cdef int* repr_pix = <int*>realloc(nodes.repr_pix,
                                       cap * sizeof(int))
    if repr_pix == NULL:
        return False
    nodes.repr_pix = repr_pix
    cdef int* child = <int*>realloc(nodes.child, cap * sizeof(int))
    if child == NULL:
        return False
    nodes.child = child
    cdef int* sib = <int*>realloc(nodes.sib, cap * sizeof(int))
    if sib == NULL:
        return False
    nodes.sib = sib
    cdef unsigned char* kept = <unsigned char*>realloc(
        nodes.kept, cap * sizeof(unsigned char))
    if kept == NULL:
        return False
    nodes.kept = kept
    nodes.cap = cap
    return True


cdef inline Py_ssize_t _find(int* parent, Py_ssize_t x) noexcept nogil:
    """
    Return the union-find root of ``x``, compressing the path.
    """
    cdef Py_ssize_t root = x
    cdef Py_ssize_t nxt
    while parent[root] != root:
        root = parent[root]
    while parent[x] != root:
        nxt = parent[x]
        parent[x] = <int>root
        x = nxt
    return root


cdef inline void _union(int* parent, int* size, Py_ssize_t a,
                        Py_ssize_t b) noexcept nogil:
    """
    Union the components containing ``a`` and ``b`` by size.
    """
    cdef Py_ssize_t ra = _find(parent, a)
    cdef Py_ssize_t rb = _find(parent, b)
    cdef Py_ssize_t tmp
    if ra == rb:
        return
    if size[ra] < size[rb]:
        tmp = ra
        ra = rb
        rb = tmp
    parent[rb] = <int>ra
    size[ra] += size[rb]


cdef Py_ssize_t _markers_core(const int* qflat, Py_ssize_t ny,
                              Py_ssize_t nx, int n_pixels, bint conn8,
                              int* parent, int* size,
                              unsigned char* added, int* stamp,
                              int* node_of_root, int* order,
                              int* flood, int* markers) noexcept nogil:
    """
    Build the deblending markers for one level-quantized cutout.

    The caller must provide the pixel-sized workspace arrays with
    ``added`` all zero and ``stamp`` all -1; both invariants are
    restored before returning. The marker labels are written into
    ``markers`` at the pixels of the cutout that have a nonzero
    quantized level (the caller is responsible for treating the
    other ``markers`` entries as zero).

    Returns the number of markers (0 if no threshold level has at
    least two sufficiently large components), or -1 if a memory
    allocation failed.
    """
    cdef Py_ssize_t n_tot = ny * nx
    cdef Py_ssize_t result = 0
    cdef Py_ssize_t p, i, j, s, m, batch_start, nid, pid, root, nb
    cdef Py_ssize_t prev_lo, prev_hi, n_snaps, kept_cnt, base
    cdef Py_ssize_t node, chain, child, first_kept, n_kept_children
    cdef Py_ssize_t n_stack, n_final, fs, vthr, mn, rank
    cdef Py_ssize_t py, px, dy, dx
    cdef int v, lbl

    # Count the active (nonzero) pixels and the maximum level
    cdef int qmax = 0
    cdef Py_ssize_t n_active = 0
    for p in range(n_tot):
        if qflat[p] > 0:
            n_active += 1
            if qflat[p] > qmax:
                qmax = qflat[p]
    if n_active == 0:
        return 0

    cdef _Nodes nodes
    nodes.n = 0
    nodes.cap = 256
    nodes.level = <int*>malloc(nodes.cap * sizeof(int))
    nodes.repr_pix = <int*>malloc(nodes.cap * sizeof(int))
    nodes.child = <int*>malloc(nodes.cap * sizeof(int))
    nodes.sib = <int*>malloc(nodes.cap * sizeof(int))
    nodes.kept = <unsigned char*>malloc(nodes.cap
                                        * sizeof(unsigned char))

    # Counting-sort bookkeeping and per-snapshot bookkeeping
    cdef Py_ssize_t* counts = <Py_ssize_t*>malloc(
        (qmax + 1) * sizeof(Py_ssize_t))
    cdef Py_ssize_t* fill = <Py_ssize_t*>malloc(
        (qmax + 1) * sizeof(Py_ssize_t))
    cdef Py_ssize_t* snap_lo = <Py_ssize_t*>malloc(
        qmax * sizeof(Py_ssize_t))
    cdef Py_ssize_t* snap_hi = <Py_ssize_t*>malloc(
        qmax * sizeof(Py_ssize_t))
    cdef Py_ssize_t* snap_kept = <Py_ssize_t*>malloc(
        qmax * sizeof(Py_ssize_t))
    cdef Py_ssize_t* stack = NULL
    cdef Py_ssize_t* final = NULL
    cdef Py_ssize_t* min_pix = NULL
    cdef Py_ssize_t* order_rank = NULL
    cdef int* lut = NULL

    if (nodes.level == NULL or nodes.repr_pix == NULL
            or nodes.child == NULL or nodes.sib == NULL
            or nodes.kept == NULL or counts == NULL or fill == NULL
            or snap_lo == NULL or snap_hi == NULL
            or snap_kept == NULL):
        result = -1

    if result == 0:
        # Sort the active pixels by decreasing level with a counting
        # sort; pixels within a level stay in raster order
        for v in range(qmax + 1):
            counts[v] = 0
        for p in range(n_tot):
            if qflat[p] > 0:
                counts[qflat[p]] += 1
        i = 0
        for v in range(qmax, 0, -1):
            fill[v] = i
            i += counts[v]
        for p in range(n_tot):
            v = qflat[p]
            if v > 0:
                order[fill[v]] = <int>p
                fill[v] += 1

        # The caller only guarantees zeros at inactive pixels
        for i in range(n_active):
            markers[order[i]] = 0

        # Build the component tree by adding pixels in decreasing
        # level order, snapshotting components at populated levels
        i = 0
        prev_lo = 0
        prev_hi = 0
        n_snaps = 0
        while i < n_active:
            v = qflat[order[i]]
            batch_start = i

            # Add pixels at this level and union with added neighbors
            while i < n_active and qflat[order[i]] == v:
                p = order[i]
                parent[p] = <int>p
                size[p] = 1
                added[p] = 1
                py = p // nx
                px = p % nx
                for dy in range(-1, 2):
                    if py + dy < 0 or py + dy >= ny:
                        continue
                    for dx in range(-1, 2):
                        if px + dx < 0 or px + dx >= nx:
                            continue
                        if dy == 0 and dx == 0:
                            continue
                        if not conn8 and dy != 0 and dx != 0:
                            continue
                        nb = p + dy * nx + dx
                        if added[nb]:
                            _union(parent, size, p, nb)
                i += 1

            # Snapshot the components of the level set with threshold
            # index v - 1; attach the previous (higher) level
            # components as children
            kept_cnt = 0
            for nid in range(prev_lo, prev_hi):
                root = _find(parent, nodes.repr_pix[nid])
                if stamp[root] != n_snaps:
                    stamp[root] = <int>n_snaps
                    if nodes.n == nodes.cap and not _nodes_grow(&nodes):
                        result = -1
                        break
                    nodes.level[nodes.n] = v - 1
                    nodes.repr_pix[nodes.n] = <int>root
                    nodes.child[nodes.n] = -1
                    nodes.kept[nodes.n] = size[root] >= n_pixels
                    kept_cnt += nodes.kept[nodes.n]
                    node_of_root[root] = <int>nodes.n
                    nodes.n += 1
                pid = node_of_root[root]
                nodes.sib[nid] = nodes.child[pid]
                nodes.child[pid] = <int>nid
            if result != 0:
                break

            # Create nodes for components new at this level
            for j in range(batch_start, i):
                root = _find(parent, order[j])
                if stamp[root] != n_snaps:
                    stamp[root] = <int>n_snaps
                    if nodes.n == nodes.cap and not _nodes_grow(&nodes):
                        result = -1
                        break
                    nodes.level[nodes.n] = v - 1
                    nodes.repr_pix[nodes.n] = <int>root
                    nodes.child[nodes.n] = -1
                    nodes.kept[nodes.n] = size[root] >= n_pixels
                    kept_cnt += nodes.kept[nodes.n]
                    node_of_root[root] = <int>nodes.n
                    nodes.n += 1
            if result != 0:
                break

            snap_lo[n_snaps] = prev_hi
            snap_hi[n_snaps] = nodes.n
            snap_kept[n_snaps] = kept_cnt
            prev_lo = prev_hi
            prev_hi = nodes.n
            n_snaps += 1

    if result == 0:
        # Find the base snapshot: the lowest level with at least two
        # sufficiently large components
        base = -1
        for s in range(n_snaps - 1, -1, -1):
            if snap_kept[s] >= 2:
                base = s
                break

        if base >= 0:
            # Descend the tree: replace each marker by its
            # sufficiently large components at the first higher
            # level containing at least two of them
            stack = <Py_ssize_t*>malloc(nodes.n * sizeof(Py_ssize_t))
            final = <Py_ssize_t*>malloc(nodes.n * sizeof(Py_ssize_t))
            if stack == NULL or final == NULL:
                result = -1
            else:
                n_stack = 0
                n_final = 0
                for nid in range(snap_lo[base], snap_hi[base]):
                    if nodes.kept[nid]:
                        stack[n_stack] = nid
                        n_stack += 1
                while n_stack > 0:
                    n_stack -= 1
                    node = stack[n_stack]
                    chain = node
                    while True:
                        n_kept_children = 0
                        first_kept = -1
                        child = nodes.child[chain]
                        while child != -1:
                            if nodes.kept[child]:
                                n_kept_children += 1
                                if n_kept_children == 1:
                                    first_kept = child
                                elif n_kept_children == 2:
                                    stack[n_stack] = first_kept
                                    n_stack += 1
                                    stack[n_stack] = child
                                    n_stack += 1
                                else:
                                    stack[n_stack] = child
                                    n_stack += 1
                            child = nodes.sib[child]
                        if n_kept_children >= 2:
                            break
                        if n_kept_children == 1:
                            chain = first_kept
                            continue
                        final[n_final] = node
                        n_final += 1
                        break

                # Paint each final marker by flood filling its
                # component from the recorded representative pixel;
                # the marker regions are disjoint, so each pixel is
                # visited at most once
                min_pix = <Py_ssize_t*>malloc(n_final
                                              * sizeof(Py_ssize_t))
                order_rank = <Py_ssize_t*>malloc(n_final
                                                 * sizeof(Py_ssize_t))
                lut = <int*>malloc((n_final + 1) * sizeof(int))
                if min_pix == NULL or order_rank == NULL or lut == NULL:
                    result = -1

            if result == 0:
                for m in range(n_final):
                    nid = final[m]
                    vthr = nodes.level[nid] + 1
                    lbl = <int>(m + 1)
                    p = nodes.repr_pix[nid]
                    markers[p] = lbl
                    flood[0] = <int>p
                    fs = 1
                    mn = p
                    while fs > 0:
                        fs -= 1
                        p = flood[fs]
                        if p < mn:
                            mn = p
                        py = p // nx
                        px = p % nx
                        for dy in range(-1, 2):
                            if py + dy < 0 or py + dy >= ny:
                                continue
                            for dx in range(-1, 2):
                                if px + dx < 0 or px + dx >= nx:
                                    continue
                                if dy == 0 and dx == 0:
                                    continue
                                if not conn8 and dy != 0 and dx != 0:
                                    continue
                                nb = p + dy * nx + dx
                                if markers[nb] == 0 and qflat[nb] >= vthr:
                                    markers[nb] = lbl
                                    flood[fs] = <int>nb
                                    fs += 1
                    min_pix[m] = mn

                # Relabel the markers in raster-scan order of their
                # first pixels (an insertion sort; the first pixels
                # are distinct), matching the ordering that
                # per-level labeling would produce
                for m in range(n_final):
                    order_rank[m] = m
                for m in range(1, n_final):
                    j = m
                    while (j > 0 and min_pix[order_rank[j - 1]]
                           > min_pix[order_rank[j]]):
                        rank = order_rank[j - 1]
                        order_rank[j - 1] = order_rank[j]
                        order_rank[j] = rank
                        j -= 1
                lut[0] = 0
                for m in range(n_final):
                    lut[order_rank[m] + 1] = <int>(m + 1)
                for i in range(n_active):
                    p = order[i]
                    if markers[p] != 0:
                        markers[p] = lut[markers[p]]

                result = n_final

    # Restore the workspace invariants for the added and stamp
    # arrays (only active pixels were touched)
    for i in range(n_active):
        p = order[i]
        added[p] = 0
        stamp[p] = -1

    free(nodes.level)
    free(nodes.repr_pix)
    free(nodes.child)
    free(nodes.sib)
    free(nodes.kept)
    free(counts)
    free(fill)
    free(snap_lo)
    free(snap_hi)
    free(snap_kept)
    free(stack)
    free(final)
    free(min_pix)
    free(order_rank)
    free(lut)

    return result


def make_deblend_markers(const int[:, ::1] quantized, int n_pixels,
                         int connectivity):
    """
    Build the deblending watershed markers for a single source.

    Parameters
    ----------
    quantized : 2D int `~numpy.ndarray`
        The level-quantized source cutout. Each pixel value is the
        number of multithreshold levels below the pixel data value
        (i.e., the pixel is above threshold level ``i`` if ``i <
        quantized``). Pixels outside the source segment and NaN
        pixels must be 0.

    n_pixels : int
        The minimum number of connected pixels an above-threshold
        component must have to be considered a source.

    connectivity : {8, 4}
        The pixel connectivity.

    Returns
    -------
    markers : 2D int `~numpy.ndarray`
        The marker image, with markers labeled consecutively from 1
        in raster-scan order. All values are 0 if no markers were
        found.

    n_markers : int
        The number of markers. Zero means no threshold level had at
        least two components with ``n_pixels`` or more pixels.
    """
    cdef Py_ssize_t ny = quantized.shape[0]
    cdef Py_ssize_t nx = quantized.shape[1]
    cdef Py_ssize_t n_tot = ny * nx

    markers_arr = np.zeros((ny, nx), dtype=np.int32)
    parent_arr = np.empty(n_tot, dtype=np.int32)
    size_arr = np.zeros(n_tot, dtype=np.int32)
    added_arr = np.zeros(n_tot, dtype=np.uint8)
    stamp_arr = np.full(n_tot, -1, dtype=np.int32)
    node_of_root_arr = np.zeros(n_tot, dtype=np.int32)
    order_arr = np.empty(n_tot, dtype=np.int32)
    flood_arr = np.empty(n_tot, dtype=np.int32)

    cdef const int* qflat = &quantized[0, 0]
    cdef int[:, ::1] markers_mv = markers_arr
    cdef int[::1] parent_mv = parent_arr
    cdef int[::1] size_mv = size_arr
    cdef unsigned char[::1] added_mv = added_arr
    cdef int[::1] stamp_mv = stamp_arr
    cdef int[::1] node_of_root_mv = node_of_root_arr
    cdef int[::1] order_mv = order_arr
    cdef int[::1] flood_mv = flood_arr

    cdef Py_ssize_t n_markers
    with nogil:
        n_markers = _markers_core(qflat, ny, nx, n_pixels,
                                  connectivity == 8,
                                  &parent_mv[0], &size_mv[0],
                                  &added_mv[0], &stamp_mv[0],
                                  &node_of_root_mv[0], &order_mv[0],
                                  &flood_mv[0], &markers_mv[0, 0])
    if n_markers < 0:
        raise MemoryError

    return markers_arr, int(n_markers)


cdef void _compute_thresholds(double smin, double smax, double ratio,
                              double delta_scale, int n_levels,
                              int mode, double* thresholds) noexcept nogil:
    """
    Compute the multithreshold levels for one source.

    This replicates, operation for operation, the NumPy arithmetic
    of the _SingleSourceDeblender threshold computation: the interior
    values of ``np.linspace(smin, smax, n_levels + 2)`` and the
    linear, exponential, or sinh spacing applied to the normalized
    thresholds. ``ratio`` and ``delta_scale`` are the ``smax / smin``
    and ``smax - smin`` scalars, rounded in the data dtype exactly as
    NumPy rounds them.
    """
    cdef double div = n_levels + 1
    cdef double delta = smax - smin
    cdef double step = delta / div
    cdef double sinh_norm = sinh(1.0 / 0.25)
    cdef double lin, norm
    cdef int k

    for k in range(1, n_levels + 1):
        if step != 0:
            lin = k * step + smin
        else:
            # np.linspace special case for subnormal steps
            lin = (k / div) * delta + smin
        if mode == MODE_LINEAR:
            thresholds[k - 1] = lin
        elif mode == MODE_EXPONENTIAL:
            norm = (lin - smin) / delta_scale
            thresholds[k - 1] = smin * pow(ratio, norm)
        else:
            norm = (lin - smin) / delta_scale
            norm = sinh(norm / 0.25) / sinh_norm
            norm = norm * delta_scale
            thresholds[k - 1] = norm + smin


cdef inline int _count_below(const double* thresholds, int n_levels,
                             double value) noexcept nogil:
    """
    Return the number of thresholds strictly below ``value``.

    This matches ``np.searchsorted(thresholds, value, side='left')``
    for non-NaN values.
    """
    cdef int lo = 0
    cdef int hi = n_levels
    cdef int mid
    while lo < hi:
        mid = (lo + hi) // 2
        if thresholds[mid] < value:
            lo = mid + 1
        else:
            hi = mid
    return lo


cdef Py_ssize_t _source_markers(const data_t* data, const segm_t* segm,
                                Py_ssize_t img_nx, long long label,
                                Py_ssize_t y0, Py_ssize_t y1,
                                Py_ssize_t x0, Py_ssize_t x1,
                                int n_pixels, bint conn8, int n_levels,
                                int mode, int max_markers,
                                double* thresholds, int* q,
                                int* parent, int* size,
                                unsigned char* added, int* stamp,
                                int* node_of_root, int* order,
                                int* flood, int* markers,
                                unsigned char* flag) noexcept nogil:
    """
    Build the deblending markers for one source of a chunk.

    Computes the source minimum and maximum over the segment (NaN
    pixels excluded), applies the exponential-mode fallback for
    non-positive minima and the too-many-markers fallback to linear
    mode (recording both in ``flag``), quantizes the cutout, and runs
    the component-tree core. Returns the number of markers (0 means
    the source does not split) or -1 if a memory allocation failed.
    """
    cdef Py_ssize_t ny_c = y1 - y0
    cdef Py_ssize_t nx_c = x1 - x0
    cdef Py_ssize_t iy, ix, idx
    cdef double value, smin, smax, ratio, delta_scale
    cdef bint has_value = False
    cdef int use_mode
    cdef Py_ssize_t n_markers

    flag[0] = 0
    smin = 0.0
    smax = 0.0
    for iy in range(ny_c):
        for ix in range(nx_c):
            idx = (y0 + iy) * img_nx + x0 + ix
            if segm[idx] != label:
                continue
            value = <double>data[idx]
            if isnan(value):
                continue
            if not has_value:
                smin = value
                smax = value
                has_value = True
            elif value < smin:
                smin = value
            elif value > smax:
                smax = value

    # No deblending for constant (or all-NaN) sources
    if not has_value or smin == smax:
        return 0

    use_mode = mode
    if mode == MODE_EXPONENTIAL and smin <= 0:
        # Fall back to linear mode for non-positive minima
        flag[0] |= FLAG_NONPOSMIN
        use_mode = MODE_LINEAR

    # The maximum/minimum ratio and difference scalars are rounded
    # in the data dtype, exactly as NumPy computes them from the
    # dtype-preserving nanmin/nanmax reductions
    if data_t is float:
        delta_scale = <double>(<float>smax - <float>smin)
        if smin != 0:
            ratio = <double>(<float>smax / <float>smin)
        else:
            ratio = 0.0
    else:
        delta_scale = smax - smin
        if smin != 0:
            ratio = smax / smin
        else:
            ratio = 0.0

    _compute_thresholds(smin, smax, ratio, delta_scale, n_levels,
                        use_mode, thresholds)

    # Quantize the cutout: the number of thresholds strictly below
    # each pixel value; pixels outside the segment and NaN pixels
    # are 0
    for iy in range(ny_c):
        for ix in range(nx_c):
            idx = (y0 + iy) * img_nx + x0 + ix
            if segm[idx] != label:
                q[iy * nx_c + ix] = 0
                continue
            value = <double>data[idx]
            if isnan(value):
                q[iy * nx_c + ix] = 0
                continue
            q[iy * nx_c + ix] = _count_below(thresholds, n_levels,
                                             value)

    n_markers = _markers_core(q, ny_c, nx_c, n_pixels, conn8, parent,
                              size, added, stamp, node_of_root, order,
                              flood, markers)

    # If there are too many markers, the watershed step would be very
    # slow, so retry with linear mode (which has fewer levels at low
    # thresholds)
    if n_markers > max_markers and use_mode != MODE_LINEAR:
        flag[0] |= FLAG_NMARKERS
        _compute_thresholds(smin, smax, ratio, delta_scale, n_levels,
                            MODE_LINEAR, thresholds)
        for iy in range(ny_c):
            for ix in range(nx_c):
                idx = (y0 + iy) * img_nx + x0 + ix
                if segm[idx] != label:
                    q[iy * nx_c + ix] = 0
                    continue
                value = <double>data[idx]
                if isnan(value):
                    q[iy * nx_c + ix] = 0
                    continue
                q[iy * nx_c + ix] = _count_below(thresholds, n_levels,
                                                 value)
        n_markers = _markers_core(q, ny_c, nx_c, n_pixels, conn8,
                                  parent, size, added, stamp,
                                  node_of_root, order, flood, markers)

    return n_markers


def deblend_markers_chunk(const data_t[:, ::1] data,
                          const segm_t[:, ::1] segm_data,
                          const long long[::1] labels,
                          const long long[::1] y0,
                          const long long[::1] y1,
                          const long long[::1] x0,
                          const long long[::1] x1, *, int n_pixels,
                          int connectivity, int n_levels, int mode,
                          int max_markers=200):
    """
    Build the deblending watershed markers for a chunk of sources.

    The multithreshold levels, the level quantization, the
    component-tree marker construction, and the mode fallbacks
    (non-positive minimum and too many markers) are all computed in
    compiled code that releases the GIL, reusing one workspace sized
    to the largest cutout in the chunk.

    Parameters
    ----------
    data : 2D float `~numpy.ndarray`
        The full data array.

    segm_data : 2D int `~numpy.ndarray`
        The full segmentation array.

    labels : 1D int64 `~numpy.ndarray`
        The label of each source in the chunk.

    y0, y1, x0, x1 : 1D int64 `~numpy.ndarray`
        The bounding-box slice bounds of each source.

    n_pixels : int
        The minimum number of connected pixels an above-threshold
        component must have to be considered a source.

    connectivity : {8, 4}
        The pixel connectivity.

    n_levels : int
        The number of multithreshold levels.

    mode : {0, 1, 2}
        The threshold-spacing mode code: 0 (linear),
        1 (exponential), or 2 (sinh).

    max_markers : int, optional
        The maximum number of markers before falling back to linear
        mode.

    Returns
    -------
    markers_list : list of 2D int `~numpy.ndarray` or `None`
        The cutout marker image for each source that splits into two
        or more markers, otherwise `None`.

    flags : 1D uint8 `~numpy.ndarray`
        The per-source mode-fallback flags
        (``DEBLEND_FLAG_NONPOSMIN`` and ``DEBLEND_FLAG_NMARKERS``).
    """
    cdef Py_ssize_t n_src = labels.shape[0]
    cdef Py_ssize_t img_nx = data.shape[1]
    cdef Py_ssize_t isrc, n_tot, max_ntot, ny_c, nx_c
    cdef Py_ssize_t n_markers

    max_ntot = 1
    for isrc in range(n_src):
        n_tot = (y1[isrc] - y0[isrc]) * (x1[isrc] - x0[isrc])
        if n_tot > max_ntot:
            max_ntot = n_tot

    thresholds_arr = np.empty(n_levels, dtype=np.float64)
    q_arr = np.empty(max_ntot, dtype=np.int32)
    parent_arr = np.empty(max_ntot, dtype=np.int32)
    size_arr = np.zeros(max_ntot, dtype=np.int32)
    added_arr = np.zeros(max_ntot, dtype=np.uint8)
    stamp_arr = np.full(max_ntot, -1, dtype=np.int32)
    node_of_root_arr = np.zeros(max_ntot, dtype=np.int32)
    order_arr = np.empty(max_ntot, dtype=np.int32)
    flood_arr = np.empty(max_ntot, dtype=np.int32)
    markers_arr = np.zeros(max_ntot, dtype=np.int32)
    flags_arr = np.zeros(n_src, dtype=np.uint8)

    cdef double[::1] thresholds_mv = thresholds_arr
    cdef int[::1] q_mv = q_arr
    cdef int[::1] parent_mv = parent_arr
    cdef int[::1] size_mv = size_arr
    cdef unsigned char[::1] added_mv = added_arr
    cdef int[::1] stamp_mv = stamp_arr
    cdef int[::1] node_of_root_mv = node_of_root_arr
    cdef int[::1] order_mv = order_arr
    cdef int[::1] flood_mv = flood_arr
    cdef int[::1] markers_mv = markers_arr
    cdef unsigned char[::1] flags_mv = flags_arr

    markers_list = []
    for isrc in range(n_src):
        ny_c = y1[isrc] - y0[isrc]
        nx_c = x1[isrc] - x0[isrc]
        with nogil:
            n_markers = _source_markers(
                &data[0, 0], &segm_data[0, 0], img_nx, labels[isrc],
                y0[isrc], y1[isrc], x0[isrc], x1[isrc], n_pixels,
                connectivity == 8, n_levels, mode, max_markers,
                &thresholds_mv[0], &q_mv[0], &parent_mv[0],
                &size_mv[0], &added_mv[0], &stamp_mv[0],
                &node_of_root_mv[0], &order_mv[0], &flood_mv[0],
                &markers_mv[0], &flags_mv[isrc])
        if n_markers < 0:
            raise MemoryError
        if n_markers >= 2:
            n_tot = ny_c * nx_c
            quantized = q_arr[:n_tot].reshape(ny_c, nx_c)
            markers = markers_arr[:n_tot].reshape(ny_c, nx_c)
            markers_list.append(np.where(quantized > 0, markers,
                                         np.int32(0)))
        else:
            markers_list.append(None)

    return markers_list, flags_arr
