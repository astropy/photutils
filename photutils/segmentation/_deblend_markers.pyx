# Licensed under a 3-clause BSD style license - see LICENSE.rst
# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
# cython: freethreading_compatible=True
"""
Cython kernel that builds the deblending watershed markers from a
level-quantized source cutout in a single component-tree pass.

The multithreshold marker construction is defined level by level.
At each threshold level, the connected components (with fewer than
``n_pixels`` pixels removed) of the pixels above the threshold are the
candidate sources, and a marker is replaced by its components at a
higher level whenever it contains at least two of them. Computing this
directly requires one full labeling pass per level.

This kernel instead builds the quantized component tree once, by
adding pixels in decreasing level order to a union-find structure and
snapshotting the components at each populated level (levels between
populated values have identical components and are provably no-ops in
the per-level construction, so they need no snapshots). The marker set
is then derived by descending the tree. Starting from the components
of the lowest level that has at least two components with ``n_pixels``
or more pixels, each marker is replaced by its sufficiently large
components at the first higher level that contains at least two of them.
The resulting markers are identical to the per-level construction,
including the raster-scan ordering of the marker labels.

The multithreshold levels themselves are computed by the caller
(vectorized in NumPy over all the sources of a chunk, see
``photutils.segmentation.deblend``) and passed in, so that they are
bitwise identical to the pure-Python reference implementation on every
platform. The kernels here compute the per-source data extrema and
flux, quantize each cutout against its levels, and build the markers.

The kernels run without the GIL and use no global mutable state, so
this module is safe to use from multiple threads, including on
free-threaded Python builds.
"""

import numpy as np

from libc.math cimport NAN, isnan
from libc.stdlib cimport free, malloc, realloc

__all__ = ['deblend_markers_chunk', 'deblend_source_stats',
           'make_deblend_markers']

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
    int* area
    double* flux
    unsigned char* kept
    Py_ssize_t n
    Py_ssize_t cap


cdef struct _SaddleArgs:
    # Inputs of the saddle contrast criterion (see _markers_core).
    # Only enabled is read when the criterion is not in use
    bint enabled
    const double* thresholds
    double limit
    double* posimg
    double* fsum


cdef inline bint _nodes_grow(_Nodes* nodes) noexcept nogil:
    """
    Double the node storage capacity and return False on failure.
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
    cdef int* area = <int*>realloc(nodes.area, cap * sizeof(int))
    if area == NULL:
        return False
    nodes.area = area
    cdef double* flux = <double*>realloc(nodes.flux,
                                         cap * sizeof(double))
    if flux == NULL:
        return False
    nodes.flux = flux
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


cdef inline void _union(int* parent, int* size, double* fsum,
                        Py_ssize_t a, Py_ssize_t b) noexcept nogil:
    """
    Union the components containing ``a`` and ``b`` by size.

    The per-component flux sums are also merged when a ``fsum``
    workspace is provided.
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
    if fsum != NULL:
        fsum[ra] += fsum[rb]


cdef Py_ssize_t _markers_core(const int* qflat, Py_ssize_t ny,
                              Py_ssize_t nx, int n_pixels, bint conn8,
                              int* parent, int* size,
                              const _SaddleArgs* saddle,
                              unsigned char* added, int* stamp,
                              int* node_of_root, int* order,
                              int* flood, int* markers) noexcept nogil:
    """
    Build the deblending markers for one level-quantized cutout.

    The caller must provide the pixel-sized workspace arrays with
    ``added`` all zero and ``stamp`` all -1. Both invariants are
    restored before returning. The marker labels are written into
    ``markers`` at the pixels of the cutout that have a nonzero
    quantized level (the caller is responsible for treating the
    other ``markers`` entries as zero).

    With ``saddle.enabled``, the markers are instead selected with
    the saddle contrast criterion, which requires the ``posimg`` pixel
    values, the ``thresholds`` level values, the ``fsum`` pixel-sized
    workspace, and the ``limit`` flux (the contrast times the total
    source flux) of the ``saddle`` arguments. A branch passes when the
    flux it holds above the level at which it separates from its
    siblings exceeds the limit. The component fluxes are accumulated
    in the union-find merge order, so they can differ in the last bits
    from the raster-order sums of the pure-Python reference
    implementation. This only matters for a branch whose flux above
    its level equals the limit to within rounding.

    Returns the number of markers (0 if the source does not split),
    or -1 if a memory allocation failed.
    """
    cdef Py_ssize_t n_tot = ny * nx
    cdef Py_ssize_t result = 0
    cdef Py_ssize_t p, i, j, s, m, batch_start, nid, pid, root, nb
    cdef Py_ssize_t prev_lo, prev_hi, n_snaps, kept_cnt, base
    cdef Py_ssize_t node, chain, child, first_kept, n_kept_children
    cdef Py_ssize_t n_stack, n_final, fs, vthr, mn, rank, n_pass
    cdef Py_ssize_t py, px, dy, dx
    cdef int v, lbl
    cdef bint splitting
    cdef double prominence

    # Unpack the saddle criterion inputs
    cdef bint use_saddle = saddle.enabled
    cdef const double* posimg = NULL
    cdef const double* thresholds = NULL
    cdef double* fsum = NULL
    cdef double saddle_limit = 0.0
    if use_saddle:
        posimg = saddle.posimg
        thresholds = saddle.thresholds
        fsum = saddle.fsum
        saddle_limit = saddle.limit

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
    nodes.area = <int*>malloc(nodes.cap * sizeof(int))
    nodes.flux = <double*>malloc(nodes.cap * sizeof(double))
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
    cdef unsigned char* passer = NULL
    cdef unsigned char* has_split = NULL

    if (nodes.level == NULL or nodes.repr_pix == NULL
            or nodes.child == NULL or nodes.sib == NULL
            or nodes.area == NULL or nodes.flux == NULL
            or nodes.kept == NULL or counts == NULL or fill == NULL
            or snap_lo == NULL or snap_hi == NULL
            or snap_kept == NULL):
        result = -1

    if result == 0:
        # Sort the active pixels by decreasing level with a counting
        # sort. Pixels within a level stay in raster order
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
                if use_saddle:
                    fsum[p] = posimg[p]
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
                            _union(parent, size, fsum, p, nb)
                i += 1

            # Snapshot the components of the level set with threshold
            # index v - 1 and attach the previous (higher) level
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
                    nodes.area[nodes.n] = size[root]
                    if use_saddle:
                        nodes.flux[nodes.n] = fsum[root]
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
                    nodes.area[nodes.n] = size[root]
                    if use_saddle:
                        nodes.flux[nodes.n] = fsum[root]
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

    n_final = 0
    if result == 0 and use_saddle:
        # Saddle criterion: a junction splits where at least two
        # sufficiently large children hold more flux above the junction
        # level than the saddle limit. The markers are the passing
        # children of splitting junctions that contain no deeper split
        # themselves.
        final = <Py_ssize_t*>malloc(nodes.n * sizeof(Py_ssize_t))
        passer = <unsigned char*>malloc(nodes.n
                                        * sizeof(unsigned char))
        has_split = <unsigned char*>malloc(nodes.n
                                           * sizeof(unsigned char))
        if final == NULL or passer == NULL or has_split == NULL:
            result = -1
        else:
            # A branch passes when the flux it holds above the level at
            # which it separates from its siblings (the level above its
            # parent junction) exceeds the saddle limit. A node
            # represents its component over every level between two
            # populated ones, so this is the lowest of those levels,
            # as in the per-level reference construction. The
            # lowest-level components separate at the first threshold
            # level. Every other node is the child of exactly one node
            for nid in range(snap_lo[n_snaps - 1],
                             snap_hi[n_snaps - 1]):
                prominence = (nodes.flux[nid]
                              - thresholds[0] * nodes.area[nid])
                passer[nid] = (nodes.kept[nid]
                               and prominence > saddle_limit)
            for nid in range(nodes.n):
                child = nodes.child[nid]
                while child != -1:
                    prominence = (nodes.flux[child]
                                  - thresholds[nodes.level[nid] + 1]
                                  * nodes.area[child])
                    passer[child] = (nodes.kept[child]
                                     and prominence > saddle_limit)
                    child = nodes.sib[child]

            # Node ids ascend from the treetops down, so children are
            # always visited before their parents
            for nid in range(nodes.n):
                n_pass = 0
                child = nodes.child[nid]
                while child != -1:
                    if passer[child]:
                        n_pass += 1
                    child = nodes.sib[child]
                splitting = n_pass >= 2
                if splitting:
                    child = nodes.child[nid]
                    while child != -1:
                        if passer[child] and not has_split[child]:
                            final[n_final] = child
                            n_final += 1
                        child = nodes.sib[child]
                has_split[nid] = splitting
                child = nodes.child[nid]
                while child != -1:
                    if has_split[child]:
                        has_split[nid] = 1
                    child = nodes.sib[child]

            # The lowest-level components form a virtual root
            # junction at the first threshold level
            n_pass = 0
            for nid in range(snap_lo[n_snaps - 1],
                             snap_hi[n_snaps - 1]):
                if passer[nid]:
                    n_pass += 1
            if n_pass >= 2:
                for nid in range(snap_lo[n_snaps - 1],
                                 snap_hi[n_snaps - 1]):
                    if passer[nid] and not has_split[nid]:
                        final[n_final] = nid
                        n_final += 1

            if n_final < 2:
                # A splitting junction always yields at least two
                # markers, so this only happens with none at all
                n_final = 0
    elif result == 0:
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

    if result == 0 and n_final >= 2:
        # Paint each final marker by flood filling its component from
        # the recorded representative pixel. The marker regions are
        # disjoint, so each pixel is visited at most once.
        min_pix = <Py_ssize_t*>malloc(n_final * sizeof(Py_ssize_t))
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

            # Relabel the markers in raster-scan order of their first
            # pixels (an insertion sort, as the first pixels are
            # distinct), matching the ordering that per-level labeling
            # would produce.
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
    # arrays. The active pixels are found from qflat rather than from
    # order, which is not filled if an early allocation failed
    for p in range(n_tot):
        if qflat[p] > 0:
            added[p] = 0
            stamp[p] = -1

    free(nodes.level)
    free(nodes.repr_pix)
    free(nodes.child)
    free(nodes.sib)
    free(nodes.area)
    free(nodes.flux)
    free(nodes.kept)
    free(passer)
    free(has_split)
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

    This single-source entry point is exported only for the pure-Python
    reference implementation in ``_deblend_reference`` and the
    cross-implementation tests. Production deblending goes through
    ``deblend_markers_chunk``.

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

    cdef _SaddleArgs saddle
    saddle.enabled = False
    saddle.thresholds = NULL
    saddle.limit = 0.0
    saddle.posimg = NULL
    saddle.fsum = NULL

    cdef Py_ssize_t n_markers
    with nogil:
        n_markers = _markers_core(qflat, ny, nx, n_pixels,
                                  connectivity == 8, &parent_mv[0],
                                  &size_mv[0], &saddle, &added_mv[0],
                                  &stamp_mv[0], &node_of_root_mv[0],
                                  &order_mv[0], &flood_mv[0],
                                  &markers_mv[0, 0])
    if n_markers < 0:
        raise MemoryError

    return markers_arr, int(n_markers)


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


cdef void _source_stats(const data_t* data, const segm_t* segm,
                        Py_ssize_t img_nx, long long label,
                        Py_ssize_t y0, Py_ssize_t y1, Py_ssize_t x0,
                        Py_ssize_t x1, double* smin, double* smax,
                        double* ssum) noexcept nogil:
    """
    Compute the minimum, maximum, and flux of one source segment.

    NaN pixels are excluded. The flux is accumulated sequentially in
    float64 in raster order. The minimum and maximum are NaN, and the
    flux is 0, if the segment has no finite pixel.
    """
    cdef Py_ssize_t iy, ix, idx
    cdef double value
    cdef bint has_value = False

    smin[0] = NAN
    smax[0] = NAN
    ssum[0] = 0.0
    for iy in range(y1 - y0):
        for ix in range(x1 - x0):
            idx = (y0 + iy) * img_nx + x0 + ix
            if segm[idx] != label:
                continue
            value = <double>data[idx]
            if isnan(value):
                continue
            ssum[0] += value
            if not has_value:
                smin[0] = value
                smax[0] = value
                has_value = True
            elif value < smin[0]:
                smin[0] = value
            elif value > smax[0]:
                smax[0] = value


def deblend_source_stats(const data_t[:, ::1] data,
                         const segm_t[:, ::1] segm_data,
                         const long long[::1] labels,
                         const long long[::1] y0,
                         const long long[::1] y1,
                         const long long[::1] x0,
                         const long long[::1] x1):
    """
    Compute the minimum, maximum, and flux of each source segment.

    NaN pixels are excluded. The minimum and maximum are identical to
    the ``nanmin`` and ``nanmax`` reductions over the segment pixels.
    The flux is accumulated sequentially in float64 in raster order,
    which is what ``np.cumsum(values, dtype=np.float64)[-1]`` computes.

    Parameters
    ----------
    data : 2D float `~numpy.ndarray`
        The full data array.

    segm_data : 2D int `~numpy.ndarray`
        The full segmentation array.

    labels : 1D int64 `~numpy.ndarray`
        The label of each source.

    y0, y1, x0, x1 : 1D int64 `~numpy.ndarray`
        The bounding-box slice bounds of each source.

    Returns
    -------
    source_min, source_max, source_sum : 1D float64 `~numpy.ndarray`
        The minimum, maximum, and flux of each source segment. The
        minimum and maximum are NaN, and the flux is 0, for a segment
        without any finite pixel.
    """
    cdef Py_ssize_t n_src = labels.shape[0]
    cdef Py_ssize_t img_nx = data.shape[1]
    cdef Py_ssize_t isrc

    smin_arr = np.empty(n_src, dtype=np.float64)
    smax_arr = np.empty(n_src, dtype=np.float64)
    ssum_arr = np.empty(n_src, dtype=np.float64)
    cdef double[::1] smin_mv = smin_arr
    cdef double[::1] smax_mv = smax_arr
    cdef double[::1] ssum_mv = ssum_arr

    with nogil:
        for isrc in range(n_src):
            _source_stats(&data[0, 0], &segm_data[0, 0], img_nx,
                          labels[isrc], y0[isrc], y1[isrc], x0[isrc],
                          x1[isrc], &smin_mv[isrc], &smax_mv[isrc],
                          &ssum_mv[isrc])

    return smin_arr, smax_arr, ssum_arr


cdef Py_ssize_t _source_markers(const data_t* data, const segm_t* segm,
                                Py_ssize_t img_nx, long long label,
                                Py_ssize_t y0, Py_ssize_t y1,
                                Py_ssize_t x0, Py_ssize_t x1,
                                int n_pixels, bint conn8, int n_levels,
                                const double* thresholds,
                                const _SaddleArgs* saddle, int* q,
                                int* parent, int* size,
                                unsigned char* added, int* stamp,
                                int* node_of_root, int* order,
                                int* flood, int* markers) noexcept nogil:
    """
    Build the deblending markers for one source of a chunk.

    Quantizes the cutout against the multithreshold levels of the
    source (the number of levels strictly below each pixel value, with
    the pixels outside the segment and the NaN pixels at 0) and runs
    the component-tree core with the selected contrast criterion.
    Returns the number of markers (0 means the source does not split)
    or -1 if a memory allocation failed.
    """
    cdef Py_ssize_t ny_c = y1 - y0
    cdef Py_ssize_t nx_c = x1 - x0
    cdef Py_ssize_t iy, ix, idx
    cdef double value

    for iy in range(ny_c):
        for ix in range(nx_c):
            idx = (y0 + iy) * img_nx + x0 + ix
            if saddle.enabled:
                saddle.posimg[iy * nx_c + ix] = <double>data[idx]
            if segm[idx] != label:
                q[iy * nx_c + ix] = 0
                continue
            value = <double>data[idx]
            if isnan(value):
                q[iy * nx_c + ix] = 0
                continue
            q[iy * nx_c + ix] = _count_below(thresholds, n_levels,
                                             value)

    return _markers_core(q, ny_c, nx_c, n_pixels, conn8, parent, size,
                         saddle, added, stamp, node_of_root, order,
                         flood, markers)


def deblend_markers_chunk(const data_t[:, ::1] data,
                          const segm_t[:, ::1] segm_data,
                          const long long[::1] labels,
                          const long long[::1] y0,
                          const long long[::1] y1,
                          const long long[::1] x0,
                          const long long[::1] x1,
                          const double[:, ::1] thresholds,
                          int[::1] packed,
                          const Py_ssize_t[::1] starts, *,
                          int n_pixels, int connectivity,
                          int max_markers, saddle_limits=None):
    """
    Build the deblending watershed markers for a chunk of sources.

    Each cutout is quantized against its own multithreshold levels and
    its markers are built by the component-tree kernel, in compiled
    code that releases the GIL and reuses one workspace sized to the
    largest cutout in the chunk.

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

    thresholds : 2D float64 `~numpy.ndarray`
        The multithreshold levels of each source, with shape
        ``(n_sources, n_levels)`` and ascending along the second axis.

    packed : 1D int32 `~numpy.ndarray`
        The buffer that receives the marker image of every source. The
        region of source ``i`` starts at ``starts[i]`` and holds its
        ``(y1 - y0) * (x1 - x0)`` cutout pixels in raster order. Each
        region is zeroed, and then the markers are written for the
        sources that split into two or more markers (and no more than
        ``max_markers`` when it is not negative). The other regions are
        left at zero.

    starts : 1D intp `~numpy.ndarray`
        The start index of each source's region in ``packed``.

    n_pixels : int
        The minimum number of connected pixels an above-threshold
        component must have to be considered a source.

    connectivity : {8, 4}
        The pixel connectivity.

    max_markers : int
        The number of markers above which the marker image of a source
        is not built. Its marker count is still returned, so that the
        caller can retry the source with other levels. A negative value
        disables the limit.

    saddle_limits : 1D float64 `~numpy.ndarray` or `None`, optional
        If given, the markers are selected with the saddle contrast
        criterion instead of building all candidate markers. A
        junction of the component tree splits only where at least two
        sufficiently large components each hold more flux above the
        junction level than this per-source limit (the contrast times
        the total source flux).

    Returns
    -------
    n_markers : 1D intp `~numpy.ndarray`
        The number of markers found for each source.
    """
    cdef Py_ssize_t n_src = labels.shape[0]
    cdef Py_ssize_t img_nx = data.shape[1]
    cdef int n_levels = thresholds.shape[1]
    cdef Py_ssize_t isrc, n_tot, max_ntot, ny_c, nx_c, start, p
    cdef Py_ssize_t n_markers
    cdef bint use_saddle = saddle_limits is not None

    if thresholds.shape[0] != n_src:
        msg = 'thresholds must have one row per source'
        raise ValueError(msg)
    if starts.shape[0] != n_src:
        msg = 'starts must have one entry per source'
        raise ValueError(msg)

    max_ntot = 1
    for isrc in range(n_src):
        n_tot = (y1[isrc] - y0[isrc]) * (x1[isrc] - x0[isrc])
        if n_tot > max_ntot:
            max_ntot = n_tot

    q_arr = np.empty(max_ntot, dtype=np.int32)
    parent_arr = np.empty(max_ntot, dtype=np.int32)
    size_arr = np.zeros(max_ntot, dtype=np.int32)
    added_arr = np.zeros(max_ntot, dtype=np.uint8)
    stamp_arr = np.full(max_ntot, -1, dtype=np.int32)
    node_of_root_arr = np.zeros(max_ntot, dtype=np.int32)
    order_arr = np.empty(max_ntot, dtype=np.int32)
    flood_arr = np.empty(max_ntot, dtype=np.int32)
    n_markers_arr = np.zeros(n_src, dtype=np.intp)

    cdef int[::1] q_mv = q_arr
    cdef int[::1] parent_mv = parent_arr
    cdef int[::1] size_mv = size_arr
    cdef unsigned char[::1] added_mv = added_arr
    cdef int[::1] stamp_mv = stamp_arr
    cdef int[::1] node_of_root_mv = node_of_root_arr
    cdef int[::1] order_mv = order_arr
    cdef int[::1] flood_mv = flood_arr
    cdef Py_ssize_t[::1] n_markers_mv = n_markers_arr

    # The saddle criterion inputs, with workspaces used only by it
    cdef _SaddleArgs saddle
    cdef const double[::1] saddle_limits_mv = None
    cdef double[::1] posimg_mv = None
    cdef double[::1] fsum_mv = None
    saddle.enabled = use_saddle
    saddle.thresholds = NULL
    saddle.limit = 0.0
    saddle.posimg = NULL
    saddle.fsum = NULL
    if use_saddle:
        saddle_limits_mv = saddle_limits
        posimg_arr = np.empty(max_ntot, dtype=np.float64)
        fsum_arr = np.empty(max_ntot, dtype=np.float64)
        posimg_mv = posimg_arr
        fsum_mv = fsum_arr
        saddle.posimg = &posimg_mv[0]
        saddle.fsum = &fsum_mv[0]

    for isrc in range(n_src):
        ny_c = y1[isrc] - y0[isrc]
        nx_c = x1[isrc] - x0[isrc]
        n_tot = ny_c * nx_c
        start = starts[isrc]
        if use_saddle:
            saddle.limit = saddle_limits_mv[isrc]
            saddle.thresholds = &thresholds[isrc, 0]
        with nogil:
            for p in range(n_tot):
                packed[start + p] = 0
            n_markers = _source_markers(
                &data[0, 0], &segm_data[0, 0], img_nx, labels[isrc],
                y0[isrc], y1[isrc], x0[isrc], x1[isrc], n_pixels,
                connectivity == 8, n_levels, &thresholds[isrc, 0],
                &saddle, &q_mv[0],
                &parent_mv[0], &size_mv[0], &added_mv[0],
                &stamp_mv[0], &node_of_root_mv[0], &order_mv[0],
                &flood_mv[0], &packed[start])
            if n_markers < 2 or (max_markers >= 0
                                 and n_markers > max_markers):
                for p in range(n_tot):
                    packed[start + p] = 0
        if n_markers < 0:
            raise MemoryError
        n_markers_mv[isrc] = n_markers

    return n_markers_arr
