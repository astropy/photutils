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
snapshotting the components at each populated level. The marker set is
then derived by descending the tree: starting from the components of the
lowest level that has at least two components with ``n_pixels`` or more
pixels, each marker is replaced by its sufficiently large components at
the first higher level that contains at least two of them. The resulting
markers are identical to the per-level construction, including the
raster-scan ordering of the marker labels.

The kernel uses no global mutable state, so it is safe to use from
multiple threads, including on free-threaded Python builds.
"""

import numpy as np

__all__ = ['make_deblend_markers']


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
    if ra == rb:
        return
    if size[ra] < size[rb]:
        ra, rb = rb, ra
    parent[rb] = <int>ra
    size[ra] += size[rb]


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
    cdef const int* qflat = &quantized[0, 0]

    markers_arr = np.zeros((ny, nx), dtype=np.int32)
    cdef int[:, ::1] markers_mv = markers_arr
    cdef int* markers = &markers_mv[0, 0]

    # Count the active (nonzero) pixels and the maximum level
    cdef Py_ssize_t p
    cdef int qmax = 0
    cdef Py_ssize_t n_active = 0
    for p in range(n_tot):
        if qflat[p] > 0:
            n_active += 1
            if qflat[p] > qmax:
                qmax = qflat[p]
    if n_active == 0:
        return markers_arr, 0

    # Sort the active pixels by decreasing level with a counting
    # sort; pixels within a level stay in raster order
    counts_arr = np.zeros(qmax + 1, dtype=np.int64)
    cdef long long[::1] counts = counts_arr
    for p in range(n_tot):
        if qflat[p] > 0:
            counts[qflat[p]] += 1
    starts_arr = np.zeros(qmax + 2, dtype=np.int64)
    cdef long long[::1] starts = starts_arr
    cdef Py_ssize_t pos = 0
    cdef int v
    for v in range(qmax, 0, -1):
        starts[v] = pos
        pos += counts[v]
    fill_arr = starts_arr[:qmax + 1].copy()
    cdef long long[::1] fill = fill_arr
    order_arr = np.empty(n_active, dtype=np.int32)
    cdef int[::1] order_mv = order_arr
    cdef int* order = &order_mv[0]
    for p in range(n_tot):
        v = qflat[p]
        if v > 0:
            order[fill[v]] = <int>p
            fill[v] += 1

    # Union-find state
    parent_arr = np.empty(n_tot, dtype=np.int32)
    size_arr = np.zeros(n_tot, dtype=np.int32)
    added_arr = np.zeros(n_tot, dtype=np.uint8)
    stamp_arr = np.full(n_tot, -1, dtype=np.int32)
    node_of_root_arr = np.zeros(n_tot, dtype=np.int32)
    cdef int[::1] parent_mv = parent_arr
    cdef int[::1] size_mv = size_arr
    cdef unsigned char[::1] added_mv = added_arr
    cdef int[::1] stamp_mv = stamp_arr
    cdef int[::1] node_of_root_mv = node_of_root_arr
    cdef int* parent = &parent_mv[0]
    cdef int* size = &size_mv[0]
    cdef unsigned char* added = &added_mv[0]
    cdef int* stamp = &stamp_mv[0]
    cdef int* node_of_root = &node_of_root_mv[0]

    # Component-tree node storage, grown geometrically as needed
    cdef Py_ssize_t node_cap = n_active + 16
    node_level_arr = np.empty(node_cap, dtype=np.int32)
    node_repr_arr = np.empty(node_cap, dtype=np.int32)
    node_child_arr = np.empty(node_cap, dtype=np.int32)
    node_sib_arr = np.empty(node_cap, dtype=np.int32)
    node_kept_arr = np.empty(node_cap, dtype=np.uint8)
    cdef int[::1] node_level = node_level_arr
    cdef int[::1] node_repr = node_repr_arr
    cdef int[::1] node_child = node_child_arr
    cdef int[::1] node_sib = node_sib_arr
    cdef unsigned char[::1] node_kept = node_kept_arr
    cdef Py_ssize_t n_nodes = 0

    # Per-snapshot bookkeeping (one snapshot per populated level)
    snap_lo_arr = np.empty(qmax, dtype=np.int64)
    snap_hi_arr = np.empty(qmax, dtype=np.int64)
    snap_kept_arr = np.zeros(qmax, dtype=np.int64)
    cdef long long[::1] snap_lo = snap_lo_arr
    cdef long long[::1] snap_hi = snap_hi_arr
    cdef long long[::1] snap_kept = snap_kept_arr

    cdef bint conn8 = connectivity == 8
    cdef Py_ssize_t i = 0
    cdef Py_ssize_t batch_start, j, nid, pid, root, nb
    cdef Py_ssize_t prev_lo = 0
    cdef Py_ssize_t prev_hi = 0
    cdef Py_ssize_t n_snaps = 0
    cdef Py_ssize_t py, px, dy, dx, kept_cnt

    while i < n_active:
        v = qflat[order[i]]
        batch_start = i

        # Add all pixels at this level and union with added neighbors
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
        # index v - 1. Levels between populated values have identical
        # components, so they need no snapshots.
        kept_cnt = 0

        # Attach the previous (higher) level components as children
        for nid in range(prev_lo, prev_hi):
            root = _find(parent, node_repr[nid])
            if stamp[root] != n_snaps:
                stamp[root] = <int>n_snaps
                if n_nodes == node_cap:
                    node_cap *= 2
                    node_level_arr = np.resize(node_level_arr, node_cap)
                    node_repr_arr = np.resize(node_repr_arr, node_cap)
                    node_child_arr = np.resize(node_child_arr, node_cap)
                    node_sib_arr = np.resize(node_sib_arr, node_cap)
                    node_kept_arr = np.resize(node_kept_arr, node_cap)
                    node_level = node_level_arr
                    node_repr = node_repr_arr
                    node_child = node_child_arr
                    node_sib = node_sib_arr
                    node_kept = node_kept_arr
                node_level[n_nodes] = v - 1
                node_repr[n_nodes] = <int>root
                node_child[n_nodes] = -1
                node_kept[n_nodes] = size[root] >= n_pixels
                kept_cnt += node_kept[n_nodes]
                node_of_root[root] = <int>n_nodes
                n_nodes += 1
            pid = node_of_root[root]
            node_sib[nid] = node_child[pid]
            node_child[pid] = <int>nid

        # Create nodes for components new at this level
        for j in range(batch_start, i):
            root = _find(parent, order[j])
            if stamp[root] != n_snaps:
                stamp[root] = <int>n_snaps
                if n_nodes == node_cap:
                    node_cap *= 2
                    node_level_arr = np.resize(node_level_arr, node_cap)
                    node_repr_arr = np.resize(node_repr_arr, node_cap)
                    node_child_arr = np.resize(node_child_arr, node_cap)
                    node_sib_arr = np.resize(node_sib_arr, node_cap)
                    node_kept_arr = np.resize(node_kept_arr, node_cap)
                    node_level = node_level_arr
                    node_repr = node_repr_arr
                    node_child = node_child_arr
                    node_sib = node_sib_arr
                    node_kept = node_kept_arr
                node_level[n_nodes] = v - 1
                node_repr[n_nodes] = <int>root
                node_child[n_nodes] = -1
                node_kept[n_nodes] = size[root] >= n_pixels
                kept_cnt += node_kept[n_nodes]
                node_of_root[root] = <int>n_nodes
                n_nodes += 1

        # The nodes created for this snapshot are contiguous,
        # starting where the previous snapshot ended
        snap_lo[n_snaps] = prev_hi
        snap_hi[n_snaps] = n_nodes
        snap_kept[n_snaps] = kept_cnt
        prev_lo = prev_hi
        prev_hi = n_nodes
        n_snaps += 1

    # Find the base snapshot: the lowest level with at least two
    # sufficiently large components
    cdef Py_ssize_t base = -1
    cdef Py_ssize_t s
    for s in range(n_snaps - 1, -1, -1):
        if snap_kept[s] >= 2:
            base = s
            break
    if base == -1:
        return markers_arr, 0

    # Descend the tree: replace each marker by its sufficiently
    # large components at the first higher level containing at least
    # two of them
    stack_arr = np.empty(n_nodes, dtype=np.int64)
    final_arr = np.empty(n_nodes, dtype=np.int64)
    cdef long long[::1] stack = stack_arr
    cdef long long[::1] final = final_arr
    cdef Py_ssize_t n_stack = 0
    cdef Py_ssize_t n_final = 0
    cdef Py_ssize_t node, chain, child, first_kept, n_kept_children

    for nid in range(snap_lo[base], snap_hi[base]):
        if node_kept[nid]:
            stack[n_stack] = nid
            n_stack += 1

    while n_stack > 0:
        n_stack -= 1
        node = stack[n_stack]
        chain = node
        while True:
            n_kept_children = 0
            first_kept = -1
            child = node_child[chain]
            while child != -1:
                if node_kept[child]:
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
                child = node_sib[child]
            if n_kept_children >= 2:
                break
            if n_kept_children == 1:
                chain = first_kept
                continue
            final[n_final] = node
            n_final += 1
            break

    # Paint each final marker by flood filling its component from
    # the recorded representative pixel; the marker regions are
    # disjoint, so each pixel is visited at most once
    flood_arr = np.empty(n_active, dtype=np.int32)
    cdef int[::1] flood_mv = flood_arr
    cdef int* flood = &flood_mv[0]
    min_pix_arr = np.empty(n_final, dtype=np.int64)
    cdef long long[::1] min_pix = min_pix_arr
    cdef Py_ssize_t m, fs, vthr, mn
    cdef int lbl

    for m in range(n_final):
        nid = final[m]
        vthr = node_level[nid] + 1
        lbl = <int>(m + 1)
        p = node_repr[nid]
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

    # Relabel the markers in raster-scan order of their first pixels,
    # matching the ordering that per-level labeling would produce
    raster_order_arr = np.argsort(min_pix_arr, kind='stable')
    cdef long long[::1] raster_order = raster_order_arr
    lut_arr = np.zeros(n_final + 1, dtype=np.int32)
    cdef int[::1] lut = lut_arr
    for m in range(n_final):
        lut[raster_order[m] + 1] = <int>(m + 1)
    for p in range(n_tot):
        if markers[p] != 0:
            markers[p] = lut[markers[p]]

    return markers_arr, n_final
