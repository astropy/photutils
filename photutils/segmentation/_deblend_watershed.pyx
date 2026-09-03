# Licensed under a 3-clause BSD style license - see LICENSE.rst
# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
# cython: freethreading_compatible=True
"""
Cython marker-based watershed kernel for source deblending.

This implements the classic priority-flood watershed (Soille 1990)
used for deblending. Pixels are flooded from the markers in order of
increasing image value, with the queue-entry age breaking ties so
that plateaus are split between the markers that reach them first.
The algorithm, the neighbor ordering (orthogonal neighbors before
diagonal ones, each group in raster order), and the tie-breaking
match ``skimage.segmentation.watershed`` (with ``compactness=0``
and ``watershed_line=False``), so the results are identical, but
without the per-call validation, padding, and cropping overhead of the
general-purpose function, which dominates for small cutouts.

The flood order is only defined for ordered image values, so the
deblending entry point maps NaN data pixels to a +inf flooding cost.
Such pixels are assigned to the basins that reach them after all the
finite pixels have been assigned.

The flood core runs without the GIL and uses no global mutable state,
so this module is safe to use from multiple threads, including on
free-threaded Python builds.
"""

import numpy as np

from libc.math cimport INFINITY, isnan
from libc.stdlib cimport free, malloc

__all__ = ['deblend_source_contrast', 'deblend_watershed']

ctypedef fused data_t:
    float
    double

ctypedef fused segm_t:
    int
    long long


cdef struct _Heap:
    # Binary min-heap on (value, age)
    double* value
    long long* age
    int* index
    Py_ssize_t n


cdef inline bint _heap_less(_Heap* heap, Py_ssize_t a,
                            Py_ssize_t b) noexcept nogil:
    """
    Compare two heap slots by (value, age).
    """
    if heap.value[a] != heap.value[b]:
        return heap.value[a] < heap.value[b]
    return heap.age[a] < heap.age[b]


cdef inline void _heap_swap(_Heap* heap, Py_ssize_t a,
                            Py_ssize_t b) noexcept nogil:
    """
    Swap two heap slots.
    """
    cdef double value = heap.value[a]
    cdef long long age = heap.age[a]
    cdef int index = heap.index[a]
    heap.value[a] = heap.value[b]
    heap.age[a] = heap.age[b]
    heap.index[a] = heap.index[b]
    heap.value[b] = value
    heap.age[b] = age
    heap.index[b] = index


cdef inline void _heap_push(_Heap* heap, double value, long long age,
                            Py_ssize_t index) noexcept nogil:
    """
    Push an item onto the heap.
    """
    cdef Py_ssize_t pos = heap.n
    cdef Py_ssize_t parent
    heap.value[pos] = value
    heap.age[pos] = age
    heap.index[pos] = <int>index
    heap.n += 1
    while pos > 0:
        parent = (pos - 1) // 2
        if _heap_less(heap, pos, parent):
            _heap_swap(heap, pos, parent)
            pos = parent
        else:
            break


cdef inline Py_ssize_t _heap_pop(_Heap* heap,
                                 double* value) noexcept nogil:
    """
    Pop the smallest item, returning its pixel index and value.
    """
    cdef Py_ssize_t result = heap.index[0]
    cdef Py_ssize_t pos = 0
    cdef Py_ssize_t child
    value[0] = heap.value[0]
    heap.n -= 1
    if heap.n > 0:
        _heap_swap(heap, 0, heap.n)
        while True:
            child = 2 * pos + 1
            if child >= heap.n:
                break
            if (child + 1 < heap.n
                    and _heap_less(heap, child + 1, child)):
                child += 1
            if _heap_less(heap, child, pos):
                _heap_swap(heap, pos, child)
                pos = child
            else:
                break
    return result


cdef int _watershed_core(const double* image, unsigned char* mask,
                         int* output, Py_ssize_t ny, Py_ssize_t nx,
                         bint conn8) noexcept nogil:
    """
    Flood the masked pixels of ``output`` from its nonzero markers.

    Returns 0 on success or -1 if a memory allocation failed.
    """
    # Neighbor offsets: orthogonal neighbors before diagonal ones,
    # each group in raster order (the stable distance ordering)
    cdef Py_ssize_t[8] off_y
    cdef Py_ssize_t[8] off_x
    cdef Py_ssize_t n_off
    off_y[0] = -1
    off_x[0] = 0
    off_y[1] = 0
    off_x[1] = -1
    off_y[2] = 0
    off_x[2] = 1
    off_y[3] = 1
    off_x[3] = 0
    if conn8:
        off_y[4] = -1
        off_x[4] = -1
        off_y[5] = -1
        off_x[5] = 1
        off_y[6] = 1
        off_x[6] = -1
        off_y[7] = 1
        off_x[7] = 1
        n_off = 8
    else:
        n_off = 4

    # Each masked pixel enters the queue at most once
    cdef Py_ssize_t n_tot = ny * nx
    cdef Py_ssize_t cap = 0
    cdef Py_ssize_t p
    for p in range(n_tot):
        if mask[p]:
            cap += 1
    if cap == 0:
        return 0

    cdef _Heap heap
    heap.n = 0
    heap.value = <double*>malloc(cap * sizeof(double))
    heap.age = <long long*>malloc(cap * sizeof(long long))
    heap.index = <int*>malloc(cap * sizeof(int))
    if heap.value == NULL or heap.age == NULL or heap.index == NULL:
        free(heap.value)
        free(heap.age)
        free(heap.index)
        return -1

    # Push the marker pixels in raster order with age 0
    cdef long long age = 0
    for p in range(n_tot):
        if output[p] != 0 and mask[p]:
            _heap_push(&heap, image[p], 0, p)

    cdef Py_ssize_t py, px, ny_i, nx_i, nb, i
    cdef double pop_value, push_value
    while heap.n > 0:
        p = _heap_pop(&heap, &pop_value)
        py = p // nx
        px = p % nx
        for i in range(n_off):
            ny_i = py + off_y[i]
            nx_i = px + off_x[i]
            if ny_i < 0 or ny_i >= ny or nx_i < 0 or nx_i >= nx:
                continue
            nb = ny_i * nx + nx_i
            if not mask[nb]:
                continue
            if output[nb] != 0:
                continue
            age += 1
            output[nb] = output[p]
            # The flooding cost of a pixel is at least the cost of
            # the pixel it was reached from, so that plateaus and
            # basins below the current flood level are distributed
            # by queue-entry age between contesting markers
            push_value = image[nb]
            if push_value < pop_value:
                push_value = pop_value
            _heap_push(&heap, push_value, age, nb)

    free(heap.value)
    free(heap.age)
    free(heap.index)
    return 0


def deblend_watershed(image, markers, mask, connectivity):
    """
    Compute the marker-based watershed of an image.

    This is equivalent to ``skimage.segmentation.watershed(image,
    markers, mask=mask, connectivity=footprint)`` for the deblending
    use case (markers inside the mask, ``compactness=0``, and
    ``watershed_line=False``), but avoids the per-call validation,
    padding, and cropping overhead.

    This entry point is exported only for the pure-Python
    reference implementation in ``_deblend_reference`` and the
    cross-implementation tests. The production contrast loop calls the
    watershed core directly through ``deblend_source_contrast``.

    Parameters
    ----------
    image : 2D `~numpy.ndarray`
        The image to flood (the lowest values are flooded first). It
        must not contain NaN values. The deblending callers map NaN
        data pixels to +inf so that they are flooded last.

    markers : 2D int `~numpy.ndarray`
        The marker image. Zero means not a marker. All markers must
        lie inside the mask.

    mask : 2D bool `~numpy.ndarray`
        Only pixels where the mask is `True` are labeled.

    connectivity : {8, 4}
        The pixel connectivity.

    Returns
    -------
    output : 2D int `~numpy.ndarray`
        The labeled basins, with the same shape as ``image``.
    """
    image_arr = np.ascontiguousarray(image, dtype=np.float64)
    output_arr = np.array(markers, dtype=np.int32, copy=True, order='C')
    mask_arr = np.ascontiguousarray(mask, dtype=np.uint8)

    cdef const double[:, ::1] image_mv = image_arr
    cdef int[:, ::1] output_mv = output_arr
    cdef unsigned char[:, ::1] mask_mv = mask_arr
    cdef Py_ssize_t ny = image_mv.shape[0]
    cdef Py_ssize_t nx = image_mv.shape[1]

    cdef bint conn8 = connectivity == 8
    cdef int status
    with nogil:
        status = _watershed_core(&image_mv[0, 0], &mask_mv[0, 0],
                                 &output_mv[0, 0], ny, nx, conn8)
    if status < 0:
        raise MemoryError

    return output_arr


cdef int _contrast_core(const double* posimg, const double* negimg,
                        unsigned char* mask, int* output,
                        Py_ssize_t ny, Py_ssize_t nx, bint conn8,
                        double contrast, double source_sum,
                        double source_min,
                        Py_ssize_t n_max_labels) noexcept nogil:
    """
    Run the watershed contrast loop for one source in place.

    ``output`` holds the markers on input and the final relabeled
    (consecutive from 1) basins on output. Returns the number of
    final labels, or -1 if a memory allocation failed, or -2 if the
    flooded basins do not cover the segment mask (a connectivity
    mismatch).

    The loop replicates the NumPy implementation operation for
    operation. The basin fluxes are accumulated in raster order in
    float64 (as np.bincount does), the below-contrast markers are
    removed one at a time or in the largest provably equivalent
    batch of the faintest markers, and NaN basin fluxes compare
    false against every threshold, with np.argmin's first-NaN
    behavior for the single-marker removal. The only divergence is
    the order of bitwise-equal basin fluxes in the batch sort (NumPy
    uses an unstable sort there, while this uses a stable one).
    """
    cdef Py_ssize_t n_tot = ny * nx
    cdef Py_ssize_t n_cap = n_max_labels + 1
    cdef long long* counts = <long long*>malloc(
        n_cap * sizeof(long long))
    cdef double* flux = <double*>malloc(n_cap * sizeof(double))
    cdef int* lab = <int*>malloc(n_cap * sizeof(int))
    cdef double* frac = <double*>malloc(n_cap * sizeof(double))
    cdef Py_ssize_t* order = <Py_ssize_t*>malloc(
        n_cap * sizeof(Py_ssize_t))
    cdef double* csum = <double*>malloc(n_cap * sizeof(double))
    cdef unsigned char* removed = <unsigned char*>malloc(
        n_cap * sizeof(unsigned char))
    cdef int* lut = <int*>malloc(n_cap * sizeof(int))

    cdef Py_ssize_t p, i, j, k, n_labels, n_remove, min_idx, last_ok
    cdef Py_ssize_t pos
    cdef int status, current
    cdef bint remove_marker, a_nan, b_nan
    cdef double value_a, value_b

    if (counts == NULL or flux == NULL or lab == NULL or frac == NULL
            or order == NULL or csum == NULL or removed == NULL
            or lut == NULL):
        status = -1
    else:
        status = 0

    n_labels = 0
    while status == 0:
        status = _watershed_core(negimg, mask, output, ny, nx, conn8)
        if status != 0:
            break

        # Present labels (ascending) and their fluxes, accumulated
        # in raster order in float64 as np.bincount does.
        for i in range(n_cap):
            counts[i] = 0
            flux[i] = 0.0
        for p in range(n_tot):
            current = output[p]
            if current != 0:
                counts[current] += 1
                flux[current] += posimg[p]
        n_labels = 0
        for i in range(1, n_cap):
            if counts[i] > 0:
                lab[n_labels] = <int>i
                frac[n_labels] = flux[i] / source_sum
                n_labels += 1

        if n_labels == 1:  # only 1 source left
            break

        remove_marker = False
        for i in range(n_labels):
            if frac[i] < contrast:
                remove_marker = True
                break
        if not remove_marker:
            break

        # Remove the faintest below-contrast marker(s). See
        # _SingleSourceDeblender._remove_faint_markers.
        n_remove = 1
        if source_min >= 0 and n_labels > 2:
            # Stable sort of the label positions by flux fraction,
            # with NaN values sorting to the end.
            for i in range(n_labels):
                order[i] = i
            for i in range(1, n_labels):
                j = i
                while j > 0:
                    value_a = frac[order[j]]
                    value_b = frac[order[j - 1]]
                    a_nan = isnan(value_a)
                    b_nan = isnan(value_b)
                    if ((not a_nan and b_nan)
                            or (not a_nan and value_a < value_b)):
                        pos = order[j - 1]
                        order[j - 1] = order[j]
                        order[j] = pos
                        j -= 1
                    else:
                        break
            csum[0] = frac[order[0]]
            for i in range(1, n_labels):
                csum[i] = csum[i - 1] + frac[order[i]]
            # A batch of the n faintest markers (2 <= n < N) is
            # valid if its total flux fraction is below both the
            # contrast and the next-faintest marker flux fraction.
            last_ok = -1
            for k in range(1, n_labels - 1):
                if (csum[k] < contrast
                        and csum[k] < frac[order[k + 1]]):
                    last_ok = k
            if last_ok >= 0:
                n_remove = last_ok + 1

        for i in range(n_cap):
            removed[i] = 0
        if n_remove == 1:
            # np.argmin: the first NaN if any, else the first minimum
            min_idx = -1
            for i in range(n_labels):
                if isnan(frac[i]):
                    min_idx = i
                    break
            if min_idx == -1:
                min_idx = 0
                for i in range(1, n_labels):
                    if frac[i] < frac[min_idx]:
                        min_idx = i
            removed[lab[min_idx]] = 1
        else:
            for j in range(n_remove):
                removed[lab[order[j]]] = 1
        for p in range(n_tot):
            if output[p] != 0 and removed[output[p]]:
                output[p] = 0

    if status == 0:
        # The flooded basins must cover the segment mask exactly
        # (they cannot with mismatched detection and deblending
        # connectivities).
        for p in range(n_tot):
            if mask[p] and output[p] == 0:
                status = -2
                break

    if status == 0:
        # Relabel the surviving labels consecutively from 1 in
        # ascending label order.
        for i in range(n_labels):
            lut[lab[i]] = <int>(i + 1)
        for p in range(n_tot):
            if output[p] != 0:
                output[p] = lut[output[p]]
        status = <int>n_labels

    free(counts)
    free(flux)
    free(lab)
    free(frac)
    free(order)
    free(csum)
    free(removed)
    free(lut)

    return status


def deblend_source_contrast(const data_t[:, ::1] data,
                            const segm_t[:, ::1] segm_data,
                            long long label, Py_ssize_t y0,
                            Py_ssize_t y1, Py_ssize_t x0,
                            Py_ssize_t x1, markers, *,
                            int connectivity, double contrast,
                            double source_sum, double source_min):
    """
    Apply the watershed contrast loop to one source's markers.

    The watershed flooding, the basin flux measurements, the
    below-contrast marker removal (single or batched), and the final
    consecutive relabeling all run in compiled code that releases
    the GIL, producing results identical to the per-step NumPy
    implementation in ``_SingleSourceDeblender``.

    Parameters
    ----------
    data : 2D float `~numpy.ndarray`
        The full data array. NaN pixels within the segment are flooded
        after all the finite pixels, so they are assigned to a
        neighboring basin. They contribute NaN to the flux of that
        basin, as in the NumPy implementation.

    segm_data : 2D int `~numpy.ndarray`
        The full segmentation array.

    label : int
        The label of the source segment.

    y0, y1, x0, x1 : int
        The bounding-box slice bounds of the source.

    markers : 2D int `~numpy.ndarray`
        The marker image cutout, with markers labeled from 1. It is
        not modified.

    connectivity : {8, 4}
        The pixel connectivity.

    contrast : float
        The contrast criterion (the minimum fraction of the total
        source flux that a watershed basin must contain).

    source_sum : float
        The total flux of the source segment (NaN pixels excluded).

    source_min : float
        The minimum data value of the source segment (NaN pixels
        excluded).

    Returns
    -------
    output : 2D int `~numpy.ndarray` or `None`
        The deblended cutout with consecutive labels starting from
        1, or `None` if only one basin remains after applying the
        contrast criterion.

    Raises
    ------
    ValueError
        If the flooded basins do not cover the segment mask, which
        happens when the detection and deblending connectivities
        differ.
    """
    cdef Py_ssize_t ny_c = y1 - y0
    cdef Py_ssize_t nx_c = x1 - x0
    cdef Py_ssize_t img_nx = data.shape[1]

    posimg_arr = np.empty((ny_c, nx_c), dtype=np.float64)
    negimg_arr = np.empty((ny_c, nx_c), dtype=np.float64)
    mask_arr = np.empty((ny_c, nx_c), dtype=np.uint8)
    output_arr = np.array(markers, dtype=np.int32, copy=True, order='C')

    cdef double[:, ::1] posimg_mv = posimg_arr
    cdef double[:, ::1] negimg_mv = negimg_arr
    cdef unsigned char[:, ::1] mask_mv = mask_arr
    cdef int[:, ::1] output_mv = output_arr
    cdef double* posimg = &posimg_mv[0, 0]
    cdef double* negimg = &negimg_mv[0, 0]
    cdef unsigned char* mask = &mask_mv[0, 0]
    cdef int* output = &output_mv[0, 0]
    cdef const data_t* data_ptr = &data[0, 0]
    cdef const segm_t* segm_ptr = &segm_data[0, 0]

    cdef bint conn8 = connectivity == 8
    cdef Py_ssize_t iy, ix, idx, p
    cdef Py_ssize_t n_max_labels = 0
    cdef int status

    with nogil:
        for iy in range(ny_c):
            for ix in range(nx_c):
                idx = (y0 + iy) * img_nx + x0 + ix
                p = iy * nx_c + ix
                posimg[p] = <double>data_ptr[idx]
                if isnan(posimg[p]):
                    # NaN pixels are flooded after all finite pixels
                    negimg[p] = INFINITY
                else:
                    negimg[p] = -posimg[p]
                mask[p] = segm_ptr[idx] == label
                if output[p] > n_max_labels:
                    n_max_labels = output[p]
        status = _contrast_core(posimg, negimg, mask, output, ny_c,
                                nx_c, conn8, contrast, source_sum,
                                source_min, n_max_labels)

    if status == -1:
        raise MemoryError
    if status == -2:
        msg = (f'Deblending failed for source {int(label)!r}. '
               'Please ensure you used the same pixel connectivity '
               'in detect_sources and deblend_sources.')
        raise ValueError(msg)
    if status == 1:  # no deblending
        return None
    return output_arr
