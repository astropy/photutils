# Licensed under a 3-clause BSD style license - see LICENSE.rst
# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
# cython: freethreading_compatible=True
"""
Cython marker-based watershed kernel for source deblending.

This implements the classic priority-flood watershed (Soille 1990)
used for deblending: pixels are flooded from the markers in order of
increasing image value, with the queue-entry age breaking ties so
that plateaus are split between the markers that reach them first.
The algorithm, the neighbor ordering (orthogonal neighbors before
diagonal ones, each group in raster order), and the tie-breaking
match ``skimage.segmentation.watershed`` (with ``compactness=0``
and ``watershed_line=False``), so the results are identical, but
without the per-call validation, padding, and cropping overhead of the
general-purpose function, which dominates for small cutouts.

The flood core runs without the GIL and uses no global mutable state,
so this module is safe to use from multiple threads, including on
free-threaded Python builds.
"""

import numpy as np

from libc.stdlib cimport free, malloc

__all__ = ['deblend_watershed']


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

    Parameters
    ----------
    image : 2D `~numpy.ndarray`
        The image to flood (the lowest values are flooded first).

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
