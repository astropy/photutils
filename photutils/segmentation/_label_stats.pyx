# Licensed under a 3-clause BSD style license - see LICENSE.rst
# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
# cython: freethreading_compatible=True
"""
Cython kernel that measures the labels of a segmentation array.

The pixel count and the bounding box of every label are accumulated in
one pass over the array. Each row is walked as runs of equal values, so
the per-label arrays are updated once per run instead of once per pixel.

The per-label arrays are indexed by label value, so their size is set
by the largest label and not by the number of labels. The caller gives
an upper limit of the largest label that it accepts, and falls back to
a sort-based method for arrays with larger labels.

The kernel runs without the GIL and uses no global mutable state, so
this module is safe to use from multiple threads, including on
free-threaded Python builds.
"""

import numpy as np

__all__ = ['label_stats']

ctypedef fused segm_t:
    int
    long long


def label_stats(const segm_t[:, ::1] segm, long long max_label_limit):
    """
    Calculate the pixel count and bounding box of every label in a
    segmentation array.

    Parameters
    ----------
    segm : 2D int32 or int64 `~numpy.ndarray`
        The C-contiguous segmentation array. A value of zero is the
        background.

    max_label_limit : int
        The largest label value for which the per-label arrays are
        allocated.

    Returns
    -------
    min_value, max_value : int
        The minimum and maximum values in ``segm``. Both are zero for
        an array of zero size.

    counts : 1D intp `~numpy.ndarray` or `None`
        The number of pixels of each label value, indexed by the label
        value. The background element is zero. `None` if ``segm`` has
        a negative value or a value larger than ``max_label_limit``.
        The label arrays below are then also `None`.

    ymin, ymax, xmin, xmax : 1D intp `~numpy.ndarray` or `None`
        The inclusive bounds of the minimal bounding box of each label
        value, indexed by the label value. The elements of the values
        that are not present are undefined.
    """
    cdef Py_ssize_t ny = segm.shape[0]
    cdef Py_ssize_t nx = segm.shape[1]
    cdef Py_ssize_t n_pix = ny * nx
    cdef Py_ssize_t i, y, x, x0
    cdef segm_t value, vmin, vmax
    cdef const segm_t* segm_ptr
    cdef const segm_t* row

    if n_pix == 0:
        return 0, 0, None, None, None, None, None
    segm_ptr = &segm[0, 0]

    vmin = segm_ptr[0]
    vmax = segm_ptr[0]
    with nogil:
        for i in range(n_pix):
            value = segm_ptr[i]
            if value < vmin:
                vmin = value
            if value > vmax:
                vmax = value

    if vmin < 0 or vmax > max_label_limit:
        return int(vmin), int(vmax), None, None, None, None, None

    counts_arr = np.zeros(vmax + 1, dtype=np.intp)
    ymin_arr = np.empty(vmax + 1, dtype=np.intp)
    ymax_arr = np.empty(vmax + 1, dtype=np.intp)
    xmin_arr = np.full(vmax + 1, nx, dtype=np.intp)
    xmax_arr = np.full(vmax + 1, -1, dtype=np.intp)
    cdef Py_ssize_t[::1] counts = counts_arr
    cdef Py_ssize_t[::1] ymin = ymin_arr
    cdef Py_ssize_t[::1] ymax = ymax_arr
    cdef Py_ssize_t[::1] xmin = xmin_arr
    cdef Py_ssize_t[::1] xmax = xmax_arr

    with nogil:
        for y in range(ny):
            row = segm_ptr + y * nx
            x = 0
            while x < nx:
                value = row[x]
                if value == 0:
                    x += 1
                    continue

                # A run of equal labels
                x0 = x
                x += 1
                while x < nx and row[x] == value:
                    x += 1

                # The rows are visited in increasing order, so the
                # first run of a label is on its first row
                if counts[value] == 0:
                    ymin[value] = y
                counts[value] += x - x0
                ymax[value] = y
                if x0 < xmin[value]:
                    xmin[value] = x0
                if x - 1 > xmax[value]:
                    xmax[value] = x - 1

    return (int(vmin), int(vmax), counts_arr, ymin_arr, ymax_arr, xmin_arr,
            xmax_arr)
