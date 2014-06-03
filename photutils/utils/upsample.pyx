# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import numpy as np
cimport numpy as np

DTYPE = np.float
ctypedef np.float_t DTYPE_t
cimport cython


def upsample(np.ndarray[DTYPE_t, ndim=2] array, int factor):
    cdef int nx = array.shape[1]
    cdef int ny = array.shape[0]
    cdef int nx_new = nx * factor
    cdef int ny_new = ny * factor
    cdef unsigned int i, j
    cdef np.ndarray[DTYPE_t, ndim=2] result = np.zeros((ny_new, nx_new), dtype=DTYPE)
    cdef float factor_sq = factor * factor

    for i in range(nx_new):
        for j in range(ny_new):
            result[j, i] += array[int(j / 2), int(i / 2)] / factor_sq
    return result
