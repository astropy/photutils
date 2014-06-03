# Licensed under a 3-clause BSD style license - see LICENSE.rst
#cython: boundscheck=False
#cython: wraparound=False
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import numpy as np

cimport cython
cimport numpy as np
DTYPE = np.float
ctypedef np.float_t DTYPE_t


def downsample(np.ndarray[DTYPE_t, ndim=2] array, int factor):
    cdef int nx = array.shape[1]
    cdef int ny = array.shape[0]
    cdef int nx_new = nx // factor
    cdef int ny_new = ny // factor
    cdef unsigned int i, j, ii, jj
    cdef np.ndarray[DTYPE_t, ndim=2] result = np.zeros([ny_new, nx_new],
                                                       dtype=DTYPE)
    for i in range(nx_new):
        for j in range(ny_new):
            for ii in range(factor):
                for jj in range(factor):
                    result[j, i] += array[j * factor + jj, i * factor + ii]
    return result
