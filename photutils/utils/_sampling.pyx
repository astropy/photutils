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

__all__ = ['_downsample', '_upsample']


def _downsample(np.ndarray[DTYPE_t, ndim=2] array, int block_size):
    """See docstring of wrapper in sampling.py."""
    cdef int nx = array.shape[1]
    cdef int ny = array.shape[0]
    cdef int nx_new = nx // block_size
    cdef int ny_new = ny // block_size
    cdef unsigned int i, j, ii, jj
    cdef np.ndarray[DTYPE_t, ndim=2] result = np.zeros([ny_new, nx_new],
                                                       dtype=DTYPE)
    for i in range(nx_new):
        for j in range(ny_new):
            for ii in range(block_size):
                for jj in range(block_size):
                    result[j, i] += array[j * block_size + jj,
                                          i * block_size + ii]
    return result


def _upsample(np.ndarray[DTYPE_t, ndim=2] array, int block_size):
    """See docstring of wrapper in sampling.py."""
    cdef int nx = array.shape[1]
    cdef int ny = array.shape[0]
    cdef int nx_new = nx * block_size
    cdef int ny_new = ny * block_size
    cdef unsigned int i, j
    cdef np.ndarray[DTYPE_t, ndim=2] result = np.zeros((ny_new, nx_new),
                                                       dtype=DTYPE)
    cdef float block_size_sq = block_size * block_size
    for i in range(nx_new):
        for j in range(ny_new):
            result[j, i] += array[int(j / block_size),
                                  int(i / block_size)] / block_size_sq
    return result
