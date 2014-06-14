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

#BTYPE = np.int16
#ctypedef np.int_t BTYPE_t

__all__ = ['_masked_median_filter']

#ctypedef fused numeric_array:
#    np.ndarray[np.int_t, ndim=2]
#    np.ndarray[np.float32_t, ndim=2]
#    np.ndarray[np.float64_t, ndim=2]


#def _masked_median_filter(np.ndarray[DTYPE_t, ndim=2] array, np.ndarray[DTYPE_t, ndim=2] mask, int size):
def _masked_median_filter(np.ndarray[DTYPE_t, ndim=2] array, np.ndarray[np.int_t, ndim=2] mask, int size):
#def _masked_median_filter(numeric_array array, numeric_array mask, int size):
    # this is likely to be slow -> cython
    cdef int nx = array.shape[1]
    cdef int ny = array.shape[0]
    #cdef unsigned int i, j, ii, jj, minx, maxx, miny, maxy
    cdef int i, j, ii, jj
    cdef minx, maxx, miny, maxy
    cdef np.ndarray[DTYPE_t, ndim=2] result = np.zeros([ny, nx],
                                                       dtype=DTYPE)
    for jj in range(ny):
        for ii in range(nx):
            # NOTE:  this simply clips image at image boundaries
            #minx = max([ii - size/2, 0])
            #minx = ii - size/2
            #minx = size / 2
            #minx = 4 // 2
            #minx = ii
            minx, maxx = max([ii - size//2, 0]), min([ii + size//2 + 1, nx])
            miny, maxy = max([jj - size//2, 0]), min([jj + size//2 + 1, ny])
            image_region = array[miny:maxy, minx:maxx]
            mask_region = mask[miny:maxy, minx:maxx]
            #cdef np.ndarray[DTYPE_t, ndim=2] image_region = array[miny:maxy, minx:maxx]
            #cdef np.ndarray[DTYPE_t, ndim=2] mask_region = mask[miny:maxy, minx:maxx]
            #print(jj, ii)
            #result[jj, ii] = np.median(image_region[mask_region == 0])
            result[jj, ii] = np.mean(image_region[mask_region == 0])
            #result[jj, ii] = np.median(image_region)
            #result[jj, ii] = 1.0
    return result
