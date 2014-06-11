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


def _downsample(np.ndarray[DTYPE_t, ndim=2] array, int factor):
    """
    Downsample an image by combining image pixels.  This process
    conserves image flux.  If the dimensions of `image` are
    not a whole-multiple of `factor`, the extra rows/columns will
    not be included in the output downsampled image.

    Parameters
    ----------
    image : array_like, float
        The 2D array of the image.

    factor : int
        Downsampling integer factor along each axis.

    Returns
    -------
    output : array_like
        The 2D array of the resampled image.

    Examples
    --------
    >>> from photutils import utils
    >>> img = np.arange(16.).reshape((4, 4))
    >>> utils.downsample(img, 2)
    array([[ 10.,  18.],
           [ 42.,  50.]])
    """

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


def _upsample(np.ndarray[DTYPE_t, ndim=2] array, int factor):
    """
    Upsample an image by replicating image pixels.  This process
    conserves image flux.

    Parameters
    ----------
    image : array_like, float
        The 2D array of the image.

    factor : int
        Upsampling integer factor along each axis.

    Returns
    -------
    output : array_like
        The 2D array of the resampled image.

    Examples
    --------
    >>> from photutils import utils
    >>> img = np.array([[0., 1.], [2., 3.]])
    >>> utils.upsample(img, 2)
    array([[ 0.  ,  0.  ,  0.25,  0.25],
           [ 0.  ,  0.  ,  0.25,  0.25],
           [ 0.5 ,  0.5 ,  0.75,  0.75],
           [ 0.5 ,  0.5 ,  0.75,  0.75]])
    """

    cdef int nx = array.shape[1]
    cdef int ny = array.shape[0]
    cdef int nx_new = nx * factor
    cdef int ny_new = ny * factor
    cdef unsigned int i, j
    cdef np.ndarray[DTYPE_t, ndim=2] result = np.zeros((ny_new, nx_new),
                                                       dtype=DTYPE)
    cdef float factor_sq = factor * factor
    for i in range(nx_new):
        for j in range(ny_new):
            result[j, i] += array[int(j / factor), int(i / factor)] / factor_sq
    return result
