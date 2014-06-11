# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import numpy as np
from ._sampling import _downsample, _upsample

__all__ = ['downsample', 'upsample']


def downsample(array, factor):
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
    >>> img = np.arange(16.).reshape((4,4))
    >>> utils.downsample(img, 2)
    array([[ 10.,  18.],
           [ 42.,  50.]])
    """

    return _downsample(array, factor)


def upsample(array, factor):
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

    return _upsample(array, factor)
