# Licensed under a 3-clause BSD style license - see LICENSE.rst

import warnings

import numpy as np
from astropy.convolution import Kernel2D
from astropy.utils.exceptions import AstropyUserWarning


__all__ = ['filter_data']


def filter_data(data, kernel, mode='constant', fill_value=0.0,
                check_normalization=False):
    """
    Convolve a 2D image with a 2D kernel.

    The kernel may either be a 2D `~numpy.ndarray` or a
    `~astropy.convolution.Kernel2D` object.

    Parameters
    ----------
    data : array_like
        The 2D array of the image.

    kernel : array-like (2D) or `~astropy.convolution.Kernel2D`
        The 2D kernel used to filter the input ``data``. Filtering the
        ``data`` will smooth the noise and maximize detectability of
        objects with a shape similar to the kernel.

    mode : {'constant', 'reflect', 'nearest', 'mirror', 'wrap'}, optional
        The ``mode`` determines how the array borders are handled.  For
        the ``'constant'`` mode, values outside the array borders are
        set to ``fill_value``.  The default is ``'constant'``.

    fill_value : scalar, optional
        Value to fill data values beyond the array borders if ``mode``
        is ``'constant'``.  The default is ``0.0``.

    check_normalization : bool, optional
        If `True` then a warning will be issued if the kernel is not
        normalized to 1.
    """

    from scipy import ndimage

    if kernel is not None:
        if isinstance(kernel, Kernel2D):
            kernel_array = kernel.array
        else:
            kernel_array = kernel

        if check_normalization:
            if not np.allclose(np.sum(kernel_array), 1.0):
                warnings.warn('The kernel is not normalized.',
                              AstropyUserWarning)

        # NOTE:  astropy.convolution.convolve fails with zero-sum
        # kernels (used in findstars) (cf. astropy #1647)
        return ndimage.convolve(data, kernel_array, mode=mode,
                                cval=fill_value)
    else:
        return data
