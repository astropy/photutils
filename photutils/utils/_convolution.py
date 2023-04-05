# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
This module provides tools for convolving images with a kernel.
"""

import warnings

import numpy as np
from astropy.convolution import Kernel2D
from astropy.units import Quantity
from astropy.utils.exceptions import AstropyUserWarning


def _filter_data(data, kernel, mode='constant', fill_value=0.0,
                 check_normalization=False):
    """
    Convolve a 2D image with a 2D kernel.

    The kernel may either be a 2D `~numpy.ndarray` or a
    `~astropy.convolution.Kernel2D` object.

    Parameters
    ----------
    data : array_like
        The 2D array of the image.

    kernel : array_like (2D) or `~astropy.convolution.Kernel2D`
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

    Returns
    -------
    result : `~numpy.ndarray`
        The convolved image.
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

        # scipy.ndimage.convolve currently strips units, but be explicit
        # in case that behavior changes
        unit = None
        if isinstance(data, Quantity):
            unit = data.unit
            data = data.value

        # NOTE: if data is int and kernel is float, ndimage.convolve
        # will return an int image. If the data dtype is int, we make the
        # data float so that a float image is always returned
        if np.issubdtype(data.dtype, np.integer):
            data = data.astype(float)

        # NOTE: astropy.convolution.convolve fails with zero-sum kernels
        # (used in findstars) (cf. astropy #1647)
        result = ndimage.convolve(data, kernel_array, mode=mode,
                                  cval=fill_value)

        # reapply the input unit
        if unit is not None:
            result <<= unit

        return result
    else:
        return data
