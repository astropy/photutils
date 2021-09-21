# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
This module contains tools to validate and handle photometry inputs.
"""


import numpy as np


def _validate_inputs(data, error):
    """
    Validate inputs.

    ``data`` and ``error`` are converted to a `~numpy.ndarray`, if
    necessary.

    Used to parse inputs to `~photutils.aperture.aperture_photometry`
    and `~photutils.aperture.PixelAperture.do_photometry`.
    """
    data = np.asanyarray(data)
    if data.ndim != 2:
        raise ValueError('data must be a 2D array.')

    if error is not None:
        error = np.asanyarray(error)
        if error.shape != data.shape:
            raise ValueError('error and data must have the same shape.')

    return data, error


def _handle_units(data, error):
    """
    Handle Quantity inputs.

    Any units on ``data`` and ``error` are removed.  ``data`` and
    ``error`` are returned as `~numpy.ndarray`.  The returned ``unit``
    represents the unit for both ``data`` and ``error``.

    Used to parse inputs to `~photutils.aperture.aperture_photometry`
    and `~photutils.aperture.PixelAperture.do_photometry`.
    """
    # check Quantity inputs
    inputs = (data, error)
    has_unit = [hasattr(x, 'unit') for x in inputs if x is not None]
    use_units = all(has_unit)
    if any(has_unit) and not use_units:
        raise ValueError('If data or error has units, then they both must '
                         'have the same units.')

    # strip data and error units for performance
    if use_units:
        unit = data.unit
        data = data.value

        if error is not None:
            error = error.value
    else:
        unit = None

    return data, error, unit


def _prepare_photometry_data(data, error, mask):
    """
    Prepare data and error arrays for photometry.

    Error is converted to variance and masked values are set to zero in
    the output data and variance arrays.

    Used to parse inputs to `~photutils.aperture.aperture_photometry`
    and `~photutils.aperture.PixelAperture.do_photometry`.

    Parameters
    ----------
    data : `~numpy.ndarray`
        The 2D array on which to perform photometry.

    error : `~numpy.ndarray` or `None`
        The pixel-wise Gaussian 1-sigma errors of the input ``data``.

    mask : array_like (bool) or `None`
        A boolean mask with the same shape as ``data`` where a `True`
        value indicates the corresponding element of ``data`` is masked.

    Returns
    -------
    data : `~numpy.ndarray`
        The 2D array on which to perform photometry, where masked values
        have been set to zero.

    variance : `~numpy.ndarray` or `None`
        The pixel-wise Gaussian 1-sigma variance of the input ``data``,
        where masked values have been set to zero.
    """
    if error is not None:
        variance = error ** 2
    else:
        variance = None

    if mask is not None:
        mask = np.asanyarray(mask)
        if mask.shape != data.shape:
            raise ValueError('mask and data must have the same shape.')

        data = data.copy()  # do not modify input data
        data[mask] = 0.

        if variance is not None:
            variance[mask] = 0.

    return data, variance
