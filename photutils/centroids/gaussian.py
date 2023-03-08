# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
The module contains tools for centroiding sources using Gaussians.
"""

import warnings

import numpy as np
from astropy.modeling.fitting import LevMarLSQFitter
from astropy.modeling.models import Const1D, Const2D, Gaussian1D, Gaussian2D
from astropy.utils.exceptions import AstropyUserWarning

__all__ = ['centroid_1dg', 'centroid_2dg']


def centroid_1dg(data, error=None, mask=None):
    """
    Calculate the centroid of a 2D array by fitting 1D Gaussians to the
    marginal ``x`` and ``y`` distributions of the array.

    Non-finite values (e.g., NaN or inf) in the ``data`` or ``error``
    arrays are automatically masked. These masks are combined.

    Parameters
    ----------
    data : 2D `~numpy.ndarray`
        The 2D image data. The image should be a background-subtracted
        cutout image containing a single source.

    error : 2D `~numpy.ndarray`, optional
        The 2D array of the 1-sigma errors of the input ``data``.

    mask : 2D bool `~numpy.ndarray`, optional
        A boolean mask, with the same shape as ``data``, where a `True`
        value indicates the corresponding element of ``data`` is masked.

    Returns
    -------
    centroid : `~numpy.ndarray`
        The ``x, y`` coordinates of the centroid.
    """
    data = np.ma.asanyarray(data)

    if mask is not None and mask is not np.ma.nomask:
        mask = np.asanyarray(mask)
        if data.shape != mask.shape:
            raise ValueError('data and mask must have the same shape.')
        data.mask |= mask

    if np.any(~np.isfinite(data)):
        data = np.ma.masked_invalid(data)
        warnings.warn('Input data contains non-finite values (e.g., NaN or '
                      'inf) that were automatically masked.',
                      AstropyUserWarning)

    if error is not None:
        error = np.ma.masked_invalid(error)
        if data.shape != error.shape:
            raise ValueError('data and error must have the same shape.')
        data.mask |= error.mask
        error.mask = data.mask

        xy_error = [np.sqrt(np.ma.sum(error**2, axis=i)) for i in (0, 1)]
        xy_weights = [(1.0 / xy_error[i].clip(min=1.0e-30)) for i in (0, 1)]
    else:
        xy_weights = [np.ones(data.shape[i]) for i in (1, 0)]

    # assign zero weight where an entire row or column is masked
    if np.any(data.mask):
        bad_idx = [np.all(data.mask, axis=i) for i in (0, 1)]
        for i in (0, 1):
            xy_weights[i][bad_idx[i]] = 0.0

    xy_data = [np.ma.sum(data, axis=i).data for i in (0, 1)]

    constant_init = np.ma.min(data)
    centroid = []
    for (data_i, weights_i) in zip(xy_data, xy_weights):
        params_init = _gaussian1d_moments(data_i)
        g_init = Const1D(constant_init) + Gaussian1D(*params_init)
        fitter = LevMarLSQFitter()
        x = np.arange(data_i.size)
        g_fit = fitter(g_init, x, data_i, weights=weights_i)
        centroid.append(g_fit.mean_1.value)

    return np.array(centroid)


def _gaussian1d_moments(data, mask=None):
    """
    Estimate 1D Gaussian parameters from the moments of 1D data.

    This function can be useful for providing initial parameter values
    when fitting a 1D Gaussian to the ``data``.

    Parameters
    ----------
    data : 1D `~numpy.ndarray`
        The 1D data array.

    mask : 1D bool `~numpy.ndarray`, optional
        A boolean mask, with the same shape as ``data``, where a `True`
        value indicates the corresponding element of ``data`` is masked.

    Returns
    -------
    amplitude, mean, stddev : float
        The estimated parameters of a 1D Gaussian.
    """
    if np.any(~np.isfinite(data)):
        data = np.ma.masked_invalid(data)
        warnings.warn('Input data contains non-finite values (e.g., NaN or '
                      'inf) that were automatically masked.',
                      AstropyUserWarning)
    else:
        data = np.ma.array(data)

    if mask is not None and mask is not np.ma.nomask:
        mask = np.asanyarray(mask)
        if data.shape != mask.shape:
            raise ValueError('data and mask must have the same shape.')
        data.mask |= mask

    data.fill_value = 0.0
    data = data.filled()

    x = np.arange(data.size)
    x_mean = np.sum(x * data) / np.sum(data)
    x_stddev = np.sqrt(abs(np.sum(data * (x - x_mean) ** 2) / np.sum(data)))
    amplitude = np.ptp(data)

    return amplitude, x_mean, x_stddev


def centroid_2dg(data, error=None, mask=None):
    """
    Calculate the centroid of a 2D array by fitting a 2D Gaussian (plus
    a constant) to the array.

    Non-finite values (e.g., NaN or inf) in the ``data`` or ``error``
    arrays are automatically masked. These masks are combined.

    Parameters
    ----------
    data : 2D `~numpy.ndarray`
        The 2D image data. The image should be a background-subtracted
        cutout image containing a single source.

    error : 2D `~numpy.ndarray`, optional
        The 2D array of the 1-sigma errors of the input ``data``.

    mask : 2D bool `~numpy.ndarray`, optional
        A boolean mask, with the same shape as ``data``, where a `True`
        value indicates the corresponding element of ``data`` is masked.

    Returns
    -------
    centroid : `~numpy.ndarray`
        The ``x, y`` coordinates of the centroid.
    """
    # prevent circular import
    from photutils.morphology import data_properties

    data = np.ma.asanyarray(data)

    if mask is not None and mask is not np.ma.nomask:
        mask = np.asanyarray(mask)
        if data.shape != mask.shape:
            raise ValueError('data and mask must have the same shape.')
        data.mask |= mask

    if np.any(~np.isfinite(data)):
        data = np.ma.masked_invalid(data)
        warnings.warn('Input data contains non-finite values (e.g., NaN or '
                      'inf) that were automatically masked.',
                      AstropyUserWarning)

    if error is not None:
        error = np.ma.masked_invalid(error)
        if data.shape != error.shape:
            raise ValueError('data and error must have the same shape.')
        data.mask |= error.mask
        weights = 1.0 / error.clip(min=1.0e-30)
    else:
        weights = np.ones(data.shape)

    if np.ma.count(data) < 7:
        raise ValueError('Input data must have a least 7 unmasked values to '
                         'fit a 2D Gaussian plus a constant.')

    # assign zero weight to masked pixels
    if data.mask is not np.ma.nomask:
        weights[data.mask] = 0.0

    mask = data.mask
    data.fill_value = 0.0
    data = data.filled()

    # Subtract the minimum of the data as a rough background estimate.
    # This will also make the data values positive, preventing issues with
    # the moment estimation in data_properties. Moments from negative data
    # values can yield undefined Gaussian parameters, e.g., x/y_stddev.
    props = data_properties(data - np.min(data), mask=mask)

    constant_init = 0.0  # subtracted data minimum above
    g_init = (Const2D(constant_init)
              + Gaussian2D(amplitude=np.ptp(data),
                           x_mean=props.xcentroid,
                           y_mean=props.ycentroid,
                           x_stddev=props.semimajor_sigma.value,
                           y_stddev=props.semiminor_sigma.value,
                           theta=props.orientation.value))
    fitter = LevMarLSQFitter()
    y, x = np.indices(data.shape)
    gfit = fitter(g_init, x, y, data, weights=weights)
    return np.array([gfit.x_mean_1.value, gfit.y_mean_1.value])
