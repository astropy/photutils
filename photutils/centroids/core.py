# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Functions for centroiding sources and measuring their morphological
properties.
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import warnings

import numpy as np
from astropy.modeling import Fittable2DModel, Parameter
from astropy.modeling.models import (Gaussian1D, Gaussian2D, Const1D,
                                     Const2D, CONSTRAINTS_DOC)
from astropy.modeling.fitting import LevMarLSQFitter
from astropy.utils.exceptions import AstropyUserWarning

from ..morphology import data_properties


__all__ = ['GaussianConst2D', 'centroid_com', 'gaussian1d_moments',
           'fit_2dgaussian', 'centroid_1dg', 'centroid_2dg']


class _GaussianConst1D(Const1D + Gaussian1D):
    """A model for a 1D Gaussian plus a constant."""


class GaussianConst2D(Fittable2DModel):
    """
    A model for a 2D Gaussian plus a constant.

    Parameters
    ----------
    constant : float
        Value of the constant.
    amplitude : float
        Amplitude of the Gaussian.
    x_mean : float
        Mean of the Gaussian in x.
    y_mean : float
        Mean of the Gaussian in y.
    x_stddev : float
        Standard deviation of the Gaussian in x.
        ``x_stddev`` and ``y_stddev`` must be specified unless a covariance
        matrix (``cov_matrix``) is input.
    y_stddev : float
        Standard deviation of the Gaussian in y.
        ``x_stddev`` and ``y_stddev`` must be specified unless a covariance
        matrix (``cov_matrix``) is input.
    theta : float, optional
        Rotation angle in radians. The rotation angle increases
        counterclockwise.
    """

    constant = Parameter(default=1)
    amplitude = Parameter(default=1)
    x_mean = Parameter(default=0)
    y_mean = Parameter(default=0)
    x_stddev = Parameter(default=1)
    y_stddev = Parameter(default=1)
    theta = Parameter(default=0)

    @staticmethod
    def evaluate(x, y, constant, amplitude, x_mean, y_mean, x_stddev,
                 y_stddev, theta):
        """Two dimensional Gaussian plus constant function."""

        model = Const2D(constant)(x, y) + Gaussian2D(amplitude, x_mean,
                                                     y_mean, x_stddev,
                                                     y_stddev, theta)(x, y)
        return model


GaussianConst2D.__doc__ += CONSTRAINTS_DOC


def centroid_com(data, mask=None):
    """
    Calculate the centroid of a 2D array as its "center of mass"
    determined from image moments.

    Invalid values (e.g. NaNs or infs) in the ``data`` array are
    automatically masked.

    Parameters
    ----------
    data : array_like
        The 2D array of the image.

    mask : array_like (bool), optional
        A boolean mask, with the same shape as ``data``, where a `True`
        value indicates the corresponding element of ``data`` is masked.

    Returns
    -------
    centroid : `~numpy.ndarray`
        The ``x, y`` coordinates of the centroid.
    """

    from skimage.measure import moments

    if np.any(~np.isfinite(data)):
        data = np.ma.masked_invalid(data)
        warnings.warn('Input data contains input values (e.g. NaNs or infs), '
                      'which were automatically masked.', AstropyUserWarning)
    else:
        data = np.ma.array(data)

    if mask is not None and mask is not np.ma.nomask:
        mask = np.asanyarray(mask)
        if data.shape != mask.shape:
            raise ValueError('data and mask must have the same shape.')
        data.mask |= mask

    # Convert the data to a float64 (double) `numpy.ndarray`,
    # which is required for input to `skimage.measure.moments`.
    data = data.astype(np.float)
    data.fill_value = 0.
    data = data.filled()

    m = moments(data, 1)
    xcen = m[1, 0] / m[0, 0]
    ycen = m[0, 1] / m[0, 0]

    return np.array([xcen, ycen])


def gaussian1d_moments(data, mask=None):
    """
    Estimate 1D Gaussian parameters from the moments of 1D data.

    This function can be useful for providing initial parameter values
    when fitting a 1D Gaussian to the ``data``.

    Parameters
    ----------
    data : array_like (1D)
        The 1D array.

    mask : array_like (1D bool), optional
        A boolean mask, with the same shape as ``data``, where a `True`
        value indicates the corresponding element of ``data`` is masked.

    Returns
    -------
    amplitude, mean, stddev : float
        The estimated parameters of a 1D Gaussian.
    """

    if mask is not None and mask is not np.ma.nomask:
        mask = np.asanyarray(mask)
        if data.shape != mask.shape:
            raise ValueError('data and mask must have the same shape.')
        data = data.copy()
        data[mask] = 0.

    x = np.arange(data.size)
    x_mean = np.sum(x * data) / np.sum(data)
    x_stddev = np.sqrt(abs(np.sum(data * (x - x_mean)**2) / np.sum(data)))
    amplitude = np.nanmax(data) - np.nanmin(data)

    return amplitude, x_mean, x_stddev


def fit_2dgaussian(data, error=None, mask=None):
    """
    Fit a 2D Gaussian plus a constant to a 2D image.

    Invalid values (e.g. NaNs or infs) in the ``data`` or ``error``
    arrays are automatically masked.  The mask for invalid values
    represents the combination of the invalid-value masks for the
    ``data`` and ``error`` arrays.

    Parameters
    ----------
    data : array_like
        The 2D array of the image.

    error : array_like, optional
        The 2D array of the 1-sigma errors of the input ``data``.

    mask : array_like (bool), optional
        A boolean mask, with the same shape as ``data``, where a `True`
        value indicates the corresponding element of ``data`` is masked.

    Returns
    -------
    result : A `GaussianConst2D` model instance.
        The best-fitting Gaussian 2D model.
    """

    if np.any(~np.isfinite(data)):
        data = np.ma.masked_invalid(data)
        warnings.warn('Input data contains input values (e.g. NaNs or infs), '
                      'which were automatically masked.', AstropyUserWarning)
    else:
        data = np.ma.array(data)

    if mask is not None and mask is not np.ma.nomask:
        mask = np.asanyarray(mask)
        if data.shape != mask.shape:
            raise ValueError('data and mask must have the same shape.')
        data.mask |= mask

    if error is not None:
        error = np.ma.masked_invalid(error)
        if data.shape != error.shape:
            raise ValueError('data and error must have the same shape.')
        data.mask |= error.mask
        weights = 1.0 / error
    else:
        weights = np.ones(data.shape)

    if np.ma.count(data) < 7:
        raise ValueError('Input data must have a least 7 unmasked values to '
                         'fit a 2D Gaussian plus a constant.')

    # down-weight masked pixels
    if data.mask is not np.ma.nomask:
        weights[data.mask] = 1.e-30

    # Subtract the minimum of the data as a crude background estimate.
    # This will also make the data values positive, preventing issues with
    # the moment estimation in data_properties (moments from negative data
    # values can yield undefined Gaussian parameters, e.g. x/y_stddev).
    props = data_properties(data.data - np.ma.min(data), mask=data.mask)

    init_const = 0.    # subtracted data minimum above
    init_amplitude = np.ma.max(data) - np.ma.min(data)
    g_init = GaussianConst2D(constant=init_const, amplitude=init_amplitude,
                             x_mean=props.xcentroid.value,
                             y_mean=props.ycentroid.value,
                             x_stddev=props.semimajor_axis_sigma.value,
                             y_stddev=props.semiminor_axis_sigma.value,
                             theta=props.orientation.value)
    fitter = LevMarLSQFitter()
    y, x = np.indices(data.shape)
    gfit = fitter(g_init, x, y, data.data, weights=weights)

    return gfit


def centroid_1dg(data, error=None, mask=None):
    """
    Calculate the centroid of a 2D array by fitting 1D Gaussians to the
    marginal ``x`` and ``y`` distributions of the array.

    Invalid values (e.g. NaNs or infs) in the ``data`` or ``error``
    arrays are automatically masked.  The mask for invalid values
    represents the combination of the invalid-value masks for the
    ``data`` and ``error`` arrays.

    Parameters
    ----------
    data : array_like
        The 2D data array.

    error : array_like, optional
        The 2D array of the 1-sigma errors of the input ``data``.

    mask : array_like (bool), optional
        A boolean mask, with the same shape as ``data``, where a `True`
        value indicates the corresponding element of ``data`` is masked.

    Returns
    -------
    centroid : `~numpy.ndarray`
        The ``x, y`` coordinates of the centroid.
    """

    if np.any(~np.isfinite(data)):
        data = np.ma.masked_invalid(data)
        warnings.warn('Input data contains input values (e.g. NaNs or infs), '
                      'which were automatically masked.', AstropyUserWarning)
    else:
        data = np.ma.array(data)

    if mask is not None and mask is not np.ma.nomask:
        mask = np.asanyarray(mask)
        if data.shape != mask.shape:
            raise ValueError('data and mask must have the same shape.')
        data.mask |= mask

    if error is not None:
        error = np.ma.masked_invalid(error)
        if data.shape != error.shape:
            raise ValueError('data and error must have the same shape.')
        data.mask |= error.mask
    else:
        error = np.ma.masked_array(np.ones_like(data))

    xy_data = np.array([np.ma.sum(data, axis=i) for i in [0, 1]])

    error.mask = data.mask
    error.fill_value = 1.e5
    error = error.filled()
    xy_error = np.array([np.sqrt(np.ma.sum(error**2, axis=i))
                         for i in [0, 1]])

    xy_weights = [(1.0 / xy_error[i].clip(min=1.e-30)) for i in [0, 1]]

    constant_init = np.min(data)
    centroid = []
    for (data_i, weights_i) in zip(xy_data, xy_weights):
        params_init = gaussian1d_moments(data_i)
        g_init = _GaussianConst1D(constant_init, *params_init)
        fitter = LevMarLSQFitter()
        x = np.arange(data_i.size)
        g_fit = fitter(g_init, x, data_i, weights=weights_i)
        centroid.append(g_fit.mean_1.value)

    return np.array(centroid)


def centroid_2dg(data, error=None, mask=None):
    """
    Calculate the centroid of a 2D array by fitting a 2D Gaussian (plus
    a constant) to the array.

    Invalid values (e.g. NaNs or infs) in the ``data`` or ``error``
    arrays are automatically masked.  The mask for invalid values
    represents the combination of the invalid-value masks for the
    ``data`` and ``error`` arrays.

    Parameters
    ----------
    data : array_like
        The 2D data array.

    error : array_like, optional
        The 2D array of the 1-sigma errors of the input ``data``.

    mask : array_like (bool), optional
        A boolean mask, with the same shape as ``data``, where a `True`
        value indicates the corresponding element of ``data`` is masked.

    Returns
    -------
    centroid : `~numpy.ndarray`
        The ``x, y`` coordinates of the centroid.
    """

    gfit = fit_2dgaussian(data, error=error, mask=mask)

    return np.array([gfit.x_mean.value, gfit.y_mean.value])
