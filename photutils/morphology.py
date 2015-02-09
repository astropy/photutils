# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Functions for centroiding sources and measuring their morphological
properties.
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import numpy as np
from astropy.modeling.models import Gaussian1D, Gaussian2D, Const1D, Const2D
from astropy.modeling.fitting import LevMarLSQFitter
from .segmentation import SegmentProperties


__all__ = ['centroid_com', 'gaussian1d_moments', 'marginalize_data2d',
           'centroid_1dg', 'centroid_2dg', 'fit_2dgaussian',
           'data_properties', 'GaussianConst2D']


class GaussianConst1D(Const1D + Gaussian1D):
    """A 1D Gaussian plus a constant."""


class GaussianConst2D(Const2D + Gaussian2D):
    """A 2D Gaussian plus a constant."""


def _convert_image(data, mask=None):
    """
    Convert the input data to a float64 (double) `numpy.ndarray`,
    required for input to `skimage.measure.moments` and
    `skimage.measure.moments_central`.

    The input ``data`` is copied unless it already has that
    `numpy.dtype`.

    If ``mask`` is input, then masked pixels are set to zero in the
    output ``data``.

    Parameters
    ----------
    data : array_like
        The 2D array of the image.

    mask : array_like (bool), optional
        A boolean mask, with the same shape as ``data``, where a `True`
        value indicates the corresponding element of ``data`` is masked.
        Masked pixels are set to zero in the output ``data``.

    Returns
    -------
    image : `numpy.ndarray`, float64
        The converted 2D array of the image, where masked pixels have
        been set to zero.
    """

    try:
        if mask is None:
            copy = False
        else:
            copy = True
        image = np.asarray(data).astype(np.float, copy=copy)
    except TypeError:    # pragma: no cover
        image = np.asarray(data).astype(np.float)    # for numpy <= 1.6
    if mask is not None:
        mask = np.asanyarray(mask)
        if data.shape != mask.shape:
            raise ValueError('data and mask must have the same shape')
        image[mask] = 0.0
    return image


def centroid_com(data, mask=None):
    """
    Calculate the centroid of a 2D array as its center of mass
    determined from image moments.

    Parameters
    ----------
    data : array_like
        The 2D array of the image.

    mask : array_like (bool), optional
        A boolean mask, with the same shape as ``data``, where a `True`
        value indicates the corresponding element of ``data`` is masked.

    Returns
    -------
    xcen, ycen : float
        (x, y) coordinates of the centroid.
    """

    from skimage.measure import moments
    data = _convert_image(data, mask=mask)
    m = moments(data, 1)
    xcen = m[1, 0] / m[0, 0]
    ycen = m[0, 1] / m[0, 0]
    return xcen, ycen


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

    if mask is not None:
        mask = np.asanyarray(mask)
        data = data.copy()
        data[mask] = 0.
    x = np.arange(data.size)
    x_mean = np.sum(x * data) / np.sum(data)
    x_stddev = np.sqrt(abs(np.sum(data * (x - x_mean)**2) / np.sum(data)))
    amplitude = np.nanmax(data) - np.nanmin(data)
    return amplitude, x_mean, x_stddev


def marginalize_data2d(data, error=None, mask=None):
    """
    Generate the marginal x and y distributions from a 2D distribution.
    """

    if error is not None:
        marginal_error = np.array(
            [np.sqrt(np.sum(error**2, axis=i)) for i in [0, 1]])
    else:
        marginal_error = [None, None]

    if mask is not None:
        mask = np.asanyarray(mask)
        marginal_mask = [mask.sum(axis=i).astype(np.bool) for i in [0, 1]]
        if error is None:
            marginal_error = np.array(
                [np.zeros(data.shape[1]), np.zeros(data.shape[0])])
        for i in [0, 1]:
            # give masked pixels a huge error
            marginal_error[i][marginal_mask[i]] = 1.e+30
    else:
        marginal_mask = [None, None]

    marginal_data = [data.sum(axis=i) for i in [0, 1]]

    return marginal_data, marginal_error, marginal_mask


def centroid_1dg(data, error=None, mask=None):
    """
    Calculate the centroid of a 2D array by fitting 1D Gaussians to the
    marginal x and y distributions of the array.

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
    xcen, ycen : float
        (x, y) coordinates of the centroid.
    """

    mdata, merror, mmask = marginalize_data2d(data, error=error, mask=mask)

    if merror[0] is None:
        mweights = [None, None]
    else:
        mweights = [(1.0 / merror[i]) for i in [0, 1]]

    const_init = np.min(data)
    centroid = []
    for (mdata_i, mweights_i, mmask_i) in zip(mdata, mweights, mmask):
        params_init = gaussian1d_moments(mdata_i, mask=mmask_i)
        g_init = GaussianConst1D(const_init, *params_init)
        fitter = LevMarLSQFitter()
        x = np.arange(mdata_i.size)
        g_fit = fitter(g_init, x, mdata_i, weights=mweights_i)
        centroid.append(g_fit.mean_1.value)
    return tuple(centroid)


def centroid_2dg(data, error=None, mask=None):
    """
    Calculate the centroid of a 2D array by fitting a 2D Gaussian (plus
    a constant) to the array.

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
    xcen, ycen : float
        (x, y) coordinates of the centroid.
    """

    gfit = fit_2dgaussian(data, error=error, mask=mask)
    return gfit.x_mean_1.value, gfit.y_mean_1.value


def fit_2dgaussian(data, error=None, mask=None):
    """
    Fit a 2D Gaussian plus a constant to a 2D image.

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
    result : `~astropy.modeling.functional_models.Gaussian2D` instance
        The best-fitting Gaussian 2D model.
    """

    if error is not None:
        weights = 1.0 / error
    else:
        weights = None

    if mask is not None:
        mask = np.asanyarray(mask)
        if weights is None:
            weights = np.ones_like(data)
        # down-weight masked pixels
        weights[mask] = 1.e-30

    # Subtract the minimum of the data as a crude background estimate.
    # This will also make the data values positive, preventing issues with
    # the moment estimation in data_properties (moments from negative data
    # values can yield undefined Gaussian parameters, e.g. x/y_stddev).
    shift = np.min(data)
    data = np.copy(data) - shift
    props = data_properties(data, mask=mask)
    init_values = np.array([props.xcentroid.value, props.ycentroid.value,
                            props.semimajor_axis_sigma.value,
                            props.semiminor_axis_sigma.value,
                            props.orientation.value])

    # if any init_values are np.nan, then estimate the initial parameters
    # using the marginal distributions
    if np.any(~np.isfinite(init_values)):
        mdata, merror, mmask = marginalize_data2d(data - shift, error=error,
                                                  mask=mask)
        x_ampl, x_mean, x_stddev = gaussian1d_moments(mdata[0], mask=mmask[0])
        y_ampl, y_mean, y_stddev = gaussian1d_moments(mdata[1], mask=mmask[1])
        init_values = np.array([x_mean, y_mean, x_stddev, y_stddev, 0.])

    init_const = 0.    # subtracted data minimum above
    init_amplitude = np.nanmax(data) - np.nanmin(data)
    g_init = GaussianConst2D(init_const, init_amplitude, *init_values)
    fitter = LevMarLSQFitter()
    y, x = np.indices(data.shape)
    gfit = fitter(g_init, x, y, data, weights=weights)
    gfit.amplitude_0 = gfit.amplitude_0 + shift
    return gfit


def data_properties(data, mask=None, background=None):
    """
    Calculate the centroid and morphological properties of a 2D array,
    e.g., an image cutout of an object.

    Parameters
    ----------
    data : array_like or `~astropy.units.Quantity`
        The 2D array of the image.

    mask : array_like (bool), optional
        A boolean mask, with the same shape as ``data``, where a
        `True` value indicates the corresponding element of ``data``
        is masked.  Masked data are excluded/ignored from all
        calculations.

    background : float, array_like, or `~astropy.units.Quantity`, optional
        The background level of the input ``data``.  ``background``
        may either be a scalar value or a 2D image with the same
        shape as the input ``data``.  If the input ``data`` has been
        background-subtracted, then set ``background`` to `None`
        (the default) or ``0.``.

    Returns
    -------
    result : `photutils.segmentation.SegmentProperties` instance
        A `photutils.segmentation.SegmentProperties` object.
    """

    segment_image = np.ones(data.shape, dtype=np.int)
    return SegmentProperties(data, segment_image, label=1, mask=mask,
                             background=background)
