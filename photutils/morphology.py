# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import numpy as np
from astropy.modeling import models
from astropy.modeling.fitting import LevMarLSQFitter


__all__ = ['centroid_com', 'gaussian1d_moments', 'centroid_1dg',
           'centroid_2dg', 'fit_2dgaussian', 'data_properties']


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
    except TypeError:
        image = np.asarray(data).astype(np.float)    # for numpy <= 1.6
    if mask is not None:
        mask = np.asarray(mask)
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
    Estimate the 1D Gaussian parameters from the moments of 1D data.
    This function can be useful for providing initial parameter values
    when fitting a 1D Gaussian to the ``data``.

    Parameters
    ----------
    data : array_like
        The 1D array of the image.

    mask : array_like (bool), optional
        A boolean mask, with the same shape as ``data``, where a `True`
        value indicates the corresponding element of ``data`` is masked.

    Returns
    -------
    amplitude, mean, stddev : float
        The estimated parameters of a 1D Gaussian.
    """

    if mask is not None:
        data = data.copy()
        data[mask] = 0.
    x = np.arange(data.size)
    xc = np.sum(x * data) / np.sum(data)
    stddev = np.sqrt(abs(np.sum(data * (x - xc)**2) / np.sum(data)))
    amplitude = np.ptp(data)
    return amplitude, xc, stddev


def centroid_1dg(data, error=None, mask=None):
    """
    Calculate the centroid of a 2D array by fitting 1D Gaussians to the
    marginal x and y distributions of the image.

    Parameters
    ----------
    data : array_like or `~astropy.nddata.NDData`
        The 2D array of the image.

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

    if error is not None:
        marginal_error = np.array(
            [np.sqrt(np.sum(error**2, axis=i)) for i in [0, 1]])
        marginal_weights = 1.0 / marginal_error
    else:
        marginal_weights = [None, None]

    if mask is not None:
        marginal_mask = [mask.sum(axis=i).astype(np.bool) for i in [0, 1]]
        if error is None:
            marginal_weights = np.array(
                [np.ones(data.shape[1]), np.ones(data.shape[0])])
        for i in [0, 1]:
            # down-weight masked pixels
            marginal_weights[i][marginal_mask[i]] = 1.e-10
    else:
        marginal_mask = [None, None]

    centroid = []
    marginal_data = [data.sum(axis=i) for i in [0, 1]]
    inputs = zip(marginal_data, marginal_weights, marginal_mask)
    for (mdata, mweights, mmask) in inputs:
        params_init = gaussian1d_moments(mdata, mask=mmask)
        g_init = models.Gaussian1D(*params_init)
        fitter = LevMarLSQFitter()
        x = np.arange(mdata.size)
        g_fit = fitter(g_init, x, mdata, weights=mweights)
        centroid.append(g_fit.mean.value)
    return centroid


def centroid_2dg(data, uncertainty=None, mask=None):
    """
    Calculate the centroid of a 2D array by fitting a 2D Gaussian to the
    image.

    Parameters
    ----------
    data : array_like or `~astropy.nddata.NDData`
        The 2D array of the image.

    uncertainty : array_like, optional
        The 2D array of the 1-sigma errors of the input ``data``.

    mask : array_like, bool, optional
        (Not yet implemented).
        A boolean mask with the same shape as ``data``, where a `True`
        value indicates the corresponding element of ``data`` is
        invalid.  If ``mask`` is input it will override ``data.mask``
        for `~astropy.nddata.NDData` inputs.

    Returns
    -------
    centroid : tuple
        (x, y) coordinates of the centroid.
    """

    gfit = fit_2dgaussian(data, uncertainty=uncertainty, mask=mask)
    return gfit.x_mean.value, gfit.y_mean.value


def fit_2dgaussian(data, uncertainty=None, mask=None):
    """
    Fit a 2D Gaussian to a 2D image.

    Parameters
    ----------
    data : array_like or `~astropy.nddata.NDData`
        The 2D array of the image.

    uncertainty : array_like, optional
        The 2D array of the 1-sigma errors of the input ``data``.

    mask : array_like, bool, optional
        (Not yet implemented).
        A boolean mask with the same shape as ``data``, where a `True`
        value indicates the corresponding element of ``data`` is
        invalid.  If ``mask`` is input it will override ``data.mask``
        for `~astropy.nddata.NDData` inputs.

    Returns
    -------
    centroid : tuple
        (x, y) coordinates of the centroid.
    """

    if uncertainty is None:
        weights = None
    else:
        weights = 1.0 / uncertainty
    init_param = shape_params(data, mask=mask)
    init_amplitude = np.ptp(data)
    g_init = models.Gaussian2D(init_amplitude, init_param['xcen'],
                               init_param['ycen'], init_param['major_axis'],
                               init_param['minor_axis'],
                               theta=init_param['angle'])
    fitter = LevMarLSQFitter()
    y, x = np.indices(data.shape)
    gfit = fitter(g_init, x, y, data, weights=weights)
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
    """

    segment_image = np.ones(data.shape)
    return SegmentProperties(data, segment_image, 1, mask=mask,
                             background=background)


def _moments_to_2DGaussian(amplitude, x_mean, y_mean, mu):
    """
    mu:  normalized second-order central moments matrix [units of pixels**2]
    mu = moments_central(image, ycen, xcen, 2) / m[0, 0]
    """

    cov_matrix = np.array([[mu[2, 0], mu[1, 1]], [mu[1, 1], mu[0, 2]]])
    return models.Gaussian2D(amplitude, x_mean, y_mean, cov_matrix=cov_matrix)


def _moments_to_2DGaussian2(amplitude, x_mean, y_mean, Ixx, Ixy, Iyy):
    """
    Ixx, Ixy, Iyy:  second-order central moments [units of pixels**2]
    """

    cov_matrix = np.array([[Ixx, Ixy], [Ixy, Iyy]])
    return models.Gaussian2D(amplitude, x_mean, y_mean, cov_matrix=cov_matrix)
