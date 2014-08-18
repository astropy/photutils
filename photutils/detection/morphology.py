# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import numpy as np
from astropy.modeling import models
from astropy.modeling.fitting import LevMarLSQFitter

__all__ = ['centroid_com', 'gaussian1d_moments', 'centroid_1dg',
           'centroid_2dg', 'fit_2dgaussian', 'shape_params']


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

    mask : array_like, bool, optional
        A boolean mask with the same shape as ``data``, where a `True`
        value indicates the corresponding element of ``data`` is
        invalid.  Masked pixels are set to zero in the output ``data``.

    Returns
    -------
    image : `numpy.ndarray`, float64
        The converted 2D array of the image.
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
    data : array_like or `~astropy.nddata.NDData`
        The 2D array of the image.

    mask : array_like, bool, optional
        A boolean mask with the same shape as ``data``, where a `True`
        value indicates the corresponding element of ``data`` is
        invalid.  If ``mask`` is input it will override ``data.mask``
        for `~astropy.nddata.NDData` inputs.

    Returns
    -------
    xcen, ycen : tuple of floats
        (x, y) coordinates of the centroid.
    """

    from skimage.measure import moments
    data = _convert_image(data, mask=mask)
    m = moments(data, 1)
    xcen = m[1, 0] / m[0, 0]
    ycen = m[0, 1] / m[0, 0]
    return xcen, ycen


def gaussian1d_moments(data):
    """
    Estimate 1D Gaussian parameters from the moments of 1D data.  This
    function can be useful for providing initial parameter values when
    fitting a 1D Gaussian to the ``data``.

    Parameters
    ----------
    data : array_like or `~astropy.nddata.NDData`
        The 1D array of the image.

    Returns
    -------
    output : tuple
        The estimated (amplitude, mean, stddev) of a 1D Gaussian.
    """

    x = np.arange(data.size)
    xc = np.sum(x * data) / np.sum(data)
    stddev = np.sqrt(abs(np.sum(data * (x - xc)**2) / np.sum(data)))
    amplitude = np.ptp(data)
    return amplitude, xc, stddev


def centroid_1dg(data, uncertainty=None, mask=None):
    """
    Calculate the centroid of a 2D array by fitting 1D Gaussians to the
    marginal x and y distributions of the image.

    Parameters
    ----------
    data : array_like or `~astropy.nddata.NDData`
        The 2D array of the image.

    uncertainty : array_like, optional
        The 2D array of the 1-sigma errors of the input ``data``.  If
        ``uncertainty`` is input it will override ``data.uncertainty``
        for `~astropy.nddata.NDData` inputs.

    mask : array_like, bool, optional
        (Not yet implemented).
        A boolean mask with the same shape as ``data``, where a `True`
        value indicates the corresponding element of ``data`` is
        invalid.  If ``mask`` is input it will override ``data.mask``
        for `~astropy.nddata.NDData` inputs.

    Returns
    -------
    xcen, ycen : tuple of floats
        (x, y) coordinates of the centroid.
    """

    data_x = data.sum(axis=0)
    data_y = data.sum(axis=1)
    if uncertainty is None:
        weights_x = None
        weights_y = None
    else:
        image_err_x = np.sqrt(np.sum(uncertainty**2, axis=0))
        image_err_y = np.sqrt(np.sum(uncertainty**2, axis=1))
        weights_x = 1.0 / image_err_x
        weights_y = 1.0 / image_err_y

    marginal_data = [data_x, data_y]
    marginal_weights = [weights_x, weights_y]
    centroid = []
    for (data, weights) in zip(marginal_data, marginal_weights):
        params_init = gaussian1d_moments(data)
        g_init = models.Gaussian1D(*params_init)
        fitter = LevMarLSQFitter()
        x = np.arange(data.size)
        g_fit = fitter(g_init, x, data, weights=weights)
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


def shape_params(data, mask=None):
    """
    Calculate the centroid and shape parameters of a 2D array (e.g., an
    image cutout of an object) using image moments.

    Parameters
    ----------
    data : array_like or `~astropy.nddata.NDData`
        The 2D array of the image.

    mask : array_like, bool, optional
        A boolean mask with the same shape as ``data``, where a `True`
        value indicates the corresponding element of ``data`` is
        invalid.  If ``mask`` is input it will override ``data.mask``
        for `~astropy.nddata.NDData` inputs.

    Returns
    -------
    params : dict
        A dictionary containing the object shape parameters:

        * ``xcen, ycen``: The object centroid (zero-based origin).
        * ``major_axis``: The length of the major axis of the ellipse
          that has the same second-order moments as the input image.
        * ``minor_axis``: The length of the minor axis of the ellipse
          that has the same second-order moments as the input image.
        * ``eccen``: The eccentricity of the ellipse that has the same
          second-order moments as the input image.  The eccentricity is
          the ratio of half the distance between the two ellipse foci to
          the length of the semimajor axis.
        * ``angle``: Angle in radians between the positive x axis and
          the major axis of the ellipse that has the same second-order
          moments as the input image.  The angle increases
          counter-clockwise.
        * ``covar``: The covariance matrix of the ellipse that has the
          same second-order moments as the input image.
        * ``linear_eccen`` : The linear eccentricity of the ellipse that
          has the same second-order moments as the input image.  Linear
          eccentricity is the distance between the ellipse center and
          either of its two foci.
    """

    from skimage.measure import moments, moments_central
    data = _convert_image(data, mask=mask)
    xcen, ycen = centroid_com(data)
    m = moments(data, 1)
    mu = moments_central(data, ycen, xcen, 2) / m[0, 0]
    result = {}
    result['xcen'] = xcen
    result['ycen'] = ycen
    mudiff = mu[2, 0] - mu[0, 2]
    angle = 0.5 * np.arctan2(2.0 * mu[1, 1], mudiff) * (180.0 / np.pi)
    if angle < 0.0:
        angle += np.pi
    result['angle'] = angle
    covar = np.array([[mu[2, 0], mu[1, 1]], [mu[1, 1], mu[0, 2]]])
    result['covar'] = covar
    eigvals, eigvecs = np.linalg.eigh(covar)
    majsq = np.max(eigvals)
    minsq = np.min(eigvals)
    result['major_axis'] = np.sqrt(majsq)
    result['minor_axis'] = np.sqrt(minsq)
    # equivalent calculation of major/minor axes:
    #     tmp = np.sqrt(4.0*mu[1,1]**2 + mudiff**2)
    #     musum = mu[2, 0] + mu[0, 2]
    #     majsq = 0.5 * (musum + tmp)
    #     minsq = 0.5 * (musum - tmp)
    #     result['major_axis2'] = np.sqrt(majsq)
    #     result['minor_axis2'] = np.sqrt(minsq)
    result['eccen'] = np.sqrt(1.0 - (minsq / majsq))
    result['linear_eccen'] = np.sqrt(majsq - minsq)
    return result


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
