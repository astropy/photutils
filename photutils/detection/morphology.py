# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import numpy as np
from astropy.modeling import models, fitting

__all__ = ['centroid_com', 'gaussian1d_moments',
           'centroid_1dg', 'centroid_2dg',
           'fit_2dgaussian', 'shape_params'
           ]


def centroid_com(data, data_mask=None):
    """
    Calculate the centroid of an array as its center of mass determined
    from image moments.

    Parameters
    ----------
    data : array_like
        The image data.

    data_mask : array_like, bool, optional
        A boolean mask with the same shape as `data`, where a `True`
        value indicates the corresponding element of `data` is invalid.

    Returns
    -------
    centroid : tuple
        (x, y) coordinates of the centroid.
    """
    from skimage.measure import moments

    if data_mask is not None:
        if data.shape != data_mask.shape:
            raise ValueError('data and data_mask must have the same shape')
        data[data_mask] = 0.

    m = moments(data, 1)
    xcen = m[1, 0] / m[0, 0]
    ycen = m[0, 1] / m[0, 0]
    return xcen, ycen


def gaussian1d_moments(data):
    """
    Estimate 1D Gaussian parameters from moments.
    """
    x = np.arange(data.size)
    xc = np.sum(x * data) / np.sum(data)
    stddev = np.sqrt(abs(np.sum(data * (x - xc)**2) / np.sum(data)))
    amplitude = np.max(data) - np.median(data)
    return amplitude, xc, stddev


def centroid_1dg(data, data_err=None, data_mask=None):
    """
    Calculate the centroid of an array from fitting 1D Gaussians
    to the marginal x and y distributions of the data.

    Parameters
    ----------
    data : array_like
        The image data.  (should be background subtracted)

    data_err : array_like, optional
        The 1-sigma errors for `data`.

    data_mask : array_like, bool, optional
        A boolean mask with the same shape as `data`, where a `True`
        value indicates the corresponding element of `data` is invalid.

    Returns
    -------
    centroid : tuple
        (x, y) coordinates of the centroid.
    """

    gaussian_x = data.sum(axis=0)
    gaussian_y = data.sum(axis=1)
    if data_err is None:
        weights_x = None
        weights_y = None
    else:
        data_err_x = np.sqrt(np.sum(data_err**2, axis=0))
        data_err_y = np.sqrt(np.sum(data_err**2, axis=1))
        weights_x = 1. / data_err_x
        weights_y = 1. / data_err_y

    gaussians = [gaussian_x, gaussian_y]
    data_weights = [weights_x, weights_y]
    centroid = []
    for (data, weights) in zip(gaussians, data_weights):
        params_init = gaussian1d_moments(data)
        g_init = models.Gaussian1D(*params_init)
        f = fitting.NonLinearLSQFitter()
        x = np.arange(data.size)
        g_fit = f(g_init, x, data, weights=weights)
        centroid.append(g_fit.mean.value)
    return centroid


def centroid_2dg(data, data_err=None, data_mask=None):
    """
    Calculate the centroid of an array from fitting a
    2D Gaussian to the data.

    Parameters
    ----------
    data : array_like
        The image data.  (should be background subtracted)

    data_err : array_like, optional
        The 1-sigma errors for `data`.

    data_mask : array_like, bool, optional
        A boolean mask with the same shape as `data`, where a `True`
        value indicates the corresponding element of `data` is invalid.

    Returns
    -------
    centroid : tuple
        (x, y) coordinates of the centroid.
    """

    gfit = fit_2dgaussian(data, data_err=data_err, data_mask=data_mask)
    return gfit.x_mean.value, gfit.y_mean.value


def fit_2dgaussian(data, data_err=None, data_mask=None):
    """
    Fit a 2D Gaussian to data.

    Parameters
    ----------
    data : array_like
        The image data.  (should be background subtracted)

    data_err : array_like, optional
        The 1-sigma errors for `data`.

    data_mask : array_like, bool, optional
        A boolean mask with the same shape as `data`, where a `True`
        value indicates the corresponding element of `data` is invalid.

    Returns
    -------
    centroid : tuple
        (x, y) coordinates of the centroid.
    """

    if data_err is None:
        weights = None
    else:
        weights = 1. / data_err
    gparams = shape_params(data)
    amplitude = np.max(data) - np.median(data)
    g_init = models.Gaussian2D(amplitude, gparams['xcen'], gparams['ycen'],
                               gparams['major_axis'], gparams['minor_axis'],
                               theta=gparams['pa'])
    f = fitting.NonLinearLSQFitter()
    y, x = np.indices(data.shape)
    gfit = f(g_init, x, y, data, weights=weights)
    return gfit


def shape_params(data, data_mask=None):
    """
    Calculate the centroid and shape parameters for an object using
    image moments.

    Parameters
    ----------
    data : array_like
        The 2D image data.

    data_mask : array_like, bool, optional
        A boolean mask with the same shape as ``data``, where a `True`
        value indicates the corresponding element of ``data`` is
        invalid.

    Returns
    -------
    dict :  A dictionary containing the object shape parameters:

        * ``xcen, ycen``: object centroid (zero-based origin).
        * ``major_axis``: length of the major axis
        * ``minor_axis``: length of the minor axis
        * ``eccen``: eccentricity.  The ratio of half the distance
          between its two ellipse foci to the length of the the
          semimajor axis.
        * ``pa``: position angle of the major axis.  Increases
          clockwise from the positive x axis.
        * ``covar``: corresponding covariance matrix for a 2D Gaussian
        * ``linear_eccen`` : linear eccentricity is the distance between
          the object center and either of its two ellipse foci.
    """
    from skimage.measure import moments, moments_central

    if data_mask is not None:
        if data.shape != data_mask.shape:
            raise ValueError('data and data_mask must have the same shape')
        data[data_mask] = 0.

    result = {}
    xcen, ycen = centroid_com(data)
    m = moments(data, 1)
    mu = moments_central(data, ycen, xcen, 2) / m[0, 0]
    result['xcen'] = xcen
    result['ycen'] = ycen
    # musum = mu[2, 0] + mu[0, 2]
    mudiff = mu[2, 0] - mu[0, 2]
    pa = 0.5 * np.arctan2(2.0*mu[1, 1], mudiff) * (180.0 / np.pi)
    if pa < 0.0:
        pa += 180.0
    result['pa'] = pa
    covar = np.array([[mu[2, 0], mu[1, 1]], [mu[1, 1], mu[0, 2]]])
    result['covar'] = covar
    eigvals, eigvecs = np.linalg.eigh(covar)
    majsq = np.max(eigvals)
    minsq = np.min(eigvals)
    result['major_axis'] = np.sqrt(majsq)
    result['minor_axis'] = np.sqrt(minsq)
    # if True:   # equivalent calculation
    #     tmp = np.sqrt(4.0*mu[1,1]**2 + mudiff**2)
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
    mu = moments_central(data, ycen, xcen, 2) / m[0, 0]
    """

    cov_matrix = np.array([[mu[2, 0], mu[1, 1]], [mu[1, 1], mu[0, 2]]])
    return models.Gaussian2D(amplitude, x_mean, y_mean, cov_matrix=cov_matrix)


def _moments_to_2DGaussian2(amplitude, x_mean, y_mean, Ixx, Ixy, Iyy):
    """
    Ixx, Ixy, Iyy:  second-order central moments [units of pixels**2]
    """

    cov_matrix = np.array([[Ixx, Ixy], [Ixy, Iyy]])
    return models.Gaussian2D(amplitude, x_mean, y_mean, cov_matrix=cov_matrix)
