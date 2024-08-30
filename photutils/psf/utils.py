# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
This module provides utilities for PSF-fitting photometry.
"""
import warnings

import numpy as np
from astropy.modeling import Model
from astropy.table import QTable
from astropy.units import Quantity
from astropy.utils.exceptions import AstropyUserWarning

from photutils.centroids import centroid_com
from photutils.psf.functional_models import CircularGaussianPSF
from photutils.utils import CutoutImage
from photutils.utils._parameters import as_pair

__all__ = ['fit_2dgaussian', 'fit_fwhm']


def fit_2dgaussian(data, *, xypos=None, fit_shape=None, mask=None, error=None,
                   fix_fwhm=True):
    """
    Fit a 2D Gaussian model to one or more sources in an image.

    This convenience function uses a
    `~photutils.psf.CircularGaussianPSF` model to fit the sources using
    the `~photutils.psf.PSFPhotometry` class.

    Non-finite values (e.g., NaN or inf) in the ``data`` array are
    automatically masked.

    Parameters
    ----------
    data : 2D array
        The 2D array of the image. The input array must be background
        subtracted.

    xypos : array-like, optional
        The (x, y) pixel coordinates of the sources. If `None`, then
        one source will be fit with an initial position using the
        center-of-mass centroid of the ``data`` array.

    fit_shape : int or tuple of two ints, optional
        The shape of the fitting region. If a scalar, then it is assumed
        to be a square. If `None`, then the shape of the input ``data``
        will be used.

    mask : array-like (bool), optional
        A boolean mask with the same shape as the input ``data``, where
        a `True` value indicates the corresponding element of ``data``
        is masked.

    error : 2D array, optional
        The pixel-wise Gaussian 1-sigma errors of the input
        ``data``. ``error`` is assumed to include *all* sources
        of error, including the Poisson error of the sources (see
        `~photutils.utils.calc_total_error`) . ``error`` must have the
        same shape as the input ``data``. If a `~astropy.units.Quantity`
        array, then ``data`` must also be a `~astropy.units.Quantity`
        array with the same units.

    fix_fwhm : bool, optional
        Whether to fix the FWHM of the Gaussian PSF model during the
        fitting process. If `False`, the initial guess for the FWHM is
        the half the mean of the x and y sizes of the ``fit_shape``
        values.

    Returns
    -------
    result : `~photutils.psf.PSFPhotometry`
        The PSF-fitting photometry results.

    Notes
    -----
    The source(s) are fit with a `~photutils.psf.CircularGaussianPSF`
    model using the `~photutils.psf.PSFPhotometry` class. The initial
    guess for the flux is the sum of the pixel values within the fitting
    region. The initial guess for the FWHM is half the mean of the x and
    y sizes of the ``fit_shape`` values.
    """
    # prevent circular import
    from photutils.psf.photometry import PSFPhotometry

    if xypos is None:
        xypos = centroid_com(data)
    xypos = np.atleast_2d(xypos)

    if fit_shape is None:
        fit_shape = data.shape
    else:
        fit_shape = as_pair('fit_shape', fit_shape, lower_bound=(1, 0),
                            check_odd=True)

    yxpos = xypos[:, ::-1]
    flux_init = []
    for yxpos_ in yxpos:
        cutout = CutoutImage(data, yxpos_, tuple(fit_shape))
        flux_init.append(np.sum(cutout.data))

    if isinstance(data, Quantity):
        flux_init <<= data.unit

    init_params = QTable()
    init_params['x'] = xypos[:, 0]
    init_params['y'] = xypos[:, 1]
    init_params['flux'] = flux_init

    model = CircularGaussianPSF()
    if not fix_fwhm:
        init_params['fwhm'] = np.mean(fit_shape) / 2.0
        model.fwhm.fixed = False

    phot = PSFPhotometry(model, fit_shape)
    _ = phot(data, mask=mask, error=error, init_params=init_params)

    return phot


def fit_fwhm(data, *, xypos=None, fit_shape=None, mask=None, error=None):
    """
    Fit the FWHM of one or more sources in an image.

    This convenience function uses a
    `~photutils.psf.CircularGaussianPSF` model to fit the sources using
    the `~photutils.psf.PSFPhotometry` class.

    Non-finite values (e.g., NaN or inf) in the ``data`` array are
    automatically masked.

    Parameters
    ----------
    data : 2D array
        The 2D array of the image. The input array must be background
        subtracted.

    xypos : array-like, optional
        The (x, y) pixel coordinates of the sources. If `None`, then
        one source will be fit with an initial position using the
        center-of-mass centroid of the ``data`` array.

    fit_shape : int or tuple of two ints, optional
        The shape of the fitting region. If a scalar, then it is assumed
        to be a square. If `None`, then the shape of the input ``data``
        will be used.

    mask : array-like (bool), optional
        A boolean mask with the same shape as the input ``data``, where
        a `True` value indicates the corresponding element of ``data``
        is masked.

    error : 2D array, optional
        The pixel-wise Gaussian 1-sigma errors of the input
        ``data``. ``error`` is assumed to include *all* sources
        of error, including the Poisson error of the sources (see
        `~photutils.utils.calc_total_error`) . ``error`` must have the
        same shape as the input ``data``. If a `~astropy.units.Quantity`
        array, then ``data`` must also be a `~astropy.units.Quantity`
        array with the same units.

    Returns
    -------
    fwhm : `~numpy.ndarray`
        The FWHM of the sources. Note that the returned FWHM values are
        always positive.

    Notes
    -----
    The source(s) are fit using the :func:`fit_2dgaussian` function,
    which uses a `~photutils.psf.CircularGaussianPSF` model with the
    `~photutils.psf.PSFPhotometry` class. The initial guess for the
    flux is the sum of the pixel values within the fitting region. The
    initial guess for the FWHM is half the mean of the x and y sizes of
    the ``fit_shape`` values.
    """
    with warnings.catch_warnings(record=True) as fit_warnings:
        phot = fit_2dgaussian(data, xypos=xypos, fit_shape=fit_shape,
                              mask=mask, error=error, fix_fwhm=False)

    if len(fit_warnings) > 0:
        warnings.warn('One or more fit(s) may not have converged. Please '
                      'carefully check your results. You may need to change '
                      'the input "xypos" and "fit_shape" parameters.',
                      AstropyUserWarning)

    return np.abs(np.array(phot.results['fwhm_fit']))


def _interpolate_missing_data(data, mask, method='cubic'):
    """
    Interpolate missing data as identified by the ``mask`` keyword.

    Parameters
    ----------
    data : 2D `~numpy.ndarray`
        An array containing the 2D image.

    mask : 2D bool `~numpy.ndarray`
        A 2D boolean mask array with the same shape as the input
        ``data``, where a `True` value indicates the corresponding
        element of ``data`` is masked. The masked data points are those
        that will be interpolated.

    method : {'cubic', 'nearest'}, optional
        The method of used to interpolate the missing data:

        * ``'cubic'``:  Masked data are interpolated using 2D cubic
            splines. This is the default.

        * ``'nearest'``:  Masked data are interpolated using
            nearest-neighbor interpolation.

    Returns
    -------
    data_interp : 2D `~numpy.ndarray`
        The interpolated 2D image.
    """
    from scipy import interpolate

    data_interp = np.array(data, copy=True)

    if len(data_interp.shape) != 2:
        raise ValueError("'data' must be a 2D array.")

    if mask.shape != data.shape:
        raise ValueError("'mask' and 'data' must have the same shape.")

    y, x = np.indices(data_interp.shape)
    xy = np.dstack((x[~mask].ravel(), y[~mask].ravel()))[0]
    z = data_interp[~mask].ravel()

    if method == 'nearest':
        interpol = interpolate.NearestNDInterpolator(xy, z)
    elif method == 'cubic':
        interpol = interpolate.CloughTocher2DInterpolator(xy, z)
    else:
        raise ValueError('Unsupported interpolation method.')

    xy_missing = np.dstack((x[mask].ravel(), y[mask].ravel()))[0]
    data_interp[mask] = interpol(xy_missing)

    return data_interp


def _validate_psf_model(psf_model):
    """
    Validate the PSF model.

    The PSF model must be a subclass of `astropy
    .modeling.Fittable2DModel`. It must also be two-dimensional and
    have a single output.

    Parameters
    ----------
    psf_model : `astropy.modeling.Fittable2DModel`
        The PSF model to validate.

    Returns
    -------
    psf_model : `astropy.modeling.Model`
        The validated PSF model.

    Raises
    ------
    TypeError
        If the PSF model is not an Astropy Model subclass.

    ValueError
        If the PSF model is not two-dimensional with n_inputs=2 and
        n_outputs=1.
    """
    if not isinstance(psf_model, Model):
        raise TypeError('psf_model must be an Astropy Model subclass.')

    if psf_model.n_inputs != 2 or psf_model.n_outputs != 1:
        raise ValueError('psf_model must be two-dimensional with '
                         'n_inputs=2 and n_outputs=1.')

    return psf_model


def _get_psf_model_params(psf_model):
    """
    Get the names of the PSF model parameters corresponding to x, y, and
    flux.

    The PSF model must have parameters called 'x_0', 'y_0', and
    'flux' or it must have 'x_name', 'y_name', and 'flux_name'
    attributes (i.e., output from `make_psf_model`). Otherwise, a
    `ValueError` is raised.

    The PSF model must be a subclass of `astropy.modeling.Model`. It
    must also be two-dimensional and have a single output.

    Parameters
    ----------
    psf_model : `astropy.modeling.Model`
        The PSF model to validate.

    Returns
    -------
    model_params : tuple
        A tuple of the PSF model parameter names.
    """
    psf_model = _validate_psf_model(psf_model)

    params1 = ('x_0', 'y_0', 'flux')
    params2 = ('x_name', 'y_name', 'flux_name')
    if all(name in psf_model.param_names for name in params1):
        model_params = params1
    elif all(params := [getattr(psf_model, name, None) for name in params2]):
        model_params = tuple(params)
    else:
        msg = 'Invalid PSF model - could not find PSF parameter names.'
        raise ValueError(msg)

    return model_params
