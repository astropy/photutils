# Licensed under a 3-clause BSD style license - see LICENSE.rst

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import warnings
import copy
import numpy as np

from astropy.modeling.fitting import LevMarLSQFitter

from .utils import py2round


__all__ = ['EPSFFitter']


def _calc_res(psf, star):
    ovx = star.pixel_scale[0] / psf.pixel_scale[0]
    ovy = star.pixel_scale[1] / psf.pixel_scale[1]
    gy, gx = np.indices((star.ny, star.nx), dtype=np.float)
    gx = ovx * (gx - star.x_center)
    gy = ovy * (gy - star.y_center)
    psfval = psf.evaluate(gx, gy, flux=1.0, x_0=0.0, y_0=0.0)
    return (star.data - star.flux * (ovx * ovy) * psfval)


def compute_residuals(psf, stars):
    """
    Register the ``psf`` to intput ``stars`` and compute the difference.

    Parameters
    ----------
    psf : FittableImageModel2D
        Model of the PSF.

    stars : Star, list of Star
        A single :py:class:`~psfutils.catalogs.Star` object or a list of stars
        for which resuduals need to be computed.

    Returns
    -------
    res : numpy.ndarray, list of numpy.ndarray
        A list of `numpy.ndarray` of residuals when input is a list of
        :py:class:`~psfutils.catalogs.Star` objects and a single
        `numpy.ndarray` when input is a single
        :py:class:`~psfutils.catalogs.Star` object.

    """
    if isinstance(stars, Star):
        res = _calc_res(psf, star)
        return res

    else:
        res = []
        for s in stars:
            res.append(_calc_res(psf, s))

    return res


class EPSFFitter(object):
    def __init__(self, psf_fit_box=5, fitter=LevMarLSQFitter(),
                 residuals=False, **kwargs):
        self.psf_fit_box = psf_fit_box
        self.fitter = fitter
        self.residuals = residuals
        self.fitter_kwargs = kwargs

    def __call__(self, stars, psf):
        return self.fit_psf(stars, psf)

    def fit_psf(self, stars, psf):
        return fit_stars(stars, psf)


def fit_stars(stars, psf, psf_fit_box=5, fitter=LevMarLSQFitter(),
              fitter_kwargs={}, residuals=False):
    """
    Fit a PSF model to stars.

    .. note::
        When models in ``stars`` contain weights, a weighted fit of the PSF to
        the stars will be performed.

    Parameters
    ----------
    stars : Star, list of Star
        A list of :py:class:`~psfutils.catalogs.Star` objects
        containing image data of star cutouts to which the PSF must be fitted.
        Fitting procedure relies on correct coordinates of the center of the
        PSF and as close as possible to the correct center positions of stars.
        Star positions are derived from ``x_0`` and ``y_0`` parameters of the
        `PSF2DModel` model.

    psf : PSF2DModel
        A PSF model to be fitted to the stars.

    psf_fit_box : int, tuple of int, None, optional
        The size of the innermost box centered on stars center to be used for
        PSF fitting. This allows using only a small number of central pixels
        of the star for fitting processed thus ignoring wings. A tuple of
        two integers can be used to indicate separate sizes of the fitting
        box for ``X-`` and ``Y-`` axes. When ``psf_fit_box`` is `None`, the
        entire star's image will be used for fitting.

    fitter : astropy.modeling.fitting.Fitter, optional
        A :py:class:`~astropy.modeling.fitting.Fitter`-subclassed fitter
        class or initialized object.

    fitter_kwargs : dict-like, optional
        Additional optional keyword arguments to be passed directly to
        fitter's ``__call__()`` method.

    residuals : bool, optional
        Enable/disable computation of residuals between star's data and fitted
        PSF model. Residual image can be retrieved using
        :py:attr:`~psfutils.catalogs.Star.fit_residual` of returned star(s).


    Returns
    -------
    fitted_stars : list of FittableImageModel2D
        A list of `~psfutils.models.FittableImageModel2D` of stars with model
        parameters `~psfutils.models.FittableImageModel2D.x_0` and
        `~psfutils.models.FittableImageModel2D.y_0` set to 0 and
        `~psfutils.models.FittableImageModel2D.origin` will show fitted
        center of the star. If `update_flux` was `True`, the
        `~psfutils.models.FittableImageModel2D.flux`
        model parameter will contain fitted flux and the original star's
        flux otherwise.

    """
    if not hasattr(stars, '__iter__'):
        stars = [stars]

    if len(stars) == 0:
        return []

    # get all stars including linked stars as a flat list:
    all_stars = []
    for s in stars:
        all_stars += s.get_linked_list()

    # analize psf_fit_box:
    snx = [s.nx for s in all_stars]
    sny = [s.ny for s in all_stars]
    minfbx = min(snx)
    minfby = min(sny)

    if psf_fit_box is None:
        # use full grid defined by stars' data size:
        psf_fit_box = (minfbx, minfby)

    elif hasattr(psf_fit_box, '__iter__'):
        if len(psf_fit_box) != 2:
            raise ValueError("'psf_fit_box' must be a tuple of two integers, "
                             "a single integer, or None")

        psf_fit_box = (min(minfbx, psf_fit_box[0]),
                       min(minfby, psf_fit_box[0]))

    else:
        psf_fit_box = min(minfbx, minfby, psf_fit_box)
        psf_fit_box = (psf_fit_box, psf_fit_box)

    # create grid for fitting box (in stars' grid units):
    width, height = psf_fit_box
    width = int(py2round(width))
    height = int(py2round(height))
    igy, igx = np.indices((height, width), dtype=np.float)

    # perform fitting for each star:
    fitted_stars = []

    # initialize fitter (if needed):
    if isinstance(fitter, type):
        fit = fitter()
    else:
        fit = fitter

    # remove fitter's keyword arguments that we set ourselves:
    rem_kwd = ['x', 'y', 'z', 'weights']
    fitter_kwargs = copy.deepcopy(fitter_kwargs)
    for k in rem_kwd:
        if k in fitter_kwargs:
            del fitter_kwargs[k]

    fitter_has_fit_info = hasattr(fit, 'fit_info')

    # make a copy of the original PSF:
    psf = psf.copy()

    for st in stars:
        cst = _fit_star(st, psf, fit, fitter_kwargs,
                        fitter_has_fit_info, residuals,
                        width, height, igx, igy)

        # also fit stars linked to the left:
        lnks = st.prev
        while lnks is not None:
            lnkcst = _fit_star(lnks, psf, fit, fitter_kwargs,
                               fitter_has_fit_info, residuals,
                               width, height, igx, igy)
            cst.append_first(lnkcst)
            lnks = lnks.prev

        # ... and fit stars linked to the right:
        lnks = st.next
        while lnks is not None:
            lnkcst = _fit_star(lnks, psf, fit, fitter_kwargs,
                               fitter_has_fit_info, residuals,
                               width, height, igx, igy)
            cst.append_last(lnkcst)
            lnks = lnks.next

        cst.constrain_linked_centers(ignore_badfit_stars=True)
        fitted_stars.append(cst)

    return fitted_stars


def _fit_star(star, psf, fit, fit_kwargs, fitter_has_fit_info, residuals,
              width, height, igx, igy):
    # NOTE: input PSF may be modified by this function. Make a copy if
    #       it is important to preserve input model.

    err = 0
    ovx = star.pixel_scale[0] / psf.pixel_scale[0]
    ovy = star.pixel_scale[1] / psf.pixel_scale[1]
    ny, nx = star.shape

    rxc = int(py2round(star.x_center))
    ryc = int(py2round(star.y_center))

    x1 = rxc - (width - 1) // 2
    x2 = x1 + width
    y1 = ryc - (height - 1) // 2
    y2 = y1 + height

    # check boundaries of the fitting box:
    if x1 < 0:
        i1 = -x1
        x1 = 0

    else:
        i1 = 0

    if x2 > nx:
        i2 = width - (x2 - nx)
        x2 = nx

    else:
        i2 = width

    if y1 < 0:
        j1 = -y1
        y1 = 0

    else:
        j1 = 0

    if y2 > ny:
        j2 = height - (y2 - ny)
        y2 = ny

    else:
        j2 = height

    # initial guess for fitted flux and shifts:
    psf.flux = star.flux
    psf.x_0 = 0.0
    psf.y_0 = 0.0

    if rxc < 0 or rxc > (nx - 1) or ryc < 0 or ryc > (ny - 1):
        # star's center is outside the extraction box
        err = 1
        fit_info = None
        fitted_psf = psf
        warnings.warn("Source with coordinates ({}, {}) is being ignored "
                      "because its center is outside the image."
                      .format(star.x_center, star.y_center))

    elif (i2 - i1) < 3 or (j2 - j1) < 3:
        # star's center is too close to the edge of the star's image:
        err = 2
        fit_info = None
        fitted_psf = psf
        warnings.warn("Source with coordinates ({}, {}) is being ignored "
                      "because there are too few pixels available around "
                      "its center pixel.".format(star.x_center, star.y_center))

    else:
        # define PSF sampling grid:
        gx = (igx[j1:j2, i1:i2] - (star.x_center - x1)) * ovx
        gy = (igy[j1:j2, i1:i2] - (star.y_center - y1)) * ovy

        # fit PSF to the star:
        scaled_data = star.data[y1:y2, x1:x2] / (ovx * ovy)
        if star.weights is None:
            # a separate treatment for the case when fitters
            # do not support weights (star's models must not have
            # weights set in such cases)
            fitted_psf = fit(model=psf, x=gx, y=gy, z=scaled_data,
                             **fit_kwargs)

        else:
            wght = star.weights[y1:y2, x1:x2]
            fitted_psf = fit(model=psf, x=gx, y=gy, z=scaled_data,
                             weights=wght, **fit_kwargs)

        if fitter_has_fit_info:
            # TODO: this treatment of fit info (fit error info) may not be
            # compatible with other fitters. This code may need revising.
            fit_info = fit.fit_info
            if 'ierr' in fit_info and fit_info['ierr'] not in [1, 2, 3, 4]:
                err = 3

        else:
            fit_info = None

    # compute correction to the star's position and flux:
    cst = copy.deepcopy(star)
    cst.x_center += fitted_psf.x_0.value / ovx
    cst.y_center += fitted_psf.y_0.value / ovy

    # set "measured" star's flux based on fitted ePSF:
    cst.flux = fitted_psf.flux.value

    if residuals:
        cst.fit_residual = _calc_res(fitted_psf, cst)

    cst.fit_info = fit_info
    cst.fit_error_status = err
    return cst
