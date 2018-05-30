# Licensed under a 3-clause BSD style license - see LICENSE.rst

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import copy
import warnings

import numpy as np
from astropy.modeling.fitting import LevMarLSQFitter

from .psfstars import PSFStar, LinkedPSFStar, PSFStars


__all__ = ['EPSFFitter']


class EPSFFitter(object):
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
        `EPSFModel` model.

    psf : `EPSFModel`
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

    def __init__(self, fitter=LevMarLSQFitter(), psf_fit_box=5,
                 **kwargs):
        self.fitter = fitter
        self.psf_fit_box = psf_fit_box
        self.fitter_kwargs = kwargs

    def __call__(self, psf, psf_stars):
        return self.fit_psf(psf, psf_stars)

    def _fit_star(self, psf, star, fit, fit_kwargs, fitter_has_fit_info,
                  width, height, igx, igy):
        # NOTE: input PSF may be modified by this function. Make a copy if
        #       it is important to preserve input model.

        from .epsf import _py2intround

        err = 0
        ovx = star.pixel_scale[0] / psf.pixel_scale[0]
        ovy = star.pixel_scale[1] / psf.pixel_scale[1]
        ny, nx = star.shape

        rxc = _py2intround(star.cutout_center[0])
        ryc = _py2intround(star.cutout_center[1])

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
                          .format(star.cutout_center[0],
                                  star.cutout_center[1]))

        elif (i2 - i1) < 3 or (j2 - j1) < 3:
            # star's center is too close to the edge of the star's image:
            err = 2
            fit_info = None
            fitted_psf = psf
            warnings.warn("Source with coordinates ({}, {}) is being ignored "
                          "because there are too few pixels available around "
                          "its center pixel.".format(star.cutout_center[0],
                                                     star.cutout_center[1]))

        else:
            # define PSF sampling grid:
            gx = (igx[j1:j2, i1:i2] - (star.cutout_center[0] - x1)) * ovx
            gy = (igy[j1:j2, i1:i2] - (star.cutout_center[1] - y1)) * ovy

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
        #cst.x_center += fitted_psf.x_0.value / ovx
        #cst.y_center += fitted_psf.y_0.value / ovy

        x_center = cst.cutout_center[0] + fitted_psf.x_0.value / ovx
        y_center = cst.cutout_center[1] + fitted_psf.y_0.value / ovy
        cst.cutout_center = (x_center, y_center)

        # set "measured" star's flux based on fitted ePSF:
        cst.flux = fitted_psf.flux.value

        cst.fit_info = fit_info
        cst.fit_error_status = err

        return cst

    def fit_psf(self, psf, psf_stars):
        from .epsf import _py2intround

        if len(psf_stars) == 0:
            return []

        # get all stars including linked stars as a flat list:
        all_stars = psf_stars.all_psfstars

        # analize psf_fit_box:
        snx = [s.shape[1] for s in all_stars]
        sny = [s.shape[0] for s in all_stars]
        minfbx = min(snx)
        minfby = min(sny)

        psf_fit_box = np.copy(self.psf_fit_box)
        if psf_fit_box is not None:
            psf_fit_box = np.atleast_1d(psf_fit_box).astype(int)
            if len(psf_fit_box) == 1:
                psf_fit_box = np.repeat(psf_fit_box, 2)
        else:
            # use full grid defined by stars' data size:
            psf_fit_box = (minfbx, minfby)


        #elif hasattr(psf_fit_box, '__iter__'):
        #    if len(psf_fit_box) != 2:
        #        raise ValueError("'psf_fit_box' must be a tuple of two "
        #                         "integers, a single integer, or None")
#
#            psf_fit_box = (min(minfbx, psf_fit_box[0]),
#                           min(minfby, psf_fit_box[0]))
#
#        else:
#            psf_fit_box = min(minfbx, minfby, psf_fit_box)
#            psf_fit_box = (psf_fit_box, psf_fit_box)

        # create grid for fitting box (in stars' grid units):
        width, height = psf_fit_box
        width = _py2intround(width)
        height = _py2intround(height)
        igy, igx = np.indices((height, width), dtype=np.float)

        # perform fitting for each star:
        fitted_stars = []

        # remove fitter's keyword arguments that we set ourselves:
        rem_kwd = ['x', 'y', 'z', 'weights']
        fitter_kwargs = copy.deepcopy(self.fitter_kwargs)
        for k in rem_kwd:
            if k in fitter_kwargs:
                del fitter_kwargs[k]

        fitter_has_fit_info = hasattr(self.fitter, 'fit_info')

        # make a copy of the original PSF:
        psf = psf.copy()

        for st in psf_stars:

            if isinstance(st, PSFStar):
                cst = self._fit_star(psf, st, self.fitter, fitter_kwargs,
                                     fitter_has_fit_info,
                                     width, height, igx, igy)
                # cst = PSFStar
            elif isinstance(st, LinkedPSFStar):
                cst = self._fit_star(psf, st, self.fitter, fitter_kwargs,
                                     fitter_has_fit_info,
                                     width, height, igx, igy)
                #cst.constrain_linked_centers(ignore_badfit_stars=True)
                # cst = LinkedPSFStar
            else:
                raise ValueError('invalid psf_star type')

            fitted_stars.append(cst)

        return PSFStars(fitted_stars)
