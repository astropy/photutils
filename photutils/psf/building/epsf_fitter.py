# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import copy
import warnings

import numpy as np
from astropy.modeling.fitting import LevMarLSQFitter
from astropy.nddata.utils import (overlap_slices, PartialOverlapError,
                                  NoOverlapError)
from astropy.utils.exceptions import AstropyUserWarning

from .psfstars import PSFStar, LinkedPSFStar, PSFStars


__all__ = ['EPSFFitter']


class EPSFFitter(object):
    """
    Class to fit an ePSF model to stars.

    Parameters
    ----------
    fitter : `astropy.modeling.fitting.Fitter`, optional
        A :py:class:`~astropy.modeling.fitting.Fitter` object.  The
        default is `~astropy.modeling.fitting.LevMarLSQFitter`.

    fit_boxsize : int, tuple of int, or `None`, optional
        The size (in pixels) of the box centered on the star to be used
        for PSF fitting.  This allows using only a small number of
        central pixels of the star (i.e. where the star is brightest)
        for fitting.  If ``fit_boxsize`` is a scalar then a square box
        of size ``fit_boxsize`` will be used.  If ``fit_boxsize`` has
        two elements, they should be in ``(ny, nx)`` order.  The size
        must be greater than or equal to 3 pixels for both axes.  If
        `None`, the fitter will use the entire star image.  The default
        is 5.

    fitter_kwargs : dict-like, optional
        Any keyword arguments to be passed directly to the
        ``__call__()`` method of the input ``fitter``.
    """

    def __init__(self, fitter=LevMarLSQFitter(), fit_boxsize=5,
                 **fitter_kwargs):

        self.fitter = fitter
        self.fitter_has_fit_info = hasattr(self.fitter, 'fit_info')

        if fit_boxsize is not None:
            fit_boxsize = np.atleast_1d(fit_boxsize).astype(int)
            if len(fit_boxsize) == 1:
                fit_boxsize = np.repeat(fit_boxsize, 2)

            min_size = 3
            if any([size < min_size for size in fit_boxsize]):
                raise ValueError('size must be >= {} for x and y'.
                                 format(min_size))

        self.fit_boxsize = fit_boxsize

        # remove fitter keyword arguments that we set ourselves
        remove_kwargs = ['x', 'y', 'z', 'weights']
        fitter_kwargs = copy.deepcopy(fitter_kwargs)
        for kwarg in remove_kwargs:
            if kwarg in fitter_kwargs:
                del fitter_kwargs[kwarg]
        self.fitter_kwargs = fitter_kwargs

    def __call__(self, psf, psf_stars):
        """
        Fit an ePSF model to stars.

        Parameters
        ----------
        psf : `EPSFModel`
            A PSF model to be fitted to the stars.

        psf_stars : `PSFStars` object
            The PSF stars to be fit.

        stars : Star, list of Star
            A list of :py:class:`~psfutils.catalogs.Star` objects
            containing image data of star cutouts to which the PSF must
            be fitted.  Fitting procedure relies on correct coordinates
            of the center of the PSF and as close as possible to the
            correct center positions of stars.  Star positions are
            derived from ``x_0`` and ``y_0`` parameters of the
            `EPSFModel` model.

            When models in ``stars`` contain weights, a weighted fit of
            the PSF to the stars will be performed.


        Returns
        -------
        fitted_stars : list of FittableImageModel2D
            A list of `~psfutils.models.FittableImageModel2D` of stars
            with model parameters
            `~psfutils.models.FittableImageModel2D.x_0` and
            `~psfutils.models.FittableImageModel2D.y_0` set to 0 and
            `~psfutils.models.FittableImageModel2D.origin` will show
            fitted center of the star. If `update_flux` was `True`, the
            `~psfutils.models.FittableImageModel2D.flux` model parameter
            will contain fitted flux and the original star's flux
            otherwise.
        """

        if len(psf_stars) == 0:
            return []

        # make a copy of the input PSF
        psf = psf.copy()

        # perform the fit
        fitted_stars = []
        for psf_star in psf_stars:
            if isinstance(psf_star, PSFStar):
                fitted_star = self._fit_star(psf, psf_star, self.fitter,
                                             self.fitter_kwargs,
                                             self.fitter_has_fit_info,
                                             self.fit_boxsize)

            elif isinstance(psf_star, LinkedPSFStar):
                fitted_star = []
                for linked_star in LinkedPSFStar:
                    fitted_star.append(
                        self._fit_star(psf, linked_star, self.fitter,
                                       self.fitter_kwargs,
                                       self.fitter_has_fit_info,
                                       self.fit_boxsize))

                fitted_star = LinkedPSFStar(fitted_star)
                fitted_star.constrain_centers()

            else:
                raise ValueError('invalid star type in psf_stars')

            fitted_stars.append(fitted_star)

        return PSFStars(fitted_stars)

    def _fit_star(self, psf, psf_star, fitter, fitter_kwargs,
                  fitter_has_fit_info, fit_boxsize):
        """
        Fit a single star with a PSF model.

        The input ``psf`` will generally be modified by fitting routine
        in this function.  Make a copy if it is important to preserve
        the input ``psf`` model.
        """

        if fit_boxsize is not None:
            try:
                xcenter, ycenter = psf_star.cutout_center
                large_slc, small_slc = overlap_slices(psf_star.shape,
                                                      fit_boxsize,
                                                      (ycenter, xcenter),
                                                      mode='strict')
            except (PartialOverlapError, NoOverlapError):
                warnings.warn('The star at ({0}, {1}) is being ignored '
                              'because its fitting region extends beyond '
                              'the image.'.format(psf_star.center[0],
                                                  psf_star.center[1]),
                              AstropyUserWarning)

                psf_star = copy.deepcopy(psf_star)
                psf_star._fit_error_status = 1

                return psf_star

            data = psf_star.data[large_slc]
            weights = psf_star.weights[large_slc]

            # define the origin of the fitting region
            x0 = large_slc[1].start
            y0 = large_slc[0].start
        else:
            data = psf_star.data
            weights = psf_star.weights

            # define the origin of the fitting region
            x0 = 0
            y0 = 0

        x_oversamp = psf_star.pixel_scale[0] / psf.pixel_scale[0]
        y_oversamp = psf_star.pixel_scale[1] / psf.pixel_scale[1]
        scaled_data = data / (x_oversamp * y_oversamp)

        # define positions in the PSF oversampled grid
        yy, xx = np.indices(data.shape, dtype=np.float)
        xx = (xx - (psf_star.cutout_center[0] - x0)) * x_oversamp
        yy = (yy - (psf_star.cutout_center[1] - y0)) * y_oversamp

        # define the initial guesses for fitted flux and shifts
        psf.flux = psf_star.flux
        psf.x_0 = 0.0
        psf.y_0 = 0.0

        try:
            fitted_psf = fitter(model=psf, x=xx, y=yy, z=scaled_data,
                                weights=weights, **fitter_kwargs)
        except TypeError:
            # fitter doesn't support weights
            fitted_psf = fitter(model=psf, x=xx, y=yy, z=scaled_data,
                                **fitter_kwargs)

        fit_error_status = 0
        if fitter_has_fit_info:
            fit_info = copy.copy(fitter.fit_info)

            if 'ierr' in fit_info and fit_info['ierr'] not in [1, 2, 3, 4]:
                fit_error_status = 2    # fit solution was not found
        else:
            fit_info = None

        # compute the star's fitted position
        x_center = (psf_star.cutout_center[0] +
                    (fitted_psf.x_0.value / x_oversamp))
        y_center = (psf_star.cutout_center[1] +
                    (fitted_psf.y_0.value / y_oversamp))

        psf_star = copy.deepcopy(psf_star)
        psf_star.cutout_center = (x_center, y_center)

        # set the star's flux to the ePSF-fitted flux
        psf_star.flux = fitted_psf.flux.value

        psf_star._fit_info = fit_info
        psf_star._fit_error_status = fit_error_status

        return psf_star
