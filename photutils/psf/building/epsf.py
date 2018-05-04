# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tools to build an ePSF.
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import copy
import warnings

import numpy as np
from astropy.stats import SigmaClip
from astropy.table import Table

from .centroid import find_peak
from .epsf_fitter import EPSFFitter, compute_residuals
from .models import NonNormalizable, PSF2DModel
from .utils import (py2round, interpolate_missing_data, _pixstat, _smoothPSF,
                    _parse_tuple_pars)


__all__ = ['EPSFBuilder']


class EPSFBuilder(object):
    """
    Class to build the ePSF.

    Parameters
    ----------
        # NOTE: center_accuracy_sq applies to each star
    """

    def __init__(self, peak_fit_box=5, peak_search_box='fitbox',
                 recenter_accuracy=1.0e-4, recenter_max_iters=1000,
                 stat='median',
                 sigma_clip=SigmaClip(sigma=3., iters=10),
                 smoothing_kernel='quar', fitter=EPSFFitter(residuals=True),
                 max_iters=50, center_accuracy=1.0e-4, epsf=None):

        self.peak_fit_box = peak_fit_box
        self.peak_search_box = peak_search_box

        recenter_accuracy = float(recenter_accuracy)
        if recenter_accuracy <= 0.0:
            raise ValueError('recenter_accuracy must be a strictly positive '
                             'number.')
        self.recenter_accuracy = recenter_accuracy

        recenter_max_iters = int(recenter_max_iters)
        if recenter_max_iters <= 0:
            raise ValueError('recenter_max_iters must be a positive integer.')
        self.recenter_max_iters = recenter_max_iters

        self.stat = stat
        self.sigma_clip = sigma_clip
        self.smoothing_kernel = smoothing_kernel
        self.fitter = fitter

        max_iters = int(max_iters)
        if max_iters <= 0:
            raise ValueError('max_iters must be a positive number.')
        self.max_iters = max_iters

        if center_accuracy <= 0.0:
            raise ValueError('center_accuracy must be a positive number.')
        self.center_accuracy_sq = center_accuracy**2

        self.epsf = epsf

    def __call__(self, psfstars):
        return self.build_psf(psfstars)

    def _build_psf_step(self, psf_stars, psf=None):
        if len(psf_stars) < 1:
            raise ValueError('psf_stars must contain at least one PSFStar '
                             'or LinkedPSFStar object.')

        if psf is None:
            psf = init_psf(psf_stars)
        else:
            psf = copy.deepcopy(psf)

        cx, cy = psf.origin
        ny, nx = psf.shape
        pscale_x, pscale_y = psf.pixel_scale

        # allocate "accumulator" array (to store transformed PSFs):
        apsf = [[[] for k in range(nx)] for k in range(ny)]

        norm_psf_data = psf.normalized_data

        for psf_star in psf_stars.all_psfstars:
            if psf_star._excluded_from_fit:
                continue

            # evaluate previous PSF model at star pixel location in
            # the PSF grid
            ovx = psf_star.pixel_scale[0] / pscale_x
            ovy = psf_star.pixel_scale[1] / pscale_y
            x = ovx * psf_star._xidx_centered
            y = ovy * psf_star._yidx_centered
            old_model_vals = psf.evaluate(x=x, y=y, flux=1.0, x_0=0.0,
                                          y_0=0.0)

            # find integer location of star pixels in the PSF grid
            # and compute residuals
            ix = py2round(x + cx).astype(np.int)
            iy = py2round(y + cy).astype(np.int)
            pv = (psf_star._data_values_normalized / (ovx * ovy) -
                  old_model_vals)
            m = np.logical_and(np.logical_and(ix >= 0, ix < nx),
                               np.logical_and(iy >= 0, iy < ny))

            # add all pixel values to the corresponding accumulator
            for i, j, v in zip(ix[m], iy[m], pv[m]):
                apsf[j][i].append(v)

        psfdata = np.empty((ny, nx), dtype=np.float)
        psfdata.fill(np.nan)

        for i in range(nx):
            for j in range(ny):
                psfdata[j, i] = _pixstat(apsf[j][i], stat=self.stat,
                                         sigma_clip=self.sigma_clip,
                                         default=np.nan)

        mask = np.isfinite(psfdata)
        if not np.all(mask):
            # fill in the "holes" (=np.nan) using interpolation
            # I. using cubic spline for inner holes
            psfdata = interpolate_missing_data(psfdata, method='cubic',
                                               mask=mask)

            # II. we fill outer points with zeros
            mask = np.isfinite(psfdata)
            psfdata[np.logical_not(mask)] = 0.0

        # add residuals to old PSF data:
        psfdata += norm_psf_data

        # apply a smoothing kernel to the PSF:
        psfdata = _smoothPSF(psfdata, self.smoothing_kernel)

        shift_x = 0
        shift_y = 0
        peak_eps_sq = self.recenter_accuracy**2
        eps_sq_prev = None
        y, x = np.indices(psfdata.shape, dtype=np.float)
        ePSF = psf.make_similar_from_data(psfdata)
        ePSF.fill_value = 0.0

        for iteration in range(self.recenter_max_iters):
            # find peak location:
            peak_x, peak_y = find_peak(psfdata, xmax=cx, ymax=cy,
                                       peak_fit_box=self.peak_fit_box,
                                       peak_search_box=self.peak_search_box,
                                       mask=None)

            dx = cx - peak_x
            dy = cy - peak_y

            eps_sq = dx**2 + dy**2
            if ((eps_sq_prev is not None and eps_sq > eps_sq_prev)
                    or eps_sq < peak_eps_sq):
                break
            eps_sq_prev = eps_sq

            shift_x += dx
            shift_y += dy

            # Resample PSF data to a shifted grid such that the pick of
            # the PSF is at expected position
            psfdata = ePSF.evaluate(x=x, y=y, flux=1.0,
                                    x_0=shift_x + cx, y_0=shift_y + cy)

        # apply final shifts and fill in any missing data
        if shift_x != 0.0 or shift_y != 0.0:
            ePSF.fill_value = np.nan
            psfdata = ePSF.evaluate(x=x, y=y, flux=1.0,
                                    x_0=shift_x + cx, y_0=shift_y + cy)

            # fill in the "holes" (=np.nan) using 0 (no contribution to
            # the flux)
            mask = np.isfinite(psfdata)
            psfdata[np.logical_not(mask)] = 0.0

        norm = np.abs(np.sum(psfdata, dtype=np.float64))
        psfdata /= norm

        # Create ePSF model and return:
        ePSF = psf.make_similar_from_data(psfdata)

        return ePSF

    def build_psf(self, psf_stars, psf=None):
        """
        Iteratively build the PSF.

        Parameters
        ----------
        psf_stars : `PSFStars` object

        psf : `FittableImageModel2D` or `None`, optional

        Returns
        -------
        psf : `FittableImageModel2D`
            The constructed ePSF.

        psf_stars : `PSFStars` object
            The PSF stars with updated centers and fluxes from
            fitting the ``psf``.
        """

        self.psf_stars = psf_stars

        iter_num = 0
        center_dist_sq = self.center_accuracy_sq + 1
        centers = psf_stars.cutout_center
        nstars = psf_stars.n_psfstars
        fit_failed = np.zeros(nstars, dtype=bool)
        dx_dy = np.zeros((nstars, 2), dtype=np.float)

        while (iter_num < self.max_iters and
                np.max(center_dist_sq) >= self.center_accuracy_sq and
                not np.all(fit_failed)):

            iter_num += 1

            print('iter_num', iter_num)
            print(psf_stars)

            # build/improve the PSF
            psf = self._build_psf_step(psf_stars, psf=psf)

            # fit the new PSF to the psf_stars to find improved centers
            psf_stars = self.fitter(psf, psf_stars)

            print(psf_stars)

            # find all psf stars where the fit failed
            fit_failed = np.array([psf_star.fit_error_status > 0
                                   for psf_star in psf_stars.all_psfstars])

            # permanently exclude fitting any psf star where the fit
            # fails after 3 iterations
            if iter_num > 3 and np.any(fit_failed):
                #for (psf_star, failed) in zip(psf_stars.all_psfstars,
                #                              fit_failed_new):
                #    psf_star.fit_failed = failed
                psf_stars.all_psfstars[fit_failed]._excluded_from_fit = True

            dx_dy = psf_stars.cutout_center - centers
            dx_dy = dx_dy[np.logical_not(fit_failed)]
            center_dist_sq = np.sum(dx_dy * dx_dy, axis=1, dtype=np.float64)
            centers = psf_stars.cutout_center

        return psf, psf_stars


def init_psf(psf_stars, shape=None, oversampling=4.0, pixel_scale=None,
             psf_cls=PSF2DModel, **kwargs):
    """
    Parameters
    ----------
    stars : Star, list of Star
        A list of :py:class:`~psfutils.catalogs.Star` objects
        containing image data of star cutouts that are used to "build" a PSF.

    psf : PSF2DModel, type, None, optional
        An existing approximation to the PSF which needs to be recomputed
        using new ``stars`` (with new parameters for center and flux)
        or a class indicating the type of the ``psf`` object to be created
        (preferebly a subclass of :py:class:`PSF2DModel`).
        If `psf` is `None`, a new ``psf`` will be computed using
        :py:class:`PSF2DModel`.

    shape : tuple, optional
        Numpy-style shape of the output PSF. If shape is not specified (i.e.,
        it is set to `None`), the shape will be derived from the sizes of the
        input star models. This is ignored if `psf` is not `None`.

    oversampling : float, tuple of float, list of (float, tuple of float), \
optional
        Oversampling factor of the PSF relative to star. It indicates how many
        times a pixel in the star's image should be sampled when creating PSF.
        If a single number is provided, that value will be used for both
        ``X`` and ``Y`` axes and for all stars. When a tuple is provided,
        first number will indicate oversampling along the ``X`` axis and the
        second number will indicate oversampling along the ``Y`` axis. It is
        also possible to have individualized oversampling factors for each star
        by providing a list of integers or tuples of integers.

    """
    # check parameters:
    if pixel_scale is None and oversampling is None:
        raise ValueError(
            "'oversampling' and 'pixel_scale' cannot be 'None' together. "
            "At least one of these two parameters must be provided."
        )

    # find pixel scale:
    if pixel_scale is None:
        ovx, ovy = _parse_tuple_pars(oversampling, name='oversampling',
                                     dtype=float)

        # compute PSF's pixel scale as the smallest scale of the stars
        # divided by the requested oversampling factor:
        pscale_x, pscale_y = psf_stars.all_psfstars[0].pixel_scale
        for psf_star in psf_stars.all_psfstars[1:]:
            px, py = psf_star.pixel_scale
            if px < pscale_x:
                pscale_x = px
            if py < pscale_y:
                pscale_y = py

        pscale_x /= ovx
        pscale_y /= ovy

    else:
        pscale_x, pscale_y = _parse_tuple_pars(pixel_scale, name='pixel_scale',
                                               dtype=float)

    # if shape is None, find the minimal shape that will include input star's
    # data:
    if shape is None:
        for psf_star in psf_stars.all_psfstars:
            w = np.array([((psf_star.cutout_center[0] + 0.5) *
                           psf_star.pixel_scale[0] / pscale_x,
                           (psf_star.shape[1] - psf_star.cutout_center[0] -
                            0.5) * psf_star.pixel_scale[0] / pscale_x)])

            h = np.array([((psf_star.cutout_center[1] + 0.5) *
                           psf_star.pixel_scale[1] / pscale_y,
                           (psf_star.shape[0] - psf_star.cutout_center[1] -
                            0.5) * psf_star.pixel_scale[1] / pscale_y)])

        # size of the PSF in the input image pixels
        # (the image with min(pixel_scale))
        nx = int(np.ceil(np.amax(w / ovx + 0.5)))
        ny = int(np.ceil(np.amax(h / ovy + 0.5)))

        # account for a maximum error of 1 pix in the initial star
        # coordinates
        nx += 2
        ny += 2

        # convert to oversampled pixels
        nx = int(np.ceil(np.amax(nx * ovx + 0.5)))
        ny = int(np.ceil(np.amax(ny * ovy + 0.5)))

        # we prefer odd sized images
        nx += 1 - nx % 2
        ny += 1 - ny % 2
        shape = (ny, nx)

    else:
        (ny, nx) = _parse_tuple_pars(shape, name='shape', dtype=int)

    # center of the output grid:
    cx = (nx - 1) / 2.0
    cy = (ny - 1) / 2.0

    with warnings.catch_warnings(record=False):
        warnings.simplefilter("ignore", NonNormalizable)

        # filter out parameters set by us:
        kwargs = copy.deepcopy(kwargs)
        for kw in ['data', 'origin', 'normalize', 'pixel_scale']:
            if kw in kwargs:
                del kwargs[kw]

        data = np.zeros((ny, nx), dtype=np.float)
        psf = psf_cls(
            data=data, origin=(cx, cy), normalize=True,
            pixel_scale=(pscale_x, pscale_y), **kwargs
        )

    return psf
