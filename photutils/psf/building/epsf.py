# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tools to build an ePSF.
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import copy

import numpy as np
from astropy.stats import SigmaClip

from .centroid import find_peak
from .epsf_fitter import EPSFFitter
from .models import PSF2DModel
from .utils import py2round, interpolate_missing_data, _pixstat, _smoothPSF


__all__ = ['EPSFBuilder']


class EPSFBuilder(object):
    """
    Class to build the ePSF.

    Parameters
    ----------
        # NOTE: center_accuracy_sq applies to each star

    oversampling : float, optional
        Determines the output ePSF pixel scale, relative to PSFstar cutout
        data.
    """

    def __init__(self, peak_fit_box=5, peak_search_box='fitbox',
                 recenter_accuracy=1.0e-4, recenter_max_iters=1000,
                 stat='median',
                 sigma_clip=SigmaClip(sigma=3., iters=10),
                 smoothing_kernel='quar', fitter=EPSFFitter(residuals=True),
                 max_iters=50, center_accuracy=1.0e-4, epsf=None,
                 epsf_shape=None, oversampling=4.):

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
        self.epsf_shape = epsf_shape
        self.oversampling = oversampling

    def __call__(self, psfstars):
        return self.build_psf(psfstars)

    def _build_psf_step(self, psf_stars, psf=None):
        """
        A single iteration of improving a PSF.

        Parameters
        ----------
        psf_stars : `PSFStars` object
        """

        if len(psf_stars) < 1:
            raise ValueError('psf_stars must contain at least one PSFStar '
                             'or LinkedPSFStar object.')

        if psf is None:
            # create an initial PSF (array of zeros)
            psf = _create_initial_psf(psf_stars,
                                      oversampling=self.oversampling,
                                      shape=self.epsf_shape)
        else:
            # improve the input PSF
            psf = copy.deepcopy(psf)

        #TODO
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

        iter_num = 0
        center_dist_sq = self.center_accuracy_sq + 1.
        centers = psf_stars.cutout_center
        nstars = psf_stars.npsfstars
        fit_failed = np.zeros(nstars, dtype=bool)
        dx_dy = np.zeros((nstars, 2), dtype=np.float)

        while (iter_num < self.max_iters and
                np.max(center_dist_sq) >= self.center_accuracy_sq and
                not np.all(fit_failed)):

            iter_num += 1

            # build/improve the PSF
            psf = self._build_psf_step(psf_stars, psf=psf)

            # fit the new PSF to the psf_stars to find improved centers
            psf_stars = self.fitter(psf, psf_stars)

            # find all psf stars where the fit failed
            fit_failed = np.array([psf_star.fit_error_status > 0
                                   for psf_star in psf_stars.all_psfstars])

            # permanently exclude fitting any psf star where the fit
            # fails after 3 iterations
            if iter_num > 3 and np.any(fit_failed):
                idx = fit_failed.nonzero()[0]
                for i in idx:
                    psf_stars.all_psfstars[i]._excluded_from_fit = True

            dx_dy = psf_stars.cutout_center - centers
            dx_dy = dx_dy[np.logical_not(fit_failed)]
            center_dist_sq = np.sum(dx_dy * dx_dy, axis=1, dtype=np.float64)
            centers = psf_stars.cutout_center

        return psf, psf_stars


def _create_initial_psf(psf_stars, pixel_scale=None, oversampling=None,
                        shape=None):
    """
    Create an initial `PSF2DModel` object.

    The initial PSF data are all zeros.  The PSF pixel scale is
    determined either from the input ``pixel_scale`` or ``oversampling``
    keyword.

    If ``shape`` is not input, the shape of the PSF data array is
    determined from the shape of the input ``psf_stars`` and the
    oversampling factor (derived from the ratio of the ``psf_star``
    pixel scale to the PSF pixel scale).  The output PSF will always
    have odd sizes along both axes.

    Parameters
    ----------
    psf_stars : `PSFStars` object
        The PSF stars used to build the PSF.

    pixel_scale : float or tuple of two floats, optional
        The pixel scale (in arbitrary units) of the output PSF.  The
        ``pixel_scale`` can either be a single float or tuple of two
        floats of the form ``(x_pixscale, y_pixscale)``.  If
        ``pixel_scale`` is a scalar then the pixel scale will be the
        same for both the x and y axes.  The PSF ``pixel_scale`` is used
        in conjunction with the star pixel scale when building and
        fitting the PSF.  This allows for building (and fitting) a PSF
        using images of stars with different pixel scales (e.g. velocity
        aberrations).  Either ``oversampling`` or ``pixel_scale`` must
        be input.  If both are input, ``pixel_scale`` will override the
        input ``oversampling``.

    oversampling : float or tuple of two floats, optional
        The oversampling factor(s) of the PSF relative to the input
        ``psf_stars`` along the x and y axes.  The ``oversampling`` can
        either be a single float or a tuple of two floats of the form
        ``(x_oversamp, y_oversamp)``.  If ``oversampling`` is a scalar
        then the oversampling will be the same for both the x and y
        axes.  Either ``oversampling`` or ``pixel_scale`` must be input.
        If both are input, ``oversampling`` will be ignored.

    shape : tuple, optional
        The shape of the output PSF.  If the ``shape`` is not input, it
        will be derived from the sizes of the input ``psf_stars`` and
        the PSF oversampling factor.  The output PSF will always have
        odd sizes along both axes.

    Returns
    -------
    psf : `PSF2DModel`
        The initial PSF model.
    """

    if pixel_scale is None and oversampling is None:
        raise ValueError('Either pixel_scale or oversampling must be input.')

    # define the PSF pixel scale
    if pixel_scale is not None:
        pixel_scale = np.atleast_1d(pixel_scale).astype(float)
        if len(pixel_scale) == 1:
            pixel_scale = np.repeat(pixel_scale, 2)

        oversampling = (psf_stars._min_pixel_scale[0] / pixel_scale[0],
                        psf_stars._min_pixel_scale[1] / pixel_scale[1])
    else:
        oversampling = np.atleast_1d(oversampling).astype(float)
        if len(oversampling) == 1:
            oversampling = np.repeat(oversampling, 2)

        pixel_scale = (psf_stars._min_pixel_scale[0] / oversampling[0],
                       psf_stars._min_pixel_scale[1] / oversampling[1])

    # define the PSF shape
    if shape is not None:
        shape = np.atleast_1d(shape).astype(int)
        if len(shape) == 1:
            shape = np.repeat(shape, 2)
    else:
        x_shape = np.int(np.ceil(psf_stars._max_shape[0] * oversampling[1]))
        y_shape = np.int(np.ceil(psf_stars._max_shape[1] * oversampling[0]))
        shape = np.array((y_shape, x_shape))

    shape = [(i + 1) for i in shape if i % 2 == 0]   # ensure odd sizes

    xcenter = (shape[1] - 1) / 2.
    ycenter = (shape[0] - 1) / 2.
    data = np.zeros(shape, dtype=np.float)

    return PSF2DModel(data=data, origin=(xcenter, ycenter), normalize=False,
                      pixel_scale=pixel_scale)
