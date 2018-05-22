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
from astropy.utils.exceptions import AstropyUserWarning

from .centroid import find_peak
from .epsf_fitter import EPSFFitter
from .models import PSF2DModel


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

    smoothing_kernel : {'quadratic', 'quartic'} or 2D `~numpy.ndarray`
        The smoothing kernel to apply.  The predefined kernels
        ``'quadratic'`` and ``'quartic'`` or a 2D array can be input.
    """

    def __init__(self, peak_fit_box=5, peak_search_box='fitbox',
                 recenter_accuracy=1.0e-4, recenter_max_iters=1000,
                 smoothing_kernel='quartic',
                 fitter=EPSFFitter(residuals=True), max_iters=50,
                 center_accuracy=1.0e-4, epsf=None,
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

    def _resample_residual(self, psf_star, psf):
        """
        Resample a single PSF star to the (oversampled) grid of the
        input PSF.

        Parameters
        ----------
        psf_star : `PSFStar` object
            A single PSF star object to be resampled.

        psf : `PSF2DModel` object, optional
            The PSF model.

        Returns
        -------
        image : 2D `~numpy.ndarray`
            A 2D image containing the resampled PSF star image.  The
            image contains NaNs where there is no data.
        """

        # find the integer index of PSFStar pixels in the oversampled
        # PSF grid
        x_oversamp = psf_star.pixel_scale[0] / psf.pixel_scale[0]
        y_oversamp = psf_star.pixel_scale[1] / psf.pixel_scale[1]
        x = x_oversamp * psf_star._xidx_centered
        y = y_oversamp * psf_star._yidx_centered
        psf_xcenter, psf_ycenter = psf.origin
        xidx = _py2intround(x + psf_xcenter).astype(np.int)
        yidx = _py2intround(y + psf_ycenter).astype(np.int)

        ny, nx = psf.shape
        mask = np.logical_and(np.logical_and(xidx >= 0, xidx < nx),
                              np.logical_and(yidx >= 0, yidx < ny))
        xidx = xidx[mask]
        yidx = yidx[mask]

        # Compute the normalized residual image by subtracting the
        # normalized PSF model from the normalized PSF star at the
        # location of the PSF star in the undersampled grid.  Then,
        # reample the normalized residual image in the oversampled
        # PSF grid
        # ((star_data_norm - (psf.eval(flux=1, ...) * xov * yov)) /
        #  (xov * yov)) = ((star_data_norm / (xov * yov)) -
        #                  psf.eval(flux=1, ...))
        stardata = ((psf_star._data_values_normalized /
                     (x_oversamp * y_oversamp)) -
                    psf.evaluate(x=x, y=y, flux=1.0, x_0=0.0, y_0=0.0))

        resampled_img = np.full(psf.shape, np.nan)
        resampled_img[yidx, xidx] = stardata[mask]

        return resampled_img

    def _resample_residuals(self, psf_stars, psf):
        """
        Resample PSF stars to the (oversampled) grid of the input PSF.

        Parameters
        ----------
        psf_stars : `PSFStars` object
            The PSF stars used to build the PSF.

        psf : `PSF2DModel` object, optional
            The PSF model.

        Returns
        -------
        star_imgs : 3D `~numpy.ndarray`
            A 3D cube containing the resampled PSF star images.
        """

        star_imgs = np.zeros((psf_stars.n_good_psfstars, *psf.shape))
        for i, psf_star in enumerate(psf_stars.all_good_psfstars):
            star_imgs[i, :, :] = self._resample_residual(psf_star, psf)

        return star_imgs

    @staticmethod
    def _interpolate_missing_data(data, mask, method='cubic'):
        """
        Interpolate missing data as identified by the ``mask`` keyword.

        Parameters
        ----------
        data : 2D `~numpy.ndarray`
            An array containing the 2D image.

        mask : 2D bool `~numpy.ndarray`
            A 2D booleen mask array with the same shape as the input
            ``data``, where a `True` value indicates the corresponding
            element of ``data`` is masked.  The masked data points are
            those that will be interpolated.

        method : {'cubic', 'nearest'}, optional
            The method of used to "interpolate" the  missing data:

            - ``'cubic'``:  Masked data are interpolated using 2D cubic
              splines.  This is the default.

            - ``'nearest'``:  Masked data are interpolated using
              nearest-neighbor interpolation.

        Returns
        -------
        data_interp : 2D `~numpy.ndarray`
            The resulting interpolated 2D image.
        """

        from scipy import interpolate

        data_interp = np.array(data, copy=True)

        if len(data_interp.shape) != 2:
            raise ValueError('data must be a 2D array.')

        if mask.shape != data.shape:
            raise ValueError('mask and data must have the same shape.')

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

    def _smooth_psf(self, psf_data):
        """
        Smooth the PSF array by convolving it with a kernel.

        Parameters
        ----------
        psf_data : 2D `~numpy.ndarray`
            A 2D array containing the PSF image.

        Returns
        -------
        result : 2D `~numpy.ndarray`
            The smoothed (convolved) PSF data.
        """

        from scipy.ndimage import convolve

        if self.smoothing_kernel == 'quadratic':
            kernel = np.array(
                [[-0.07428311, 0.01142786, 0.03999952, 0.01142786,
                  -0.07428311],
                 [+0.01142786, 0.09714283, 0.12571449, 0.09714283,
                  +0.01142786],
                 [+0.03999952, 0.12571449, 0.15428215, 0.12571449,
                  +0.03999952],
                 [+0.01142786, 0.09714283, 0.12571449, 0.09714283,
                  +0.01142786],
                 [-0.07428311, 0.01142786, 0.03999952, 0.01142786,
                  -0.07428311]])

        elif self.smoothing_kernel == 'quartic':
            kernel = np.array(
                [[+0.041632, -0.080816, 0.078368, -0.080816, +0.041632],
                 [-0.080816, -0.019592, 0.200816, -0.019592, -0.080816],
                 [+0.078368, +0.200816, 0.441632, +0.200816, +0.078368],
                 [-0.080816, -0.019592, 0.200816, -0.019592, -0.080816],
                 [+0.041632, -0.080816, 0.078368, -0.080816, +0.041632]])

        elif isinstance(self.smoothing_kernel, np.ndarray):
            kernel = self.kernel

        else:
            raise TypeError("Unsupported kernel.")

        return convolve(psf_data, kernel)

    def _recenter_psf(self, psf_data, psf):
        """
        Recenter the PSF.
        """

        shift_x = 0
        shift_y = 0
        peak_eps_sq = self.recenter_accuracy**2
        eps_sq_prev = None
        y, x = np.indices(psf_data.shape, dtype=np.float)

        ePSF = psf.make_similar_from_data(psf_data)
        ePSF.fill_value = 0.0

        cx, cy = psf.origin
        for iteration in range(self.recenter_max_iters):
            # find peak location:
            peak_x, peak_y = find_peak(psf_data, xmax=cx, ymax=cy,
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
            psf_data = ePSF.evaluate(x=x, y=y, flux=1.0, x_0=shift_x + cx,
                                     y_0=shift_y + cy)

        # apply final shifts and fill in any missing data
        if shift_x != 0.0 or shift_y != 0.0:
            ePSF.fill_value = np.nan
            psf_data = ePSF.evaluate(x=x, y=y, flux=1.0, x_0=shift_x + cx,
                                     y_0=shift_y + cy)

            # fill in the "holes" (=np.nan) using 0 (no contribution to
            # the flux)
            psf_data[~np.isfinite(psf_data)] = 0.

        return psf_data

    def _build_psf_step(self, psf_stars, psf=None):
        """
        A single iteration of improving a PSF.

        Parameters
        ----------
        psf_stars : `PSFStars` object
            The PSF stars used to build the PSF.

        psf : `PSF2DModel` object, optional
            The initial PSF model.  If not input, then the PSF will be
            built from scratch.

        Returns
        -------
        psf : `PSF2DModel` object
            The improved PSF.
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

        # compute a 3D stack of 2D residual images
        residuals = self._resample_residuals(psf_stars, psf)

        # compute the sigma-clipped median along the 3D stack
        # TODO: allow custom SigmaClip/statistic class
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
            warnings.simplefilter('ignore', category=AstropyUserWarning)
            sigclip = SigmaClip(sigma=3., cenfunc=np.ma.median, iters=10)
            residuals = sigclip(residuals, axis=0)
            residuals = np.ma.median(residuals, axis=0)
            residuals = residuals.filled(np.nan)

        # interpolate any missing data (np.nan)
        mask = ~np.isfinite(residuals)
        if np.any(mask):
            residuals = self._interpolate_missing_data(residuals, mask,
                                                       method='cubic')

            # fill any remaining nans (outer points) with zeros
            residuals[~np.isfinite(residuals)] = 0.

        # add the residuals to the previous PSF image
        new_psf = psf.normalized_data + residuals

        # smooth the PSF
        new_psf = self._smooth_psf(new_psf)

        # recenter the PSF
        new_psf = self._recenter_psf(new_psf, psf)

        norm = np.abs(np.sum(new_psf, dtype=np.float64))
        new_psf /= norm

        # Create ePSF model and return:
        ePSF = psf.make_similar_from_data(new_psf)

        return ePSF

    def build_psf(self, psf_stars, init_psf=None):
        """
        Iteratively build a PSF from star cutouts.

        If the optional ``psf`` is input, then it will be used as the
        initial PSF.

        Parameters
        ----------
        psf_stars : `PSFStars` object
            The PSF stars used to build the PSF.

        init_psf : `PSF2DModel` object, optional
            The initial PSF model.  If not input, then the PSF will be
            built from scratch.

        Returns
        -------
        psf : `PSF2DModel` object
            The constructed PSF.

        fit_psf_stars : `PSFStars` object
            The input PSF stars with updated centers and fluxes derived
            by fitting the output ``psf``.
        """

        iter_num = 0
        center_dist_sq = self.center_accuracy_sq + 1.
        centers = psf_stars.cutout_center
        n_stars = psf_stars.n_psfstars
        fit_failed = np.zeros(n_stars, dtype=bool)
        dx_dy = np.zeros((n_stars, 2), dtype=np.float)
        psf = init_psf

        while (iter_num < self.max_iters and
                np.max(center_dist_sq) >= self.center_accuracy_sq and
                not np.all(fit_failed)):

            iter_num += 1
            print('iter', iter_num)

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


def _py2intround(a):
    """
    Round the input to the nearest integer (returned as a float).

    If two integers are equally close, rounding is done away from 0.
    """

    data = np.asanyarray(a)
    value = np.where(data >= 0, np.floor(data + 0.5), np.ceil(data - 0.5))

    if not hasattr(a, '__iter__'):
        value = float(value)

    return value
