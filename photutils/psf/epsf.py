# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
This module provides tools to build and fit an effective PSF (ePSF)
based on Anderson and King (2000; PASP 112, 1360) and Anderson (2016;
WFC3 ISR 2016-12).
"""

import copy
import warnings

import numpy as np
from astropy.modeling.fitting import LevMarLSQFitter
from astropy.nddata.utils import (NoOverlapError, PartialOverlapError,
                                  overlap_slices)
from astropy.stats import SigmaClip
from astropy.utils.exceptions import AstropyUserWarning

from photutils.centroids import centroid_com
from photutils.psf.epsf_stars import EPSFStar, EPSFStars, LinkedEPSFStar
from photutils.psf.models import EPSFModel
from photutils.psf.utils import _interpolate_missing_data
from photutils.utils._optional_deps import HAS_BOTTLENECK
from photutils.utils._parameters import as_pair
from photutils.utils._progress_bars import add_progress_bar
from photutils.utils._round import py2intround

__all__ = ['EPSFFitter', 'EPSFBuilder']


class EPSFFitter:
    """
    Class to fit an ePSF model to one or more stars.

    Parameters
    ----------
    fitter : `astropy.modeling.fitting.Fitter`, optional
        A `~astropy.modeling.fitting.Fitter` object.

    fit_boxsize : int, tuple of int, or `None`, optional
        The size (in pixels) of the box centered on the star to be used
        for ePSF fitting. This allows using only a small number of
        central pixels of the star (i.e., where the star is brightest)
        for fitting. If ``fit_boxsize`` is a scalar then a square box of
        size ``fit_boxsize`` will be used. If ``fit_boxsize`` has two
        elements, they must be in ``(ny, nx)`` order. ``fit_boxsize``
        must have odd values and be greater than or equal to 3 for both
        axes. If `None`, the fitter will use the entire star image.

    **fitter_kwargs : dict-like, optional
        Any additional keyword arguments (except ``x``, ``y``, ``z``, or
        ``weights``) to be passed directly to the ``__call__()`` method
        of the input ``fitter``.
    """

    def __init__(self, *, fitter=LevMarLSQFitter(), fit_boxsize=5,
                 **fitter_kwargs):

        self.fitter = fitter
        self.fitter_has_fit_info = hasattr(self.fitter, 'fit_info')
        self.fit_boxsize = as_pair('fit_boxsize', fit_boxsize,
                                   lower_bound=(3, 0), check_odd=True)

        # remove any fitter keyword arguments that we need to set
        remove_kwargs = ['x', 'y', 'z', 'weights']
        fitter_kwargs = copy.deepcopy(fitter_kwargs)
        for kwarg in remove_kwargs:
            if kwarg in fitter_kwargs:
                del fitter_kwargs[kwarg]
        self.fitter_kwargs = fitter_kwargs

    def __call__(self, epsf, stars):
        """
        Fit an ePSF model to stars.

        Parameters
        ----------
        epsf : `EPSFModel`
            An ePSF model to be fitted to the stars.

        stars : `EPSFStars` object
            The stars to be fit.  The center coordinates for each star
            should be as close as possible to actual centers.  For stars
            than contain weights, a weighted fit of the ePSF to the star
            will be performed.

        Returns
        -------
        fitted_stars : `EPSFStars` object
            The fitted stars.  The ePSF-fitted center position and flux
            are stored in the ``center`` (and ``cutout_center``) and
            ``flux`` attributes.
        """
        if len(stars) == 0:
            return stars

        if not isinstance(epsf, EPSFModel):
            raise TypeError('The input epsf must be an EPSFModel.')

        # make a copy of the input ePSF
        epsf = epsf.copy()

        # perform the fit
        fitted_stars = []
        for star in stars:
            if isinstance(star, EPSFStar):
                fitted_star = self._fit_star(epsf, star, self.fitter,
                                             self.fitter_kwargs,
                                             self.fitter_has_fit_info,
                                             self.fit_boxsize)

            elif isinstance(star, LinkedEPSFStar):
                fitted_star = []
                for linked_star in star:
                    fitted_star.append(
                        self._fit_star(epsf, linked_star, self.fitter,
                                       self.fitter_kwargs,
                                       self.fitter_has_fit_info,
                                       self.fit_boxsize))

                fitted_star = LinkedEPSFStar(fitted_star)
                fitted_star.constrain_centers()

            else:
                raise TypeError('stars must contain only EPSFStar and/or '
                                'LinkedEPSFStar objects.')

            fitted_stars.append(fitted_star)

        return EPSFStars(fitted_stars)

    def _fit_star(self, epsf, star, fitter, fitter_kwargs,
                  fitter_has_fit_info, fit_boxsize):
        """
        Fit an ePSF model to a single star.

        The input ``epsf`` will usually be modified by the fitting
        routine in this function.  Make a copy before calling this
        function if the original is needed.
        """
        if fit_boxsize is not None:
            try:
                xcenter, ycenter = star.cutout_center
                large_slc, _ = overlap_slices(star.shape, fit_boxsize,
                                              (ycenter, xcenter),
                                              mode='strict')
            except (PartialOverlapError, NoOverlapError):
                warnings.warn(f'The star at ({star.center[0]}, '
                              f'{star.center[1]}) cannot be fit because '
                              'its fitting region extends beyond the star '
                              'cutout image.', AstropyUserWarning)

                star = copy.deepcopy(star)
                star._fit_error_status = 1

                return star

            data = star.data[large_slc]
            weights = star.weights[large_slc]

            # define the origin of the fitting region
            x0 = large_slc[1].start
            y0 = large_slc[0].start
        else:
            # use the entire cutout image
            data = star.data
            weights = star.weights

            # define the origin of the fitting region
            x0 = 0
            y0 = 0

        # Define positions in the undersampled grid. The fitter will
        # evaluate on the defined interpolation grid, currently in the
        # range [0, len(undersampled grid)].
        yy, xx = np.indices(data.shape, dtype=float)
        xx = xx + x0 - star.cutout_center[0]
        yy = yy + y0 - star.cutout_center[1]

        # define the initial guesses for fitted flux and shifts
        epsf.flux = star.flux
        epsf.x_0 = 0.0
        epsf.y_0 = 0.0

        try:
            fitted_epsf = fitter(model=epsf, x=xx, y=yy, z=data,
                                 weights=weights, **fitter_kwargs)
        except TypeError:
            # fitter doesn't support weights
            fitted_epsf = fitter(model=epsf, x=xx, y=yy, z=data,
                                 **fitter_kwargs)

        fit_error_status = 0
        if fitter_has_fit_info:
            fit_info = copy.copy(fitter.fit_info)

            if 'ierr' in fit_info and fit_info['ierr'] not in [1, 2, 3, 4]:
                fit_error_status = 2  # fit solution was not found
        else:
            fit_info = None

        # compute the star's fitted position
        x_center = star.cutout_center[0] + fitted_epsf.x_0.value
        y_center = star.cutout_center[1] + fitted_epsf.y_0.value

        star = copy.deepcopy(star)
        star.cutout_center = (x_center, y_center)

        # set the star's flux to the ePSF-fitted flux
        star.flux = fitted_epsf.flux.value

        star._fit_info = fit_info
        star._fit_error_status = fit_error_status

        return star


class EPSFBuilder:
    """
    Class to build an effective PSF (ePSF).

    See `Anderson and King (2000; PASP 112, 1360)
    <https://ui.adsabs.harvard.edu/abs/2000PASP..112.1360A/abstract>`_
    and `Anderson (2016; WFC3 ISR 2016-12)
    <https://ui.adsabs.harvard.edu/abs/2016wfc..rept...12A/abstract>`_
    for details.

    Parameters
    ----------
    oversampling : int or array_like (int)
        The integer oversampling factor(s) of the ePSF relative to the
        input ``stars`` along each axis. If ``oversampling`` is a scalar
        then it will be used for both axes. If ``oversampling`` has two
        elements, they must be in ``(y, x)`` order.

    shape : float, tuple of two floats, or `None`, optional
        The shape of the output ePSF.  If the ``shape`` is not `None`,
        it will be derived from the sizes of the input ``stars`` and the
        ePSF oversampling factor.  If the size is even along any axis,
        it will be made odd by adding one.  The output ePSF will always
        have odd sizes along both axes to ensure a well-defined central
        pixel.

    smoothing_kernel : {'quartic', 'quadratic'}, 2D `~numpy.ndarray`, or `None`
        The smoothing kernel to apply to the ePSF.  The predefined
        ``'quartic'`` and ``'quadratic'`` kernels are derived from
        fourth and second degree polynomials, respectively.
        Alternatively, a custom 2D array can be input.  If `None` then
        no smoothing will be performed.

    recentering_func : callable, optional
        A callable object (e.g., function or class) that is used to
        calculate the centroid of a 2D array. The callable must accept
        a 2D `~numpy.ndarray`, have a ``mask`` keyword and optionally
        ``error`` and ``oversampling`` keywords. The callable object
        must return a tuple of two 1D `~numpy.ndarray` variables,
        representing the x and y centroids.

    recentering_maxiters : int, optional
        The maximum number of recentering iterations to perform during
        each ePSF build iteration.

    fitter : `EPSFFitter` object, optional
        A `EPSFFitter` object use to fit the ePSF to stars. To set
        custom fitter options, input a new `EPSFFitter` object. See the
        `EPSFFitter` documentation for options.

    maxiters : int, optional
        The maximum number of iterations to perform.

    progress_bar : bool, option
        Whether to print the progress bar during the build
        iterations. The progress bar requires that the `tqdm
        <https://tqdm.github.io/>`_ optional dependency be installed.
        Note that the progress bar does not currently work in the
        Jupyter console due to limitations in ``tqdm``.

    norm_radius : float, optional
        The pixel radius over which the ePSF is normalized.

    recentering_boxsize : float or tuple of two floats, optional
            The size (in pixels) of the box used to calculate the
            centroid of the ePSF during each build iteration. If a
            single integer number is provided, then a square box will
            be used. If two values are provided, then they must be in
            ``(ny, nx)`` order. ``recentering_boxsize`` must have odd
            must have odd values and be greater than or equal to 3 for
            both axes.

    center_accuracy : float, optional
            The desired accuracy for the centers of stars.  The building
            iterations will stop if the centers of all the stars change by
            less than ``center_accuracy`` pixels between iterations.  All
            stars must meet this condition for the loop to exit.

    sigma_clip : `astropy.stats.SigmaClip` instance, optional
        A `~astropy.stats.SigmaClip` object that defines the sigma
        clipping parameters used to determine which pixels are ignored
        when stacking the ePSF residuals in each iteration step. If
        `None` then no sigma clipping will be performed.

    Notes
    -----
    If your image image contains NaN values, you may see better
    performance if you have the `bottleneck`_ package installed.

    .. _bottleneck:  https://github.com/pydata/bottleneck
    """

    def __init__(self, *, oversampling=4, shape=None,
                 smoothing_kernel='quartic', recentering_func=centroid_com,
                 recentering_maxiters=20, fitter=EPSFFitter(), maxiters=10,
                 progress_bar=True, norm_radius=5.5,
                 recentering_boxsize=(5, 5), center_accuracy=1.0e-3,
                 sigma_clip=SigmaClip(sigma=3, cenfunc='median', maxiters=10)):

        if oversampling is None:
            raise ValueError("'oversampling' must be specified.")
        self.oversampling = as_pair('oversampling', oversampling,
                                    lower_bound=(0, 1))
        self._norm_radius = norm_radius
        if shape is not None:
            self.shape = as_pair('shape', shape, lower_bound=(0, 1))
        else:
            self.shape = shape

        self.recentering_func = recentering_func
        self.recentering_maxiters = recentering_maxiters
        self.recentering_boxsize = as_pair('recentering_boxsize',
                                           recentering_boxsize,
                                           lower_bound=(3, 0), check_odd=True)
        self.smoothing_kernel = smoothing_kernel

        if not isinstance(fitter, EPSFFitter):
            raise TypeError('fitter must be an EPSFFitter instance.')
        self.fitter = fitter

        if center_accuracy <= 0.0:
            raise ValueError('center_accuracy must be a positive number.')
        self.center_accuracy_sq = center_accuracy**2

        maxiters = int(maxiters)
        if maxiters <= 0:
            raise ValueError("'maxiters' must be a positive number.")
        self.maxiters = maxiters

        self.progress_bar = progress_bar

        if not isinstance(sigma_clip, SigmaClip):
            raise ValueError('sigma_clip must be an '
                             'astropy.stats.SigmaClip instance.')
        self._sigma_clip = sigma_clip

        # store each ePSF build iteration
        self._epsf = []

    def __call__(self, stars):
        return self.build_epsf(stars)

    def _create_initial_epsf(self, stars):
        """
        Create an initial `EPSFModel` object.

        The initial ePSF data are all zeros.

        If ``shape`` is not specified, the shape of the ePSF data array
        is determined from the shape of the input ``stars`` and the
        oversampling factor. If the size is even along any axis, it will
        be made odd by adding one. The output ePSF will always have odd
        sizes along both axes to ensure a central pixel.

        Parameters
        ----------
        stars : `EPSFStars` object
            The stars used to build the ePSF.

        Returns
        -------
        epsf : `EPSFModel`
            The initial ePSF model.
        """
        norm_radius = self._norm_radius
        oversampling = self.oversampling
        shape = self.shape

        # define the ePSF shape
        if shape is not None:
            shape = as_pair('shape', shape, lower_bound=(0, 1), check_odd=True)
        else:
            # Stars class should have odd-sized dimensions, and thus we
            # get the oversampled shape as oversampling * len + 1; if
            # len=25, then newlen=101, for example.
            x_shape = (np.ceil(stars._max_shape[0]) * oversampling[1]
                       + 1).astype(int)
            y_shape = (np.ceil(stars._max_shape[1]) * oversampling[0]
                       + 1).astype(int)

            shape = np.array((y_shape, x_shape))

        # verify odd sizes of shape
        shape = [(i + 1) if i % 2 == 0 else i for i in shape]

        data = np.zeros(shape, dtype=float)

        # ePSF origin should be in the undersampled pixel units, not the
        # oversampled grid units. The middle, fractional (as we wish for
        # the center of the pixel, so the center should be at (v.5, w.5)
        # detector pixels) value is simply the average of the two values
        # at the extremes.
        xcenter = stars._max_shape[0] / 2.0
        ycenter = stars._max_shape[1] / 2.0

        epsf = EPSFModel(data=data, origin=(xcenter, ycenter),
                         oversampling=oversampling, norm_radius=norm_radius)

        return epsf

    def _resample_residual(self, star, epsf):
        """
        Compute a normalized residual image in the oversampled ePSF
        grid.

        A normalized residual image is calculated by subtracting the
        normalized ePSF model from the normalized star at the location
        of the star in the undersampled grid.  The normalized residual
        image is then resampled from the undersampled star grid to the
        oversampled ePSF grid.

        Parameters
        ----------
        star : `EPSFStar` object
            A single star object.

        epsf : `EPSFModel` object
            The ePSF model.

        Returns
        -------
        image : 2D `~numpy.ndarray`
            A 2D image containing the resampled residual image.  The
            image contains NaNs where there is no data.
        """
        # Compute the normalized residual by subtracting the ePSF model
        # from the normalized star at the location of the star in the
        # undersampled grid.

        x = star._xidx_centered
        y = star._yidx_centered

        stardata = (star._data_values_normalized
                    - epsf.evaluate(x=x, y=y, flux=1.0, x_0=0.0, y_0=0.0))

        x = epsf.oversampling[1] * star._xidx_centered
        y = epsf.oversampling[0] * star._yidx_centered

        epsf_xcenter, epsf_ycenter = (int((epsf.data.shape[1] - 1) / 2),
                                      int((epsf.data.shape[0] - 1) / 2))
        xidx = py2intround(x + epsf_xcenter)
        yidx = py2intround(y + epsf_ycenter)

        resampled_img = np.full(epsf.shape, np.nan)

        mask = np.logical_and(np.logical_and(xidx >= 0, xidx < epsf.shape[1]),
                              np.logical_and(yidx >= 0, yidx < epsf.shape[0]))
        xidx_ = xidx[mask]
        yidx_ = yidx[mask]

        resampled_img[yidx_, xidx_] = stardata[mask]

        return resampled_img

    def _resample_residuals(self, stars, epsf):
        """
        Compute normalized residual images for all the input stars.

        Parameters
        ----------
        stars : `EPSFStars` object
            The stars used to build the ePSF.

        epsf : `EPSFModel` object
            The ePSF model.

        Returns
        -------
        epsf_resid : 3D `~numpy.ndarray`
            A 3D cube containing the resampled residual images.
        """
        shape = (stars.n_good_stars, epsf.shape[0], epsf.shape[1])
        epsf_resid = np.zeros(shape)
        for i, star in enumerate(stars.all_good_stars):
            epsf_resid[i, :, :] = self._resample_residual(star, epsf)

        return epsf_resid

    def _smooth_epsf(self, epsf_data):
        """
        Smooth the ePSF array by convolving it with a kernel.

        Parameters
        ----------
        epsf_data : 2D `~numpy.ndarray`
            A 2D array containing the ePSF image.

        Returns
        -------
        result : 2D `~numpy.ndarray`
            The smoothed (convolved) ePSF data.
        """
        from scipy.ndimage import convolve

        if self.smoothing_kernel is None:
            return epsf_data

        # do this check first as comparing a ndarray to string causes a warning
        elif isinstance(self.smoothing_kernel, np.ndarray):
            kernel = self.smoothing_kernel

        elif self.smoothing_kernel == 'quartic':
            # from Polynomial2D fit with degree=4 to 5x5 array of
            # zeros with 1.0 at the center
            # Polynomial2D(4, c0_0=0.04163265, c1_0=-0.76326531,
            #              c2_0=0.99081633, c3_0=-0.4, c4_0=0.05,
            #              c0_1=-0.76326531, c0_2=0.99081633, c0_3=-0.4,
            #              c0_4=0.05, c1_1=0.32653061, c1_2=-0.08163265,
            #              c1_3=0.0, c2_1=-0.08163265, c2_2=0.02040816,
            #              c3_1=-0.0)>
            kernel = np.array(
                [[+0.041632, -0.080816, 0.078368, -0.080816, +0.041632],
                 [-0.080816, -0.019592, 0.200816, -0.019592, -0.080816],
                 [+0.078368, +0.200816, 0.441632, +0.200816, +0.078368],
                 [-0.080816, -0.019592, 0.200816, -0.019592, -0.080816],
                 [+0.041632, -0.080816, 0.078368, -0.080816, +0.041632]])

        elif self.smoothing_kernel == 'quadratic':
            # from Polynomial2D fit with degree=2 to 5x5 array of
            # zeros with 1.0 at the center
            # Polynomial2D(2, c0_0=-0.07428571, c1_0=0.11428571,
            #              c2_0=-0.02857143, c0_1=0.11428571,
            #              c0_2=-0.02857143, c1_1=-0.0)
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

        else:
            raise TypeError('Unsupported kernel.')

        return convolve(epsf_data, kernel)

    def _recenter_epsf(self, epsf, centroid_func=centroid_com,
                       box_size=(5, 5), maxiters=20, center_accuracy=1.0e-4):
        """
        Calculate the center of the ePSF data and shift the data so the
        ePSF center is at the center of the ePSF data array.

        Parameters
        ----------
        epsf : `EPSFModel` object
            The ePSF model.

        centroid_func : callable, optional
            A callable object (e.g., function or class) that is used
            to calculate the centroid of a 2D array. The callable must
            accept a 2D `~numpy.ndarray`, have a ``mask`` keyword
            and optionally an ``error`` keyword. The callable object
            must return a tuple of two 1D `~numpy.ndarray` variables,
            representing the x and y centroids.

        box_size : float or tuple of two floats, optional
            The size (in pixels) of the box used to calculate the
            centroid of the ePSF during each build iteration. If a
            single integer number is provided, then a square box will
            be used. If two values are provided, then they must be in
            ``(ny, nx)`` order. ``box_size`` must have odd values and be
            greater than or equal to 3 for both axes.

        maxiters : int, optional
            The maximum number of recentering iterations to perform.

        center_accuracy : float, optional
            The desired accuracy for the centers of stars.  The building
            iterations will stop if the center of the ePSF changes by
            less than ``center_accuracy`` pixels between iterations.

        Returns
        -------
        result : 2D `~numpy.ndarray`
            The recentered ePSF data.
        """
        epsf_data = epsf._data

        epsf = EPSFModel(data=epsf._data, origin=epsf.origin,
                         oversampling=epsf.oversampling,
                         norm_radius=epsf._norm_radius, normalize=False)

        xcenter, ycenter = epsf.origin

        y, x = np.indices(epsf._data.shape, dtype=float)
        x /= epsf.oversampling[1]
        y /= epsf.oversampling[0]

        dx_total, dy_total = 0, 0
        iter_num = 0
        center_accuracy_sq = center_accuracy**2
        center_dist_sq = center_accuracy_sq + 1.0e6
        center_dist_sq_prev = center_dist_sq + 1
        while (iter_num < maxiters and center_dist_sq >= center_accuracy_sq):
            iter_num += 1

            # Anderson & King (2000) recentering function depends
            # on specific pixels, and thus does not need a cutout
            slices_large, _ = overlap_slices(epsf_data.shape, box_size,
                                             (ycenter * self.oversampling[0],
                                              xcenter * self.oversampling[1]))
            epsf_cutout = epsf_data[slices_large]
            mask = ~np.isfinite(epsf_cutout)

            # find a new center position
            xcenter_new, ycenter_new = centroid_func(epsf_cutout,
                                                     mask=mask)
            xcenter_new /= self.oversampling[1]
            ycenter_new /= self.oversampling[0]

            xcenter_new += slices_large[1].start / self.oversampling[1]
            ycenter_new += slices_large[0].start / self.oversampling[0]

            # Calculate the shift; dx = i - x_star so if dx was positively
            # incremented then x_star was negatively incremented for a given i.
            # We will therefore actually subsequently subtract dx from xcenter
            # (or x_star).
            dx = xcenter_new - xcenter
            dy = ycenter_new - ycenter

            center_dist_sq = dx**2 + dy**2

            if center_dist_sq >= center_dist_sq_prev:  # don't shift
                break
            center_dist_sq_prev = center_dist_sq

            dx_total += dx
            dy_total += dy

            epsf_data = epsf.evaluate(x=x, y=y, flux=1.0,
                                      x_0=xcenter - dx_total,
                                      y_0=ycenter - dy_total)

        return epsf_data

    def _build_epsf_step(self, stars, epsf=None):
        """
        A single iteration of improving an ePSF.

        Parameters
        ----------
        stars : `EPSFStars` object
            The stars used to build the ePSF.

        epsf : `EPSFModel` object, optional
            The initial ePSF model.  If not input, then the ePSF will be
            built from scratch.

        Returns
        -------
        epsf : `EPSFModel` object
            The updated ePSF.
        """
        if len(stars) < 1:
            raise ValueError('stars must contain at least one EPSFStar or '
                             'LinkedEPSFStar object.')

        if epsf is None:
            # create an initial ePSF (array of zeros)
            epsf = self._create_initial_epsf(stars)
        else:
            # improve the input ePSF
            epsf = copy.deepcopy(epsf)

        # compute a 3D stack of 2D residual images
        residuals = self._resample_residuals(stars, epsf)

        # compute the sigma-clipped average along the 3D stack
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
            warnings.simplefilter('ignore', category=AstropyUserWarning)
            residuals = self._sigma_clip(residuals, axis=0, masked=False,
                                         return_bounds=False)
            if HAS_BOTTLENECK:
                import bottleneck
                residuals = bottleneck.nanmedian(residuals, axis=0)
            else:
                residuals = np.nanmedian(residuals, axis=0)

        # interpolate any missing data (np.nan)
        mask = ~np.isfinite(residuals)
        if np.any(mask):
            residuals = _interpolate_missing_data(residuals, mask,
                                                  method='cubic')

            # fill any remaining nans (outer points) with zeros
            residuals[~np.isfinite(residuals)] = 0.0

        # add the residuals to the previous ePSF image
        new_epsf = epsf._data + residuals

        # smooth and recenter the ePSF
        new_epsf = self._smooth_epsf(new_epsf)

        epsf = EPSFModel(data=new_epsf, origin=epsf.origin,
                         oversampling=epsf.oversampling,
                         norm_radius=epsf._norm_radius, normalize=False)

        epsf._data = self._recenter_epsf(
            epsf, centroid_func=self.recentering_func,
            box_size=self.recentering_boxsize,
            maxiters=self.recentering_maxiters)

        # Return the new ePSF object, but with undersampled grid pixel
        # coordinates.
        xcenter = (epsf._data.shape[1] - 1) / 2.0 / epsf.oversampling[1]
        ycenter = (epsf._data.shape[0] - 1) / 2.0 / epsf.oversampling[0]

        return EPSFModel(data=epsf._data, origin=(xcenter, ycenter),
                         oversampling=epsf.oversampling,
                         norm_radius=epsf._norm_radius)

    def build_epsf(self, stars, *, init_epsf=None):
        """
        Build iteratively an ePSF from star cutouts.

        Parameters
        ----------
        stars : `EPSFStars` object
            The stars used to build the ePSF.

        init_epsf : `EPSFModel` object, optional
            The initial ePSF model.  If not input, then the ePSF will be
            built from scratch.

        Returns
        -------
        epsf : `EPSFModel` object
            The constructed ePSF.

        fitted_stars : `EPSFStars` object
            The input stars with updated centers and fluxes derived
            from fitting the output ``epsf``.
        """
        iter_num = 0
        n_stars = stars.n_stars
        fit_failed = np.zeros(n_stars, dtype=bool)
        epsf = init_epsf
        center_dist_sq = self.center_accuracy_sq + 1.0
        centers = stars.cutout_center_flat

        pbar = None
        if self.progress_bar:
            desc = f'EPSFBuilder ({self.maxiters} maxiters)'
            pbar = add_progress_bar(total=self.maxiters, desc=desc)  # pragma: no cover

        while (iter_num < self.maxiters and not np.all(fit_failed)
               and np.max(center_dist_sq) >= self.center_accuracy_sq):

            iter_num += 1

            # build/improve the ePSF
            epsf = self._build_epsf_step(stars, epsf=epsf)

            # fit the new ePSF to the stars to find improved centers
            # we catch fit warnings here -- stars with unsuccessful fits
            # are excluded from the ePSF build process
            with warnings.catch_warnings():
                message = '.*The fit may be unsuccessful;.*'
                warnings.filterwarnings('ignore', message=message,
                                        category=AstropyUserWarning)
                stars = self.fitter(epsf, stars)

            # find all stars where the fit failed
            fit_failed = np.array([star._fit_error_status > 0
                                   for star in stars.all_stars])
            if np.all(fit_failed):
                raise ValueError('The ePSF fitting failed for all stars.')

            # permanently exclude fitting any star where the fit fails
            # after 3 iterations
            if iter_num > 3 and np.any(fit_failed):
                idx = fit_failed.nonzero()[0]
                for i in idx:  # pylint: disable=not-an-iterable
                    stars.all_stars[i]._excluded_from_fit = True

            # if no star centers have moved by more than pixel accuracy,
            # stop the iteration loop early
            dx_dy = stars.cutout_center_flat - centers
            dx_dy = dx_dy[np.logical_not(fit_failed)]
            center_dist_sq = np.sum(dx_dy * dx_dy, axis=1, dtype=np.float64)
            centers = stars.cutout_center_flat

            self._epsf.append(epsf)

            if pbar is not None:
                pbar.update()

        if pbar is not None:
            if iter_num < self.maxiters:
                pbar.write(f'EPSFBuilder converged after {iter_num} '
                           f'iterations (of {self.maxiters} maximum '
                           'iterations)')
            pbar.close()

        return epsf, stars
