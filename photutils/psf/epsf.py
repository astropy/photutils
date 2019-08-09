# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
This module provides tools to build and fit an effective PSF (ePSF)
based on Anderson and King (2000; PASP 112, 1360).
"""

import copy
import time
import warnings

from astropy.modeling.fitting import LevMarLSQFitter
from astropy.nddata.utils import (overlap_slices, PartialOverlapError,
                                  NoOverlapError)
from astropy.utils.exceptions import AstropyUserWarning
import numpy as np

from .epsf_stars import EPSFStar, EPSFStars, LinkedEPSFStar
from .models import EPSFModel
from ..centroids import centroid_epsf
from ..extern.sigma_clipping import SigmaClip

try:
    import bottleneck  # pylint: disable=W0611
    HAS_BOTTLENECK = True
except ImportError:
    HAS_BOTTLENECK = False

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
        for ePSF fitting.  This allows using only a small number of
        central pixels of the star (i.e. where the star is brightest)
        for fitting.  If ``fit_boxsize`` is a scalar then a square box
        of size ``fit_boxsize`` will be used.  If ``fit_boxsize`` has
        two elements, they should be in ``(ny, nx)`` order.  The size
        must be greater than or equal to 3 pixels for both axes.  If
        `None`, the fitter will use the entire star image.

    fitter_kwargs : dict-like, optional
        Any additional keyword arguments (except ``x``, ``y``, ``z``, or
        ``weights``) to be passed directly to the ``__call__()`` method
        of the input ``fitter``.
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
                raise ValueError('size must be >= {} for x and y'
                                 .format(min_size))

        self.fit_boxsize = fit_boxsize

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
                warnings.warn('The star at ({0}, {1}) cannot be fit because '
                              'its fitting region extends beyond the star '
                              'cutout image.'.format(star.center[0],
                                                     star.center[1]),
                              AstropyUserWarning)

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
        yy, xx = np.indices(data.shape, dtype=np.float)
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
    <http://adsabs.harvard.edu/abs/2000PASP..112.1360A>`_ for details.

    Parameters
    ----------
    oversampling : int or tuple of two int, optional
        The oversampling factor(s) of the ePSF relative to the input
        ``stars`` along the x and y axes. The ``oversampling`` can
        either be a single float or a tuple of two floats of the form
        ``(x_oversamp, y_oversamp)``.  If ``oversampling`` is a scalar
        then the oversampling will be the same for both the x and y
        axes.

    shape : float, tuple of two floats, or `None`, optional
        The shape of the output ePSF.  If the ``shape`` is not `None`,
        it will be derived from the sizes of the input ``stars`` and the
        ePSF oversampling factor.  If the size is even along any axis,
        it will be made odd by adding one.  The output ePSF will always
        have odd sizes along both axes to ensure a well-defined central
        pixel.

    norm_radius : float, optional
        The pixel radius over which the ePSF is normalized.

    smoothing_kernel : {'quartic', 'quadratic'}, 2D `~numpy.ndarray`, or `None`
        The smoothing kernel to apply to the ePSF.  The predefined
        ``'quartic'`` and ``'quadratic'`` kernels are derived from
        fourth and second degree polynomials, respectively.
        Alternatively, a custom 2D array can be input.  If `None` then
        no smoothing will be performed.

    recentering_func : callable, optional
        A callable object (e.g. function or class) that is used to
        calculate the centroid of a 2D array.  The callable must accept
        a 2D `~numpy.ndarray`, have a ``mask`` keyword and optionally
        ``error`` and ``oversampling`` keywords.  The callable object must return
        a tuple of two 1D `~numpy.ndarray` variables, representing the x and y
        centroids.

    recentering_maxiters : int, optional
        The maximum number of recentering iterations to perform during
        each ePSF build iteration.

    fitter : `EPSFFitter` object, optional
        A `EPSFFitter` object use to fit the ePSF to stars.  To set fitter
        options, a new object with specific options should be passed in - the
        default uses simply the default options.  To see more of these options,
        see the `EPSFFitter` documentation.

    maxiters : int, optional
        The maximum number of iterations to perform.

    progress_bar : bool, option
        Whether to print the progress bar during the build iterations.
    """

    def __init__(self, oversampling=4., shape=None,
                 smoothing_kernel='quartic', norm_radius=5.5, shift_val=0.5,
                 recentering_func=centroid_epsf, recentering_maxiters=20,
                 fitter=EPSFFitter(), maxiters=10, progress_bar=True):

        if oversampling is None:
            raise ValueError("'oversampling' must be specified.")
        oversampling = np.atleast_1d(oversampling).astype(int)
        if len(oversampling) == 1:
            oversampling = np.repeat(oversampling, 2)
        if np.any(oversampling % 2 != 0):
                raise ValueError('Oversampling factor must be a multiple of two')
        if np.any(oversampling <= 0.0):
            raise ValueError('oversampling must be a positive number.')
        self._norm_radius = norm_radius
        self._shift_val = shift_val
        self.oversampling = oversampling
        self.shape = self._init_img_params(shape)
        if self.shape is not None:
            self.shape = self.shape.astype(int)

        self.recentering_func = recentering_func
        self.recentering_maxiters = recentering_maxiters

        self.smoothing_kernel = smoothing_kernel

        if not isinstance(fitter, EPSFFitter):
            raise TypeError('fitter must be an EPSFFitter instance.')
        self.fitter = fitter

        maxiters = int(maxiters)
        if maxiters <= 0:
            raise ValueError("'maxiters' must be a positive number.")
        self.maxiters = maxiters

        self.progress_bar = progress_bar

        # TODO: allow custom SigmaClip object after faster SigmaClip
        # is available in astropy (>=3.1)
        self.sigclip = SigmaClip(sigma=2.5, cenfunc='mean', maxiters=10)

        # store some data during each ePSF build iteration
        self._nfit_failed = []
        self._center_dist_sq = []
        self._max_center_dist_sq = []
        self._epsf = []
        self._residuals = []
        self._residuals_sigclip = []
        self._residuals_interp = []

    def __call__(self, stars):
        return self.build_epsf(stars)

    @staticmethod
    def _init_img_params(param):
        """
        Initialize 2D image-type parameters that can accept either a
        single or two values.
        """

        if param is not None:
            param = np.atleast_1d(param)
            if len(param) == 1:
                param = np.repeat(param, 2)

        return param

    def _create_initial_epsf(self, stars):
        """
        Create an initial `EPSFModel` object.

        The initial ePSF data are all zeros.

        If ``shape`` is not specified, the shape of the ePSF data array
        is determined from the shape of the input ``stars`` and the
        oversampling factor.  If the size is even along any axis, it
        will be made odd by adding one.  The output ePSF will always
        have odd sizes along both axes to ensure a central pixel.

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
        shift_val = self._shift_val
        oversampling = self.oversampling
        shape = self.shape

        # define the ePSF shape
        if shape is not None:
            shape = np.atleast_1d(shape).astype(int)
            if len(shape) == 1:
                shape = np.repeat(shape, 2)
        else:
            # Stars class should have odd-sized dimensions, and thus we get the
            # oversampled shape as oversampling * len + 1; if len=25, then
            # newlen=101, for example.
            x_shape = np.int(np.ceil(stars._max_shape[0]) * oversampling[0] + 1)
            y_shape = np.int(np.ceil(stars._max_shape[1]) * oversampling[1] + 1)
            shape = np.array((y_shape, x_shape))

        # verify odd sizes of shape
        shape = [(i + 1) if i % 2 == 0 else i for i in shape]

        data = np.zeros(shape, dtype=np.float)
        # ePSF origin should be in the undersampled pixel units, not the oversampled
        # grid units. The middle, fractional (as we wish for the center of the
        # pixel, so the center should be at (v.5, w.5) detector pixels) value is
        # simply the average of the two values at the extremes.
        xcenter = stars._max_shape[0] / 2.
        ycenter = stars._max_shape[1] / 2.

        epsf = EPSFModel(data=data, origin=(xcenter, ycenter),
                         oversampling=oversampling, norm_radius=norm_radius,
                         shift_val=shift_val)

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

        x = epsf.oversampling[0] * star._xidx_centered
        y = epsf.oversampling[1] * star._yidx_centered
        epsf_xcenter, epsf_ycenter = (int((epsf.data.shape[1] - 1) / 2),
                                      int((epsf.data.shape[0] - 1) / 2))
        xidx = _py2intround(x + epsf_xcenter)
        yidx = _py2intround(y + epsf_ycenter)

        mask = np.logical_and(np.logical_and(xidx >= 0, xidx < epsf.shape[1]),
                              np.logical_and(yidx >= 0, yidx < epsf.shape[0]))
        xidx = xidx[mask]
        yidx = yidx[mask]

        x = star._xidx_centered
        y = star._yidx_centered

        # Compute the normalized residual by subtracting the ePSF model from
        # the normalized star at the location of the star in the undersampled grid.
        stardata = (star._data_values_normalized -
                    epsf.evaluate(x=x, y=y, flux=1.0, x_0=0.0, y_0=0.0))

        resampled_img = np.full(epsf.shape, np.nan)
        resampled_img[yidx, xidx] = stardata[mask]

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

        elif self.smoothing_kernel == 'quartic':
            # from Polynomial2D fit with degree=4 to 5x5 array of
            # zeros with 1. at the center
            # Polynomial2D(4, c0_0=0.04163265, c1_0=-0.76326531,
            #              c2_0=0.99081633, c3_0=-0.4, c4_0=0.05,
            #              c0_1=-0.76326531, c0_2=0.99081633, c0_3=-0.4,
            #              c0_4=0.05, c1_1=0.32653061, c1_2=-0.08163265,
            #              c1_3=0., c2_1=-0.08163265, c2_2=0.02040816,
            #              c3_1=-0.)>
            kernel = np.array(
                [[+0.041632, -0.080816, 0.078368, -0.080816, +0.041632],
                 [-0.080816, -0.019592, 0.200816, -0.019592, -0.080816],
                 [+0.078368, +0.200816, 0.441632, +0.200816, +0.078368],
                 [-0.080816, -0.019592, 0.200816, -0.019592, -0.080816],
                 [+0.041632, -0.080816, 0.078368, -0.080816, +0.041632]])

        elif self.smoothing_kernel == 'quadratic':
            # from Polynomial2D fit with degree=2 to 5x5 array of
            # zeros with 1. at the center
            # Polynomial2D(2, c0_0=-0.07428571, c1_0=0.11428571,
            #              c2_0=-0.02857143, c0_1=0.11428571,
            #              c0_2=-0.02857143, c1_1=-0.)
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

        elif isinstance(self.smoothing_kernel, np.ndarray):
            kernel = self.kernel

        else:
            raise TypeError('Unsupported kernel.')

        return convolve(epsf_data, kernel)

    def _recenter_epsf(self, epsf, centroid_func=centroid_epsf):
        """
        Calculate the center of the ePSF data and shift the data so the
        ePSF center is at the center of the ePSF data array.

        Parameters
        ----------
        epsf : `EPSFModel` object
            The ePSF model.

        centroid_func : callable, optional
            A callable object (e.g. function or class) that is used to
            calculate the centroid of a 2D array.  The callable must
            accept a 2D `~numpy.ndarray`, have a ``mask`` keyword and
            optionally an ``error`` keyword.  The callable object must
            return a tuple of two 1D `~numpy.ndarray` variables, representing
            the x and y centroids.

        Returns
        -------
        result : 2D `~numpy.ndarray`
            The recentered ePSF data.
        """

        xcenter, ycenter = epsf.origin

        y, x = np.indices(epsf._data.shape, dtype=np.float)
        x /= epsf.oversampling[0]
        y /= epsf.oversampling[1]

        mask = ~np.isfinite(epsf._data)

        try:
            # find a new center position
            xcenter_new, ycenter_new = centroid_func(epsf._data, mask=mask,
                                                     oversampling=epsf.oversampling,
                                                     shift_val=epsf._shift_val)
        except TypeError:
            pass
        try:
            xcenter_new, ycenter_new = centroid_func(epsf._data, mask=mask,
                                                     oversampling=epsf.oversampling)
        # default centroid_epsf overloaded, or otherwise a function that does not accept
        # oversampling or shift_val, in which case just pass data and mask
        except TypeError:
            xcenter_new, ycenter_new = centroid_func(epsf._data, mask=mask)

        # Calculate the shift; dx = i - x_star so if dx was positively
        # incremented then x_star was negatively incremented for a given i.
        # We will therefore actually subsequently subtract dx from xcenter
        # (or x_star).
        dx = xcenter_new - xcenter
        dy = ycenter_new - ycenter

        epsf_data = epsf.evaluate(x=x, y=y, flux=1.0,
                                  x_0=xcenter - dx,  # subtract dx from x_0
                                  y_0=ycenter - dy)  # even if positive

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

        for _ in range(self.recentering_maxiters):
            # compute a 3D stack of 2D residual images
            residuals = self._resample_residuals(stars, epsf)
            self._residuals.append(residuals)

            # compute the sigma-clipped mean along the 3D stack
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', category=RuntimeWarning)
                warnings.simplefilter('ignore', category=AstropyUserWarning)
                residuals = self.sigclip(residuals, axis=0, masked=False,
                                         return_bounds=False)
                if HAS_BOTTLENECK:
                    residuals = bottleneck.nanmean(residuals, axis=0)
                else:
                    residuals = np.nanmean(residuals, axis=0)

            self._residuals_sigclip.append(residuals)

            # interpolate any missing data (np.nan)
            mask = ~np.isfinite(residuals)
            if np.any(mask):
                residuals = _interpolate_missing_data(residuals, mask,
                                                      method='cubic')

                # fill any remaining nans (outer points) with zeros
                residuals[~np.isfinite(residuals)] = 0.

            self._residuals_interp.append(residuals)

            # add the residuals to the previous ePSF image
            new_epsf = epsf._data + residuals

            epsf = EPSFModel(data=new_epsf, origin=epsf.origin,
                             oversampling=epsf.oversampling,
                             norm_radius=epsf._norm_radius,
                             shift_val=epsf._shift_val, normalize=False)

            # smooth and recenter the ePSF
            epsf._data = self._smooth_epsf(epsf._data)
            epsf._data = self._recenter_epsf(epsf, centroid_func=self.recentering_func)

        # return the new ePSF object, but with undersampled grid pixel coordinates
        xcenter = (epsf._data.shape[1] - 1) / 2. / epsf.oversampling[0]
        ycenter = (epsf._data.shape[0] - 1) / 2. / epsf.oversampling[1]

        return EPSFModel(data=epsf._data, origin=(xcenter, ycenter),
                         oversampling=epsf.oversampling,
                         norm_radius=epsf._norm_radius,
                         shift_val=epsf._shift_val)

    def build_epsf(self, stars, init_epsf=None):
        """
        Iteratively build an ePSF from star cutouts.

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
        centers = stars.cutout_center_flat
        n_stars = stars.n_stars
        fit_failed = np.zeros(n_stars, dtype=bool)
        epsf = init_epsf
        dt = 0.

        while (iter_num < self.maxiters and not np.all(fit_failed)):

            t_start = time.time()
            iter_num += 1

            if self.progress_bar:
                if iter_num == 1:
                    dt_str = ' [? s/iter]'
                else:
                    dt_str = ' [{:.1f} s/iter]'.format(dt)

                print('PROGRESS: iteration {0:d} (of max {1}){2}'
                      .format(iter_num, self.maxiters, dt_str), end='\r')

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
                for i in idx:
                    stars.all_stars[i]._excluded_from_fit = True

            dx_dy = stars.cutout_center_flat - centers
            dx_dy = dx_dy[np.logical_not(fit_failed)]
            center_dist_sq = np.sum(dx_dy * dx_dy, axis=1, dtype=np.float64)
            centers = stars.cutout_center_flat

            self._nfit_failed.append(np.count_nonzero(fit_failed))
            self._center_dist_sq.append(center_dist_sq)
            self._max_center_dist_sq.append(np.max(center_dist_sq))
            self._epsf.append(epsf)

            dt = time.time() - t_start

        return epsf, stars


def _py2intround(a):
    """
    Round the input to the nearest integer.

    If two integers are equally close, rounding is done away from 0.
    """

    data = np.asanyarray(a)
    value = np.where(data >= 0, np.floor(data + 0.5),
                     np.ceil(data - 0.5)).astype(int)

    if not hasattr(a, '__iter__'):
        value = np.asscalar(value)

    return value


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
        The method of used to interpolate the missing data:

        * ``'cubic'``:  Masked data are interpolated using 2D cubic
            splines.  This is the default.

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
