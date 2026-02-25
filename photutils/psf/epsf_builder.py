# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tools to build and fit an effective PSF (ePSF) based on Anderson and
King 2000 (PASP 112, 1360) and Anderson 2016 (WFC3 ISR 2016-12).
"""

import copy
import inspect
import warnings
from dataclasses import dataclass

import numpy as np
from astropy.modeling.fitting import TRFLSQFitter
from astropy.nddata import NoOverlapError, PartialOverlapError, overlap_slices
from astropy.stats import SigmaClip
from astropy.utils.decorators import deprecated
from astropy.utils.exceptions import (AstropyDeprecationWarning,
                                      AstropyUserWarning)
from scipy.ndimage import convolve

from photutils.centroids import centroid_com
from photutils.psf.epsf_stars import EPSFStar, EPSFStars, LinkedEPSFStar
from photutils.psf.image_models import ImagePSF
from photutils.psf.utils import _interpolate_missing_data
from photutils.utils._parameters import (SigmaClipSentinelDefault, as_pair,
                                         create_default_sigmaclip)
from photutils.utils._progress_bars import add_progress_bar
from photutils.utils._round import round_half_away
from photutils.utils._stats import nanmedian

__all__ = ['EPSFBuildResult', 'EPSFBuilder', 'EPSFFitter']

SIGMA_CLIP = SigmaClipSentinelDefault(sigma=3.0, maxiters=10)


class _SmoothingKernel:
    """
    Utility class for ePSF smoothing kernel generation and convolution.

    This class encapsulates the creation of smoothing kernels used in
    ePSF building and provides consistent smoothing operations.
    """

    # Pre-computed kernels based on polynomial fits
    QUARTIC_KERNEL = np.array([
        [+0.041632, -0.080816, 0.078368, -0.080816, +0.041632],
        [-0.080816, -0.019592, 0.200816, -0.019592, -0.080816],
        [+0.078368, +0.200816, 0.441632, +0.200816, +0.078368],
        [-0.080816, -0.019592, 0.200816, -0.019592, -0.080816],
        [+0.041632, -0.080816, 0.078368, -0.080816, +0.041632]])

    QUADRATIC_KERNEL = np.array([
        [-0.07428311, 0.01142786, 0.03999952, 0.01142786, -0.07428311],
        [+0.01142786, 0.09714283, 0.12571449, 0.09714283, +0.01142786],
        [+0.03999952, 0.12571449, 0.15428215, 0.12571449, +0.03999952],
        [+0.01142786, 0.09714283, 0.12571449, 0.09714283, +0.01142786],
        [-0.07428311, 0.01142786, 0.03999952, 0.01142786, -0.07428311]])

    @classmethod
    def get_kernel(cls, kernel_type):
        """
        Get a smoothing kernel by type.

        Parameters
        ----------
        kernel_type : {'quartic', 'quadratic'} or array_like
            The type of kernel to retrieve or a custom kernel array.

        Returns
        -------
        kernel : 2D `numpy.ndarray`
            The smoothing kernel.

        Raises
        ------
        TypeError
            If `kernel_type` is not supported.

        Notes
        -----
        The predefined kernels are derived from polynomial fits:

        - 'quartic': From Polynomial2D fit with degree=4 to 5x5 array of
          zeros with 1.0 at the center. Based on fourth degree polynomial.

        - 'quadratic': From Polynomial2D fit with degree=2 to 5x5
          array of zeros with 1.0 at the center. Based on second degree
          polynomial.
        """
        if isinstance(kernel_type, np.ndarray):
            return kernel_type
        if kernel_type == 'quartic':
            return cls.QUARTIC_KERNEL
        if kernel_type == 'quadratic':
            return cls.QUADRATIC_KERNEL

        msg = (f'Unsupported kernel type: {kernel_type}. Supported types '
               'are "quartic", "quadratic", or ndarray.')
        raise TypeError(msg)

    @staticmethod
    def apply_smoothing(data, kernel_type):
        """
        Apply smoothing to data using the specified kernel.

        Parameters
        ----------
        data : 2D `numpy.ndarray`
            The data to smooth.

        kernel_type : {'quartic', 'quadratic'}, array_like, or `None`
            The type of kernel to use for smoothing, or `None` for no
            smoothing.

        Returns
        -------
        smoothed_data : 2D `numpy.ndarray`
            The smoothed data. Returns original data if `kernel_type` is
            `None`.
        """
        if kernel_type is None:
            return data

        kernel = _SmoothingKernel.get_kernel(kernel_type)
        return convolve(data, kernel)


class _EPSFValidator:
    """
    Class to validate ePSF building parameters and data.

    This class centralizes all validation logic with context-aware error
    messages.
    """

    @staticmethod
    def validate_oversampling(oversampling, context=''):
        """
        Validate oversampling parameters.

        Parameters
        ----------
        oversampling : int or tuple
            The oversampling factor(s).

        context : str, optional
            Additional context for error messages.

        Raises
        ------
        ValueError
            If oversampling is invalid.
        """
        if oversampling is None:
            msg = "'oversampling' must be specified"
            raise ValueError(msg)

        try:
            oversampling = as_pair('oversampling', oversampling,
                                   lower_bound=(0, 1))
        except (TypeError, ValueError) as e:
            msg = f'Invalid oversampling parameter - {e}'
            if context:
                msg = f'{context}: {msg}'
            raise ValueError(msg) from None

        return oversampling

    @staticmethod
    def validate_shape_compatibility(stars, oversampling, shape=None):
        """
        Validate that ePSF shape is compatible with star dimensions.

        Performs validation of shape compatibility between requested
        ePSF shape and star cutout dimensions, accounting for
        oversampling factors and providing detailed diagnostics.

        Parameters
        ----------
        stars : EPSFStars
            The input stars.

        oversampling : tuple
            The oversampling factors (y, x).

        shape : tuple, optional
            Requested ePSF shape (height, width).

        Raises
        ------
        ValueError
            If shape is incompatible with stars and oversampling.
            Error messages include suggested minimum shapes and
            detailed diagnostic information.
        """
        if not stars:
            msg = ('Cannot validate shape compatibility with empty star list. '
                   'Please provide at least one star for ePSF building.')
            raise ValueError(msg)

        # Collect star dimension statistics
        star_heights = [star.shape[0] for star in stars]
        star_widths = [star.shape[1] for star in stars]
        max_height = max(star_heights)
        max_width = max(star_widths)

        # Check for extremely small stars that may cause issues
        min_star_size = 3  # minimum reasonable star cutout size
        problematic_stars = []
        for i, star in enumerate(stars):
            if min(star.shape) < min_star_size:
                problematic_stars.append(f'Star {i}: {star.shape}')

        if problematic_stars:
            msg = (f"Found {len(problematic_stars)} star(s) with very small "
                   f"dimensions (< {min_star_size}x{min_star_size}): "
                   f"{', '.join(problematic_stars)}. Consider using larger "
                   'star cutouts for better ePSF quality.')
            raise ValueError(msg)

        # Compute minimum required ePSF shape with proper padding
        # The +1 ensures odd dimensions for proper centering
        min_epsf_height = max_height * oversampling[0] + 1
        min_epsf_width = max_width * oversampling[1] + 1

        # Validate requested shape if provided
        if shape is not None:
            shape = np.array(shape)
            if shape.ndim != 1 or len(shape) != 2:
                msg = 'Shape must be a 2-element sequence'
                raise ValueError(msg)

            if shape[0] < min_epsf_height or shape[1] < min_epsf_width:
                # Provide detailed diagnostic information
                msg = (f'Requested ePSF shape {shape} is incompatible with '
                       f'star dimensions and oversampling.\n\n'
                       f'  Oversampling factors: {oversampling}\n'
                       f'  Minimum required ePSF shape: '
                       f'({min_epsf_height}, {min_epsf_width})\n'
                       f'Solution: Use shape >= '
                       f'({min_epsf_height}, {min_epsf_width}) '
                       f'or reduce oversampling factors.')
                raise ValueError(msg)

            # Check for odd dimensions (for proper centering)
            if shape[0] % 2 == 0 or shape[1] % 2 == 0:
                msg = (f'Requested ePSF shape {shape} has even dimensions. '
                       f'Odd dimensions are recommended for proper ePSF '
                       f'centering. Consider using '
                       f'({shape[0] + shape[0] % 2}, '
                       f'{shape[1] + shape[1] % 2}) instead.')
                warnings.warn(msg, AstropyUserWarning)

    @staticmethod
    def validate_stars(stars, context=''):
        """
        Validate EPSFStars object and individual star data.

        Parameters
        ----------
        stars : EPSFStars
            The stars to validate.

        context : str, optional
            Additional context for error messages.

        Raises
        ------
        ValueError, TypeError
            If stars are invalid.
        """
        # Check basic type and structure
        if not hasattr(stars, '__len__') or len(stars) == 0:
            msg = 'EPSFStars object must contain at least one star'
            if context:
                msg = f'{context}: {msg}'
            raise ValueError(msg)

        # Validate individual stars
        invalid_stars = []
        for i, star in enumerate(stars):
            try:
                # Check for valid data
                if not hasattr(star, 'data') or star.data is None:
                    invalid_stars.append((i, 'missing data'))
                    continue

                # Check for finite values
                if not np.any(np.isfinite(star.data)):
                    invalid_stars.append((i, 'no finite data values'))
                    continue

                # Check for reasonable dimensions
                if min(star.shape) < 3:
                    invalid_stars.append((i, f'too small ({star.shape})'))
                    continue

                # Check for center coordinates
                if not hasattr(star, 'cutout_center'):
                    invalid_stars.append((i, 'missing cutout_center'))
                    continue

            except (AttributeError, TypeError, ValueError) as e:
                invalid_stars.append((i, f'validation error: {e}'))

        if invalid_stars:
            error_details = [f'Star {i}: {issue}'
                             for i, issue in invalid_stars[:5]]
            if len(invalid_stars) > 5:
                error_details.append(f'... and {len(invalid_stars) - 5} more')

            msg = (f'Found {len(invalid_stars)} invalid stars out of '
                   f'{len(stars)} total:\n' + '\n'.join(error_details))
            if context:
                msg = f'{context}: {msg}'
            raise ValueError(msg)

    @staticmethod
    def validate_center_accuracy(center_accuracy):
        """
        Validate center accuracy parameter.

        Parameters
        ----------
        center_accuracy : float
            The center accuracy threshold.

        Raises
        ------
        ValueError
            If center accuracy is invalid.
        """
        if not isinstance(center_accuracy, (int, float)):
            msg = (f'center_accuracy must be a number, got '
                   f'{type(center_accuracy)}')
            raise TypeError(msg)

        if center_accuracy <= 0.0:
            msg = ('center_accuracy must be positive, got '
                   f'{center_accuracy}. Typical values are 1e-3 to 1e-4.')
            raise ValueError(msg)

        if center_accuracy > 1.0:
            msg = (f'center_accuracy {center_accuracy} seems unusually large. '
                   'Values > 1.0 may prevent convergence. '
                   'Typical values are 1e-3 to 1e-4.')
            warnings.warn(msg, AstropyUserWarning)

    @staticmethod
    def validate_maxiters(maxiters):
        """
        Validate maximum iterations parameter.

        Parameters
        ----------
        maxiters : int
            The maximum number of iterations.

        Raises
        ------
        ValueError, TypeError
            If maxiters is invalid.
        """
        if not isinstance(maxiters, int):
            msg = f'maxiters must be an integer, got {type(maxiters)}'
            raise TypeError(msg)

        if maxiters <= 0:
            msg = 'maxiters must be a positive number'
            raise ValueError(msg)

        maxiters_warn_threshold = 100
        if maxiters > maxiters_warn_threshold:
            msg = (f'maxiters {maxiters} seems unusually large. '
                   f'Values > {maxiters_warn_threshold} may indicate '
                   'convergence issues. Consider checking your data and '
                   'parameters.')
            warnings.warn(msg, AstropyUserWarning)


class _CoordinateTransformer:
    """
    Handle coordinate transformations between pixel and oversampled
    spaces.

    This class centralizes all coordinate system conversions used in
    ePSF building, providing consistent transformations between the
    input star coordinate system and the oversampled ePSF coordinate
    system.

    Parameters
    ----------
    oversampling : tuple of int
        The (y, x) oversampling factors for the ePSF.
    """

    def __init__(self, oversampling):
        self.oversampling = np.asarray(oversampling)

    def star_to_epsf_coords(self, star_x, star_y, epsf_origin):
        """
        Transform star-relative coordinates to ePSF grid coordinates.

        Parameters
        ----------
        star_x, star_y : array_like
            Star coordinates in undersampled units relative to star
            center.

        epsf_origin : tuple
            The (x, y) origin of the ePSF in oversampled coordinates.

        Returns
        -------
        epsf_x, epsf_y : array_like
            Integer coordinates in the oversampled ePSF grid.
        """
        # Apply oversampling transformation
        x_oversampled = self.oversampling[1] * star_x
        y_oversampled = self.oversampling[0] * star_y

        # Add ePSF center offset
        epsf_xcenter, epsf_ycenter = epsf_origin
        epsf_x = round_half_away(
            x_oversampled + epsf_xcenter).astype(int)
        epsf_y = round_half_away(
            y_oversampled + epsf_ycenter).astype(int)

        return epsf_x, epsf_y

    def compute_epsf_shape(self, star_shapes):
        """
        Compute the appropriate ePSF shape from input star shapes.

        Parameters
        ----------
        star_shapes : list of tuple
            List of (height, width) tuples for each star.

        Returns
        -------
        epsf_shape : tuple
            The (height, width) shape for the oversampled ePSF.
        """
        if not star_shapes:
            msg = 'Need at least one star to compute ePSF shape'
            raise ValueError(msg)

        # Find maximum star dimensions
        max_height = max(shape[0] for shape in star_shapes)
        max_width = max(shape[1] for shape in star_shapes)

        # Apply oversampling (both are integers, so product is integer)
        epsf_height = max_height * self.oversampling[0]
        epsf_width = max_width * self.oversampling[1]

        # Ensure odd dimensions for centered origin
        if epsf_height % 2 == 0:
            epsf_height += 1
        if epsf_width % 2 == 0:
            epsf_width += 1

        return (epsf_height, epsf_width)

    def compute_epsf_origin(self, epsf_shape):
        """
        Compute the geometric origin (center) coordinates for an ePSF.

        Parameters
        ----------
        epsf_shape : tuple
            The (height, width) shape of the ePSF. The shape should have
            odd dimensions to ensure a well-defined center.

        Returns
        -------
        origin : tuple
            The (x, y) origin coordinates in the ePSF coordinate system.
        """
        origin_x = (epsf_shape[1] - 1) / 2.0
        origin_y = (epsf_shape[0] - 1) / 2.0
        return (origin_x, origin_y)

    def oversampled_to_undersampled(self, x, y):
        """
        Convert oversampled coordinates to undersampled coordinates.

        Parameters
        ----------
        x, y : array_like or float
            Coordinates in the oversampled grid.

        Returns
        -------
        x_under, y_under : array_like or float
            Coordinates in the undersampled (original) grid.
        """
        return x / self.oversampling[1], y / self.oversampling[0]

    def undersampled_to_oversampled(self, x, y):
        """
        Convert undersampled coordinates to oversampled coordinates.

        Parameters
        ----------
        x, y : array_like or float
            Coordinates in the undersampled (original) grid.

        Returns
        -------
        x_over, y_over : array_like or float
            Coordinates in the oversampled grid.
        """
        return x * self.oversampling[1], y * self.oversampling[0]


class _ProgressReporter:
    """
    Utility class for managing progress reporting during ePSF building.

    This class encapsulates all progress bar functionality, providing a
    clean interface for setting up, updating, and finalizing progress
    reporting during the iterative ePSF building process.

    Parameters
    ----------
    enabled : bool
        Whether progress reporting is enabled.

    maxiters : int
        Maximum number of iterations for progress tracking.

    Attributes
    ----------
    enabled : bool
        Whether progress reporting is active.

    maxiters : int
        Maximum iterations for progress bar setup.

    _pbar : progress bar or `None`
        The underlying progress bar instance.
    """

    def __init__(self, enabled, maxiters):
        """
        Initialize a _ProgressReporter.

        Parameters
        ----------
        enabled : bool
            Whether progress reporting is enabled.

        maxiters : int
            The maximum number of iterations.
        """
        self.enabled = enabled
        self.maxiters = maxiters
        self._pbar = None

    def setup(self):
        """
        Initialize the progress bar for ePSF building.

        Sets up the progress bar with appropriate description and
        maximum iterations if progress reporting is enabled.

        Returns
        -------
        self : _ProgressReporter
            Returns `self` for method chaining.
        """
        if not self.enabled:
            self._pbar = None
            return self

        desc = f'EPSFBuilder ({self.maxiters} maxiters)'
        self._pbar = add_progress_bar(total=self.maxiters,
                                      desc=desc)
        return self

    def update(self):
        """
        Update the progress bar by one iteration.

        Only updates if progress reporting is enabled and progress bar
        is initialized.
        """
        if self._pbar is not None:
            self._pbar.update()

    def write_convergence_message(self, iteration):
        """
        Write convergence message to progress bar.

        Parameters
        ----------
        iteration : int
            The iteration number at which convergence occurred.
        """
        if self._pbar is not None:
            self._pbar.write(f'EPSFBuilder converged after {iteration} '
                             f'iterations (of {self.maxiters} maximum '
                             'iterations)')

    def close(self):
        """
        Close and finalize the progress bar.

        Should be called when ePSF building is complete, regardless of
        convergence status.
        """
        if self._pbar is not None:
            self._pbar.close()


@dataclass
class EPSFBuildResult:
    """
    Container for ePSF building results.

    This class provides structured access to the results of the ePSF
    building process, including convergence information and diagnostic
    data that can help users understand and validate the building
    process.

    Attributes
    ----------
    epsf : `ImagePSF` object
        The final constructed ePSF model.

    fitted_stars : `EPSFStars` object
        The input stars with updated centers and fluxes derived from
        fitting the final ePSF.

    iterations : int
        The number of iterations performed during the building process.
        This will be <= maxiters specified in EPSFBuilder.

    converged : bool
        Whether the building process converged based on the center
        accuracy criterion. `True` if star centers moved less than the
        specified accuracy between the final iterations.

    final_center_accuracy : float
        The maximum center displacement in the final iteration, in
        pixels. This indicates how much the star centers changed in the
        last iteration and can be used to assess convergence quality.

    n_excluded_stars : int
        The number of individual stars (including those from linked
        stars) that were excluded from fitting due to repeated fit
        failures.

    excluded_star_indices : list
        Indices of stars that were excluded from fitting during the
        building process. These correspond to positions in the flattened
        star list (stars.all_stars).

    Notes
    -----
    This result object maintains backward compatibility by implementing
    tuple unpacking, so existing code like:

        epsf, stars = epsf_builder(stars)

    will continue to work unchanged. The additional information is
    available as attributes for users who want more detailed results.

    Examples
    --------
    >>> from photutils.psf import EPSFBuilder
    >>> epsf_builder = EPSFBuilder(oversampling=4)  # doctest: +SKIP
    >>> result = epsf_builder(stars)  # doctest: +SKIP
    >>> print(result.iterations)  # doctest: +SKIP
    >>> print(result.final_center_accuracy)  # doctest: +SKIP
    >>> print(result.n_excluded_stars)  # doctest: +SKIP
    """

    epsf: 'ImagePSF'
    fitted_stars: 'EPSFStars'
    iterations: int
    converged: bool
    final_center_accuracy: float
    n_excluded_stars: int
    excluded_star_indices: list

    def __iter__(self):
        """
        Allow tuple unpacking for backward compatibility.

        Returns
        -------
        iterator
            An iterator that yields (epsf, fitted_stars) for
            compatibility with existing code that expects a 2-tuple.
        """
        return iter((self.epsf, self.fitted_stars))

    def __getitem__(self, index):
        """
        Allow indexing for backward compatibility.

        Parameters
        ----------
        index : int
            Index to access (0 for epsf, 1 for fitted_stars).

        Returns
        -------
        value
            The ePSF (index 0) or fitted stars (index 1).
        """
        if index == 0:
            return self.epsf
        if index == 1:
            return self.fitted_stars

        msg = 'EPSFBuildResult index must be 0 (epsf) or 1 (fitted_stars)'
        raise IndexError(msg)


@deprecated(since='3.0',
            message=('EPSFFitter is deprecated and will be removed in a '
                     'future version. Use EPSFBuilder with the fitter, '
                     'fit_shape, and fitter_maxiters parameters instead.'))
class EPSFFitter:
    """
    Class to fit an ePSF model to one or more stars.

    Parameters
    ----------
    fitter : `astropy.modeling.fitting.Fitter`, optional
        A `~astropy.modeling.fitting.Fitter` object. If `None`, then the
        default `~astropy.modeling.fitting.TRFLSQFitter` will be used.

    fit_boxsize : int, tuple of int, or `None`, optional
        The size (in pixels) of the box centered on the star to be used
        for ePSF fitting. This allows using only a small number of
        central pixels of the star (i.e., where the star is brightest)
        for fitting. If ``fit_boxsize`` is a scalar then a square box of
        size ``fit_boxsize`` will be used. If ``fit_boxsize`` has two
        elements, they must be in ``(ny, nx)`` order. ``fit_boxsize``
        must have odd values and be greater than or equal to 3 for both
        axes. If `None`, the fitter will use the entire star image.

    **fitter_kwargs : dict, optional
        Any additional keyword arguments (except ``x``, ``y``, ``z``, or
        ``weights``) to be passed directly to the ``__call__()`` method
        of the input ``fitter``.
    """

    def __init__(self, *, fitter=None, fit_boxsize=5, **fitter_kwargs):

        if fitter is None:
            fitter = TRFLSQFitter()
        self.fitter = fitter
        self.fitter_has_fit_info = hasattr(self.fitter, 'fit_info')
        if fit_boxsize is not None:
            self.fit_boxsize = as_pair('fit_boxsize', fit_boxsize,
                                       lower_bound=(3, 0), check_odd=True)
        else:
            self.fit_boxsize = None

        # Remove any fitter keyword arguments that we need to set
        remove_kwargs = {'x', 'y', 'z', 'weights'}
        self.fitter_kwargs = {
            k: v for k, v in fitter_kwargs.items()
            if k not in remove_kwargs
        }

    def __call__(self, epsf, stars):
        """
        Fit an ePSF model to stars.

        Parameters
        ----------
        epsf : `ImagePSF`
            An ePSF model to be fitted to the stars.

        stars : `EPSFStars` object
            The stars to be fit. The center coordinates for each star
            should be as close as possible to actual centers. For stars
            than contain weights, a weighted fit of the ePSF to the star
            will be performed.

        Returns
        -------
        fitted_stars : `EPSFStars` object
            The fitted stars. The ePSF-fitted center position and flux
            are stored in the ``center`` (and ``cutout_center``) and
            ``flux`` attributes.
        """
        if len(stars) == 0:
            return stars

        if not isinstance(epsf, ImagePSF):
            msg = 'The input epsf must be an ImagePSF'
            raise TypeError(msg)

        # Perform the fit
        fitted_stars = []
        for star in stars:
            if isinstance(star, EPSFStar):
                # Skip fitting stars that have been excluded; return
                # directly since no modification is needed
                if star._excluded_from_fit:
                    fitted_star = star
                else:
                    fitted_star = self._fit_star(epsf, star, self.fitter,
                                                 self.fitter_kwargs,
                                                 self.fitter_has_fit_info,
                                                 self.fit_boxsize)

            elif isinstance(star, LinkedEPSFStar):
                fitted_star = []
                for linked_star in star:
                    # Skip fitting stars that have been excluded; return
                    # directly since no modification is needed
                    if linked_star._excluded_from_fit:
                        fitted_star.append(linked_star)
                    else:
                        fitted_star.append(
                            self._fit_star(epsf, linked_star, self.fitter,
                                           self.fitter_kwargs,
                                           self.fitter_has_fit_info,
                                           self.fit_boxsize))

                fitted_star = LinkedEPSFStar(fitted_star)
                fitted_star.constrain_centers()

            else:
                msg = ('stars must contain only EPSFStar and/or '
                       'LinkedEPSFStar objects')
                raise TypeError(msg)

            fitted_stars.append(fitted_star)

        return EPSFStars(fitted_stars)

    def _fit_star(self, epsf, star, fitter, fitter_kwargs,
                  fitter_has_fit_info, fit_boxsize):
        """
        Fit an ePSF model to a single star.
        """
        # Create a shallow copy to avoid mutating the input star. This
        # is a shallow copy, so the large numpy arrays (_data, weights,
        # mask) are shared and not duplicated; only the object wrapper
        # and small scalar attributes are new.
        star = copy.copy(star)

        if fit_boxsize is not None:
            try:
                xcenter, ycenter = star.cutout_center
                large_slc, _ = overlap_slices(star.shape, fit_boxsize,
                                              (ycenter, xcenter),
                                              mode='strict')
            except (PartialOverlapError, NoOverlapError):
                star._fit_error_status = 1

                return star

            data = star.data[large_slc]
            weights = star.weights[large_slc]

            # Define the origin of the fitting region
            x0 = large_slc[1].start
            y0 = large_slc[0].start
        else:
            # Use the entire cutout image
            data = star.data
            weights = star.weights

            # Define the origin of the fitting region
            x0 = 0
            y0 = 0

        # Define positions in the undersampled grid. The fitter will
        # evaluate on the defined interpolation grid, currently in the
        # range [0, len(undersampled grid)].
        yy, xx = np.indices(data.shape, dtype=float)
        xx = xx + x0 - star.cutout_center[0]
        yy = yy + y0 - star.cutout_center[1]

        # Define the initial guesses for fitted flux and shifts
        epsf.flux = star.flux
        epsf.x_0 = 0.0
        epsf.y_0 = 0.0

        try:
            fitted_epsf = fitter(model=epsf, x=xx, y=yy, z=data,
                                 weights=weights, **fitter_kwargs)
        except TypeError:
            # Handle case where the fitter does not support weights
            fitted_epsf = fitter(model=epsf, x=xx, y=yy, z=data,
                                 **fitter_kwargs)

        fit_error_status = 0
        if fitter_has_fit_info:
            fit_info = fitter.fit_info

            if 'ierr' in fit_info and fit_info['ierr'] not in [1, 2, 3, 4]:
                fit_error_status = 2  # fit solution was not found
        else:
            fit_info = None

        # Compute the star's fitted position
        x_center = star.cutout_center[0] + fitted_epsf.x_0.value
        y_center = star.cutout_center[1] + fitted_epsf.y_0.value

        # Check if fitted position is outside the data cutout
        if (x_center < 0 or x_center >= star.shape[1]
                or y_center < 0 or y_center >= star.shape[0]):
            fit_error_status = 3  # pragma: no cover

        if fit_error_status != 3:
            star.cutout_center = (x_center, y_center)
            # Set the star's flux to the ePSF-fitted flux
            star.flux = fitted_epsf.flux.value

        star._fit_info = fit_info
        star._fit_error_status = fit_error_status

        return star


class EPSFBuilder:
    """
    Class to build an effective PSF (ePSF).

    See `Anderson and King 2000 (PASP 112, 1360)
    <https://ui.adsabs.harvard.edu/abs/2000PASP..112.1360A/abstract>`_
    and `Anderson 2016 (WFC3 ISR 2016-12)
    <https://ui.adsabs.harvard.edu/abs/2016wfc..rept...12A/abstract>`_
    for details.

    Parameters
    ----------
    oversampling : int or array_like (int)
        The integer oversampling factor(s) of the output ePSF relative
        to the input ``stars`` along each axis. If ``oversampling`` is a
        scalar then it will be used for both axes. If ``oversampling``
        has two elements, they must be in ``(y, x)`` order.

    shape : float, tuple of two floats, or `None`, optional
        The (ny, nx) shape of the output ePSF. If the input shape is
        even along any axis, it will be made odd by adding one. If the
        ``shape`` is `None`, it will be derived from the sizes of the
        input ``stars`` and the ePSF ``oversampling`` factor. The output
        ePSF will always have odd sizes along both axes to ensure a
        well-defined central pixel.

    smoothing_kernel : {'quartic', 'quadratic'}, 2D `~numpy.ndarray`, or `None`
        The smoothing kernel to apply to the ePSF during each iteration
        step. The predefined ``'quartic'`` and ``'quadratic'`` kernels
        are derived from fourth and second degree polynomials,
        respectively. Alternatively, a custom 2D array can be input. If
        `None` then no smoothing will be performed.

    sigma_clip : `astropy.stats.SigmaClip` instance, optional
        A `~astropy.stats.SigmaClip` object that defines the sigma
        clipping parameters used to determine which pixels are ignored
        when stacking the ePSF residuals in each iteration step. If
        `None` then no sigma clipping will be performed.

    recentering_func : callable, optional
        A callable object that is used to calculate the centroid of a
        2D array. The callable must accept a 2D `~numpy.ndarray`, have
        a ``mask`` keyword and optionally an ``error`` keyword. The
        callable object must return a tuple of (x, y) centroids.

    recentering_boxsize : float or tuple of two floats, optional
        The size (in pixels) of the box used to calculate the centroid
        of the ePSF during each build iteration. The size is in
        the input star (i.e., undersampled) pixel space; it is
        automatically scaled by the oversampling factor when applied
        to the oversampled ePSF grid. If a single integer number
        is provided, then a square box will be used. If two values
        are provided, then they must be in ``(ny, nx)`` order.
        ``recentering_boxsize`` must have odd values and be greater than
        or equal to 3 for both axes.

    recentering_maxiters : int, optional
        The maximum number of recentering iterations to perform during
        each ePSF build iteration.

    center_accuracy : float, optional
        The desired accuracy for the centers of stars. The building
        iterations will stop if the centers of all the stars change by
        less than ``center_accuracy`` pixels between iterations. All
        stars must meet this condition for the building iterations to
        stop.

    fitter : `~astropy.modeling.fitting.Fitter` or `EPSFFitter`, optional
        A `~astropy.modeling.fitting.Fitter` object used to fit the
        ePSF to stars. If `None`, then the default
        `~astropy.modeling.fitting.TRFLSQFitter` will be used.

        .. deprecated:: 3.0
            Passing an `EPSFFitter` instance is deprecated; use
            the ``fitter``, ``fit_shape``, and ``fitter_maxiters``
            parameters instead.

    fit_shape : int, tuple of int, or `None`, optional
        The size (in pixels) of the box centered on the star to be
        used for ePSF fitting. This allows using only a small number
        of central pixels of the star (i.e., where the star is
        brightest) for fitting. If ``fit_shape`` is a scalar then a
        square box of size ``fit_shape`` will be used. If ``fit_shape``
        has two elements, they must be in ``(ny, nx)`` order.
        ``fit_shape`` must have odd values and be greater than or equal
        to 3 for both axes. If `None`, the fitter will use the entire
        star image.

    fitter_maxiters : int, optional
        The maximum number of iterations in which the ``fitter`` is
        called for each star. The value can be increased if the fit
        is not converging. This parameter is passed to the ``fitter``
        if it supports the ``maxiter`` parameter and ignored otherwise.

    maxiters : int, optional
        The maximum number of ePSF building iterations to perform.

    progress_bar : bool, option
        Whether to print the progress bar during the build
        iterations. The progress bar requires that the `tqdm
        <https://tqdm.github.io/>`_ optional dependency be installed.

    Notes
    -----
    If your image contains NaN values, you may see better performance if
    you have the `bottleneck`_ package installed.

    .. _bottleneck:  https://github.com/pydata/bottleneck
    """

    def __init__(self, *, oversampling=4, shape=None,
                 smoothing_kernel='quartic', sigma_clip=SIGMA_CLIP,
                 recentering_func=centroid_com, recentering_boxsize=(5, 5),
                 recentering_maxiters=20, center_accuracy=1.0e-3,
                 fitter=None, fit_shape=5, fitter_maxiters=100,
                 maxiters=10, progress_bar=True):

        # Validate and store oversampling using the validator
        self.oversampling = _EPSFValidator.validate_oversampling(
            oversampling, 'EPSFBuilder initialization')

        # Initialize coordinate transformer for consistent transformations
        self.coord_transformer = _CoordinateTransformer(self.oversampling)

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

        # Handle fitter parameter - accept both astropy Fitter and
        # deprecated EPSFFitter for backward compatibility
        if isinstance(fitter, EPSFFitter):
            warnings.warn('Passing an EPSFFitter instance to '
                          'EPSFBuilder is deprecated. Use the fitter, '
                          'fit_shape, and fitter_maxiters parameters '
                          'instead.',
                          AstropyDeprecationWarning)
            self.fitter = fitter.fitter
            self.fit_shape = fitter.fit_boxsize
            self.fitter_maxiters = None
            self._fitter_kwargs = fitter.fitter_kwargs
        else:
            if fitter is None:
                fitter = TRFLSQFitter()
            if not callable(fitter):
                msg = 'fitter must be a callable astropy Fitter instance'
                raise TypeError(msg)
            self.fitter = fitter

            # Validate fit_shape
            if fit_shape is not None:
                self.fit_shape = as_pair('fit_shape', fit_shape,
                                         lower_bound=(3, 0),
                                         check_odd=True)
            else:
                self.fit_shape = None

            # Validate fitter_maxiters
            self.fitter_maxiters = self._validate_fitter_maxiters(
                fitter_maxiters)

            # Build fitter keyword arguments
            self._fitter_kwargs = {}
            if self.fitter_maxiters is not None:
                self._fitter_kwargs['maxiter'] = self.fitter_maxiters

        self._fitter_has_fit_info = hasattr(self.fitter, 'fit_info')

        # Validate center accuracy using the validator
        _EPSFValidator.validate_center_accuracy(center_accuracy)
        self.center_accuracy_sq = center_accuracy**2

        # Validate maxiters using the validator
        _EPSFValidator.validate_maxiters(maxiters)
        self.maxiters = maxiters

        self.progress_bar = progress_bar

        if sigma_clip is SIGMA_CLIP:
            sigma_clip = create_default_sigmaclip(sigma=SIGMA_CLIP.sigma,
                                                  maxiters=SIGMA_CLIP.maxiters)
        if not isinstance(sigma_clip, SigmaClip):
            msg = 'sigma_clip must be an astropy.stats.SigmaClip instance'
            raise TypeError(msg)
        self._sigma_clip = sigma_clip

        # store each ePSF build iteration
        self._epsf = []

    def __call__(self, stars):
        """
        Build an ePSF from input stars.

        Parameters
        ----------
        stars : `EPSFStars`
            The stars used to build the ePSF.

        Returns
        -------
        result : `EPSFBuildResult`
            The result of the ePSF building process.
        """
        return self._build_epsf(stars)

    def _validate_fitter_maxiters(self, fitter_maxiters):
        """
        Validate the ``fitter_maxiters`` parameter.

        Parameters
        ----------
        fitter_maxiters : int
            Maximum number of fitter iterations to validate.

        Returns
        -------
        fitter_maxiters : int or `None`
            The validated value, or `None` if the fitter does not
            support the ``maxiter`` parameter.
        """
        spec = inspect.signature(self.fitter.__call__)
        has_maxiter = ('maxiter' in spec.parameters
                       or any(p.kind == inspect.Parameter.VAR_KEYWORD
                              for p in spec.parameters.values()))
        if not has_maxiter:
            warnings.warn('"fitter_maxiters" will be ignored because '
                          'it is not accepted by the input fitter',
                          AstropyUserWarning)
            return None
        return fitter_maxiters

    def _create_initial_epsf(self, stars):
        """
        Create an initial `ImagePSF` object with zero data.

        This method initializes the ePSF building process by creating a
        blank ImagePSF model with the appropriate size and coordinate
        system. The initial ePSF data are all zeros and will be
        populated through the iterative building process.

        Shape Determination Algorithm
        -----------------------------
        1. If shape is explicitly provided, use it (ensuring odd
           dimensions)

        2. Otherwise, determine shape from input stars and oversampling:
           - Take the maximum star cutout dimensions
           - Apply oversampling factor: new_size = old_size * oversampling
           - Ensure resulting dimensions are odd (add 1 if even)

        This ensures that oversampled arrays have a well-defined center
        pixel, which is crucial for PSF modeling and fitting.

        Coordinate System Setup
        -----------------------
        The method establishes the coordinate system for the ImagePSF.
        The origin is set to the geometric center of the data array,
        which ensures that the PSF center aligns with the array center.
        The coordinate system is consistent with the expectations of the
        ImagePSF class and allows for straightforward mapping between
        star-relative coordinates and ePSF grid coordinates during the
        building process.

        Parameters
        ----------
        stars : `EPSFStars` object
            The stars used to build the ePSF. The method uses
            stars._max_shape to ensure the ePSF is large enough to
            contain all stars.

        Returns
        -------
        epsf : `ImagePSF` object
            The initial ePSF model with:
            - data: Zero-filled array of appropriate dimensions
            - origin: Set to the array center in (x, y) order
            - oversampling: Copied from the EPSFBuilder configuration
            - fill_value: Set to 0.0 for regions outside the PSF

        Notes
        -----
        The initial ePSF has zero flux and data values. These will be
        populated through the iterative building process as residuals
        from individual stars are combined.

        The method ensures that:
        - Array dimensions are always odd (ensuring a center pixel)
        - The coordinate system is properly established
        - All necessary attributes are set for downstream processing

        Examples
        --------
        For stars with maximum shape (25, 25) and oversampling=4:
        - x_shape = 25 * 4 = 100 (even), add 1 -> 101
        - y_shape = 25 * 4 = 100 (even), add 1 -> 101
        - Final shape: (101, 101)
        - Origin: (50.0, 50.0)

        For stars with maximum shape (25, 25) and oversampling=3:
        - x_shape = 25 * 3 = 75 (already odd)
        - y_shape = 25 * 3 = 75 (already odd)
        - Final shape: (75, 75)
        - Origin: (37.0, 37.0)
        """
        oversampling = self.oversampling
        shape = self.shape

        # Define the ePSF shape using coordinate transformer
        if shape is not None:
            shape = as_pair('shape', shape, lower_bound=(0, 1), check_odd=True)
        else:
            # Use coordinate transformer to compute shape from star
            # dimensions
            star_shapes = [star.shape for star in stars]
            shape = self.coord_transformer.compute_epsf_shape(star_shapes)

        # Initialize with zeros
        data = np.zeros(shape, dtype=float)

        # Use coordinate transformer to compute origin
        origin_xy = self.coord_transformer.compute_epsf_origin(shape)

        return ImagePSF(data=data, origin=origin_xy, oversampling=oversampling,
                        fill_value=0.0)

    def _resample_residual(self, star, epsf, out_image=None):
        """
        Compute a normalized residual image in the oversampled ePSF
        grid.

        A normalized residual image is calculated by subtracting the
        normalized ePSF model from the normalized star at the location
        of the star in the undersampled grid. The normalized residual
        image is then resampled from the undersampled star grid to the
        oversampled ePSF grid.

        Parameters
        ----------
        star : `EPSFStar` object
            A single star object.

        epsf : `ImagePSF` object
            The ePSF model.

        out_image : 2D `~numpy.ndarray`, optional
            A 2D array to hold the resampled residual image. If `None`,
            a new array will be created.

        Returns
        -------
        image : 2D `~numpy.ndarray`
            A 2D image containing the resampled residual image. The
            image contains NaNs where there is no data.
        """
        # Compute the normalized residual by subtracting the ePSF model
        # from the normalized star at the location of the star in the
        # undersampled grid.
        xidx_centered, yidx_centered = star._xyidx_centered
        stardata = (star._data_values_normalized
                    - epsf.evaluate(x=xidx_centered,
                                    y=yidx_centered,
                                    flux=1.0, x_0=0.0, y_0=0.0))

        # Use coordinate transformer to map to the oversampled ePSF grid
        xidx, yidx = self.coord_transformer.star_to_epsf_coords(
            xidx_centered, yidx_centered, epsf.origin)

        epsf_shape = epsf.data.shape
        if out_image is None:
            out_image = np.full(epsf_shape, np.nan)

        mask = np.logical_and(np.logical_and(xidx >= 0, xidx < epsf_shape[1]),
                              np.logical_and(yidx >= 0, yidx < epsf_shape[0]))
        xidx_ = xidx[mask]
        yidx_ = yidx[mask]

        out_image[yidx_, xidx_] = stardata[mask]

        return out_image

    def _resample_residuals(self, stars, epsf):
        """
        Compute normalized residual images for all the input stars.

        Optimized to minimize memory allocations.

        Parameters
        ----------
        stars : `EPSFStars` object
            The stars used to build the ePSF.

        epsf : `ImagePSF` object
            The ePSF model.

        Returns
        -------
        epsf_resid : 3D `~numpy.ndarray`
            A 3D cube containing the resampled residual images.
        """
        epsf_shape = epsf.data.shape
        n_good_stars = stars.n_good_stars

        if n_good_stars == 0:
            # Return empty array with correct shape
            return np.zeros((0, epsf_shape[0], epsf_shape[1]))

        # Pre-allocate with NaN (default for missing data)
        shape = (n_good_stars, epsf_shape[0], epsf_shape[1])
        epsf_resid = np.full(shape, np.nan)

        # Loop over stars and compute residuals directly into the
        # pre-allocated array
        for i, star in enumerate(stars.all_good_stars):
            self._resample_residual(star, epsf, out_image=epsf_resid[i])

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
        return _SmoothingKernel.apply_smoothing(epsf_data,
                                                self.smoothing_kernel)

    def _normalize_epsf(self, epsf_data):
        """
        Normalize the ePSF data so that the sum of the array values
        equals the product of the oversampling factors.

        The normalization accounts for oversampling. For proper
        normalization with flux=1.0, the sum of the ePSF data array
        should equal the product of the oversampling factors.

        Parameters
        ----------
        epsf_data : 2D `~numpy.ndarray`
            A 2D array containing the ePSF image.

        Returns
        -------
        result : 2D `~numpy.ndarray`
            The normalized ePSF data.

        Notes
        -----
        For an oversampled PSF image, the sum of array values should
        equal the product of the oversampling factors (e.g., for
        oversampling=(4, 4), sum should be 16.0). This ensures that the
        ImagePSF model with flux=1.0 represents a properly normalized
        PSF.
        """
        oversampling_product = np.prod(self.oversampling)
        current_sum = np.sum(epsf_data)

        if current_sum == 0:
            msg = 'Cannot normalize ePSF: data sum is zero'
            raise ValueError(msg)

        return epsf_data * (oversampling_product / current_sum)

    def _recenter_epsf(self, epsf, centroid_func=None, box_size=None,
                       maxiters=None, center_accuracy=None):
        """
        Recenter the ePSF data by shifting to the array center.

        This method uses iterative centroiding to find the center of
        the ePSF and applies sub-pixel shifts using interpolation.
        This provides accurate centering even when the PSF is offset
        by fractional pixels.

        Algorithm Overview
        ------------------
        1. Find the centroid of the ePSF using the centroid function
        2. Calculate the sub-pixel shift needed to center the PSF
        3. Apply the shift using spline interpolation via epsf.evaluate()
        4. Iterate until convergence or max iterations reached

        Parameters
        ----------
        epsf : `ImagePSF` object
            The ePSF model containing the data to be recentered.

        centroid_func : callable, optional
            A callable object (e.g., function or class) that is used
            to calculate the centroid of a 2D array. The callable must
            accept a 2D `~numpy.ndarray`, have a ``mask`` keyword
            and optionally an ``error`` keyword. The callable object
            must return a tuple of two 1D `~numpy.ndarray` variables,
            representing the x and y centroids. If `None`, uses the
            builder's configured recentering_func.

        box_size : float or tuple of two floats, optional
            The size (in pixels) of the box used to calculate the
            centroid of the ePSF during each iteration. The size is
            in the input star (i.e., undersampled) pixel space; it is
            automatically scaled by the oversampling factor when applied
            to the oversampled ePSF grid. If a single integer number
            is provided, then a square box will be used. If two values
            are provided, then they must be in ``(ny, nx)`` order.
            ``box_size`` must have odd values and be greater than or
            equal to 3 for both axes. If `None`, uses the builder's
            configured recentering_boxsize.

        maxiters : int, optional
            The maximum number of recentering iterations to perform. If
            `None`, uses the builder's configured recentering_maxiters .

        center_accuracy : float, optional
            The desired accuracy for the center position. The centering
            iterations will stop if the center of the ePSF changes by
            less than ``center_accuracy`` pixels between iterations. If
            `None`, uses 1.0e-4.

        Returns
        -------
        result : 2D `~numpy.ndarray`
            The recentered ePSF data array with the same shape as input.

        Notes
        -----
        This method uses spline interpolation to apply sub-pixel shifts,
        which preserves the PSF shape more accurately than integer
        pixel shifting. The interpolation is done using the ImagePSF's
        evaluate method.
        """
        # Use instance defaults if not specified
        if centroid_func is None:
            centroid_func = self.recentering_func
        if box_size is None:
            box_size = self.recentering_boxsize
        if maxiters is None:
            maxiters = self.recentering_maxiters
        if center_accuracy is None:
            center_accuracy = 1.0e-4

        # Scale box_size from undersampled (input star) space to
        # oversampled ePSF space, ensuring odd dimensions.
        box_size = np.asarray(box_size)
        oversampled_box = box_size * self.oversampling
        # Ensure odd dimensions so the box is centered on a pixel
        oversampled_box = tuple(s + 1 if s % 2 == 0 else s
                                for s in oversampled_box)
        oversampled_box = np.array(oversampled_box, dtype=int)

        # The center of the ePSF in oversampled pixel coordinates.
        # This is where we want the PSF center to be.
        xcenter, ycenter = self.coord_transformer.compute_epsf_origin(
            epsf.data.shape)

        # Create coordinate grids in undersampled units for evaluate()
        y, x = np.indices(epsf.data.shape, dtype=float)
        x, y = self.coord_transformer.oversampled_to_undersampled(x, y)

        # The origin in undersampled units (for use with evaluate)
        x_origin, y_origin = (
            self.coord_transformer.oversampled_to_undersampled(xcenter,
                                                               ycenter))

        dx_total, dy_total = 0.0, 0.0
        iter_num = 0
        center_accuracy_sq = center_accuracy ** 2
        center_dist_sq = center_accuracy_sq + 1.0e6
        center_dist_sq_prev = center_dist_sq + 1

        epsf_data = epsf.data
        while (iter_num < maxiters and center_dist_sq >= center_accuracy_sq):
            iter_num += 1

            # Get a cutout around the expected center for centroiding
            slices_large, _ = overlap_slices(
                epsf_data.shape, oversampled_box,
                (ycenter, xcenter))
            epsf_cutout = epsf_data[slices_large]
            mask = ~np.isfinite(epsf_cutout)

            # Find the centroid in the cutout (in oversampled pixel coords)
            xcenter_new, ycenter_new = centroid_func(epsf_cutout, mask=mask)

            # Convert cutout coordinates to full array coordinates
            xcenter_new += slices_large[1].start
            ycenter_new += slices_large[0].start

            # Calculate the shift in oversampled pixels
            dx = xcenter_new - xcenter
            dy = ycenter_new - ycenter

            center_dist_sq = dx ** 2 + dy ** 2

            if center_dist_sq >= center_dist_sq_prev:
                # Shift is getting larger, stop iterating
                break
            center_dist_sq_prev = center_dist_sq

            # Accumulate total shift in undersampled units
            dx_under, dy_under = (
                self.coord_transformer.oversampled_to_undersampled(dx, dy))
            dx_total += dx_under
            dy_total += dy_under

            # Apply the shift using evaluate (uses spline
            # interpolation). The shift is applied by moving the origin.
            epsf_data = epsf.evaluate(x=x, y=y, flux=1.0,
                                      x_0=x_origin - dx_total,
                                      y_0=y_origin - dy_total)

        return epsf_data

    def _build_epsf_step(self, stars, epsf=None):
        """
        A single iteration of improving an ePSF.

        Parameters
        ----------
        stars : `EPSFStars` object
            The stars used to build the ePSF.

        epsf : `ImagePSF` object, optional
            The initial ePSF model. If not input, then the ePSF will be
            built from scratch.

        Returns
        -------
        epsf : `ImagePSF` object
            The updated ePSF.
        """
        if epsf is None:
            # Create an initial ePSF (array of zeros)
            epsf = self._create_initial_epsf(stars)

        # Compute a 3D stack of 2D residual images
        residuals = self._resample_residuals(stars, epsf)

        # Compute the sigma-clipped median along the 3D stack
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
            warnings.simplefilter('ignore', category=AstropyUserWarning)
            residuals = self._sigma_clip(residuals, axis=0, masked=False,
                                         return_bounds=False)
            residuals = nanmedian(residuals, axis=0)

        # Interpolate any missing data (np.nan values) in the residual
        # image
        mask = ~np.isfinite(residuals)
        if np.any(mask):
            residuals = _interpolate_missing_data(residuals, mask,
                                                  method='cubic')

        # Add the residuals to the previous ePSF image
        new_epsf = epsf.data + residuals

        # Smooth the ePSF
        smoothed_data = self._smooth_epsf(new_epsf)

        # Recenter the ePSF
        # Create an intermediate ePSF for recentering operations.
        # Use the current epsf's origin if it exists, otherwise compute
        # center.
        temp_epsf = ImagePSF(data=smoothed_data,
                             origin=epsf.origin,
                             oversampling=self.oversampling,
                             fill_value=0.0)

        # Apply recentering to the smoothed data
        recentered_data = self._recenter_epsf(temp_epsf)

        # Normalize the ePSF data
        normalized_data = self._normalize_epsf(recentered_data)

        return ImagePSF(data=normalized_data,
                        oversampling=self.oversampling,
                        fill_value=0.0)

    def _check_convergence(self, stars, centers, fit_failed):
        """
        Check if the ePSF building has converged.

        Convergence is determined by checking the movement of star
        centers between iterations. The method calculates the squared
        distance of center movements for successfully fitted stars and
        applies enhanced convergence criteria that consider both the
        maximum movement and the overall stability of the star centers.
        This provides a more robust convergence detection mechanism that
        is less sensitive to outliers and provides better diagnostic
        information on the quality of convergence.

        Parameters
        ----------
        stars : `EPSFStars` object
            The stars used to build the ePSF.

        centers : `~numpy.ndarray`
            Previous star center positions.

        fit_failed : `~numpy.ndarray`
            Boolean array tracking failed fits.

        Returns
        -------
        converged : bool
            `True` if convergence criteria are met.

        center_dist_sq : `~numpy.ndarray`
            Squared distances of center movements.

        new_centers : `~numpy.ndarray`
            Updated star center positions.
        """
        # Calculate center movements for successfully fitted stars only
        new_centers = stars.cutout_center_flat
        dx_dy = new_centers - centers

        # Filter out failed fits for convergence calculation
        good_stars = np.logical_not(fit_failed)

        if not np.any(good_stars):
            # No good stars - cannot determine convergence
            # Return high values to prevent false convergence
            return False, np.array([self.center_accuracy_sq * 10]), new_centers

        dx_dy_good = dx_dy[good_stars]
        center_dist_sq = np.sum(dx_dy_good * dx_dy_good, axis=1,
                                dtype=np.float64)

        # Enhanced convergence criteria
        max_movement = np.max(center_dist_sq)

        # Primary convergence check
        primary_converged = max_movement < self.center_accuracy_sq

        # Secondary check: ensure most stars are stable
        # 80% of stars must be stable
        stable_fraction_threshold = 0.8
        stable_fraction = (np.sum(center_dist_sq < self.center_accuracy_sq)
                           / len(center_dist_sq))
        stability_converged = stable_fraction > stable_fraction_threshold

        # Combined convergence: both criteria must be met for robust
        # results
        converged = primary_converged and stability_converged

        return converged, center_dist_sq, new_centers

    def _fit_stars(self, epsf, stars):
        """
        Fit an ePSF model to stars.

        Parameters
        ----------
        epsf : `ImagePSF`
            An ePSF model to be fitted to the stars.

        stars : `EPSFStars` object
            The stars to be fit. The center coordinates for each star
            should be as close as possible to actual centers. For stars
            that contain weights, a weighted fit of the ePSF to the
            star will be performed.

        Returns
        -------
        fitted_stars : `EPSFStars` object
            The fitted stars. The ePSF-fitted center position and flux
            are stored in the ``center`` (and ``cutout_center``) and
            ``flux`` attributes.
        """
        if len(stars) == 0:
            return stars

        if not isinstance(epsf, ImagePSF):
            msg = 'The input epsf must be an ImagePSF'
            raise TypeError(msg)

        fitted_stars = []
        for star in stars:
            if isinstance(star, EPSFStar):
                if star._excluded_from_fit:
                    fitted_star = star
                else:
                    fitted_star = self._fit_star(epsf, star)

            elif isinstance(star, LinkedEPSFStar):
                fitted_star = []
                for linked_star in star:
                    if linked_star._excluded_from_fit:
                        fitted_star.append(linked_star)
                    else:
                        fitted_star.append(self._fit_star(epsf,
                                                          linked_star))

                fitted_star = LinkedEPSFStar(fitted_star)
                fitted_star.constrain_centers()

            else:
                msg = ('stars must contain only EPSFStar and/or '
                       'LinkedEPSFStar objects')
                raise TypeError(msg)

            fitted_stars.append(fitted_star)

        return EPSFStars(fitted_stars)

    def _fit_star(self, epsf, star):
        """
        Fit an ePSF model to a single star.

        Parameters
        ----------
        epsf : `ImagePSF`
            An ePSF model to be fitted to the star.

        star : `EPSFStar`
            The star to be fit.

        Returns
        -------
        star : `EPSFStar`
            The fitted star with updated cutout center and flux.
        """
        fit_shape = self.fit_shape
        fitter = self.fitter
        fitter_kwargs = self._fitter_kwargs
        fitter_has_fit_info = self._fitter_has_fit_info

        # Create a shallow copy to avoid mutating the input star. This
        # is a shallow copy, so the large numpy arrays (_data, weights,
        # mask) are shared and not duplicated; only the object wrapper
        # and small scalar attributes are new.
        star = copy.copy(star)

        if fit_shape is not None:
            try:
                xcenter, ycenter = star.cutout_center
                large_slc, _ = overlap_slices(star.shape, fit_shape,
                                              (ycenter, xcenter),
                                              mode='strict')
            except (PartialOverlapError, NoOverlapError):
                star._fit_error_status = 1
                return star

            data = star.data[large_slc]
            weights = star.weights[large_slc]

            # Define the origin of the fitting region
            x0 = large_slc[1].start
            y0 = large_slc[0].start
        else:
            # Use the entire cutout image
            data = star.data
            weights = star.weights

            # Define the origin of the fitting region
            x0 = 0
            y0 = 0

        # Define positions in the undersampled grid. The fitter will
        # evaluate on the defined interpolation grid, currently in the
        # range [0, len(undersampled grid)].
        yy, xx = np.indices(data.shape, dtype=float)
        xx = xx + x0 - star.cutout_center[0]
        yy = yy + y0 - star.cutout_center[1]

        # Define the initial guesses for fitted flux and shifts
        epsf.flux = star.flux
        epsf.x_0 = 0.0
        epsf.y_0 = 0.0

        try:
            fitted_epsf = fitter(model=epsf, x=xx, y=yy, z=data,
                                 weights=weights, **fitter_kwargs)
        except TypeError:
            # Handle case where the fitter does not support weights
            fitted_epsf = fitter(model=epsf, x=xx, y=yy, z=data,
                                 **fitter_kwargs)

        fit_error_status = 0
        if fitter_has_fit_info:
            fit_info = fitter.fit_info
            if 'ierr' in fit_info and fit_info['ierr'] not in [1, 2, 3, 4]:
                fit_error_status = 2  # fit solution was not found
        else:
            fit_info = None

        # Compute the star's fitted position
        x_center = star.cutout_center[0] + fitted_epsf.x_0.value
        y_center = star.cutout_center[1] + fitted_epsf.y_0.value

        # Check if fitted position is outside the data cutout
        if (x_center < 0 or x_center >= star.shape[1]
                or y_center < 0 or y_center >= star.shape[0]):
            fit_error_status = 3  # fitted position outside cutout

        if fit_error_status != 3:
            star.cutout_center = (x_center, y_center)
            # Set the star's flux to the ePSF-fitted flux
            star.flux = fitted_epsf.flux.value

        star._fit_info = fit_info
        star._fit_error_status = fit_error_status

        return star

    def _process_iteration(self, stars, epsf, iter_num):
        """
        Process a single iteration of ePSF building.

        Parameters
        ----------
        stars : `EPSFStars` object
            The stars used to build the ePSF.

        epsf : `ImagePSF` object
            Current ePSF model.

        iter_num : int
            Current iteration number.

        Returns
        -------
        epsf : `ImagePSF` object
            Updated ePSF model.

        stars : `EPSFStars` object
            Updated stars with new fitted centers.

        fit_failed : `~numpy.ndarray`
            Boolean array tracking failed fits.
        """
        # Build/improve the ePSF
        epsf = self._build_epsf_step(stars, epsf=epsf)

        # Fit the new ePSF to the stars to find improved centers
        with warnings.catch_warnings():
            message = '.*The fit may be unsuccessful;.*'
            warnings.filterwarnings('ignore', message=message,
                                    category=AstropyUserWarning)

            stars = self._fit_stars(epsf, stars)

        # Reset ePSF flux to 1.0 after fitting (fitting modifies the
        # flux)
        epsf.flux = 1.0

        # Find all stars where the fit failed
        fit_failed = np.array([star._fit_error_status > 0
                              for star in stars.all_stars])

        if np.all(fit_failed):
            msg = 'The ePSF fitting failed for all stars.'
            raise ValueError(msg)

        # Permanently exclude fitting any star where the fit fails
        # after 3 iterations
        if iter_num > 3 and np.any(fit_failed):
            for i in fit_failed.nonzero()[0]:
                star = stars.all_stars[i]
                # Only warn for stars being newly excluded
                if not star._excluded_from_fit:
                    if star._fit_error_status == 1:
                        reason = ('its fitting region extends beyond the '
                                  'star cutout image')
                    elif star._fit_error_status == 3:
                        reason = ('its fitted position is outside the '
                                  'data cutout')
                    else:  # _fit_error_status == 2
                        reason = 'the fit did not converge'

                    msg = (f'The star at ({star._center_original[0]:.2f}, '
                           f'{star._center_original[1]:.2f}) (index='
                           f'{star.id_label - 1}) has been excluded from '
                           f'ePSF fitting because {reason}.')
                    warnings.warn(msg, AstropyUserWarning)
                star._excluded_from_fit = True

        # Store the ePSF from this iteration
        self._epsf.append(epsf)

        return epsf, stars, fit_failed

    def _finalize_build(self, epsf, stars, progress_reporter, iter_num,
                        converged, final_center_accuracy,
                        excluded_star_indices):
        """
        Finalize the ePSF building process and create result object.

        Parameters
        ----------
        epsf : `ImagePSF` object
            Final ePSF model.

        stars : `EPSFStars` object
            Final fitted stars.

        progress_reporter : `_ProgressReporter`
            Progress reporter instance for handling completion messages.

        iter_num : int
            Number of completed iterations.

        converged : bool
            Whether the building process converged.

        final_center_accuracy : float
            Final center accuracy achieved.

        excluded_star_indices : list
            Indices of excluded stars.

        Returns
        -------
        result : `EPSFBuildResult`
            Structured result containing ePSF, stars, and build
            diagnostics.
        """
        # Handle progress reporting completion
        if iter_num < self.maxiters:
            progress_reporter.write_convergence_message(iter_num)
        progress_reporter.close()

        # Create structured result
        return EPSFBuildResult(
            epsf=epsf,
            fitted_stars=stars,
            iterations=iter_num,
            converged=converged,
            final_center_accuracy=final_center_accuracy,
            n_excluded_stars=len(excluded_star_indices),
            excluded_star_indices=excluded_star_indices,
        )

    def _build_epsf(self, stars, *, epsf=None):
        """
        Build iteratively an ePSF from star cutouts.

        Parameters
        ----------
        stars : `EPSFStars` object
            The stars used to build the ePSF.

        epsf : `ImagePSF` object, optional
            The initial ePSF model. If not input, then the ePSF will be
            built from scratch.

        Returns
        -------
        result : `EPSFBuildResult` or tuple
            The ePSF building results. Returns an `EPSFBuildResult` object
            with detailed information about the building process. For
            backward compatibility, the result can be unpacked as a tuple:
            ``(epsf, fitted_stars) = epsf_builder(stars)``.

        Notes
        -----
        The structured result object contains:
        - epsf: The final constructed ePSF
        - fitted_stars: Stars with updated centers/fluxes
        - iterations: Number of iterations performed
        - converged: Whether convergence was achieved
        - final_center_accuracy: Final center movement accuracy
        - n_excluded_stars: Number of stars excluded due to fit failures
        - excluded_star_indices: Indices of excluded stars
        """
        _EPSFValidator.validate_stars(stars, 'ePSF building')
        _EPSFValidator.validate_shape_compatibility(stars, self.oversampling,
                                                    self.shape)

        # Initialize variables for building process
        fit_failed = np.zeros(stars.n_stars, dtype=bool)
        centers = stars.cutout_center_flat

        # Setup progress tracking
        progress_reporter = _ProgressReporter(self.progress_bar,
                                              self.maxiters).setup()

        # Initialize iteration variables and tracking
        iter_num = 0
        converged = False
        center_dist_sq = np.array([self.center_accuracy_sq + 1.0])
        excluded_star_indices = []

        # Main iteration loop
        while (iter_num < self.maxiters and not np.all(fit_failed)
               and not converged):

            iter_num += 1

            # Process one iteration
            epsf, stars, fit_failed = self._process_iteration(
                stars, epsf, iter_num)

            # Track newly excluded stars
            if iter_num > 3 and np.any(fit_failed):
                new_excluded = fit_failed.nonzero()[0]
                for idx in new_excluded:
                    if idx not in excluded_star_indices:
                        excluded_star_indices.append(idx)

            # Check convergence based on center movements
            converged, center_dist_sq, centers = self._check_convergence(
                stars, centers, fit_failed)

            # Update progress bar
            progress_reporter.update()

        # Calculate the final center accuracy
        final_converged = converged
        final_center_accuracy = np.max(center_dist_sq) ** 0.5

        # Finalize and return structured results
        return self._finalize_build(epsf, stars, progress_reporter,
                                    iter_num, final_converged,
                                    final_center_accuracy,
                                    excluded_star_indices)
