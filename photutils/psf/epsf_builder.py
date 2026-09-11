# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tools to build and fit an effective PSF (ePSF) based on Anderson and
King 2000 (PASP 112, 1360) and Anderson 2016 (WFC3 ISR 2016-12).
"""

import copy
import inspect
import numbers
import warnings
from dataclasses import dataclass, field

import numpy as np
from astropy.modeling.fitting import TRFLSQFitter
from astropy.nddata import NoOverlapError, PartialOverlapError, overlap_slices
from astropy.stats import SigmaClip
from astropy.utils.decorators import deprecated, deprecated_attribute
from astropy.utils.exceptions import AstropyUserWarning
from scipy.ndimage import convolve
from scipy.stats import chi2 as chi2_dist

from photutils.centroids import centroid_com
from photutils.psf.epsf_stars import EPSFStar, EPSFStars, LinkedEPSFStar
from photutils.psf.image_models import ImagePSF
from photutils.psf.utils import _interpolate_missing_data
from photutils.utils._parameters import (SigmaClipSentinelDefault, as_pair,
                                         create_default_sigmaclip)
from photutils.utils._progress_bars import add_progress_bar
from photutils.utils._round import round_half_away
from photutils.utils._stats import nanmedian
from photutils.utils.exceptions import PhotutilsDeprecationWarning

__all__ = ['EPSFBuildResults', 'EPSFBuilder', 'EPSFFitter']

SIGMA_CLIP = SigmaClipSentinelDefault(sigma=3.0, maxiters=10)

# Parameters of the automatic ('auto') smoothing kernel and fit shape.
# The smoothing window is this fraction of the ePSF FWHM (in oversampled
# grid points) and no smoothing is applied below the minimum size. The
# fit shape is this multiple of the ePSF FWHM (in detector pixels), with
# the given minimum size.
_AUTO_KERNEL_FWHM_FRACTION = 0.7
_AUTO_KERNEL_MIN_SIZE = 5
_AUTO_FIT_FWHM_FRACTION = 2.0
_AUTO_FIT_MIN_SIZE = 5


def _fitter_accepts_weights(fitter):
    """
    Determine whether a fitter accepts a ``weights`` keyword.

    Parameters
    ----------
    fitter : callable
        The fitter to inspect.

    Returns
    -------
    result : bool
        `True` if the fitter call signature accepts a ``weights``
        keyword (directly or via ``**kwargs``).
    """
    spec = inspect.signature(fitter.__call__)
    return ('weights' in spec.parameters
            or any(p.kind == inspect.Parameter.VAR_KEYWORD
                   for p in spec.parameters.values()))


def _suppress_alias_modes(data, oversampling, *, nu_pass=0.7, nu_stop=1.0):
    """
    Low-pass filter an oversampled ePSF above the input pixel sampling
    frequency.

    The filter is applied independently along each axis with an
    oversampling factor greater than one. It has unit gain up to
    ``nu_pass`` cycles per input (undersampled) pixel, a raised-cosine
    transition, and zero gain at and above ``nu_stop`` cycles per input
    pixel. Frequencies at integer cycles per input pixel are the zeros
    of the pixel response, so a pixel-integrated PSF has essentially no
    power there or above. However, those are exactly the frequencies at
    which the star-pixel sampling lattice aliases onto the oversampled
    grid, so noise at those frequencies can grow into a checkerboard
    pattern during the ePSF build iterations.

    Parameters
    ----------
    data : 2D `~numpy.ndarray`
        The oversampled ePSF data.

    oversampling : tuple of int
        The (y, x) oversampling factors.

    nu_pass : float, optional
        The end of the passband in cycles per input pixel.

    nu_stop : float, optional
        The start of the stopband in cycles per input pixel.

    Returns
    -------
    result : 2D `~numpy.ndarray`
        The filtered data. The input is returned unchanged if both
        oversampling factors are one.
    """
    data = np.asarray(data, dtype=float)
    for axis in (0, 1):
        factor = int(oversampling[axis])
        if factor < 2:
            continue

        npts = data.shape[axis]
        # Frequency in cycles per input (undersampled) pixel
        nu = np.abs(np.fft.fftfreq(npts)) * factor
        frac = np.clip((nu - nu_pass) / (nu_stop - nu_pass), 0.0, 1.0)
        gain = 0.5 * (1.0 + np.cos(np.pi * frac))
        shape = [1, 1]
        shape[axis] = npts

        spectrum = np.fft.fft(data, axis=axis) * gain.reshape(shape)
        data = np.fft.ifft(spectrum, axis=axis).real

    return data


def _odd_size(value):
    """
    Round a positive value to the nearest integer and then up to an odd
    integer.

    The result is the nearest integer when that is odd and the next
    larger integer when it is even, so the rounding is biased upward
    rather than to the nearest odd integer (e.g., 5.9 gives 7).

    Parameters
    ----------
    value : float
        The input value.

    Returns
    -------
    size : int
        The odd integer, at least 1.
    """
    size = round(value)
    if size % 2 == 0:
        size += 1
    return max(size, 1)


def _profile_fwhm(profile):
    """
    Measure the full width at half maximum of a 1D profile.

    The width is measured between the first crossings of the half
    maximum on each side of the peak, using linear interpolation between
    the neighboring samples.

    Parameters
    ----------
    profile : 1D `~numpy.ndarray`
        The profile.

    Returns
    -------
    width : float or `None`
        The width in samples, or `None` if the peak is not finite and
        positive or the profile does not cross the half maximum on both
        sides of the peak within the array.
    """
    if not np.any(np.isfinite(profile)):
        return None

    peak_index = int(np.nanargmax(profile))
    peak = profile[peak_index]
    if not np.isfinite(peak) or peak <= 0:
        return None

    half_max = 0.5 * peak
    edges = []
    for step in (-1, 1):
        i = peak_index
        while (0 <= i + step < len(profile)
               and profile[i + step] > half_max):
            i += step
        j = i + step
        if j < 0 or j >= len(profile) or not np.isfinite(profile[j]):
            return None
        # Linear interpolation between the last sample above the half
        # maximum and the first sample at or below it.
        edges.append(i + step * (profile[i] - half_max)
                     / (profile[i] - profile[j]))

    return abs(edges[1] - edges[0])


def _measure_fwhm(data):
    """
    Measure the FWHM of a 2D ePSF along each axis.

    The FWHM is measured along the row and the column through the
    peak, from the first crossings of the half maximum on each side of
    the peak. To reduce the sensitivity to noise, the row and column
    profiles are averaged over the three rows and columns centered on
    the peak.

    Parameters
    ----------
    data : 2D `~numpy.ndarray`
        The ePSF data.

    Returns
    -------
    fwhm : tuple of float or `None`
        The (y, x) FWHM in grid points, or `None` if the FWHM could not
        be measured (e.g., a non-positive peak or a profile that does
        not fall below half of the peak within the array).
    """
    data = np.asarray(data, dtype=float)
    if data.size == 0 or not np.any(np.isfinite(data)):
        return None

    iy, ix = np.unravel_index(np.nanargmax(data), data.shape)
    if data[iy, ix] <= 0:
        return None

    # The rows (columns) of a separable PSF are scaled copies of each
    # other, so averaging the three rows (columns) around the peak
    # reduces the noise without broadening the profiles.
    profile_x = data[max(iy - 1, 0):iy + 2, :].mean(axis=0)
    profile_y = data[:, max(ix - 1, 0):ix + 2].mean(axis=1)
    fwhm_x = _profile_fwhm(profile_x)
    fwhm_y = _profile_fwhm(profile_y)
    if fwhm_x is None or fwhm_y is None:
        return None

    return fwhm_y, fwhm_x


def _phase_uniformity_pvalue(centers, *, nbins=4):
    """
    Compute a chi-square p-value for the uniformity of the subpixel
    phases of star centers.

    Parameters
    ----------
    centers : 2D `~numpy.ndarray`
        The (x, y) star centers, one row per star.

    nbins : int, optional
        The number of phase bins per axis.

    Returns
    -------
    pvalue : float
        The p-value of the chi-square test of a uniform phase
        distribution, combining both axes.
    """
    chi2 = 0.0
    for axis in (0, 1):
        phases = np.mod(centers[:, axis], 1.0)
        hist = np.histogram(phases, bins=nbins, range=(0.0, 1.0))[0]
        expected = len(phases) / nbins
        chi2 += np.sum((hist - expected) ** 2) / expected

    return chi2_dist.sf(chi2, 2 * (nbins - 1))


def _make_polynomial_kernel(size, *, degree=4):
    """
    Make the kernel of a least-squares polynomial smoother.

    Convolving an array with this kernel replaces each value by the
    value at the center of a 2D polynomial of the given degree fit by
    least squares to the ``size`` x ``size`` values centered on it. The
    5x5 ``'quartic'`` and ``'quadratic'`` kernels of `_SmoothingKernel`
    are the ``degree=4`` and ``degree=2`` cases. The quartic kernel
    is the smoothing kernel of equation 8 of Anderson and King 2000
    (PASP 112, 1360).

    Parameters
    ----------
    size : int
        The odd size of the square kernel.

    degree : int, optional
        The degree of the 2D polynomial. The number of polynomial terms
        must be smaller than the number of kernel elements.

    Returns
    -------
    kernel : 2D `~numpy.ndarray`
        The smoothing kernel.
    """
    if size < 3 or size % 2 == 0:
        msg = 'size must be an odd integer greater than or equal to 3'
        raise ValueError(msg)

    nterms = (degree + 1) * (degree + 2) // 2
    if degree < 1 or nterms >= size**2:
        msg = ('degree must be at least 1 and the number of polynomial '
               'terms must be smaller than the number of kernel elements')
        raise ValueError(msg)

    half = size // 2
    yy, xx = np.mgrid[-half:half + 1, -half:half + 1]
    design = np.array([xx.ravel()**i * yy.ravel()**j
                       for i in range(degree + 1)
                       for j in range(degree + 1 - i)], dtype=float).T

    # The smoothed value at the center is the center row of the
    # least-squares projection matrix applied to the data.
    center = half * size + half
    weights = design @ np.linalg.solve(design.T @ design, design[center])
    return weights.reshape(size, size)


class _SmoothingKernel:
    """
    Utility class for ePSF smoothing kernel generation and convolution.

    This class encapsulates the creation of smoothing kernels used in
    ePSF building and provides consistent smoothing operations.
    """

    # The 5x5 least-squares polynomial kernels. The quartic kernel is
    # equation 8 of Anderson and King 2000.
    QUARTIC_KERNEL = _make_polynomial_kernel(5, degree=4)
    QUADRATIC_KERNEL = _make_polynomial_kernel(5, degree=2)

    # The kernel constructor is a module-level function so that the
    # class attributes above can be generated when the class is created.
    make_polynomial_kernel = staticmethod(_make_polynomial_kernel)

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

        ValueError
            If a custom kernel array is not 2D.

        Notes
        -----
        The predefined kernels are 5x5 least-squares polynomial
        smoothing kernels generated by ``make_polynomial_kernel``. The
        ``'quartic'`` kernel (degree 4) is the smoothing kernel of
        equation 8 of Anderson and King 2000 (PASP 112, 1360), and the
        ``'quadratic'`` kernel is the degree 2 counterpart.
        """
        if isinstance(kernel_type, np.ndarray):
            if kernel_type.ndim != 2:
                msg = 'smoothing_kernel must be a 2D array'
                raise ValueError(msg)
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
    def validate_oversampling(oversampling, *, context=''):
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
                                   lower_bound=(0, 0))
        except (TypeError, ValueError) as e:
            msg = f'Invalid oversampling parameter - {e}'
            if context:
                msg = f'{context}: {msg}'
            raise ValueError(msg) from None

        return oversampling

    @staticmethod
    def validate_shape_compatibility(stars, oversampling, *, shape=None):
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

        # Iterate over the flat star list so that the stars within
        # LinkedEPSFStar objects are validated individually
        star_list = getattr(stars, 'all_stars', stars)

        # Collect star dimension statistics
        star_heights = [star.shape[0] for star in star_list]
        star_widths = [star.shape[1] for star in star_list]
        max_height = max(star_heights)
        max_width = max(star_widths)

        # Check for extremely small stars that may cause issues
        min_star_size = 3  # minimum reasonable star cutout size
        problematic_stars = []
        for i, star in enumerate(star_list):
            if min(star.shape) < min_star_size:
                problematic_stars.append(f'Star {i}: {star.shape}')

        if problematic_stars:
            msg = (f'Found {len(problematic_stars)} star(s) with very small '
                   f'dimensions (< {min_star_size}x{min_star_size}): '
                   f"{', '.join(problematic_stars)}. Consider using larger "
                   'star cutouts for better ePSF quality.')
            raise ValueError(msg)

        # Compute the minimum required ePSF shape, consistent with
        # _CoordinateTransformer.compute_epsf_shape (add 1 only when
        # the product is even, to ensure odd dimensions)
        min_epsf_height = max_height * oversampling[0]
        if min_epsf_height % 2 == 0:
            min_epsf_height += 1
        min_epsf_width = max_width * oversampling[1]
        if min_epsf_width % 2 == 0:
            min_epsf_width += 1

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

    @staticmethod
    def validate_stars(stars, *, context=''):
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
        ValueError
            If stars are invalid.
        """
        # Check basic type and structure
        if not hasattr(stars, '__len__') or len(stars) == 0:
            msg = 'EPSFStars object must contain at least one star'
            if context:
                msg = f'{context}: {msg}'
            raise ValueError(msg)

        # Validate the individual stars in the flat star list so that
        # the stars within LinkedEPSFStar objects are checked (their
        # container delegates attributes like shape as lists). The
        # flat indices in error messages match excluded_star_indices.
        star_list = getattr(stars, 'all_stars', stars)
        invalid_stars = []
        for i, star in enumerate(star_list):
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
                   f'{len(star_list)} total:\n' + '\n'.join(error_details))
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
        TypeError
            If center accuracy is not a number.

        ValueError
            If center accuracy is not positive.
        """
        if not isinstance(center_accuracy, numbers.Real):
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
    def validate_converged_fraction(converged_fraction):
        """
        Validate the converged fraction parameter.

        Parameters
        ----------
        converged_fraction : float
            The fraction of the successfully fitted stars that must
            converge.

        Raises
        ------
        TypeError
            If converged_fraction is not a number.

        ValueError
            If converged_fraction is not in the range (0, 1].
        """
        if (isinstance(converged_fraction, bool)
                or not isinstance(converged_fraction, numbers.Real)):
            msg = (f'converged_fraction must be a number, got '
                   f'{type(converged_fraction)}')
            raise TypeError(msg)

        if not 0.0 < converged_fraction <= 1.0:
            msg = (f'converged_fraction must be in the range (0, 1], got '
                   f'{converged_fraction}')
            raise ValueError(msg)

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
        TypeError
            If maxiters is not an integer.

        ValueError
            If maxiters is not positive.
        """
        if isinstance(maxiters, bool) or not isinstance(maxiters,
                                                        numbers.Integral):
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
class EPSFBuildResults:
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
        Whether the building process converged based on the
        center accuracy criterion. `True` if at least the
        ``converged_fraction`` of the successfully fitted stars moved
        by less than the specified center accuracy between the final
        iterations.

    final_center_accuracy : float
        The maximum center displacement in the final iteration, in
        pixels, over all of the successfully fitted stars. This includes
        the stars that the ``converged_fraction`` of the builder allows
        to remain unconverged, so it can be much larger than the
        ``center_accuracy`` for a converged build. Use it together with
        ``final_converged_fraction`` to assess the convergence quality.

    final_converged_fraction : float
        The fraction of the successfully fitted stars whose centers
        changed by less than ``center_accuracy`` in the final iteration.
        The build is converged when this fraction is at least the
        ``converged_fraction`` of the builder.

    n_excluded_stars : int
        The number of individual stars (including those from linked
        stars) that were excluded from fitting due to repeated fit
        failures.

    excluded_star_indices : list
        Indices of stars that were excluded from fitting during the
        building process. These correspond to positions in the flattened
        star list (stars.all_stars).

    smoothing_kernel : 2D `~numpy.ndarray` or `None`
        The smoothing kernel applied in the final iteration, or `None`
        if no smoothing was applied. This includes the kernel chosen
        by ``smoothing_kernel='auto'``, which can be input as a fixed
        ``smoothing_kernel`` to reproduce the build.

    fit_shape : tuple or `None`
        The (ny, nx) shape of the fitting box used in the final
        iteration, or `None` if the entire star cutouts were fit. This
        includes the shape chosen by ``fit_shape='auto'``.

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
    smoothing_kernel: np.ndarray | None = field(default=None, compare=False,
                                                repr=False)
    fit_shape: tuple | None = None
    final_converged_fraction: float | None = None

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

    def __len__(self):
        """
        Return the length of the backward-compatible 2-tuple.

        Returns
        -------
        length : int
            Always 2, for (epsf, fitted_stars).
        """
        return 2

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

        msg = 'EPSFBuildResults index must be 0 (epsf) or 1 (fitted_stars)'
        raise IndexError(msg)


@deprecated(since='3.0',
            message=('EPSFFitter is deprecated and will be removed in '
                     'version 4.0. Use EPSFBuilder with the fitter, '
                     'fit_shape, and fitter_maxiters parameters instead.'),
            warning_type=PhotutilsDeprecationWarning)
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
        self._fitter_accepts_weights = _fitter_accepts_weights(self.fitter)
        if fit_boxsize is not None:
            self.fit_boxsize = as_pair('fit_boxsize', fit_boxsize,
                                       lower_bound=(3, 1), check_odd=True)
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
            that contain weights, a weighted fit of the ePSF to the star
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
                # Skip fitting stars that have been excluded. Return
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
                    # Skip fitting stars that have been excluded. Return
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
        # mask) are shared and not duplicated. Only the object wrapper
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
            mask = star.mask[large_slc]

            # Define the origin of the fitting region
            x0 = large_slc[1].start
            y0 = large_slc[0].start
        else:
            # Use the entire cutout image
            data = star.data
            weights = star.weights
            mask = star.mask

            # Define the origin of the fitting region
            x0 = 0
            y0 = 0

        # Define positions in the undersampled grid. The fitter will
        # evaluate on the defined interpolation grid, currently in the
        # range [0, len(undersampled grid)].
        yy, xx = np.indices(data.shape, dtype=float)
        xx = xx + x0 - star.cutout_center[0]
        yy = yy + y0 - star.cutout_center[1]

        # Fit only the unmasked pixels. Masked pixels (non-finite data
        # or non-positive weights) are dropped from the fit because the
        # fitter objective function raises on non-finite data values
        # even where the weight is zero.
        good = ~mask
        if not np.any(good):
            star._fit_error_status = 4  # fitting region is fully masked
            return star
        xx = xx[good]
        yy = yy[good]
        data = data[good]
        weights = weights[good]

        # Define the initial guesses for fitted flux and shifts
        epsf.flux = star.flux
        epsf.x_0 = 0.0
        epsf.y_0 = 0.0

        if self._fitter_accepts_weights:
            fitted_epsf = fitter(model=epsf, x=xx, y=yy, z=data,
                                 weights=weights, **fitter_kwargs)
        else:
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
            fit_error_status = 3

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
    oversampling : int or array_like (int), optional
        The integer oversampling factor(s) of the output ePSF relative
        to the input ``stars`` along each axis. If ``oversampling`` is a
        scalar then it will be used for both axes. If ``oversampling``
        has two elements, they must be in ``(y, x)`` order. The ePSF
        should have at least about four grid points per FWHM, so a good
        rule of thumb is ``oversampling >= 4 / FWHM`` with the FWHM
        in pixels. Do not use a larger value than the data require.
        For well-sampled data (a FWHM of a few pixels or more), an
        oversampling of 1 is usually the best choice. Larger values
        require more stars, roughly ``10 * oversampling**2`` stars with
        uniformly distributed subpixel phases. See the guidelines in the
        ePSF building user guide for details.

    shape : int, tuple of two ints, or `None`, optional
        The (ny, nx) shape of the output ePSF. If the input shape is
        even along any axis, it will be made odd by adding one (with a
        warning). If the ``shape`` is `None`, it will be derived from
        the sizes of the input ``stars`` and the ePSF ``oversampling``
        factor. The output ePSF will always have odd sizes along both
        axes to ensure a well-defined central pixel.

    smoothing_kernel : {'auto', 'quartic', 'quadratic'}, 2D array, or `None`
        The smoothing kernel to apply to the ePSF during each iteration
        step. If ``'auto'``, a least-squares quartic polynomial kernel
        (see Notes) whose width is 0.7 times the FWHM of the current
        ePSF in oversampled grid points is used, and no smoothing
        is applied if that width is less than 5 grid points, i.e.,
        for heavily undersampled ePSFs. The kernel is square and the
        FWHM is measured in each iteration along the narrowest axis
        of the ePSF, so with anisotropic oversampling the axis with
        the fewer grid points per FWHM sets the kernel size. If the
        FWHM cannot be measured, the ``'quartic'`` kernel is used
        and a warning is emitted. The predefined ``'quartic'`` and
        ``'quadratic'`` kernels are 5x5 kernels (in oversampled grid
        points) derived from fourth and second degree polynomials,
        respectively. Alternatively, a custom 2D array can be input.
        If `None` then no smoothing will be performed. The kernels are
        applied on the oversampled grid, so the physical width of a
        fixed kernel depends on the oversampling factor. Power at and
        above one cycle per input pixel is always removed from the ePSF
        along oversampled axes, independently of this parameter.

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

    recentering_boxsize : int or tuple of two ints, optional
        The size (in pixels) of the box used to calculate the centroid
        of the ePSF during each build iteration. The size is in
        the input star (i.e., undersampled) pixel space. It is
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
        The desired accuracy for the centers of stars. The
        building iterations will stop when the centers of at least
        ``converged_fraction`` of the successfully fitted stars change
        by less than ``center_accuracy`` pixels between iterations.

    converged_fraction : float, optional
        The fraction of the successfully fitted stars whose centers
        must change by less than ``center_accuracy`` pixels between
        iterations for the build to be considered converged. The
        default of 0.95 allows a small number of stars (e.g., spurious
        detections or contaminated cutouts) whose centers never settle
        to not prevent convergence. Set to 1.0 to require all stars
        to converge. The fraction achieved in the final iteration is
        reported in the ``final_converged_fraction`` attribute of the
        returned `EPSFBuildResults`.

    fitter : `~astropy.modeling.fitting.Fitter` or `EPSFFitter`, optional
        A `~astropy.modeling.fitting.Fitter` object used to fit the
        ePSF to stars. If `None`, then the default
        `~astropy.modeling.fitting.TRFLSQFitter` will be used.

        .. deprecated:: 3.0
            Passing an `EPSFFitter` instance is deprecated. Use
            the ``fitter``, ``fit_shape``, and ``fitter_maxiters``
            parameters instead.

    fit_shape : {'auto'}, int, tuple of int, or `None`, optional
        The size (in detector pixels) of the box centered on the star
        to be used for ePSF fitting. This allows using only a small
        number of central pixels of the star (i.e., where the star is
        brightest) for fitting. If ``'auto'``, a square box of twice the
        FWHM of the current ePSF (measured in each iteration along its
        narrowest axis, in detector pixels), with a minimum of 5 pixels
        and a maximum of the smallest star cutout size, is used. For an
        elongated ePSF, the narrower axis sets the box size along both
        axes. If the FWHM cannot be measured, a 5x5 box is used and a
        warning is emitted. If ``fit_shape`` is a scalar then a square
        box of size ``fit_shape`` will be used. If ``fit_shape`` has two
        elements, they must be in ``(ny, nx)`` order. ``fit_shape`` must
        have odd values and be greater than or equal to 3 for both axes.
        If `None`, the fitter will use the entire star image.

    fitter_maxiters : int, optional
        The maximum number of iterations in which the ``fitter`` is
        called for each star. The value can be increased if the fit
        is not converging. This parameter is passed to the ``fitter``
        if it supports the ``maxiter`` parameter and ignored otherwise.

    constrain_fluxes : bool, optional
        Whether to constrain the fluxes of the stars within each
        `~photutils.psf.LinkedEPSFStar` (i.e., the same physical star
        observed in multiple dithered images) to their mean value after
        each fitting iteration, in addition to constraining their
        centers to the same sky coordinate. This breaks the degeneracy
        between the flux of a star and its subpixel position caused
        by intra-pixel sensitivity variations, which would otherwise
        be absorbed into the ePSF. It assumes that the linked images
        have the same flux scale (e.g., the same exposure time and
        throughput). Set to `False` if the linked images have different
        flux scales. This parameter has no effect on stars that are not
        linked.

    maxiters : int, optional
        The maximum number of ePSF building iterations to perform.

    progress_bar : bool, optional
        Whether to print the progress bar during the build
        iterations. The progress bar requires that the `tqdm
        <https://tqdm.github.io/>`_ optional dependency be installed.

    Notes
    -----
    In each build iteration, the residual between each star and the
    current ePSF model is deposited on every oversampled grid point
    inside the footprint of each star pixel, so that every star
    contributes to every grid point regardless of its subpixel phase.
    After the residuals are combined and the ePSF is smoothed, power
    at and above one cycle per input pixel is removed along each
    oversampled axis. A pixel-integrated PSF has essentially no power
    there, but those are the frequencies at which the star-pixel
    sampling lattice aliases onto the oversampled grid. Without
    these two measures, noise in the ePSF grid from heterogeneous
    or contaminated stars can bias the fitted star centers toward
    particular subpixel phases and grow into a checkerboard pattern in
    the ePSF. A warning is emitted if the subpixel phases of the fitted
    star centers are strongly non-uniform at the end of the build.

    The default ``smoothing_kernel='auto'`` and ``fit_shape='auto'``
    scale the smoothing kernel and the fitting box with the FWHM of the
    ePSF. The 5x5 ``'quartic'`` kernel (in oversampled grid points)
    and 5-pixel fitting box (in detector pixels) of Anderson and King
    2000 were designed for HST images with an oversampling factor
    of 4, where they correspond to about 0.7 and 2.5 FWHM. A fixed
    kernel oversmooths heavily undersampled ePSFs, and a fixed 5-pixel
    fitting box applied to a well-sampled star uses only its flat core,
    which biases the fitted centers and can prevent convergence. The
    polynomial kernels replace each grid value by the value at the
    center of a least-squares polynomial fit to the surrounding grid
    values. The kernel and fitting box chosen in the final iteration are
    reported in the ``smoothing_kernel`` and ``fit_shape`` attributes of
    the returned `EPSFBuildResults`.

    This class stores per-call state on the instance (e.g., the list
    of per-iteration ePSFs), so a single instance must not be called
    concurrently from multiple threads. Create one instance per thread
    for concurrent use. Sharing a single Astropy fitter instance across
    concurrently-used objects is also unsafe because Astropy fitters
    store ``fit_info`` on themselves.
    """

    coord_transformer = deprecated_attribute(
        'coord_transformer', '3.1',
        warning_type=PhotutilsDeprecationWarning)

    def __init__(self, *, oversampling=4, shape=None,
                 smoothing_kernel='auto', sigma_clip=SIGMA_CLIP,
                 recentering_func=centroid_com, recentering_boxsize=(5, 5),
                 recentering_maxiters=20, center_accuracy=1.0e-3,
                 converged_fraction=0.95, fitter=None, fit_shape='auto',
                 fitter_maxiters=100, constrain_fluxes=True, maxiters=10,
                 progress_bar=True):

        # Validate and store oversampling using the validator
        self.oversampling = _EPSFValidator.validate_oversampling(
            oversampling, context='EPSFBuilder initialization')

        # Initialize coordinate transformer for consistent transformations
        self._coord_transformer = _CoordinateTransformer(self.oversampling)

        if shape is not None:
            shape = as_pair('shape', shape, lower_bound=(0, 0))
            even = shape % 2 == 0
            if np.any(even):
                new_shape = shape + even.astype(int)
                msg = (f'The input shape {tuple(shape)} has even '
                       f'values. Using {tuple(new_shape)} instead '
                       'for proper ePSF centering.')
                warnings.warn(msg, AstropyUserWarning)
                shape = new_shape
        self.shape = shape

        self.recentering_func = recentering_func
        if (isinstance(recentering_maxiters, bool)
                or not isinstance(recentering_maxiters, numbers.Integral)
                or recentering_maxiters <= 0):
            msg = ('recentering_maxiters must be a strictly-positive '
                   'integer')
            raise ValueError(msg)
        self.recentering_maxiters = int(recentering_maxiters)
        self.recentering_boxsize = as_pair('recentering_boxsize',
                                           recentering_boxsize,
                                           lower_bound=(3, 1), check_odd=True)

        if isinstance(smoothing_kernel, str):
            if smoothing_kernel not in ('auto', 'quartic', 'quadratic'):
                msg = ("smoothing_kernel must be 'auto', 'quartic', "
                       "'quadratic', a 2D array, or None")
                raise ValueError(msg)
        elif smoothing_kernel is not None:
            # Validate early so bad kernels fail at construction
            # instead of in the middle of a build.
            _SmoothingKernel.get_kernel(smoothing_kernel)
        self.smoothing_kernel = smoothing_kernel

        # Per-call state for the automatic smoothing kernel and fit
        # shape (reset in build_epsf and updated in each iteration).
        self._auto_state = None
        self._auto_fit_max = None
        self._auto_fallback_warned = False

        # Handle fitter parameter. Accept both astropy Fitter and
        # deprecated EPSFFitter for backward compatibility.
        if isinstance(fitter, EPSFFitter):
            msg = ('Passing an EPSFFitter instance to EPSFBuilder is '
                   'deprecated. Use the fitter, fit_shape, and '
                   'fitter_maxiters parameters instead.')
            warnings.warn(msg, PhotutilsDeprecationWarning)
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
            if isinstance(fit_shape, str) and not self._is_auto(fit_shape):
                msg = ("fit_shape must be 'auto', an integer, a tuple of "
                       'integers, or None')
                raise ValueError(msg)
            if fit_shape is None or self._is_auto(fit_shape):
                self.fit_shape = fit_shape
            else:
                self.fit_shape = as_pair('fit_shape', fit_shape,
                                         lower_bound=(3, 1), check_odd=True)

            # Validate fitter_maxiters
            self.fitter_maxiters = self._validate_fitter_maxiters(
                fitter_maxiters)

            # Build fitter keyword arguments
            self._fitter_kwargs = {}
            if self.fitter_maxiters is not None:
                self._fitter_kwargs['maxiter'] = self.fitter_maxiters

        self._fitter_has_fit_info = hasattr(self.fitter, 'fit_info')
        self._fitter_accepts_weights = _fitter_accepts_weights(self.fitter)

        self.constrain_fluxes = bool(constrain_fluxes)

        # Validate center accuracy using the validator
        _EPSFValidator.validate_center_accuracy(center_accuracy)
        self.center_accuracy_sq = center_accuracy**2

        # Validate converged_fraction using the validator
        _EPSFValidator.validate_converged_fraction(converged_fraction)
        self.converged_fraction = float(converged_fraction)

        # Validate maxiters using the validator
        _EPSFValidator.validate_maxiters(maxiters)
        self.maxiters = maxiters

        self.progress_bar = progress_bar

        if sigma_clip is SIGMA_CLIP:
            sigma_clip = create_default_sigmaclip(sigma=SIGMA_CLIP.sigma,
                                                  maxiters=SIGMA_CLIP.maxiters)
        if sigma_clip is not None and not isinstance(sigma_clip, SigmaClip):
            msg = ('sigma_clip must be an astropy.stats.SigmaClip '
                   'instance or None')
            raise TypeError(msg)
        self._sigma_clip = sigma_clip

    def __call__(self, stars):
        """
        Build an ePSF from input stars.

        Parameters
        ----------
        stars : `EPSFStars`
            The stars used to build the ePSF.

        Returns
        -------
        result : `EPSFBuildResults`
            The result of the ePSF building process.
        """
        return self.build_epsf(stars)

    @staticmethod
    def _is_auto(value):
        """
        Return whether a parameter value is the string ``'auto'``.
        """
        return isinstance(value, str) and value == 'auto'

    def _auto_fallback(self):
        """
        Return the fallback state used when the ePSF FWHM cannot be
        measured, warning once per build.
        """
        fit_size = _AUTO_FIT_MIN_SIZE
        if self._auto_fit_max is not None:
            fit_size = min(fit_size, self._auto_fit_max)

        if not self._auto_fallback_warned:
            auto_kernel = self._is_auto(self.smoothing_kernel)
            auto_fit = self._is_auto(self.fit_shape)

            if auto_kernel and auto_fit:
                what = ('smoothing_kernel and fit_shape fall back to the '
                        f"'quartic' kernel and a fit shape of {fit_size}")
            elif auto_kernel:
                what = "smoothing_kernel falls back to the 'quartic' kernel"
            else:
                what = f'fit_shape falls back to a fit shape of {fit_size}'

            msg = ('The FWHM of the ePSF could not be measured, so the '
                   f"'auto' {what}.")
            warnings.warn(msg, AstropyUserWarning)

            self._auto_fallback_warned = True

        return {'kernel': _SmoothingKernel.QUARTIC_KERNEL,
                'fit_shape': (fit_size, fit_size)}

    def _update_auto_parameters(self, epsf_data):
        """
        Choose the smoothing kernel and fit shape from the FWHM of the
        current ePSF.

        The smoothing window is ``_AUTO_KERNEL_FWHM_FRACTION`` times
        the FWHM in oversampled grid points (no smoothing below
        ``_AUTO_KERNEL_MIN_SIZE``), and the fit shape is
        ``_AUTO_FIT_FWHM_FRACTION`` times the FWHM in detector pixels
        (at least ``_AUTO_FIT_MIN_SIZE`` and at most the smallest star
        cutout size). The FWHM is measured along the narrowest axis of
        the ePSF.

        Parameters
        ----------
        epsf_data : 2D `~numpy.ndarray`
            The current (unsmoothed) ePSF data.
        """
        if not (self._is_auto(self.smoothing_kernel)
                or self._is_auto(self.fit_shape)):
            return

        fwhm = _measure_fwhm(epsf_data)
        if fwhm is None:
            self._auto_state = self._auto_fallback()
            return

        fwhm_grid = min(fwhm)
        fwhm_pixels = min(fwhm[0] / self.oversampling[0],
                          fwhm[1] / self.oversampling[1])

        kernel = None
        # Cap the kernel at the largest odd size smaller than the ePSF.
        # A kernel as large as the ePSF would fit every grid value
        # mostly to edge-reflected data.
        size = _odd_size(_AUTO_KERNEL_FWHM_FRACTION * fwhm_grid)
        size = min(size, _odd_size(min(epsf_data.shape)) - 2)
        if size >= _AUTO_KERNEL_MIN_SIZE:
            kernel = _SmoothingKernel.make_polynomial_kernel(size, degree=4)

        fit_size = max(_AUTO_FIT_MIN_SIZE,
                       _odd_size(_AUTO_FIT_FWHM_FRACTION * fwhm_pixels))
        if self._auto_fit_max is not None:
            fit_size = min(fit_size, self._auto_fit_max)

        self._auto_state = {'kernel': kernel,
                            'fit_shape': (fit_size, fit_size)}

    def _current_smoothing_kernel(self):
        """
        Return the smoothing kernel for the current iteration.
        """
        if self._is_auto(self.smoothing_kernel):
            return self._auto_state['kernel']
        return self.smoothing_kernel

    def _current_fit_shape(self):
        """
        Return the fit shape for the current iteration.
        """
        if self._is_auto(self.fit_shape):
            return self._auto_state['fit_shape']
        return self.fit_shape

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
        if isinstance(fitter_maxiters, bool) or not isinstance(
                fitter_maxiters, numbers.Integral):
            msg = ('fitter_maxiters must be an integer, got '
                   f'{type(fitter_maxiters)}')
            raise TypeError(msg)
        if fitter_maxiters <= 0:
            msg = 'fitter_maxiters must be a positive integer'
            raise ValueError(msg)

        spec = inspect.signature(self.fitter.__call__)
        has_maxiter = ('maxiter' in spec.parameters
                       or any(p.kind == inspect.Parameter.VAR_KEYWORD
                              for p in spec.parameters.values()))
        if not has_maxiter:
            msg = ("'fitter_maxiters' will be ignored because "
                   'it is not accepted by the input fitter')
            warnings.warn(msg, AstropyUserWarning)
            return None
        return fitter_maxiters

    def _create_initial_epsf(self, stars):
        """
        Create an initial `ImagePSF` object with zero data.

        The ePSF shape is the configured ``shape`` if provided,
        otherwise it is derived from the maximum star cutout dimensions
        and the oversampling factors (made odd along each axis so the
        ePSF has a well-defined central pixel). The origin is set to the
        geometric center of the data array.

        Parameters
        ----------
        stars : `EPSFStars` object
            The stars used to build the ePSF.

        Returns
        -------
        epsf : `ImagePSF` object
            The initial zero-data ePSF model.
        """
        oversampling = self.oversampling
        shape = self.shape

        # Define the ePSF shape using coordinate transformer
        if shape is not None:
            shape = as_pair('shape', shape, lower_bound=(0, 0), check_odd=True)
        else:
            # Use coordinate transformer to compute shape from star
            # dimensions (use the flat star list so that stars within
            # LinkedEPSFStar objects contribute individual shapes)
            star_shapes = [star.shape for star in stars.all_stars]
            shape = self._coord_transformer.compute_epsf_shape(star_shapes)

        # Initialize with zeros
        data = np.zeros(shape, dtype=float)

        # Use coordinate transformer to compute origin
        origin_xy = self._coord_transformer.compute_epsf_origin(shape)

        return ImagePSF(data=data, origin=origin_xy, oversampling=oversampling,
                        fill_value=0.0)

    def _resample_residuals(self, stars, epsf, *, chunk_size=128):
        """
        Compute normalized residual images in the oversampled ePSF grid
        for all the input stars.

        A normalized residual image is calculated for each star by
        subtracting the normalized ePSF model from the normalized
        star at the location of the star in the undersampled grid.
        The normalized residual image is then resampled from the
        undersampled star grid to the oversampled ePSF grid by
        depositing each star pixel value on every oversampled grid
        point inside the footprint of that pixel. Every star therefore
        contributes to every grid point that its cutout covers,
        regardless of its subpixel phase. For an oversampling factor of
        one along an axis, this reduces to the nearest grid point.

        Parameters
        ----------
        stars : `EPSFStars` object
            The stars used to build the ePSF. Stars that are excluded
            from fitting are skipped.

        epsf : `ImagePSF` object
            The ePSF model.

        chunk_size : int, optional
            The number of stars processed together. The stars are
            resampled in chunks so that the temporary per-pixel arrays
            stay bounded for large star samples.

        Returns
        -------
        epsf_resid : 3D `~numpy.ndarray`
            A 3D cube containing the resampled residual images, one
            per star. The images contain NaNs where there is no data.
        """
        ny, nx = epsf.data.shape
        good_stars = stars.all_good_stars
        n_good_stars = len(good_stars)

        # Pre-allocate with NaN (default for missing data)
        epsf_resid = np.full((n_good_stars, ny, nx), np.nan)

        for start in range(0, n_good_stars, chunk_size):
            stop = min(start + chunk_size, n_good_stars)
            self._deposit_residuals(good_stars[start:stop], epsf,
                                    epsf_resid[start:stop])

        return epsf_resid

    def _deposit_residuals(self, stars, epsf, epsf_resid):
        """
        Deposit the normalized residuals of the input stars into a
        stack of oversampled ePSF grid images.

        Parameters
        ----------
        stars : list of `EPSFStar`
            The stars to resample.

        epsf : `ImagePSF` object
            The ePSF model.

        epsf_resid : 3D `~numpy.ndarray`
            The contiguous stack of residual images, with one image per
            input star, that is filled in place. Grid points that are
            not covered by a star pixel are left unchanged.
        """
        ny, nx = epsf.data.shape

        # Gather the unmasked pixels of all stars so that the ePSF model
        # is evaluated once and the residuals are deposited with a
        # single indexed assignment per footprint offset.
        star_index = []
        xidx_centered = []
        yidx_centered = []
        residuals = []
        for i, star in enumerate(stars):
            xidx, yidx = star._xyidx_centered
            star_index.append(np.full(xidx.size, i))
            xidx_centered.append(xidx)
            yidx_centered.append(yidx)
            residuals.append(star._data_values_normalized)
        star_index = np.concatenate(star_index)
        xidx_centered = np.concatenate(xidx_centered)
        yidx_centered = np.concatenate(yidx_centered)

        # The concatenation is a new array, so the model can be
        # subtracted in place.
        residuals = np.concatenate(residuals)
        residuals -= epsf.evaluate(x=xidx_centered, y=yidx_centered,
                                   flux=1.0, x_0=0.0, y_0=0.0)

        # Each star pixel covers the oversampled grid points k with
        # x_over - os / 2 < k <= x_over + os / 2 along each axis, where
        # x_over is the pixel center in the oversampled ePSF grid.
        # Compute the first covered grid point along each axis.
        ny_over, nx_over = self.oversampling
        x_over, y_over = self._coord_transformer.undersampled_to_oversampled(
            xidx_centered, yidx_centered)
        x_first = np.floor(x_over + epsf.origin[0] - nx_over / 2.0)
        y_first = np.floor(y_over + epsf.origin[1] - ny_over / 2.0)
        x_first = x_first.astype(int) + 1
        y_first = y_first.astype(int) + 1

        # Deposit each pixel residual on every grid point inside the
        # pixel footprint through a flat index into the stack, which
        # needs only two masked copies per footprint offset. The
        # in-bounds masks along the x axis are the same for every row
        # offset, so they are computed once.
        epsf_resid_flat = epsf_resid.reshape(-1)
        star_offset = star_index * (ny * nx)
        x_masks = [((x_first + i) >= 0) & ((x_first + i) < nx)
                   for i in range(nx_over)]
        for j in range(ny_over):
            yidx = y_first + j
            y_mask = (yidx >= 0) & (yidx < ny)
            row_index = star_offset + yidx * nx
            for i in range(nx_over):
                mask = y_mask & x_masks[i]
                flat_index = row_index + (x_first + i)
                epsf_resid_flat[flat_index[mask]] = residuals[mask]

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
        return _SmoothingKernel.apply_smoothing(
            epsf_data, self._current_smoothing_kernel())

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

        if not np.isfinite(current_sum) or current_sum <= 0:
            msg = (f'Cannot normalize ePSF: the data sum ({current_sum}) '
                   'is not finite and positive. This usually indicates '
                   'a problem in the recentering or smoothing steps, '
                   'e.g., a recentering function that returned '
                   'non-finite centroids.')
            raise ValueError(msg)

        return epsf_data * (oversampling_product / current_sum)

    def _recenter_epsf(self, epsf, *, centroid_func=None, box_size=None,
                       maxiters=None, center_accuracy=None):
        """
        Recenter the ePSF data by shifting to the array center.

        This method uses iterative centroiding to find the center of
        the ePSF and applies sub-pixel shifts using spline
        interpolation via the ImagePSF ``evaluate`` method.

        Parameters
        ----------
        epsf : `ImagePSF` object
            The ePSF model containing the data to be recentered.

        centroid_func : callable, optional
            A callable object used to calculate the centroid of a 2D
            array. The callable must accept a 2D `~numpy.ndarray`,
            have a ``mask`` keyword, and optionally an ``error``
            keyword. The callable object must return a tuple of (x, y)
            scalar centroids. If `None`, uses the builder's configured
            recentering_func.

        box_size : int or tuple of two ints, optional
            The size (in pixels) of the box used to calculate the
            centroid of the ePSF during each iteration, in the input
            star (i.e., undersampled) pixel space. It is automatically
            scaled by the oversampling factor when applied to the
            oversampled ePSF grid. If `None`, uses the builder's
            configured recentering_boxsize.

        maxiters : int, optional
            The maximum number of recentering iterations to perform. If
            `None`, uses the builder's configured recentering_maxiters.

        center_accuracy : float, optional
            The desired accuracy for the center position. The centering
            iterations will stop if the center of the ePSF changes by
            less than ``center_accuracy`` pixels between iterations. If
            `None`, uses 1.0e-4.

        Returns
        -------
        result : 2D `~numpy.ndarray`
            The recentered ePSF data array with the same shape as input.
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
        xcenter, ycenter = self._coord_transformer.compute_epsf_origin(
            epsf.data.shape)

        # Create coordinate grids in undersampled units for evaluate()
        y, x = np.indices(epsf.data.shape, dtype=float)
        x, y = self._coord_transformer.oversampled_to_undersampled(x, y)

        # The origin in undersampled units (for use with evaluate)
        x_origin, y_origin = (
            self._coord_transformer.oversampled_to_undersampled(
                xcenter, ycenter))

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
                self._coord_transformer.oversampled_to_undersampled(dx, dy))
            dx_total += dx_under
            dy_total += dy_under

            # Apply the shift using evaluate (uses spline
            # interpolation). The shift is applied by moving the origin.
            epsf_data = epsf.evaluate(x=x, y=y, flux=1.0,
                                      x_0=x_origin - dx_total,
                                      y_0=y_origin - dy_total)

        return epsf_data

    def _build_epsf_step(self, stars, *, epsf=None):
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
            if self._sigma_clip is not None:
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

        # Choose the automatic smoothing kernel and fit shape from the
        # FWHM of the current ePSF.
        self._update_auto_parameters(new_epsf)

        # Smooth the ePSF
        smoothed_data = self._smooth_epsf(new_epsf)

        # Remove power at and above the input pixel sampling frequency
        # along oversampled axes, where the star-pixel lattice aliases
        # onto the ePSF grid.
        smoothed_data = _suppress_alias_modes(smoothed_data, self.oversampling)

        # Recenter the ePSF using an intermediate ePSF that keeps the
        # current epsf's origin. The recentering shifts the ePSF by
        # evaluating its spline on the shifted grid, so the edge row and
        # column on one side fall outside the original grid. They are
        # extrapolated (fill_value=None) rather than set to zero, which
        # would leave an all-zero row and column at that edge.
        temp_epsf = ImagePSF(data=smoothed_data,
                             origin=epsf.origin,
                             oversampling=self.oversampling,
                             fill_value=None)

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

        Convergence is determined by the movement of the star centers
        between iterations. The build has converged when at least
        ``converged_fraction`` of the successfully fitted stars moved by
        less than the configured center accuracy.

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

        converged_fraction : float
            The fraction of the successfully fitted stars whose centers
            moved by less than the center accuracy.

        max_center_dist_sq : float
            The maximum squared center movement of the successfully
            fitted stars.

        new_centers : `~numpy.ndarray`
            Updated star center positions.
        """
        # Calculate center movements for successfully fitted stars only
        new_centers = stars.cutout_center_flat
        dx_dy = new_centers - centers

        # Filter out failed fits for convergence calculation
        good_stars = np.logical_not(fit_failed)

        if not np.any(good_stars):
            # No good stars, so convergence cannot be determined. This
            # is unreachable from build_epsf (all-failed fits raise
            # earlier), but guards direct calls. NaN indicates that no
            # center movement could be measured.
            return False, 0.0, np.nan, new_centers

        dx_dy_good = dx_dy[good_stars]
        center_dist_sq = np.sum(dx_dy_good * dx_dy_good, axis=1,
                                dtype=np.float64)

        converged_fraction = float(
            np.mean(center_dist_sq < self.center_accuracy_sq))
        converged = converged_fraction >= self.converged_fraction

        return (converged, converged_fraction, float(np.max(center_dist_sq)),
                new_centers)

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

        # Build the cached spline interpolators once so that the model
        # copies made by the fitter for every star share them instead
        # of each rebuilding the spline.
        epsf._precompute_interpolators()

        fitted_stars = []
        for star in stars:
            if isinstance(star, EPSFStar):
                if star._excluded_from_fit:
                    fitted_star = star
                else:
                    fitted_star = self._fit_star(epsf, star)

            elif isinstance(star, LinkedEPSFStar):
                if star.all_excluded:
                    # Pass through unchanged to prevent
                    # constrain_centers from warning on every iteration
                    # for a fully excluded linked star.
                    fitted_star = star
                else:
                    fitted_star = []
                    for linked_star in star:
                        if linked_star._excluded_from_fit:
                            fitted_star.append(linked_star)
                        else:
                            fitted_star.append(self._fit_star(epsf,
                                                              linked_star))

                    fitted_star = LinkedEPSFStar(fitted_star)
                    fitted_star.constrain_centers()
                    if self.constrain_fluxes:
                        fitted_star.constrain_fluxes()

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
        fit_shape = self._current_fit_shape()
        fitter = self.fitter
        fitter_kwargs = self._fitter_kwargs
        fitter_has_fit_info = self._fitter_has_fit_info

        # Create a shallow copy to avoid mutating the input star. This
        # is a shallow copy, so the large numpy arrays (_data, weights,
        # mask) are shared and not duplicated. Only the object wrapper
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
            mask = star.mask[large_slc]

            # Define the origin of the fitting region
            x0 = large_slc[1].start
            y0 = large_slc[0].start
        else:
            # Use the entire cutout image
            data = star.data
            weights = star.weights
            mask = star.mask

            # Define the origin of the fitting region
            x0 = 0
            y0 = 0

        # Define positions in the undersampled grid. The fitter will
        # evaluate on the defined interpolation grid, currently in the
        # range [0, len(undersampled grid)].
        yy, xx = np.indices(data.shape, dtype=float)
        xx = xx + x0 - star.cutout_center[0]
        yy = yy + y0 - star.cutout_center[1]

        # Fit only the unmasked pixels. Masked pixels (non-finite data
        # or non-positive weights) are dropped from the fit because the
        # fitter objective function raises on non-finite data values
        # even where the weight is zero.
        good = ~mask
        if not np.any(good):
            star._fit_error_status = 4  # fitting region is fully masked
            return star
        xx = xx[good]
        yy = yy[good]
        data = data[good]
        weights = weights[good]

        # Define the initial guesses for fitted flux and shifts
        epsf.flux = star.flux
        epsf.x_0 = 0.0
        epsf.y_0 = 0.0

        if self._fitter_accepts_weights:
            fitted_epsf = fitter(model=epsf, x=xx, y=yy, z=data,
                                 weights=weights, **fitter_kwargs)
        else:
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
                    elif star._fit_error_status == 4:
                        reason = ('its fitting region contains no '
                                  'unmasked pixels')
                    else:  # _fit_error_status == 2
                        reason = 'the fit did not converge'

                    label = ''
                    if star.id_label is not None:
                        label = f' (id={star.id_label})'
                    msg = (f'The star at '
                           f'({star._center_original[0]:.2f}, '
                           f'{star._center_original[1]:.2f}) '
                           f'(index={i}){label} has been excluded '
                           f'from ePSF fitting because {reason}.')
                    warnings.warn(msg, AstropyUserWarning)
                star._excluded_from_fit = True

        return epsf, stars, fit_failed

    @staticmethod
    def _warn_nonuniform_phases(stars, *, min_stars=32, pvalue=1.0e-6):
        """
        Warn if the subpixel phases of the fitted star centers are
        strongly non-uniform.

        Unbiased star centers have uniformly distributed subpixel
        phases. A strongly non-uniform distribution indicates that
        the fitted centers are biased toward particular phases, which
        happens when the ePSF grid is noisy (e.g., for heterogeneous,
        contaminated, or low signal-to-noise stars).

        Parameters
        ----------
        stars : `EPSFStars` object
            The fitted stars.

        min_stars : int, optional
            The minimum number of successfully fitted stars needed to
            perform the check.

        pvalue : float, optional
            The chi-square p-value below which the warning is emitted.
        """
        centers = [star.cutout_center for star in stars.all_stars
                   if not star._excluded_from_fit
                   and star._fit_error_status == 0]
        if len(centers) < min_stars:
            return

        if _phase_uniformity_pvalue(np.array(centers)) < pvalue:
            msg = ('The subpixel phases of the fitted star centers are '
                   'strongly non-uniform, which indicates that the star '
                   'centers are biased. The ePSF and the fitted star '
                   'centers may be unreliable. This can be caused by '
                   'stars with different PSFs, contaminated or saturated '
                   'star cutouts, spurious detections, or low '
                   'signal-to-noise stars. Consider using a cleaner '
                   'star sample or a lower oversampling factor.')
            warnings.warn(msg, AstropyUserWarning)

    def _finalize_build(self, epsf, stars, progress_reporter, iter_num,
                        converged, final_center_accuracy,
                        final_converged_fraction=None):
        """
        Finalize the ePSF building process and create result object.

        The excluded star indices are derived from the per-star
        exclusion flags, which are the single source of truth.

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

        final_converged_fraction : float, optional
            The fraction of the successfully fitted stars whose centers
            changed by less than the center accuracy in the final
            iteration.

        Returns
        -------
        result : `EPSFBuildResults`
            Structured result containing ePSF, stars, and build
            diagnostics.
        """
        # Handle progress reporting completion
        if iter_num < self.maxiters:
            progress_reporter.write_convergence_message(iter_num)
        progress_reporter.close()

        excluded_star_indices = [i for i, star
                                 in enumerate(stars.all_stars)
                                 if star._excluded_from_fit]

        # Warn if the fitted star centers have strongly non-uniform
        # subpixel phases.
        self._warn_nonuniform_phases(stars)

        # Create structured result. The kernel is copied so that the
        # results own it and edits cannot reach the shared predefined
        # kernels or the input array.
        kernel = self._current_smoothing_kernel()
        if kernel is not None:
            kernel = _SmoothingKernel.get_kernel(kernel).copy()

        fit_shape = self._current_fit_shape()
        if fit_shape is not None:
            fit_shape = tuple(int(size) for size in fit_shape)

        return EPSFBuildResults(
            epsf=epsf,
            fitted_stars=stars,
            iterations=iter_num,
            converged=converged,
            final_center_accuracy=final_center_accuracy,
            n_excluded_stars=len(excluded_star_indices),
            excluded_star_indices=excluded_star_indices,
            smoothing_kernel=kernel,
            fit_shape=fit_shape,
            final_converged_fraction=final_converged_fraction,
        )

    def build_epsf(self, stars, *, epsf=None):
        """
        Build iteratively an ePSF from star cutouts.

        This method builds an ePSF from an initial model when ``epsf``
        is provided, or from scratch when ``epsf`` is `None`. In the
        latter case, it is equivalent to invoking an `EPSFBuilder`
        instance on the input ``stars``.

        Parameters
        ----------
        stars : `EPSFStars` object
            The stars used to build the ePSF.

        epsf : `ImagePSF` object, optional
            The initial ePSF model. If `None`, then the ePSF will be
            built from scratch.

        Returns
        -------
        result : `EPSFBuildResults`
            The ePSF building results, with detailed information about
            the building process. For backward compatibility, the
            result can be unpacked as a tuple: ``(epsf, fitted_stars) =
            epsf_builder(stars)``.

        Notes
        -----
        The structured result object contains:

        - epsf: The final constructed ePSF
        - fitted_stars: Stars with updated centers/fluxes
        - iterations: Number of iterations performed
        - converged: Whether convergence was achieved
        - final_center_accuracy: Final center movement accuracy
        - final_converged_fraction: Fraction of stars that met the
          accuracy
        - n_excluded_stars: Number of stars excluded due to fit failures
        - excluded_star_indices: Indices of excluded stars
        - smoothing_kernel: Smoothing kernel of the final iteration
        - fit_shape: Fitting box of the final iteration
        """
        if epsf is not None:
            if not np.array_equal(epsf.oversampling, self.oversampling):
                msg = (f'The input epsf oversampling '
                       f'{tuple(epsf.oversampling)} does not match the '
                       f'builder oversampling '
                       f'{tuple(self.oversampling)}')
                raise ValueError(msg)

            # An undersized initial ePSF would silently truncate the
            # result, so validate its shape like a requested shape
            _EPSFValidator.validate_shape_compatibility(
                stars, self.oversampling, shape=epsf.data.shape)

        _EPSFValidator.validate_stars(stars, context='ePSF building')
        _EPSFValidator.validate_shape_compatibility(stars, self.oversampling,
                                                    shape=self.shape)

        # Initialize variables for building process
        fit_failed = np.zeros(stars.n_all_stars, dtype=bool)
        centers = stars.cutout_center_flat

        # Reset the per-call automatic kernel and fit shape state. The
        # automatic fit shape cannot exceed the smallest star cutout.
        self._auto_state = None
        self._auto_fallback_warned = False
        min_cutout = min(min(star.shape) for star in stars.all_stars)
        self._auto_fit_max = _odd_size(min_cutout)
        if self._auto_fit_max > min_cutout:
            self._auto_fit_max -= 2

        # Setup progress tracking
        progress_reporter = _ProgressReporter(self.progress_bar,
                                              self.maxiters).setup()

        # Initialize iteration variables
        iter_num = 0
        converged = False
        converged_fraction = 0.0
        max_center_dist_sq = self.center_accuracy_sq + 1.0

        # Main iteration loop. Note that an all-failed iteration
        # raises inside _process_iteration, so no fit_failed exit
        # condition is needed here.
        while iter_num < self.maxiters and not converged:

            iter_num += 1

            # Process one iteration
            epsf, stars, fit_failed = self._process_iteration(
                stars, epsf, iter_num)

            # Check convergence based on center movements
            (converged, converged_fraction, max_center_dist_sq,
             centers) = self._check_convergence(stars, centers, fit_failed)

            # Update progress bar
            progress_reporter.update()

        final_center_accuracy = float(max_center_dist_sq ** 0.5)

        # Finalize and return structured results
        return self._finalize_build(epsf, stars, progress_reporter,
                                    iter_num, converged,
                                    final_center_accuracy,
                                    converged_fraction)


def __getattr__(name):
    # EPSFBuildResult was renamed to EPSFBuildResults in 3.1.
    if name == 'EPSFBuildResult':
        msg = ('EPSFBuildResult is deprecated and will be removed in a '
               'future version. Use EPSFBuildResults instead.')
        warnings.warn(msg, PhotutilsDeprecationWarning, stacklevel=2)
        return EPSFBuildResults

    msg = f'module {__name__!r} has no attribute {name!r}'
    raise AttributeError(msg)
