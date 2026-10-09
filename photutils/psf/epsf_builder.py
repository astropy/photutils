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
from typing import NamedTuple

import numpy as np
from astropy.modeling.fitting import TRFLSQFitter
from astropy.nddata import NoOverlapError, PartialOverlapError, overlap_slices
from astropy.stats import SigmaClip
from astropy.table import Table
from astropy.utils.decorators import deprecated, deprecated_attribute
from astropy.utils.exceptions import AstropyUserWarning
from scipy.ndimage import convolve
from scipy.signal import fftconvolve
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

# Smoothing of the wings of the final ePSF, in units of the ePSF FWHM.
# Beyond _WING_START the ePSF is blended, over _WING_BLEND, into a
# least-squares polynomial fit of degree _WING_DEGREE in a box of width
# _WING_SMALL_BOX. Beyond _WING_LARGE_START it is blended into the fit
# in a box of width _WING_LARGE_BOX. This follows the approach of
# Anderson 2016 (WFC3 ISR 2016-12), which smooths the wings of HST ePSFs
# more strongly than their cores. A quadratic fit is used here because
# the mean of a concave profile in a box is biased high, which changes
# the encircled energy.
_WING_START = 3.5
_WING_LARGE_START = 5.0
_WING_BLEND = 1.0
_WING_SMALL_BOX = 1.25
_WING_LARGE_BOX = 1.75
_WING_DEGREE = 2
_WING_MIN_SIZE = 5

# Half width (in detector pixels) of the box around each oversampled
# grid point within which star pixels contribute to that grid point.
# The half width is never smaller than one grid spacing.
_DEPOSIT_HALF_WIDTH = 0.375

# Passband of the alias low-pass filter. It ends at the smaller of
# _ALIAS_PASS cycles per input pixel and _ALIAS_PASS_NYQUIST times the
# Nyquist frequency of the oversampled grid.
_ALIAS_PASS = 0.8
_ALIAS_PASS_NYQUIST = 0.7

# Low-pass filter of the refinement iterations (cycles per input pixel),
# used only along axes with at least the minimum oversampling factor.
# The gain falls to zero at the first zero of the transfer function of
# the residual deposit box, which is 1 / (2 * _DEPOSIT_HALF_WIDTH). The
# star residuals do not constrain the ePSF at and above that frequency.
# Each refinement iteration updates the ePSF _REFINE_STEPS times with
# the star centers and fluxes fixed and then refits the stars.
_REFINE_PASS = 1.1
_REFINE_STOP = 1.0 / (2.0 * _DEPOSIT_HALF_WIDTH)
_REFINE_MIN_OVERSAMPLING = 4
_REFINE_STEPS = 5


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


def _alias_filter_band(oversampling, *, nu_pass=None, refine=False):
    """
    Return the passband and stopband of the alias low-pass filter along
    one axis.

    Parameters
    ----------
    oversampling : int
        The oversampling factor along the axis.

    nu_pass : float or `None`, optional
        The end of the passband of the filter of the building iterations
        in cycles per input pixel. If `None`, it is the smaller of 0.8
        cycles per input pixel and 70% of the Nyquist frequency of the
        oversampled grid (0.7 cycles per input pixel for an oversampling
        factor of 2).

    refine : bool, optional
        If `True`, return the band of the filter of the refinement
        iterations, which has unit gain up to 1.1 cycles per input pixel
        and zero gain at and above 1.33 cycles per input pixel. That
        filter does not remove the signal of the ePSF near one cycle per
        input pixel, so it leaves no ripple pattern. It removes only
        the frequencies that the star residuals do not constrain. It is
        used only for an oversampling factor of at least 4. For smaller
        factors those frequencies are too close to the Nyquist frequency
        of the oversampled grid, and the band of the building iterations
        is returned.

    Returns
    -------
    nu_pass, nu_stop : float
        The end of the passband and the start of the stopband in cycles
        per input pixel.
    """
    if refine and oversampling >= _REFINE_MIN_OVERSAMPLING:
        return _REFINE_PASS, _REFINE_STOP
    if nu_pass is None:
        nu_pass = min(_ALIAS_PASS, _ALIAS_PASS_NYQUIST * oversampling / 2)
    return float(nu_pass), 1.0


def _suppress_alias_modes(data, oversampling, *, nu_pass=None, nu_stop=1.0):
    """
    Low-pass filter an oversampled ePSF near and above the input pixel
    sampling frequency.

    The filter is applied independently along each axis with an
    oversampling factor greater than one. It has unit gain up to
    ``nu_pass`` cycles per input (undersampled) pixel, a raised-cosine
    transition, and zero gain at and above ``nu_stop`` cycles per input
    pixel. Frequencies at integer cycles per input pixel are the zeros
    of the pixel response, so a pixel-integrated PSF has essentially no
    power there. However, those are exactly the frequencies at which
    the star-pixel sampling lattice aliases onto the oversampled grid,
    so noise at those frequencies can grow into a checkerboard pattern
    during the ePSF build iterations.

    The default passband is a compromise. The ePSF of an undersampled
    detector has real power just below one cycle per input pixel. For
    example, about 10% of the Fourier amplitude of the HST WFC3/IR F110W
    ePSF lies between 0.7 and 1.0 cycles per pixel. A lower ``nu_pass``
    removes that power, which lowers the peak of the ePSF and leaves
    ripples around its core. On the other hand, a star sampled once
    per pixel constrains those frequencies only weakly, so a higher
    ``nu_pass`` slows the convergence of the build, especially for
    small star samples. The passband is also limited to 70% of the
    Nyquist frequency of the oversampled grid, which matters only for an
    oversampling factor of 2.

    Parameters
    ----------
    data : 2D `~numpy.ndarray`
        The oversampled ePSF data.

    oversampling : tuple of int
        The (y, x) oversampling factors.

    nu_pass : float, tuple of 2 floats, or `None`, optional
        The end of the passband in cycles per input pixel, as a single
        value or as the (y, x) values. If `None`, the passband along
        each axis ends at the smaller of 0.8 cycles per input pixel and
        70% of the Nyquist frequency of the oversampled grid (0.7 cycles
        per input pixel for an oversampling factor of 2).

    nu_stop : float or tuple of 2 floats, optional
        The start of the stopband in cycles per input pixel, as a single
        value or as the (y, x) values.

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

        if nu_pass is None:
            axis_pass = _alias_filter_band(factor)[0]
        else:
            axis_pass = np.broadcast_to(nu_pass, 2)[axis]
        axis_stop = np.broadcast_to(nu_stop, 2)[axis]

        npts = data.shape[axis]
        # Frequency in cycles per input (undersampled) pixel
        nu = np.abs(np.fft.fftfreq(npts)) * factor
        frac = np.clip((nu - axis_pass) / (axis_stop - axis_pass), 0.0, 1.0)
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

    desc : str or `None`, optional
        The description of the progress bar. If `None`, the description
        of the building iterations is used.

    Attributes
    ----------
    enabled : bool
        Whether progress reporting is active.

    maxiters : int
        Maximum iterations for progress bar setup.

    desc : str
        The description of the progress bar.

    _pbar : progress bar or `None`
        The underlying progress bar instance.
    """

    def __init__(self, enabled, maxiters, *, desc=None):
        """
        Initialize a _ProgressReporter.

        Parameters
        ----------
        enabled : bool
            Whether progress reporting is enabled.

        maxiters : int
            The maximum number of iterations.

        desc : str or `None`, optional
            The description of the progress bar.
        """
        self.enabled = enabled
        self.maxiters = maxiters
        if desc is None:
            desc = f'EPSFBuilder ({maxiters} maxiters)'
        self.desc = desc
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

        self._pbar = add_progress_bar(total=self.maxiters,
                                      desc=self.desc)
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
            self._pbar.write('EPSFBuilder building iterations converged '
                             f'after {iteration} iterations (of '
                             f'{self.maxiters} maximum iterations)')

    def close(self):
        """
        Close and finalize the progress bar.

        Should be called when ePSF building is complete, regardless of
        convergence status.
        """
        if self._pbar is not None:
            self._pbar.close()


class _IterationRecord(NamedTuple):
    """
    The ePSF and the convergence statistics of one building or
    refinement iteration.
    """

    epsf_data: np.ndarray
    stage: str
    converged: bool
    converged_fraction: float
    max_center_dist_sq: float
    n_fit_failed: int


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
        The number of building iterations performed. This will be <=
        maxiters specified in EPSFBuilder. The refinement iterations are
        not counted.

    converged : bool
        Whether the building process converged based on the
        center accuracy criterion. `True` if at least the
        ``converged_fraction`` of the successfully fitted stars
        moved by less than the specified center accuracy between the
        final iterations. If the ePSF was refined, ``converged``,
        ``final_center_accuracy``, and ``final_converged_fraction``
        are measured in the last refinement iteration, so that they
        describe the returned ``fitted_stars``. A build whose building
        iterations converged can therefore report `False`. The first
        refit of the refinement moves some stars by more than the center
        accuracy, because the ePSF changes when the refinement restores
        its signal near one cycle per pixel. This mostly happens with
        a single refinement iteration (``refinement_iters=1``). The
        star centers settle again within a few refinement iterations.
        With different oversampling factors along the two axes they can
        need more refinement iterations than the default (about 10 in
        tests). The ``converged`` column of ``iteration_info`` gives the
        convergence of every building and refinement iteration.

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

    iteration_epsfs : list of 2D `~numpy.ndarray`
        The ePSF image after each iteration, in order. The building
        iterations come first, followed by the refinement iterations
        (if any). Each image has the shape and normalization of
        ``epsf.data``. The images can be used to check how the
        ePSF evolved and whether it stopped changing. See also
        `~photutils.psf.EPSFBuildResults.plot_iterations`. The smoothing
        of the wings of the final ePSF (see the ``wing_smoothing``
        keyword of `EPSFBuilder`) is not an iteration, so the last
        image is the ePSF before its wings were smoothed, and it
        differs from ``epsf.data`` in the wings if that smoothing
        changed the ePSF.

    iteration_info : `~astropy.table.Table`
        A table with one row for each image in ``iteration_epsfs``. The
        columns are the iteration number (``iteration``, starting at 1),
        the kind of iteration (``stage``, ``'build'`` or ``'refine'``),
        whether the star centers had converged in that iteration
        (``converged``), the fraction of the successfully fitted stars
        whose centers moved by less than the center accuracy in that
        iteration (``converged_fraction``), the largest center movement
        in pixels (``max_center_shift``), the number of stars whose
        fit failed (``n_fit_failed``), and the largest absolute change
        of the ePSF image from the previous iteration as a fraction
        of the ePSF peak (``max_epsf_change``). The change of the
        first iteration is measured from ``initial_epsf``, or from
        an empty ePSF if the ePSF was built from scratch. The last
        row gives the ``converged``, ``final_converged_fraction``,
        and ``final_center_accuracy`` values of the results. The last
        ``'build'`` row tells whether the building iterations converged,
        which ``converged`` alone does not when the ePSF was refined.

    initial_epsf : 2D `~numpy.ndarray` or `None`
        The image of the input ePSF that the build started from, or
        `None` if the ePSF was built from scratch.

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
    iteration_epsfs: list | None = field(default=None, compare=False,
                                         repr=False)
    iteration_info: Table | None = field(default=None, compare=False,
                                         repr=False)
    initial_epsf: np.ndarray | None = field(default=None, compare=False,
                                            repr=False)

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

    def plot_iterations(self, *, iterations=None, figsize=None,
                        cmap='viridis', diff_cmap='RdBu_r'):
        """
        Plot the ePSF after each iteration and its change from the
        previous iteration.

        The figure has one row for each iteration. The left panel shows
        the ePSF with a logarithmic stretch that is common to all
        the rows. The right panel shows the difference from the ePSF
        of the previous iteration, as a fraction of the peak of the
        final ePSF, with a symmetric linear stretch of its own. For
        the first iteration the right panel shows the difference from
        ``initial_epsf``, or the ePSF itself if the ePSF was built from
        scratch.

        If the wings of the final ePSF were smoothed (see the
        ``wing_smoothing`` keyword of `EPSFBuilder`), the figure has
        one more row at the bottom. It shows the returned ePSF and its
        difference from the ePSF of the last iteration, which is the
        change made by the wing smoothing. This row is always plotted,
        whatever the value of ``iterations``.

        Parameters
        ----------
        iterations : int, 1D array_like of int, or `None`, optional
            The iteration numbers (starting at 1, as in the
            ``iteration`` column of ``iteration_info``) to plot. If
            `None`, all the iterations are plotted. The figure has one
            row per iteration, so select a few iterations to get a
            compact figure.

        figsize : tuple of 2 float or `None`, optional
            The figure (width, height) in inches. If `None`, the figure
            is 7 inches wide and 2.6 inches tall per row.

        cmap : str or `matplotlib.colors.Colormap`, optional
            The colormap of the ePSF panels.

        diff_cmap : str or `matplotlib.colors.Colormap`, optional
            The colormap of the difference panels.

        Returns
        -------
        fig : `matplotlib.figure.Figure`
            The figure.

        Notes
        -----
        This method returns a figure object. If you are using this
        method in a script, you will need to call ``fig.show()`` to
        display the figure. If you are using this method in a Jupyter
        notebook, the figure will be displayed automatically.

        When in a notebook, if you do not store the return value of this
        function, the figure will be displayed twice due to the REPL
        automatically displaying the return value of the last function
        call. Alternatively, you can append a semicolon to the end of
        the function call to suppress the display of the return value.
        """
        import matplotlib.pyplot as plt
        from astropy.visualization import simple_norm

        if not self.iteration_epsfs or self.iteration_info is None:
            msg = 'There is no per-iteration history to plot'
            raise ValueError(msg)

        n_total = len(self.iteration_epsfs)
        if iterations is None:
            iterations = np.arange(1, n_total + 1)
        iterations = np.atleast_1d(iterations)
        if (iterations.ndim != 1 or iterations.size == 0
                or not np.issubdtype(iterations.dtype, np.integer)):
            msg = ('iterations must be an integer or a non-empty 1D array '
                   'of integers')
            raise ValueError(msg)
        if np.any(iterations < 1) or np.any(iterations > n_total):
            msg = f'iterations must be between 1 and {n_total}'
            raise ValueError(msg)

        final = self.iteration_epsfs[-1]
        peak = np.max(final)
        norm = simple_norm(final, 'log', percent=99.0)

        # The wing smoothing is not an iteration, so the smoothed ePSF
        # is plotted in a row of its own (None) after the iterations
        rows = list(iterations)
        if not np.array_equal(self.epsf.data, final):
            rows.append(None)

        n_rows = len(rows)
        if figsize is None:
            figsize = (7.0, 2.6 * n_rows)
        fig, axes = plt.subplots(n_rows, 2, figsize=figsize, squeeze=False)
        for row, iteration in enumerate(rows):
            if iteration is None:
                data = self.epsf.data
                previous = final
                title = 'Final ePSF (wings smoothed)'
                subtitle = ''
            else:
                data = self.iteration_epsfs[iteration - 1]
                if iteration > 1:
                    previous = self.iteration_epsfs[iteration - 2]
                elif self.initial_epsf is not None:
                    previous = self.initial_epsf
                else:
                    previous = 0.0
                info = self.iteration_info[iteration - 1]
                title = f'Iteration {iteration} ({info["stage"]})'
                subtitle = ('\nconverged fraction '
                            f'{info["converged_fraction"]:.2f}')
            diff = (data - previous) / peak
            limit = np.max(np.abs(diff))
            if limit == 0:
                limit = 1.0

            ax = axes[row, 0]
            axim = ax.imshow(data, norm=norm, origin='lower', cmap=cmap)
            fig.colorbar(axim, ax=ax)
            ax.set_title(title)

            ax = axes[row, 1]
            axim = ax.imshow(diff, origin='lower', cmap=diff_cmap,
                             vmin=-limit, vmax=limit)
            fig.colorbar(axim, ax=ax)
            ax.set_title(f'change / peak (max {limit:.2g}){subtitle}',
                         fontsize=9)
        fig.tight_layout()
        return fig


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

    The method is based on `Anderson and King 2000 (PASP 112, 1360)
    <https://ui.adsabs.harvard.edu/abs/2000PASP..112.1360A/abstract>`_
    and `Anderson 2016 (WFC3 ISR 2016-12)
    <https://ui.adsabs.harvard.edu/abs/2016wfc..rept...12A/abstract>`_.
    It differs from them in several steps (see Notes).

    Parameters
    ----------
    oversampling : int or array_like (int), optional
        The integer oversampling factor(s) of the output ePSF relative
        to the input ``stars`` along each axis. If ``oversampling`` is a
        scalar then it will be used for both axes. If ``oversampling``
        has two elements, they must be in ``(y, x)`` order. The ePSF
        should have at least about four grid points per FWHM, so the
        value should be at least ``4 / FWHM`` with the FWHM in pixels.
        The default of 4 is a good choice for a FWHM of about 1 pixel
        or more. A value above ``4 / FWHM`` increases the run time
        and the memory use, and in tests a value of 8 was no more
        accurate than 4. For well-sampled data (a FWHM of 4 pixels or
        more), a value of 1 gives the same fitted star positions and
        fluxes in less time. See the guidelines in the ePSF building
        user guide for details.

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
        (see Notes) whose width is the largest odd number of oversampled
        grid points that is not larger than 0.7 times the FWHM of the
        current ePSF is used, and no smoothing is applied if that
        width is less than 5 grid points, i.e., for ePSFs with fewer
        than about 7 grid points per FWHM. The kernel is square and
        the FWHM is measured in each iteration along the narrowest
        axis of the ePSF, so with anisotropic oversampling the axis
        with the fewer grid points per FWHM sets the kernel size. If
        the FWHM cannot be measured, the ``'quartic'`` kernel is used
        and a warning is emitted. The predefined ``'quartic'`` and
        ``'quadratic'`` kernels are 5x5 kernels (in oversampled grid
        points) derived from fourth and second degree polynomials,
        respectively. Alternatively, a custom 2D array can be input.
        If `None` then no smoothing will be performed. The kernels are
        applied on the oversampled grid, so the physical width of a
        fixed kernel depends on the oversampling factor. Structure that
        repeats with a period of about one input pixel or shorter is
        removed from the ePSF along oversampled axes independently of
        this parameter (see ``alias_passband``).

    alias_passband : {'auto'}, float, or `None`, optional
        The end of the passband, in cycles per detector pixel, of the
        low-pass filter that is applied to the ePSF in each iteration
        along the axes with an oversampling factor greater than
        one. A spatial frequency of one cycle per pixel describes
        structure that repeats with a period of one detector pixel,
        and a frequency of 0.5 cycles per pixel describes structure
        that repeats every two pixels. The filter has unit gain up to
        ``alias_passband``, a raised-cosine transition, and zero gain
        at and above one cycle per pixel. It removes the frequencies
        at which the star-pixel sampling lattice aliases onto the
        oversampled grid, which otherwise can grow into a checkerboard
        pattern (see Notes). The value must be greater than 0 and less
        than 1.

        If ``'auto'`` (default), the passband ends at 0.8 cycles per
        pixel, or at 0.7 cycles per pixel for an oversampling factor of
        2 (70% of the Nyquist frequency of the oversampled grid).

        The default is the best choice for most data. A different value
        can help in two cases, which are distinguished by the optical
        cutoff frequency of the telescope in cycles per pixel, ``cutoff
        = D * pixel_scale / wavelength``, with the telescope diameter
        ``D`` and the mean ``wavelength`` of the bandpass in the same
        units and the ``pixel_scale`` in radians per pixel:

        * ``cutoff`` greater than about 1 (strongly undersampled,
          e.g., HST WFC3/IR F110W, JWST NIRCam F070W, or Roman WFI F062
          and F106): the ePSF has real signal up to nearly one cycle per
          pixel. In tests the default recovered the peak of such ePSFs
          to within about 1 percent, except for the sharpest ones, whose
          peak was 3 percent low. A value of 0.9 recovered that peak to
          within 0.2 percent, but it increased the noise in the core of
          the other ePSFs by 10 to 65 percent. Try 0.9 if the default
          ePSF is too broad, i.e., if the stars have positive residuals
          at their centers after the fitted ePSF is subtracted. It needs
          a large star sample (a few hundred stars) and more iterations
          (``maxiters`` of 20 or more).

        * ``cutoff`` less than about 0.9 (e.g., JWST NIRCam F115W and
          redder, JWST MIRI, or most ground-based data): the ePSF has
          no signal to preserve near one cycle per pixel. A value of
          0.7 rejects more noise (10 to 40 percent lower residuals in
          the core in tests) and converges in fewer iterations. It
          leaves the peak of a strongly undersampled ePSF low by 2 to 6
          percent, so use it only when ``cutoff`` is known.

        A value larger than needed makes the build converge more slowly,
        because a star sampled once per pixel constrains the frequencies
        near one cycle per pixel only weakly, and it is not recommended
        for small star samples or for an oversampling factor of 2.

        If `None`, the filter is not applied. This is rarely
        appropriate. Without the filter, noise at the alias frequencies
        accumulates over the iterations, the build can stall before it
        converges, and heterogeneous or contaminated star samples can
        grow a checkerboard pattern. In tests with simulated HST, JWST,
        and Roman star fields, the unfiltered ePSF was less accurate
        than the filtered one in nearly every case, even for large,
        clean, and homogeneous star samples. The option is provided for
        experimentation, e.g., to check how much the filter changes
        a particular ePSF. Always compare the result with a filtered
        build. The filter is never applied along an axis with an
        oversampling factor of 1.

    sigma_clip : `astropy.stats.SigmaClip` instance, optional
        A `~astropy.stats.SigmaClip` object that defines the sigma
        clipping parameters used to determine which pixels are ignored
        when stacking the ePSF residuals in each iteration step. If
        `None` then no sigma clipping will be performed.

    recentering_func : callable, optional
        A callable object that is used to calculate the centroid of a
        2D array. The callable must accept a 2D `~numpy.ndarray`, have
        a ``mask`` keyword and optionally an ``error`` keyword. The
        callable object must return a tuple of (x, y) centroids. The
        default is `~photutils.centroids.centroid_com`, the center of
        mass. The center of mass of an asymmetric ePSF is pulled toward
        the asymmetric structure around its core. To center the ePSF on
        its core instead, use `~photutils.centroids.centroid_symmetry`,
        which is the center definition of Anderson 2016. With the
        default ``recentering_boxsize`` it measures the symmetry within
        1.5 pixels of the center.

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

    refinement_iters : int, optional
        The number of refinement iterations to perform after the
        building iterations. The alias low-pass filter of the building
        iterations (see ``alias_passband``) removes some real signal of
        an undersampled ePSF just below one cycle per pixel. The filter
        acts separately along each axis, so the missing signal shows as
        a ripple pattern with a period of about one pixel along the row
        and the column through the center of the ePSF. The refinement
        iterations restore that signal. In each refinement iteration,
        the ePSF is updated five times from the star residuals with
        the star centers and fluxes held fixed, and the stars are then
        refit with the updated ePSF. The ePSF is recentered in each
        update, as in the building iterations, which keeps the ePSF and
        the star centers from drifting together. The low-pass filter of
        these updates has unit gain up to 1.1 cycles per pixel and zero
        gain at and above 1.33 cycles per pixel, so it does not remove
        signal near one cycle per pixel. It removes only the frequencies
        that the star residuals do not constrain. Such a wide filter
        cannot be used from the start of the build, because the build
        then converges slowly and is more sensitive to the initial star
        centers. The refinement is not performed if ``refinement_iters``
        is 0, if ``alias_passband`` is `None`, or if the oversampling
        factor is less than 4 along both axes. It roughly doubles the
        run time of a build.

        The ``converged`` attribute of the results describes the
        last refinement iteration. The first refit of the refinement
        moves some stars by more than ``center_accuracy``, so with
        ``refinement_iters=1`` a build whose building iterations
        converged can report ``converged=False``. The star centers
        settle again within a few refinement iterations. With
        different oversampling factors along the two axes they can
        need more refinement iterations than the default (about 10 in
        tests).

    wing_smoothing : bool, optional
        Whether to smooth the wings of the final ePSF more strongly
        than its core. Far from the center the ePSF is faint and
        varies slowly, so its noise can be averaged over a larger area
        than in the core. If `True`, each value of the final ePSF
        beyond 3.5 FWHM from its center is blended into the value of
        a least-squares quadratic fit to the values in a box 1.25
        FWHM wide around it, and beyond 5 FWHM into the fit in a box
        1.75 FWHM wide. The ePSF within 3.5 FWHM of its center is not
        changed, except by the renormalization of the smoothed ePSF
        (less than 0.03 percent in tests). The smoothing is applied
        once, after the last iteration, so it does not affect the star
        fits or the convergence of the build. The fluxes of the returned
        stars were therefore fit before that renormalization. It is
        modeled on the approach of Anderson 2016, which smooths the
        wings of HST ePSFs more strongly than their cores. In tests
        with a few hundred stars it lowered the residuals of the wings
        of undersampled ePSFs by up to about 50 percent. This matters
        when the wings are used, e.g., to subtract bright stars, to
        make model images, or to measure encircled energies.

        The smoothing also removes real structure in the wings that
        is finer than about two FWHM, such as diffraction rings and
        spikes. Applied to noise-free ePSFs, it changed the wings by
        2 to 8 percent of their mean value for JWST and Roman ePSFs
        and by 11 to 14 percent for HST WFC3/IR ePSFs. With a few
        hundred stars the noise that it removes is larger than this
        in every case that was tested. For a large star sample of
        high signal-to-noise (thousands of stars), the noise in the
        wings can be smaller than this change, and the wings are then
        more accurate without the smoothing.

        The boxes are at least 5 oversampled grid points wide. For an
        ePSF with a FWHM of less than 4 grid points they are therefore
        wider than given above, and they remove more of the real
        structure. This matters most for an oversampling factor of 1,
        where the boxes of an undersampled ePSF (a FWHM of about 1.3
        pixels) are nearly 4 FWHM wide. In tests with an oversampling
        factor of 1, the smoothing made the wings of the most
        undersampled HST and JWST ePSFs less accurate, by up to a
        factor of about 2, and those of most other ePSFs slightly
        more accurate. With an oversampling factor of 2 or larger it
        made the wings more accurate or left them unchanged in every
        case. For an oversampling factor of 1, compare the ePSFs
        built with and without the smoothing.

        Set to `False` to keep the wings as built.

    progress_bar : bool, optional
        Whether to print the progress bar during the build
        iterations. The progress bar requires that the `tqdm
        <https://tqdm.github.io/>`_ optional dependency be installed.

    Notes
    -----
    In each build iteration, the residual between each star and the
    current ePSF model is deposited on the oversampled grid points
    within 0.375 detector pixel (and at least one grid spacing) of each
    star pixel center along each axis. Each grid point is therefore
    estimated from the star pixels in a box three quarters of a pixel
    wide around it, and every star contributes to both parities of the
    grid regardless of its subpixel phase. Anderson and King (2000)
    combined the pixels within 0.25 pixel of each grid point of an ePSF
    with an oversampling factor of 4. That narrower box gives a slightly
    sharper ePSF for clean star samples, but it grows a checkerboard
    pattern for heterogeneous or contaminated ones. After the residuals
    are combined and the ePSF is smoothed, power is removed along each
    oversampled axis with a low-pass filter that by default has unit
    gain up to 0.8 cycles per input pixel (0.7 for an oversampling
    factor of 2) and zero gain at and above one cycle per input pixel
    (see ``alias_passband``). A pixel-integrated PSF has essentially no
    power at one cycle per input pixel, but that is the frequency at
    which the star-pixel sampling lattice aliases onto the oversampled
    grid. Without these two measures, noise in the ePSF grid from
    heterogeneous or contaminated stars can bias the fitted star centers
    toward particular subpixel phases and grow into a checkerboard
    pattern in the ePSF. A warning is emitted if the subpixel phases of
    the fitted star centers are strongly non-uniform at the end of the
    build.

    The default ``smoothing_kernel='auto'`` and ``fit_shape='auto'``
    scale the smoothing kernel and the fitting box with the FWHM of the
    ePSF. The 5x5 ``'quartic'`` kernel (in oversampled grid points) of
    Anderson and King 2000 and the 5-pixel fitting box (in detector
    pixels) of Anderson 2016 were designed for HST images with an
    oversampling factor of 4, where they correspond to about 0.7 and
    2.5 FWHM. A fixed kernel oversmooths heavily undersampled ePSFs,
    and a fixed 5-pixel fitting box applied to a well-sampled star
    uses only its flat core, which biases the fitted centers and can
    prevent convergence. The polynomial kernels replace each grid value
    by the value at the center of a least-squares polynomial fit to
    the surrounding grid values. The kernel and fitting box chosen in
    the final iteration are reported in the ``smoothing_kernel`` and
    ``fit_shape`` attributes of the returned `EPSFBuildResults`.

    This class follows the ePSF building procedure of Anderson and
    King 2000 and Anderson 2016, with several modifications. The main
    differences are listed below. See the ePSF building user guide
    (:ref:`epsf-anderson-differences`) for the complete comparison and
    the reasons for each difference.

    * A single ePSF is built with any integer oversampling factor.
      Anderson builds a 3x3 array of ePSFs across each detector with
      an oversampling factor of 4.

    * The star cutouts must be background subtracted by the user.
      Anderson measures the background of each star in an annulus
      around it.

    * The residuals are combined in a box of 0.375 pixel half width
      with a sigma-clipped median (3 sigma by default). Anderson uses a
      half width of 0.25 pixel and a mean with rejection at 2.5 sigma.

    * The smoothing kernel and the wing smoothing scale with the FWHM
      of the ePSF, and the wings are smoothed only once, after the last
      iteration. Anderson uses a 5x5 quartic kernel in the core and
      stronger smoothing beyond fixed radii of 3 to 5 pixels, in every
      iteration.

    * A low-pass Fourier filter is applied in every iteration for
      oversampling factors greater than 1 (see ``alias_passband``).
      Anderson applies none.

    * The ePSF is centered on its center of mass in a 5x5 pixel box
      by default. Anderson requires equal values half a pixel on either
      side of the center (2000) or centers the ePSF on its point of
      maximal symmetry within a radius of 1.5 pixels (2016). The latter
      is available as ``recentering_func=centroid_symmetry``.

    * The ePSF is normalized so that its values sum to the product of
      the oversampling factors over the whole grid. Fitted fluxes are
      therefore the fluxes within the area of the grid. Anderson
      normalizes the ePSF to unit flux in the central 5x5 pixels
      (2000) or within a radius of 5.5 pixels (2016).

    * The stars are fit after every ePSF update until their centers
      converge, and the ePSF is then refined with five updates per
      fit (see ``refinement_iters``). Anderson uses five updates
      per fit throughout and iterates until the fitted positions and
      fluxes show no trend with pixel phase.

    * The stars are fit in a box of twice the FWHM with a nonlinear
      least-squares fitter. Anderson fits the pixels within about 2
      pixels of the center (2000) or the central 5x5 pixels (2016)
      with Poisson weights.

    * Dithered exposures are optional (see
      `~photutils.psf.LinkedEPSFStar`). They are central to Anderson's
      method, which averages the positions and fluxes of each star over
      the exposures after every fit.

    * The ePSF is evaluated with a single bicubic spline over the whole
      grid. Anderson uses a bicubic spline within 4 pixels of the
      center and bilinear interpolation farther out (2016).

    This class stores per-call state on the instance (e.g., the
    automatic smoothing kernel), so a single instance must not be called
    concurrently from multiple threads. Create one instance per thread
    for concurrent use. Sharing a single Astropy fitter instance across
    concurrently-used objects is also unsafe because Astropy fitters
    store ``fit_info`` on themselves.
    """

    coord_transformer = deprecated_attribute(
        'coord_transformer', '3.1',
        warning_type=PhotutilsDeprecationWarning)

    def __init__(self, *, oversampling=4, shape=None,
                 smoothing_kernel='auto', alias_passband='auto',
                 sigma_clip=SIGMA_CLIP,
                 recentering_func=centroid_com, recentering_boxsize=(5, 5),
                 recentering_maxiters=20, center_accuracy=1.0e-3,
                 converged_fraction=0.95, fitter=None, fit_shape='auto',
                 fitter_maxiters=100, constrain_fluxes=True, maxiters=10,
                 refinement_iters=5, wing_smoothing=True, progress_bar=True):

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
        msg = 'recentering_maxiters must be a strictly-positive integer'
        if (isinstance(recentering_maxiters, bool)
                or not isinstance(recentering_maxiters, numbers.Integral)):
            raise TypeError(msg)
        if recentering_maxiters <= 0:
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

        if not (alias_passband is None or self._is_auto(alias_passband)):
            msg = ("alias_passband must be 'auto', a number between "
                   '0 and 1 (exclusive), or None')
            if (isinstance(alias_passband, bool)
                    or not isinstance(alias_passband, (str, numbers.Real))):
                raise TypeError(msg)
            if (isinstance(alias_passband, str)
                    or not 0.0 < alias_passband < 1.0):
                raise ValueError(msg)
            alias_passband = float(alias_passband)
        self.alias_passband = alias_passband

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

        if not isinstance(constrain_fluxes, (bool, np.bool_)):
            msg = 'constrain_fluxes must be a bool'
            raise TypeError(msg)
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

        msg = 'refinement_iters must be a non-negative integer'
        if (isinstance(refinement_iters, bool)
                or not isinstance(refinement_iters, numbers.Integral)):
            raise TypeError(msg)
        if refinement_iters < 0:
            raise ValueError(msg)
        self.refinement_iters = int(refinement_iters)

        if not isinstance(wing_smoothing, (bool, np.bool_)):
            msg = 'wing_smoothing must be a bool'
            raise TypeError(msg)
        self.wing_smoothing = bool(wing_smoothing)

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

        The smoothing window is the largest odd size that is not
        larger than ``_AUTO_KERNEL_FWHM_FRACTION`` times the FWHM
        in oversampled grid points (no smoothing below
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
        # The window is rounded down to an odd size so that the kernel
        # is never wider than the requested fraction of the FWHM
        size = int(_AUTO_KERNEL_FWHM_FRACTION * fwhm_grid)
        if size % 2 == 0:
            size -= 1
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
        depositing each star pixel value on the oversampled grid points
        within 0.375 detector pixel (and at least one grid spacing)
        of the pixel center along each axis. Each grid point is
        therefore estimated from the star pixels in a box three
        quarters of a pixel wide around it. Anderson and King (2000)
        used a box half a pixel wide (pixels within 0.25 pixel of each
        grid point of an ePSF with an oversampling factor of 4). The
        wider box is needed for heterogeneous or contaminated star
        samples, which grow a checkerboard pattern with the narrower
        one. Every star contributes to both parities of the grid along
        each axis, regardless of its subpixel phase. For an
        oversampling factor of one along an axis, this reduces to the
        nearest grid point.

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

    @staticmethod
    def _deposit_half_width(oversampling):
        """
        Return the half width, in oversampled grid spacings, of the box
        around each star pixel center within which the pixel is
        deposited on the ePSF grid points along one axis.

        The half width is ``_DEPOSIT_HALF_WIDTH`` detector pixels, but
        at least one grid spacing so that every star pixel reaches both
        parities of the grid. For an oversampling factor of one it is
        half a grid spacing, i.e., the nearest grid point.

        Parameters
        ----------
        oversampling : int
            The oversampling factor along the axis.

        Returns
        -------
        half_width : float
            The half width in oversampled grid spacings.
        """
        if oversampling < 2:
            return 0.5
        return max(_DEPOSIT_HALF_WIDTH * oversampling, 1.0)

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
        # single indexed assignment per grid offset.
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

        # Each star pixel is deposited on the oversampled grid points
        # k with x_over - w < k <= x_over + w along each axis, where
        # x_over is the pixel center in the oversampled ePSF grid and w
        # is the deposit half width in grid spacings. Compute the first
        # of these grid points and the largest number of them along
        # each axis.
        ny_over, nx_over = self.oversampling
        y_width = self._deposit_half_width(ny_over)
        x_width = self._deposit_half_width(nx_over)
        x_over, y_over = self._coord_transformer.undersampled_to_oversampled(
            xidx_centered, yidx_centered)
        x_over = x_over + epsf.origin[0]
        y_over = y_over + epsf.origin[1]
        x_first = np.floor(x_over - x_width).astype(int) + 1
        y_first = np.floor(y_over - y_width).astype(int) + 1
        ny_deposit = int(np.ceil(2 * y_width))
        nx_deposit = int(np.ceil(2 * x_width))

        # Deposit each pixel residual on these grid points through a
        # flat index into the stack, which needs only two masked copies
        # per grid offset. The masks along the x axis are the same for
        # every row offset, so they are computed once.
        epsf_resid_flat = epsf_resid.reshape(-1)
        star_offset = star_index * (ny * nx)
        x_masks = [((x_first + i) >= 0) & ((x_first + i) < nx)
                   & ((x_first + i) <= x_over + x_width)
                   for i in range(nx_deposit)]
        for j in range(ny_deposit):
            yidx = y_first + j
            y_mask = (yidx >= 0) & (yidx < ny) & (yidx <= y_over + y_width)
            row_index = star_offset + yidx * nx
            for i in range(nx_deposit):
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

    def _smooth_wings(self, epsf_data):
        """
        Smooth the wings of the final ePSF more strongly than its core.

        Far from the center the ePSF is faint and varies slowly, so
        the noise there can be averaged over a larger area than in the
        core. This also removes real structure that is finer than about
        two FWHM. Beyond 3.5 FWHM from the center, each value of the
        ePSF is blended into the value at the center of a least-squares
        quadratic fit to the values in a box 1.25 FWHM wide around it,
        and beyond 5 FWHM into the fit in a box 1.75 FWHM wide. The
        ePSF within 3.5 FWHM of the center is not changed here. The
        caller renormalizes the result, which rescales the whole ePSF
        by the small change of its sum. The FWHM is measured along the
        narrowest axis of the ePSF, and the boxes are square on the
        oversampled grid and at least ``_WING_MIN_SIZE`` grid points
        wide.

        The smoothing is applied once, to the final ePSF. Applying it
        in every iteration would compound its effect and couple the
        wings to the normalization of the ePSF.

        Parameters
        ----------
        epsf_data : 2D `~numpy.ndarray`
            The ePSF image.

        Returns
        -------
        result : 2D `~numpy.ndarray`
            The ePSF image with smoothed wings. ``epsf_data`` is
            returned if it has non-finite values, if the FWHM of the
            ePSF could not be measured, or if no part of the image is
            in the wings.
        """
        # The box fits use an FFT convolution, which would spread a
        # non-finite value over the whole image
        if not np.all(np.isfinite(epsf_data)):
            return epsf_data

        fwhm = _measure_fwhm(epsf_data)
        if fwhm is None:
            return epsf_data

        oversampling = np.asarray(self.oversampling, dtype=float)
        fwhm = float(np.min(np.asarray(fwhm) / oversampling))  # pixels

        # Radius from the center in units of the FWHM
        ny, nx = epsf_data.shape
        yy, xx = np.indices((ny, nx), dtype=float)
        yy = (yy - (ny - 1) / 2.0) / oversampling[0]
        xx = (xx - (nx - 1) / 2.0) / oversampling[1]
        radius = np.hypot(xx, yy) / fwhm

        weight_small = np.clip((radius - _WING_START) / _WING_BLEND,
                               0.0, 1.0)
        if not np.any(weight_small > 0):
            return epsf_data
        weight_large = np.clip((radius - _WING_LARGE_START) / _WING_BLEND,
                               0.0, 1.0)

        def box_fit(width):
            size = max(_odd_size(width * fwhm * float(np.min(oversampling))),
                       _WING_MIN_SIZE)
            kernel = _make_polynomial_kernel(size, degree=_WING_DEGREE)
            # The kernel is symmetric and the image is extended with
            # its edge values, which is the same as a convolution with
            # mode='nearest' but much faster for large boxes.
            padded = np.pad(epsf_data, size // 2, mode='edge')
            return fftconvolve(padded, kernel, mode='valid')

        wings = box_fit(_WING_SMALL_BOX)
        if np.any(weight_large > 0):
            wings = ((1.0 - weight_large) * wings
                     + weight_large * box_fit(_WING_LARGE_BOX))
        return (1.0 - weight_small) * epsf_data + weight_small * wings

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

        This method uses iterative centroiding to find the center of the
        ePSF and applies sub-pixel shifts using spline interpolation via
        the ImagePSF ``evaluate`` method.

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

    def _build_epsf_step(self, stars, *, epsf=None, refine=False):
        """
        A single iteration of improving an ePSF.

        Parameters
        ----------
        stars : `EPSFStars` object
            The stars used to build the ePSF.

        epsf : `ImagePSF` object, optional
            The initial ePSF model. If not input, then the ePSF will be
            built from scratch.

        refine : bool, optional
            Whether this is an update of a refinement iteration. The
            alias low-pass filter then uses the wider refinement filter.

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

        # Remove power near and above the input pixel sampling frequency
        # along oversampled axes, where the star-pixel lattice aliases
        # onto the ePSF grid.
        if self.alias_passband is not None:
            nu_pass = None
            if not self._is_auto(self.alias_passband):
                nu_pass = self.alias_passband
            bands = [_alias_filter_band(factor, nu_pass=nu_pass,
                                        refine=refine)
                     for factor in self.oversampling]
            smoothed_data = _suppress_alias_modes(
                smoothed_data, self.oversampling,
                nu_pass=(bands[0][0], bands[1][0]),
                nu_stop=(bands[0][1], bands[1][1]))

        # Recenter the ePSF using an intermediate ePSF that keeps the
        # current epsf's origin. The recentering shifts the ePSF by
        # evaluating its spline on the shifted grid, so the edge row
        # and column on one side fall outside the original grid. They
        # take the spline value at the nearest point on the grid edge
        # (fill_value=None) rather than being set to zero, which would
        # leave an all-zero row and column at that edge.
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

        # Build the cached interpolators of a custom interpolator once
        # so that the model copies made by the fitter for every star
        # share them instead of each rebuilding them.
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

    def _process_iteration(self, stars, epsf, iter_num, *, refine=False):
        """
        Process a single building or refinement iteration.

        Parameters
        ----------
        stars : `EPSFStars` object
            The stars used to build the ePSF.

        epsf : `ImagePSF` object
            Current ePSF model.

        iter_num : int
            Current iteration number, counting the building and the
            refinement iterations.

        refine : bool, optional
            Whether this is a refinement iteration, in which the ePSF is
            updated ``_REFINE_STEPS`` times with the refinement filter
            and the star centers and fluxes held fixed before the stars
            are refit.

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
        n_steps = _REFINE_STEPS if refine else 1
        for _ in range(n_steps):
            epsf = self._build_epsf_step(stars, epsf=epsf, refine=refine)

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

    def _finalize_build(self, epsf, stars, iter_num, converged,
                        final_center_accuracy,
                        final_converged_fraction=None, *, history=None,
                        initial_epsf=None):
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

        history : list of `_IterationRecord` or `None`, optional
            The record of each building and refinement iteration.

        initial_epsf : 2D `~numpy.ndarray` or `None`, optional
            The image of the input ePSF that the build started from.

        Returns
        -------
        result : `EPSFBuildResults`
            Structured result containing ePSF, stars, and build
            diagnostics.
        """
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

        iteration_epsfs = None
        iteration_info = None
        if history is not None:
            iteration_epsfs = [record.epsf_data for record in history]
            iteration_info = self._make_iteration_info(
                history, initial_epsf=initial_epsf)

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
            iteration_epsfs=iteration_epsfs,
            iteration_info=iteration_info,
            initial_epsf=initial_epsf,
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
        - iteration_epsfs: ePSF image after each iteration
        - iteration_info: Table of the convergence statistics of each
          iteration
        - initial_epsf: Image of the input ePSF, if any
        """
        initial_epsf = None
        if epsf is not None:
            initial_epsf = epsf.data.copy()
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
        history = []
        while iter_num < self.maxiters and not converged:

            iter_num += 1

            # Process one iteration
            epsf, stars, fit_failed = self._process_iteration(
                stars, epsf, iter_num)

            # Check convergence based on center movements
            (converged, converged_fraction, max_center_dist_sq,
             centers) = self._check_convergence(stars, centers, fit_failed)
            history.append(_IterationRecord(
                epsf.data.copy(), 'build', converged, converged_fraction,
                max_center_dist_sq, int(np.sum(fit_failed))))

            # Update progress bar
            progress_reporter.update()

        final_center_accuracy = float(max_center_dist_sq ** 0.5)

        # Finish the progress reporting of the building iterations
        # before the refinement, which has its own progress bar.
        if iter_num < self.maxiters:
            progress_reporter.write_convergence_message(iter_num)
        progress_reporter.close()

        # Refine the ePSF. Each refinement iteration updates the ePSF
        # with the star centers and fluxes fixed and then refits the
        # stars with the updated ePSF. The convergence diagnostics then
        # describe the refit of the last refinement iteration, so that
        # they match the returned stars.
        if (self.refinement_iters > 0 and self.alias_passband is not None
                and np.any(self.oversampling >= _REFINE_MIN_OVERSAMPLING)):
            refine_reporter = _ProgressReporter(
                self.progress_bar, self.refinement_iters,
                desc='EPSFBuilder refinement').setup()
            for refine_num in range(1, self.refinement_iters + 1):
                epsf, stars, fit_failed = self._process_iteration(
                    stars, epsf, iter_num + refine_num, refine=True)
                (converged, converged_fraction, max_center_dist_sq,
                 centers) = self._check_convergence(stars, centers,
                                                    fit_failed)
                history.append(_IterationRecord(
                    epsf.data.copy(), 'refine', converged,
                    converged_fraction, max_center_dist_sq,
                    int(np.sum(fit_failed))))
                refine_reporter.update()
            refine_reporter.close()
            final_center_accuracy = float(max_center_dist_sq ** 0.5)

        # Smooth the wings of the final ePSF. The ePSF is replaced only
        # if the smoothing changed it, so that an ePSF without wings is
        # exactly the ePSF of the last iteration.
        if self.wing_smoothing:
            smoothed = self._smooth_wings(epsf.data)
            if smoothed is not epsf.data:
                epsf = ImagePSF(data=self._normalize_epsf(smoothed),
                                oversampling=self.oversampling,
                                fill_value=0.0)

        # Finalize and return structured results
        return self._finalize_build(epsf, stars, iter_num, converged,
                                    final_center_accuracy,
                                    converged_fraction, history=history,
                                    initial_epsf=initial_epsf)

    @staticmethod
    def _make_iteration_info(history, *, initial_epsf=None):
        """
        Make the table of per-iteration diagnostics.

        Parameters
        ----------
        history : list of `_IterationRecord`
            The record of each iteration. It has at least one record.

        initial_epsf : 2D `~numpy.ndarray` or `None`, optional
            The image of the input ePSF that the build started from,
            which is the reference for the change of the first
            iteration. If `None`, the reference is an empty ePSF.

        Returns
        -------
        table : `~astropy.table.Table`
            The table described in `EPSFBuildResults`.
        """
        peak = np.max(history[-1].epsf_data)
        changes = []
        previous = 0.0 if initial_epsf is None else initial_epsf
        for record in history:
            change = np.max(np.abs(record.epsf_data - previous)) / peak
            changes.append(float(change))
            previous = record.epsf_data

        table = Table()
        table['iteration'] = np.arange(1, len(history) + 1)
        table['stage'] = [record.stage for record in history]
        table['converged'] = [bool(record.converged) for record in history]
        table['converged_fraction'] = [float(record.converged_fraction)
                                       for record in history]
        table['max_center_shift'] = [float(record.max_center_dist_sq) ** 0.5
                                     for record in history]
        table['n_fit_failed'] = [record.n_fit_failed for record in history]
        table['max_epsf_change'] = changes
        table['converged_fraction'].info.format = '.3f'
        table['max_center_shift'].info.format = '.3g'
        table['max_epsf_change'].info.format = '.3g'
        return table


def __getattr__(name):
    # EPSFBuildResult was renamed to EPSFBuildResults in 3.1.
    if name == 'EPSFBuildResult':
        msg = ('EPSFBuildResult is deprecated and will be removed in a '
               'future version. Use EPSFBuildResults instead.')
        warnings.warn(msg, PhotutilsDeprecationWarning, stacklevel=2)
        return EPSFBuildResults

    msg = f'module {__name__!r} has no attribute {name!r}'
    raise AttributeError(msg)
