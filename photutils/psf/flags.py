# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tools for working with PSF photometry flags, including centralized flag
definitions and decoding utilities.
"""

from typing import ClassVar

from photutils.utils._deprecation import deprecated_positional_kwargs
from photutils.utils._flags import (FlagDefinition, FlagRegistry, decode_flags,
                                    update_flag_docstring)

__all__ = ['PSF_FLAGS', 'decode_psf_flags']


class _PSFFlags(FlagRegistry):
    """
    Centralized definition of PSF photometry flags.

    This class provides a single source of truth for all PSF flag
    definitions, including bit values, names, and descriptions. It
    enables consistent flag handling across the PSF photometry codebase
    and supports dynamic docstring generation.

    Examples
    --------
    >>> from photutils.psf.flags import _PSFFlags
    >>> flags = _PSFFlags()
    >>> flags.N_PIXELS_FIT_PARTIAL
    1
    >>> flags.get_name(1)
    'n_pixels_fit_partial'
    >>> flags.get_description(8)
    'possible non-convergence'
    """

    # Define all PSF flags with their properties
    FLAG_DEFINITIONS: ClassVar = [
        FlagDefinition(
            bit_value=1,
            name='n_pixels_fit_partial',
            description=('n_pixels_fit smaller than full fit_shape '
                         'region'),
            detailed_description=('The number of fitted pixels '
                                  '(n_pixels_fit) is smaller than the '
                                  'full fit_shape region, indicating '
                                  'partial PSF fitting'),
        ),
        FlagDefinition(
            bit_value=2,
            name='outside_bounds',
            description='fitted position outside input image bounds',
            detailed_description=('The fitted source position is outside the '
                                  'bounds of the input image'),
        ),
        FlagDefinition(
            bit_value=4,
            name='negative_flux',
            description='non-positive flux',
            detailed_description=('The fitted flux value is negative or zero, '
                                  'which is non-physical'),
        ),
        FlagDefinition(
            bit_value=8,
            name='no_convergence',
            description='possible non-convergence',
            detailed_description=('The PSF fitting algorithm may not have '
                                  'converged to a stable solution'),
        ),
        FlagDefinition(
            bit_value=16,
            name='no_covariance',
            description='missing parameter covariance',
            detailed_description=('Parameter covariance matrix is not '
                                  'available, preventing error estimation'),
        ),
        FlagDefinition(
            bit_value=32,
            name='near_bound',
            description='fitted parameter near a bound',
            detailed_description=('One or more fitted parameters are very '
                                  'close to their imposed bounds'),
        ),
        FlagDefinition(
            bit_value=64,
            name='no_overlap',
            description='no overlap with data',
            detailed_description=('The source PSF fitting region has no '
                                  'overlap with valid data pixels'),
        ),
        FlagDefinition(
            bit_value=128,
            name='fully_masked',
            description='fully masked source',
            detailed_description=('All pixels in the source fitting region '
                                  'are masked'),
        ),
        FlagDefinition(
            bit_value=256,
            name='too_few_pixels',
            description='too few pixels for fitting',
            detailed_description=('Insufficient unmasked pixels available '
                                  'for reliable PSF fitting'),
        ),
        FlagDefinition(
            bit_value=512,
            name='non_finite_position',
            description='non-finite fitted position',
            detailed_description=('The fitted x or y position is NaN or inf, '
                                  'indicating an invalid or failed fit'),
        ),
        FlagDefinition(
            bit_value=1024,
            name='non_finite_flux',
            description='non-finite fitted flux',
            detailed_description=('The fitted flux value is NaN or inf, '
                                  'indicating an invalid or failed fit'),
        ),
        FlagDefinition(
            bit_value=2048,
            name='non_finite_localbkg',
            description='non-finite local background',
            detailed_description=('The local background value is NaN or '
                                  'inf, so it was not subtracted before '
                                  'fitting'),
        ),
    ]

    domain: ClassVar = 'psf'

    # Remove in 4.0
    _DEPRECATED_FLAG_NAMES: ClassVar = {
        'npixfit_partial': 'n_pixels_fit_partial',
    }

    # Remove in 4.0
    _DEPRECATED_CONSTANT_NAMES: ClassVar = {
        'NPIXFIT_PARTIAL': 'N_PIXELS_FIT_PARTIAL',
    }

    _DEPRECATED_SINCE: ClassVar = '3.0'
    _DEPRECATED_UNTIL: ClassVar = '4.0'


# Create a singleton instance for global use
PSF_FLAGS = _PSFFlags()


def _update_decode_docstring(func):
    """
    Decorator to update a function docstring with the PSF flag
    documentation.

    The ``<flag_descriptions>`` placeholder in the function docstring
    is replaced with a bullet list generated from ``PSF_FLAGS`` (see
    `photutils.utils._flags.update_flag_docstring`).

    Parameters
    ----------
    func : function
        The function to decorate.

    Returns
    -------
    func : function
        The decorated function with updated docstring.
    """
    return update_flag_docstring(func, PSF_FLAGS)


@_update_decode_docstring
@deprecated_positional_kwargs(since='3.0', until='4.0')
def decode_psf_flags(flags, return_bit_values=False):
    # numpydoc ignore: RT05
    """
    Decode PSF photometry bit flags into individual components.

    This function takes integer flag values from PSF photometry results
    and returns a list of human-readable descriptions of the issues
    that occurred during fitting. This is useful for understanding
    what problems were encountered without needing to manually perform
    bitwise operations.

    Parameters
    ----------
    flags : int or array-like of int
        Integer flag value(s) to decode. Each bit in the flag
        represents a specific condition that occurred during
        PSF fitting.

    return_bit_values : bool, optional
        If `True`, return the decoded bit flags (integers) instead of
        the flag descriptions (strings). Default is `False`.

    Returns
    -------
    decoded : list of str, list of int, list of list of str, or \
            list of list of int
        List of active flag names (or bit values), or list of lists
        if input is an array. Each string (or integer) represents a
        specific condition that was detected during PSF fitting. If no
        flags are set, an empty list is returned. Possible flag names
        are:
        <flag_descriptions>

    Examples
    --------
    Decode a single flag value:

    >>> from photutils.psf import decode_psf_flags
    >>> issues = decode_psf_flags(5)  # bits 1 and 4 set
    >>> print(issues)
    ['n_pixels_fit_partial', 'negative_flux']
    >>> 'n_pixels_fit_partial' in issues
    True
    >>> 'no_convergence' in issues
    False

    Decode multiple flag values:

    >>> flags = [0, 8, 136]  # 0, bit 8, bits 8+128
    >>> decoded_list = decode_psf_flags(flags)
    >>> len(decoded_list)
    3
    >>> decoded_list[0]  # No issues
    []
    >>> decoded_list[1]  # Convergence issue
    ['no_convergence']
    >>> decoded_list[2]  # Multiple issues
    ['no_convergence', 'fully_masked']

    Check for specific issues:

    >>> issues = decode_psf_flags(136)
    >>> if 'no_convergence' in issues:
    ...     print("Fit may not have converged")
    Fit may not have converged
    >>> if issues:  # Any issues present
    ...     print(f"Found {len(issues)} issues: {', '.join(issues)}")
    Found 2 issues: no_convergence, fully_masked

    Working with PSF photometry results:

    >>> import numpy as np
    >>> from astropy.modeling import models
    >>> from astropy.table import Table
    >>> from photutils.psf import (CircularGaussianPRF, PSFPhotometry,
    ...                            decode_psf_flags)
    >>> # Create minimal test data
    >>> yy, xx = np.mgrid[:21, :21]
    >>> m1 = CircularGaussianPRF(flux=-10, x_0=10, y_0=10, fwhm=2)
    >>> m2 = CircularGaussianPRF(flux=10, x_0=3, y_0=3, fwhm=2)
    >>> m3 = CircularGaussianPRF(flux=10, x_0=21, y_0=21, fwhm=2)
    >>> data = m1(xx, yy) + m2(xx, yy) + m3(xx, yy)
    >>> psf_model = CircularGaussianPRF(flux=1, x_0=10, y_0=10, fwhm=2)
    >>> init_params = Table({'x': (10, 3, 21), 'y': (10, 3, 21),
    ...                      'flux': (1, 10, 10)})
    >>> photometry = PSFPhotometry(psf_model, (3, 3))
    >>> results = photometry(data, init_params=init_params)
    >>> issues_list = decode_psf_flags(results['flags'])
    >>> for i, issues in enumerate(issues_list):
    ...     if issues:
    ...         print(f"Source {i+1}: {', '.join(issues)}")
    Source 1: negative_flux
    Source 3: n_pixels_fit_partial, no_covariance, too_few_pixels, \
non_finite_position, non_finite_flux
    """
    return decode_flags(flags, PSF_FLAGS,
                        return_bit_values=return_bit_values)
