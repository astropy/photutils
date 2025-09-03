# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Utilities for decoding PSF photometry bit flags.
"""

import numpy as np

__all__ = ['decode_psf_flags']


def decode_psf_flags(flags):
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

    Returns
    -------
    decoded : list of str or list of list of str
        List of active flag names, or list of lists if input is
        an array. Each string represents a specific condition
        that was detected during PSF fitting. If no flags are
        set, returns an empty list. Possible flag names are:

        - ``'npixfit_partial'`` : bit 1, npixfit smaller than
          full fit_shape region
        - ``'outside_bounds'`` : bit 2, fitted position outside
          input image bounds
        - ``'negative_flux'`` : bit 4, non-positive flux
        - ``'no_convergence'`` : bit 8, possible non-convergence
        - ``'no_covariance'`` : bit 16, missing parameter covariance
        - ``'near_bound'`` : bit 32, fitted parameter near a bound
        - ``'no_overlap'`` : bit 64, no overlap with data
        - ``'fully_masked'`` : bit 128, fully masked source
        - ``'too_few_pixels'`` : bit 256, too few pixels for fitting

    Examples
    --------
    Decode a single flag value:

    >>> from photutils.psf import decode_psf_flags
    >>> issues = decode_psf_flags(5)  # bits 1 and 4 set
    >>> print(issues)
    ['npixfit_partial', 'negative_flux']
    >>> 'npixfit_partial' in issues
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
    Source 3: npixfit_partial, no_covariance, too_few_pixels
    """
    # Flag definitions with descriptive names
    flag_definitions = {
        1: 'npixfit_partial',   # npixfit smaller than full fit_shape
        2: 'outside_bounds',    # fitted position outside image bounds
        4: 'negative_flux',     # non-positive flux
        8: 'no_convergence',    # possible non-convergence
        16: 'no_covariance',    # missing parameter covariance
        32: 'near_bound',       # near a positional bound
        64: 'no_overlap',       # no overlap with data
        128: 'fully_masked',    # fully masked source
        256: 'too_few_pixels',  # too few pixels for fitting
    }

    def _decode_single_flag(flag_value):
        """
        Decode a single integer flag value.
        """
        if not isinstance(flag_value, (int, np.integer)):
            msg = 'Flag value must be an integer'
            raise TypeError(msg)

        active_flags = []
        for bit_value, description in flag_definitions.items():
            if flag_value & bit_value:
                active_flags.append(description)
        return active_flags

    # Handle both single values and arrays
    if np.isscalar(flags):
        return _decode_single_flag(flags)

    # Convert to numpy array for consistent handling
    flags_array = np.asarray(flags)
    if flags_array.ndim == 0:
        # Handle 0-d arrays (scalar arrays)
        return _decode_single_flag(flags_array.item())

    # Handle 1-d or higher dimensional arrays
    return [_decode_single_flag(flag) for flag in flags_array.flat]
