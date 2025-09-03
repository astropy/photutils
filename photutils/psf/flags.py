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
