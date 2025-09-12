# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Define tools for working with PSF photometry flags, including
centralized flag definitions and decoding utilities.
"""

from dataclasses import dataclass
from typing import ClassVar

import numpy as np

__all__ = ['PSF_FLAGS', 'decode_psf_flags']


@dataclass(frozen=True)
class _PSFFlagDefinition:
    """
    A single PSF flag definition.

    Attributes
    ----------
    bit_value : int
        The bit value (power of 2) for this flag.

    name : str
        Short name for the flag (used in decode_psf_flags).

    description : str
        Brief description of what this flag indicates.

    detailed_description : str
        Detailed description for use in docstrings.
    """

    bit_value: int
    name: str
    description: str
    detailed_description: str


class _PSFFlags:
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
    >>> flags.NPIXFIT_PARTIAL
    1
    >>> flags.get_name(1)
    'npixfit_partial'
    >>> flags.get_description(8)
    'possible non-convergence'
    """

    # Define all PSF flags with their properties
    FLAG_DEFINITIONS: ClassVar = [
        _PSFFlagDefinition(
            bit_value=1,
            name='npixfit_partial',
            description='npixfit smaller than full fit_shape region',
            detailed_description=('The number of fitted pixels (npixfit) is '
                                  'smaller than the full fit_shape region, '
                                  'indicating partial PSF fitting'),
        ),
        _PSFFlagDefinition(
            bit_value=2,
            name='outside_bounds',
            description='fitted position outside input image bounds',
            detailed_description=('The fitted source position is outside the '
                                  'bounds of the input image'),
        ),
        _PSFFlagDefinition(
            bit_value=4,
            name='negative_flux',
            description='non-positive flux',
            detailed_description=('The fitted flux value is negative or zero, '
                                  'which is non-physical'),
        ),
        _PSFFlagDefinition(
            bit_value=8,
            name='no_convergence',
            description='possible non-convergence',
            detailed_description=('The PSF fitting algorithm may not have '
                                  'converged to a stable solution'),
        ),
        _PSFFlagDefinition(
            bit_value=16,
            name='no_covariance',
            description='missing parameter covariance',
            detailed_description=('Parameter covariance matrix is not '
                                  'available, preventing error estimation'),
        ),
        _PSFFlagDefinition(
            bit_value=32,
            name='near_bound',
            description='fitted parameter near a bound',
            detailed_description=('One or more fitted parameters are very '
                                  'close to their imposed bounds'),
        ),
        _PSFFlagDefinition(
            bit_value=64,
            name='no_overlap',
            description='no overlap with data',
            detailed_description=('The source PSF fitting region has no '
                                  'overlap with valid data pixels'),
        ),
        _PSFFlagDefinition(
            bit_value=128,
            name='fully_masked',
            description='fully masked source',
            detailed_description=('All pixels in the source fitting region '
                                  'are masked'),
        ),
        _PSFFlagDefinition(
            bit_value=256,
            name='too_few_pixels',
            description='too few pixels for fitting',
            detailed_description=('Insufficient unmasked pixels available '
                                  'for reliable PSF fitting'),
        ),
    ]

    def __init__(self):
        for flag_def in self.FLAG_DEFINITIONS:
            # Create uppercase constants (e.g., NPIXFIT_PARTIAL = 1)
            setattr(self, flag_def.name.upper(), flag_def.bit_value)

        # Create lookup dictionaries for efficient access
        self._bit_to_def = {fd.bit_value: fd for fd in self.FLAG_DEFINITIONS}
        self._name_to_def = {fd.name: fd for fd in self.FLAG_DEFINITIONS}

    @property
    def all_flags(self):
        """
        Return all flag definitions.
        """
        return self.FLAG_DEFINITIONS.copy()

    @property
    def bit_values(self):
        """
        Return all bit values.
        """
        return [fd.bit_value for fd in self.FLAG_DEFINITIONS]

    @property
    def names(self):
        """
        Return all flag names.
        """
        return [fd.name for fd in self.FLAG_DEFINITIONS]

    @property
    def flag_dict(self):
        """
        Return dictionary mapping bit values to names.
        """
        return {fd.bit_value: fd.name for fd in self.FLAG_DEFINITIONS}

    def get_definition(self, identifier):
        """
        Get flag definition by bit value or name.

        Parameters
        ----------
        identifier : int or str
            Either the bit value (int) or name (str) of the flag.

        Returns
        -------
        definition : `_PSFFlagDefinition`
            The flag definition.

        Raises
        ------
        KeyError
            If the identifier is not found.
        """
        if isinstance(identifier, int):
            if identifier not in self._bit_to_def:
                msg = f"No flag with bit value {identifier}"
                raise KeyError(msg)
            return self._bit_to_def[identifier]

        if isinstance(identifier, str):
            if identifier not in self._name_to_def:
                msg = f"No flag with name '{identifier}'"
                raise KeyError(msg)
            return self._name_to_def[identifier]

        msg = 'identifier must be int (bit value) or str (name)'
        raise TypeError(msg)

    def get_name(self, bit_value):
        """
        Get flag name from bit value.

        Parameters
        ----------
        bit_value : int
            The bit value of the flag.

        Returns
        -------
        name : str
            The name of the flag.
        """
        return self.get_definition(bit_value).name

    def get_bit_value(self, name):
        """
        Get flag bit value from name.

        Parameters
        ----------
        name : str
            The name of the flag.

        Returns
        -------
        bit_value : int
            The bit value of the flag.
        """
        return self.get_definition(name).bit_value

    def get_description(self, bit_value):
        """
        Get flag description from bit value.

        Parameters
        ----------
        bit_value : int
            The bit value of the flag.

        Returns
        -------
        description : str
            The brief description of the flag.
        """
        return self.get_definition(bit_value).description

    def get_detailed_description(self, bit_value):
        """
        Get detailed flag description from bit value.

        Parameters
        ----------
        bit_value : int
            The bit value of the flag.

        Returns
        -------
        detailed_description : str
            The detailed description of the flag.
        """
        return self.get_definition(bit_value).detailed_description


# Create a singleton instance for global use
PSF_FLAGS = _PSFFlags()


def _update_decode_docstring(func):
    """
    Decorator to update function docstring with PSF flag documentation.

    This decorator can be applied to functions like decode_psf_flags to
    automatically replace manually defined flag lists with dynamically
    generated ones.

    Parameters
    ----------
    func : function
        The function to decorate.

    Returns
    -------
    func : function
        The decorated function with updated docstring.
    """
    if not hasattr(func, '__doc__') or func.__doc__ is None:
        return func

    docstring = func.__doc__

    # Look for the placeholder text
    placeholder = '<flag descriptions>'

    if placeholder in docstring:
        # Generate the flag descriptions
        flag_descriptions = ['']

        indent = ' ' * 4
        for flag_def in PSF_FLAGS.FLAG_DEFINITIONS:
            name = flag_def.name
            bit_val = flag_def.bit_value
            desc = flag_def.description
            line = f"{indent}- ``'{name}'`` : bit {bit_val}, {desc}"
            flag_descriptions.append(line)

        # Replace the placeholder with the flag descriptions
        flag_text = '\n'.join(flag_descriptions)
        new_docstring = docstring.replace(placeholder, flag_text)
        func.__doc__ = new_docstring

    return func


@_update_decode_docstring
def decode_psf_flags(flags):
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

    Returns
    -------
    decoded : list of str or list of list of str
        List of active flag names, or list of lists if input is an
        array. Each string represents a specific condition that was
        detected during PSF fitting. If no flags are set, an empty list
        is returned. Possible flag names are:
        <flag descriptions>

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
    # Get flag definitions from centralized source
    flag_definitions = PSF_FLAGS.flag_dict

    def _decode_single_flag(flag_value):
        """
        Decode a single integer flag value.
        """
        if not isinstance(flag_value, (int, np.integer)):
            msg = 'Flag value must be an integer'
            raise TypeError(msg)

        if flag_value < 0:
            msg = 'Flag value must be a non-negative integer'
            raise ValueError(msg)

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
