# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Generic machinery for defining and decoding bitwise quality flags.

This private module provides the shared building blocks used by the
public per-package flag registries (e.g., ``photutils.psf.flags`` and
``photutils.aperture.flags``): a flag-definition dataclass, a registry
base class, a flag decoder, and a docstring-substitution helper.
"""

import warnings
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
from astropy.utils.exceptions import AstropyDeprecationWarning

from photutils.utils._deprecation import deprecated_getattr

__all__ = []


@dataclass(frozen=True)
class FlagDefinition:
    """
    A single bitwise flag definition.

    Attributes
    ----------
    bit_value : int
        The bit value (power of 2) for this flag.

    name : str
        Short name for the flag (used by the flag decoders).

    description : str
        Brief description of what this flag indicates.

    detailed_description : str
        Detailed description for use in docstrings.
    """

    bit_value: int
    name: str
    description: str
    detailed_description: str


class FlagRegistry:
    """
    Base class for centralized definitions of bitwise quality flags.

    Subclasses define the ``FLAG_DEFINITIONS`` list of `FlagDefinition`
    objects and a short ``domain`` string identifying the flag source
    (e.g., ``'psf'`` or ``'aperture'``). The registry provides a single
    source of truth for the flag bit values, names, and descriptions,
    enabling consistent flag handling and dynamic docstring generation.

    Uppercase constants (e.g., ``NO_OVERLAP``) are created on the
    instance for each flag definition.
    """

    FLAG_DEFINITIONS: ClassVar = []
    domain: ClassVar = ''

    # Mappings of deprecated flag/constant names to their new names;
    # subclasses may override these along with the deprecation
    # versions used in the warning messages.
    _DEPRECATED_FLAG_NAMES: ClassVar = {}
    _DEPRECATED_CONSTANT_NAMES: ClassVar = {}
    _DEPRECATED_SINCE: ClassVar = None
    _DEPRECATED_UNTIL: ClassVar = None

    def __init__(self):
        for flag_def in self.FLAG_DEFINITIONS:
            # Create uppercase constants (e.g., NO_OVERLAP = 1)
            setattr(self, flag_def.name.upper(), flag_def.bit_value)

        # Create lookup dictionaries for efficient access
        self._bit_to_def = {fd.bit_value: fd
                            for fd in self.FLAG_DEFINITIONS}
        self._name_to_def = {fd.name: fd for fd in self.FLAG_DEFINITIONS}

    def __getattr__(self, name):
        return deprecated_getattr(self, name,
                                  self._DEPRECATED_CONSTANT_NAMES,
                                  since=self._DEPRECATED_SINCE,
                                  until=self._DEPRECATED_UNTIL)

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
        definition : `FlagDefinition`
            The flag definition.

        Raises
        ------
        KeyError
            If the identifier is not found.
        """
        if isinstance(identifier, int):
            if identifier not in self._bit_to_def:
                msg = f'No flag with bit value {identifier}'
                raise KeyError(msg)
            return self._bit_to_def[identifier]

        if isinstance(identifier, str):
            if identifier in self._DEPRECATED_FLAG_NAMES:
                new_name = self._DEPRECATED_FLAG_NAMES[identifier]
                warnings.warn(
                    f"The flag name '{identifier}' is deprecated "
                    f"in version {self._DEPRECATED_SINCE}. Use "
                    f"'{new_name}' instead. It will be removed in "
                    f'version {self._DEPRECATED_UNTIL}.',
                    AstropyDeprecationWarning,
                    stacklevel=2,
                )
                identifier = new_name

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


def update_flag_docstring(func, registry,
                          placeholder='<flag_descriptions>'):
    """
    Update a function docstring with flag documentation.

    The ``placeholder`` text in the function docstring is replaced with
    a bullet list of the flag names, bit values, and brief descriptions
    generated from the ``registry``.

    Parameters
    ----------
    func : function
        The function whose docstring is updated in place.

    registry : `FlagRegistry`
        The flag registry providing the flag definitions.

    placeholder : str, optional
        The placeholder text to replace in the docstring.

    Returns
    -------
    func : function
        The function with an updated docstring.
    """
    if not hasattr(func, '__doc__') or func.__doc__ is None:
        return func

    docstring = func.__doc__

    if placeholder in docstring:
        # Generate the flag descriptions
        flag_descriptions = ['']

        indent = ' ' * 4
        for flag_def in registry.FLAG_DEFINITIONS:
            name = flag_def.name
            bit_val = flag_def.bit_value
            desc = flag_def.description
            line = f"{indent}- ``'{name}'`` : bit {bit_val}, {desc}"
            flag_descriptions.append(line)

        # Replace the placeholder with the flag descriptions
        flag_text = '\n'.join(flag_descriptions)
        func.__doc__ = docstring.replace(placeholder, flag_text)

    return func


def decode_flags(flags, registry, *, return_bit_values=False):
    """
    Decode bitwise flag values into individual components.

    Parameters
    ----------
    flags : int or array-like of int
        Integer flag value(s) to decode.

    registry : `FlagRegistry`
        The flag registry providing the flag definitions.

    return_bit_values : bool, optional
        If `True`, return the decoded bit flags (integers) instead of
        the flag names (strings).

    Returns
    -------
    decoded : list
        List of active flag names (or bit values), or list of lists
        if the input is an array.
    """
    flag_definitions = registry.flag_dict

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
        for bit_value, name in flag_definitions.items():
            if flag_value & bit_value:
                if return_bit_values:
                    active_flags.append(bit_value)
                else:
                    active_flags.append(name)
        return active_flags

    # Handle both single values and arrays
    if np.isscalar(flags):
        return _decode_single_flag(flags)

    # Convert to numpy array for consistent handling
    flags_array = np.asarray(flags)
    if flags_array.ndim == 0:
        # Handle 0D arrays (scalar arrays)
        return _decode_single_flag(flags_array.item())

    # Handle 1D or higher dimensional arrays
    return [_decode_single_flag(flag) for flag in flags_array.flat]
