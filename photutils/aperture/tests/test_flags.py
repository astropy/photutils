# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the flags module.
"""

import numpy as np
import pytest

from photutils.aperture.flags import (APERTURE_FLAGS, _ApertureFlags,
                                      decode_aperture_flags)
from photutils.utils._flags import FlagDefinition

EXPECTED_FLAGS = {
    'no_overlap': 1,
    'partial_overlap': 2,
    'no_pixels': 4,
    'masked_pixels': 8,
    'all_masked': 16,
    'non_finite_data': 32,
    'non_finite_error': 64,
    'neighbor_pixels': 128,
    'uncorrected_pixels': 256,
    'sigma_clipped': 512,
    'all_clipped': 1024,
    'too_few_pixels': 2048,
}


def test_decode_aperture_flags():
    """
    Test the decode_aperture_flags standalone function.
    """
    decoded = decode_aperture_flags(0)
    assert decoded == []
    assert isinstance(decoded, list)

    # Test each single flag value
    for name, bit_value in EXPECTED_FLAGS.items():
        assert decode_aperture_flags(bit_value) == [name]

    # Test combination of flags
    decoded = decode_aperture_flags(10)  # bits 2 and 8
    assert decoded == ['partial_overlap', 'masked_pixels']

    decoded = decode_aperture_flags(5)  # bits 1 and 4
    assert decoded == ['no_overlap', 'no_pixels']

    # Test with all flags set
    all_flags = sum(EXPECTED_FLAGS.values())
    decoded = decode_aperture_flags(all_flags)
    assert decoded == list(EXPECTED_FLAGS)

    # Test with array input
    flags_array = [0, 1, 2, 10]
    decoded_list = decode_aperture_flags(flags_array)
    assert len(decoded_list) == 4
    assert decoded_list[0] == []
    assert decoded_list[1] == ['no_overlap']
    assert decoded_list[2] == ['partial_overlap']
    assert decoded_list[3] == ['partial_overlap', 'masked_pixels']

    # Test with numpy array
    decoded_list = decode_aperture_flags(np.array([8, 16, 32]))
    assert decoded_list == [['masked_pixels'], ['all_masked'],
                            ['non_finite_data']]

    # Test with 0D numpy array (scalar array)
    decoded = decode_aperture_flags(np.array(64))
    assert decoded == ['non_finite_error']

    # Test with empty array
    decoded = decode_aperture_flags(np.array([], dtype=int))
    assert decoded == []

    # Test with 2D array (flattened)
    decoded = decode_aperture_flags(np.array([[0, 1], [8, 24]]))
    assert len(decoded) == 4
    assert decoded[3] == ['masked_pixels', 'all_masked']


def test_decode_aperture_flags_return_bit_values():
    """
    Test decode_aperture_flags with return_bit_values=True.
    """
    assert decode_aperture_flags(0, return_bit_values=True) == []
    assert decode_aperture_flags(10, return_bit_values=True) == [2, 8]

    decoded_list = decode_aperture_flags([1, 24],
                                         return_bit_values=True)
    assert decoded_list == [[1], [8, 16]]


def test_decode_aperture_flags_errors():
    """
    Test decode_aperture_flags error conditions.
    """
    match = 'Flag value must be an integer'
    with pytest.raises(TypeError, match=match):
        decode_aperture_flags(3.14)
    with pytest.raises(TypeError, match=match):
        decode_aperture_flags('invalid')
    with pytest.raises(TypeError, match=match):
        decode_aperture_flags([1, 2.5, 3])

    match = 'Flag value must be a non-negative integer'
    with pytest.raises(ValueError, match=match):
        decode_aperture_flags(-2)


def test_aperture_flags_singleton():
    """
    Test APERTURE_FLAGS singleton behavior.
    """
    assert isinstance(APERTURE_FLAGS, _ApertureFlags)
    assert APERTURE_FLAGS.domain == 'aperture'

    new_flags = _ApertureFlags()
    assert isinstance(new_flags, _ApertureFlags)
    assert new_flags is not APERTURE_FLAGS


def test_aperture_flags_constants():
    """
    Test _ApertureFlags constant access.
    """
    for name, bit_value in EXPECTED_FLAGS.items():
        const_name = name.upper()
        assert hasattr(APERTURE_FLAGS, const_name)
        actual_value = getattr(APERTURE_FLAGS, const_name)
        assert actual_value == bit_value
        assert isinstance(actual_value, int)

    match = "has no attribute 'INVALID'"
    with pytest.raises(AttributeError, match=match):
        _ = APERTURE_FLAGS.INVALID


def test_aperture_flags_properties():
    """
    Test _ApertureFlags property access methods.
    """
    assert APERTURE_FLAGS.bit_values == list(EXPECTED_FLAGS.values())
    assert APERTURE_FLAGS.names == list(EXPECTED_FLAGS)
    assert APERTURE_FLAGS.flag_dict == {bit: name for name, bit
                                        in EXPECTED_FLAGS.items()}

    all_flags = APERTURE_FLAGS.all_flags
    assert isinstance(all_flags, list)
    assert len(all_flags) == len(EXPECTED_FLAGS)
    for flag_def in all_flags:
        assert isinstance(flag_def, FlagDefinition)


def test_aperture_flags_get_methods():
    """
    Test _ApertureFlags getter methods.
    """
    for name, bit_value in EXPECTED_FLAGS.items():
        assert APERTURE_FLAGS.get_name(bit_value) == name
        assert APERTURE_FLAGS.get_bit_value(name) == bit_value
        assert isinstance(APERTURE_FLAGS.get_description(bit_value), str)
        detailed = APERTURE_FLAGS.get_detailed_description(bit_value)
        assert isinstance(detailed, str)

    # Test get_definition by bit value and name
    def_by_bit = APERTURE_FLAGS.get_definition(1)
    def_by_name = APERTURE_FLAGS.get_definition('no_overlap')
    assert def_by_bit is def_by_name
    assert def_by_bit.name == 'no_overlap'

    # Test error cases
    match = 'No flag with bit value 999'
    with pytest.raises(KeyError, match=match):
        APERTURE_FLAGS.get_definition(999)

    match = "No flag with name 'invalid'"
    with pytest.raises(KeyError, match=match):
        APERTURE_FLAGS.get_definition('invalid')

    match = 'identifier must be int'
    with pytest.raises(TypeError, match=match):
        APERTURE_FLAGS.get_definition(3.14)


def test_aperture_flags_completeness():
    """
    Test that _ApertureFlags bit values and names are consistent.
    """
    bit_values = APERTURE_FLAGS.bit_values
    names = APERTURE_FLAGS.names

    # Bit values are unique powers of 2
    assert len(bit_values) == len(set(bit_values))
    for bit_val in bit_values:
        assert bit_val > 0
        assert (bit_val & (bit_val - 1)) == 0

    # Names are unique, valid snake_case identifiers
    assert len(names) == len(set(names))
    for name in names:
        assert name.isidentifier()
        assert name == name.lower()


def test_decode_aperture_flags_docstring():
    """
    Test that decode_aperture_flags has dynamic flag documentation.
    """
    docstring = decode_aperture_flags.__doc__
    assert '<flag_descriptions>' not in docstring

    for name, bit_value in EXPECTED_FLAGS.items():
        expected = f"``'{name}'`` : bit {bit_value}"
        assert expected in docstring
