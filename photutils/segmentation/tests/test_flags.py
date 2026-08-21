# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the flags module.
"""

import numpy as np
import pytest

from photutils.aperture.flags import APERTURE_FLAGS
from photutils.segmentation.flags import (SEGMENTATION_FLAGS,
                                          decode_segmentation_flags)

# Flags that intentionally share a name (and meaning, with the segment
# as the region) with APERTURE_FLAGS
SHARED_APERTURE_NAMES = ('masked_pixels', 'non_finite_data',
                         'non_finite_error', 'all_masked',
                         'undefined_shape', 'singular_covariance')

EXPECTED_BITS = {
    'masked_pixels': 1,
    'non_finite_data': 2,
    'non_finite_error': 4,
    'all_masked': 8,
    'edge_touch': 16,
    'deblended': 32,
    'deblend_nonposmin': 64,
    'deblend_n_markers': 128,
    'undefined_shape': 256,
    'singular_covariance': 512,
    'centroid_win_fallback': 1024,
    'centroid_quad_failed': 2048,
    'kron_undefined': 4096,
    'kron_minimum_radius': 8192,
    'kron_no_overlap': 16384,
    'kron_partial_overlap': 32768,
    'kron_masked_pixels': 65536,
    'kron_neighbor_pixels': 131072,
    'kron_uncorrected_pixels': 262144,
}


def test_flag_names_and_bits():
    """
    Test the frozen flag names and bit values.
    """
    assert set(SEGMENTATION_FLAGS.names) == set(EXPECTED_BITS)
    for name, bit in EXPECTED_BITS.items():
        assert SEGMENTATION_FLAGS.get_bit_value(name) == bit


def test_bit_values_unique_powers_of_two():
    """
    Test that all bit values are unique powers of two.
    """
    bits = SEGMENTATION_FLAGS.bit_values
    assert len(bits) == len(set(bits))
    for bit in bits:
        assert bit > 0
        assert bit & (bit - 1) == 0


def test_shared_aperture_names_intentional():
    """
    Test that the name overlap with APERTURE_FLAGS is exactly the
    intentional shared set.

    A shared name must describe the same condition (with the segment as
    the region). Any new accidental overlap must either be renamed or
    added to the shared set deliberately.
    """
    shared = (set(SEGMENTATION_FLAGS.names)
              & set(APERTURE_FLAGS.names))
    assert shared == set(SHARED_APERTURE_NAMES)


def test_uppercase_constants():
    """
    Test that uppercase constants exist for every flag.
    """
    for name, bit in EXPECTED_BITS.items():
        assert getattr(SEGMENTATION_FLAGS, name.upper()) == bit


def test_decode_scalar():
    """
    Test decoding a scalar flag value.
    """
    value = (SEGMENTATION_FLAGS.DEBLENDED
             | SEGMENTATION_FLAGS.EDGE_TOUCH)
    names = decode_segmentation_flags(value)
    assert set(names) == {'deblended', 'edge_touch'}

    bits = decode_segmentation_flags(value, return_bit_values=True)
    assert set(bits) == {16, 32}

    assert decode_segmentation_flags(0) == []


def test_decode_array():
    """
    Test decoding an array of flag values.
    """
    values = np.array([0, SEGMENTATION_FLAGS.MASKED_PIXELS])
    result = decode_segmentation_flags(values)
    assert result == [[], ['masked_pixels']]


def test_decode_invalid():
    """
    Test invalid decoder inputs.
    """
    match = 'must be an integer'
    with pytest.raises(TypeError, match=match):
        decode_segmentation_flags(1.5)
    match = 'must be a non-negative integer'
    with pytest.raises(ValueError, match=match):
        decode_segmentation_flags(-1)


def test_public_import():
    """
    Test that the public names are importable from the subpackage.
    """
    import photutils.segmentation as seg_mod  # noqa: PLC0415

    assert hasattr(seg_mod, 'SEGMENTATION_FLAGS')
    assert hasattr(seg_mod, 'decode_segmentation_flags')
    assert seg_mod.SEGMENTATION_FLAGS is SEGMENTATION_FLAGS
    assert seg_mod.decode_segmentation_flags is decode_segmentation_flags
