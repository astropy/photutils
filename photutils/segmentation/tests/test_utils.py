# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the _utils module.
"""

import numpy as np
from numpy.testing import assert_allclose, assert_equal

from photutils.segmentation.utils import (_make_binary_structure,
                                          _mask_to_mirrored_value,
                                          make_2dgaussian_kernel)


def test_make_2dgaussian_kernel():
    """
    Test make 2dgaussian kernel.
    """
    kernel = make_2dgaussian_kernel(1.0, size=3)
    expected = np.array([[0.01411809, 0.0905834, 0.01411809],
                         [0.0905834, 0.58119403, 0.0905834],
                         [0.01411809, 0.0905834, 0.01411809]])
    assert_allclose(kernel.array, expected, atol=1.0e-6)
    assert_allclose(kernel.array.sum(), 1.0)


def test_make_2dgaussian_kernel_modes():
    """
    Test make 2dgaussian kernel modes.
    """
    kernel = make_2dgaussian_kernel(3.0, 5)
    assert_allclose(kernel.array.sum(), 1.0)

    kernel = make_2dgaussian_kernel(3.0, 5, mode='center')
    assert_allclose(kernel.array.sum(), 1.0)

    kernel = make_2dgaussian_kernel(3.0, 5, mode='linear_interp')
    assert_allclose(kernel.array.sum(), 1.0)

    kernel = make_2dgaussian_kernel(3.0, 5, mode='integrate')
    assert_allclose(kernel.array.sum(), 1.0)


def test_make_binary_structure():
    """
    Test make binary structure.
    """
    footprint = _make_binary_structure(1, 4)
    assert_allclose(footprint, np.array([1, 1, 1]))

    footprint = _make_binary_structure(3, 4)
    assert_equal(footprint[0, 0], np.array([False, False, False]))
    expected = np.array([[[0, 0, 0],
                          [0, 1, 0],
                          [0, 0, 0]],
                         [[0, 1, 0],
                          [1, 1, 1],
                          [0, 1, 0]],
                         [[0, 0, 0],
                          [0, 1, 0],
                          [0, 0, 0]]])
    assert_equal(footprint.astype(int), expected)


def test_mask_to_mirrored_value():
    """
    Test mask to mirrored value.
    """
    center = (2.0, 2.0)
    data = np.arange(25).reshape(5, 5)
    mask = np.zeros(data.shape, dtype=bool)
    mask[0, 0] = True
    mask[1, 1] = True
    data_ref = data.copy()
    data_ref[0, 0] = data[4, 4]
    data_ref[1, 1] = data[3, 3]
    mirror_data = _mask_to_mirrored_value(data, mask, center)
    assert_allclose(mirror_data, data_ref, rtol=0, atol=1.0e-6)


def test_mask_to_mirrored_value_range():
    """
    Test mask_to_mirrored_value when mirrored pixels are outside the
    image.
    """
    center = (3.0, 3.0)
    data = np.arange(25).reshape(5, 5)
    mask = np.zeros(data.shape, dtype=bool)
    mask[0, 0] = True
    mask[1, 1] = True
    mask[2, 2] = True
    data_ref = data.copy()
    data_ref[0, 0] = 0.0
    data_ref[1, 1] = 0.0
    data_ref[2, 2] = data[4, 4]
    mirror_data = _mask_to_mirrored_value(data, mask, center)
    assert_allclose(mirror_data, data_ref, rtol=0, atol=1.0e-6)


def test_mask_to_mirrored_value_masked():
    """
    Test mask_to_mirrored_value when mirrored pixels are also in the
    ``replace_mask``.
    """
    center = (2.0, 2.0)
    data = np.arange(25).reshape(5, 5)
    mask = np.zeros(data.shape, dtype=bool)
    mask[0, 0] = True
    mask[1, 1] = True
    mask[3, 3] = True
    mask[4, 4] = True
    data_ref = data.copy()
    data_ref[0, 0] = 0.0
    data_ref[1, 1] = 0.0
    data_ref[3, 3] = 0.0
    data_ref[4, 4] = 0.0
    mirror_data = _mask_to_mirrored_value(data, mask, center)
    mirror_data = _mask_to_mirrored_value(data, mask, center)
    assert_allclose(mirror_data, data_ref, rtol=0, atol=1.0e-6)


def test_mask_to_mirrored_value_mask_keyword():
    """
    Test mask_to_mirrored_value when mirrored pixels are masked (via the
    mask keyword).
    """
    center = (2.0, 2.0)
    data = np.arange(25.0).reshape(5, 5)
    replace_mask = np.zeros(data.shape, dtype=bool)
    mask = np.zeros(data.shape, dtype=bool)
    replace_mask[0, 2] = True
    data[4, 2] = np.nan
    mask[4, 2] = True
    result = _mask_to_mirrored_value(data, replace_mask, center, mask=mask)
    assert result[0, 2] == 0


def test_mask_to_mirrored_value_return_uncorrected():
    """
    Test that return_uncorrected reports pixels whose mirror was outside
    the image.
    """
    center = (2.0, 2.0)
    data = np.arange(16.0).reshape(4, 4)
    replace_mask = np.zeros(data.shape, dtype=bool)
    replace_mask[0, 0] = True  # mirror at (4, 4) is outside the image
    replace_mask[1, 2] = True  # mirror at (3, 2) is available
    result, uncorrected = _mask_to_mirrored_value(
        data, replace_mask, center, return_uncorrected=True)
    assert uncorrected[0, 0]
    assert not uncorrected[1, 2]
    assert np.count_nonzero(uncorrected) == 1
    assert result[0, 0] == 0
    assert result[1, 2] == data[3, 2]

    # The two-value return preserves the corrected data
    result_only = _mask_to_mirrored_value(data, replace_mask, center)
    assert_equal(result, result_only)


def test_mask_to_mirrored_value_return_uncorrected_masked():
    """
    Test that return_uncorrected reports pixels whose mirror is excluded
    by the ``mask`` keyword or is itself in ``replace_mask``.
    """
    center = (2.0, 2.0)
    data = np.arange(25.0).reshape(5, 5)
    replace_mask = np.zeros(data.shape, dtype=bool)
    mask = np.zeros(data.shape, dtype=bool)
    replace_mask[0, 2] = True  # mirror at (4, 2) is in mask
    replace_mask[1, 1] = True  # mirror at (3, 3) is in replace_mask
    replace_mask[3, 3] = True
    mask[4, 2] = True
    result, uncorrected = _mask_to_mirrored_value(
        data, replace_mask, center, mask=mask, return_uncorrected=True)
    assert_equal(np.nonzero(uncorrected),
                 (np.array([0, 1, 3]), np.array([2, 1, 3])))
    assert result[0, 2] == 0
    assert result[1, 1] == 0
    assert result[3, 3] == 0
