# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the non_parametric module.
"""

import numpy as np
import pytest

from photutils.morphology.non_parametric import gini


def test_gini():
    """
    Test Gini coefficient calculation with simple cases.
    """
    data_evenly_distributed = np.ones((100, 100))
    data_point_like = np.zeros((100, 100))
    data_point_like[50, 50] = 1

    assert gini(data_evenly_distributed) == 0.0
    assert gini(data_point_like) == 1.0


def test_gini_1d():
    """
    Test Gini coefficient with 1D input.
    """
    assert gini(np.ones(100)) == 0.0
    data_1d = np.zeros(100)
    data_1d[50] = 1
    assert gini(data_1d) == 1.0


def test_gini_mask():
    """
    Test Gini coefficient calculation with a mask.
    """
    shape = (100, 100)
    data1 = np.ones(shape)
    data1[50, 50] = 0
    mask1 = np.zeros(data1.shape, dtype=bool)
    mask1[50, 50] = True

    data2 = np.zeros(shape)
    data2[50, 50] = 1
    data2[0, 0] = 100
    mask2 = np.zeros(data2.shape, dtype=bool)
    mask2[0, 0] = True

    assert gini(data1, mask=mask1) == 0.0
    assert gini(data2, mask=mask2) == 1.0


def test_gini_mask_invalid_shape():
    """
    Test that mask must have the same shape as data.
    """
    data = np.ones((10, 10))
    mask_wrong_shape = np.zeros((5, 5), dtype=bool)
    match = 'mask must have the same shape'
    with pytest.raises(ValueError, match=match):
        gini(data, mask=mask_wrong_shape)


def test_gini_invalid_values_filtered():
    """
    Test that NaN and inf are automatically excluded.
    """
    # All valid: point-like
    data = np.zeros((5, 5))
    data[2, 2] = 1.0
    assert gini(data) == 1.0

    # Same with NaNs in other pixels - should get same result
    data_nan = data.astype(float)
    data_nan[0, 0] = np.nan
    data_nan[1, 1] = np.nan
    assert gini(data_nan) == 1.0

    # Same with inf in other pixels - should get same result
    data_inf = data.astype(float)
    data_inf[0, 0] = np.inf
    data_inf[1, 1] = -np.inf
    assert gini(data_inf) == 1.0

    # All NaN returns nan
    assert np.isnan(gini(np.full((5, 5), np.nan)))

    # All inf returns nan (no finite values)
    assert np.isnan(gini(np.full((5, 5), np.inf)))

    # Mix: one valid pixel, rest NaN - Gini of single value is 0
    data_one = np.full((5, 5), np.nan)
    data_one[2, 2] = 1.0
    assert gini(data_one) == 0.0


def test_gini_all_zeros():
    """
    Test that an all-zero array returns 0.0 (normalization
    early-return).
    """
    assert gini(np.zeros((100, 100))) == 0.0
    assert gini(np.zeros(10)) == 0.0


def test_gini_bounded():
    """
    Test that Gini coefficient is in [0, 1] for diverse inputs.
    """
    rng = np.random.default_rng(seed=0)

    # Uniform random values in (0, 1) — strictly between extremes
    result = gini(rng.random((50, 50)))
    assert 0.0 < result < 1.0

    # Mixed-sign data must also be bounded
    data_mixed_sign = np.array([-4.0, 1.0, 1.0, 1.0])
    result_mixed = gini(data_mixed_sign)
    assert 0.0 <= result_mixed <= 1.0

    # Gradient array — monotonically increasing, result in (0, 1)
    result_grad = gini(np.arange(1.0, 101.0))
    assert 0.0 < result_grad < 1.0


def test_gini_negative_values():
    """
    Test that negative pixel values are treated via their absolute value
    per the Lotz et al. formula.
    """
    # Negating all values must give the same result because only |x_i|
    # and |mean| enter the formula
    data_pos = np.array([1.0, 2.0, 3.0])
    data_neg = -data_pos
    assert gini(data_neg) == gini(data_pos)

    # Mixed sign: result must be in [0, 1]
    data_mixed = np.array([-5.0, -1.0, 0.0, 2.0, 4.0])
    result = gini(data_mixed)
    assert 0.0 <= result <= 1.0

    # 2D array with negative values
    data_2d = np.array([[-3.0, -1.0], [0.0, 1.0]])
    result_2d = gini(data_2d)
    assert 0.0 <= result_2d <= 1.0
