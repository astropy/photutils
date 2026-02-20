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


def test_gini_normalization():
    """
    Test that Gini coefficient is normalized between 0 and 1.
    """
    data = np.zeros((100, 100))
    assert gini(data) == 0.0
