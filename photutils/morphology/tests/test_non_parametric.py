# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the non_parametric module.
"""

import numpy as np

from photutils.morphology.non_parametric import gini


def test_gini():
    """
    Test Gini coefficient calculation.
    """
    data_evenly_distributed = np.ones((100, 100))
    data_point_like = np.zeros((100, 100))
    data_point_like[50, 50] = 1

    assert gini(data_evenly_distributed) == 0.0
    assert gini(data_point_like) == 1.0


def test_gini_mask():
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


def test_gini_normalization():
    data = np.zeros((100, 100))
    assert gini(data) == 0.0
