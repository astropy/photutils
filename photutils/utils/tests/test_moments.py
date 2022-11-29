# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the moments module.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_equal

from photutils.utils._moments import _moments, _moments_central


def test_moments():
    data = np.array([[0, 1], [0, 1]])
    moments = _moments(data, order=2)
    result = np.array([[2, 2, 2], [1, 1, 1], [1, 1, 1]])

    assert_equal(moments, result)
    assert_allclose(moments[0, 1] / moments[0, 0], 1.0)
    assert_allclose(moments[1, 0] / moments[0, 0], 0.5)


def test_moments_central():
    data = np.array([[0, 1], [0, 1]])
    moments = _moments_central(data, order=2)
    result = np.array([[2.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
    assert_allclose(moments, result)


def test_moments_central_nonsquare():
    data = np.array([[0, 1], [0, 1], [0, 1]])
    moments = _moments_central(data, order=2)
    result = np.array([[3.0, 0.0, 0.0], [0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    assert_allclose(moments, result)


def test_moments_central_invalid_dim():
    data = np.arange(27).reshape(3, 3, 3)
    with pytest.raises(ValueError):
        _moments_central(data, order=3)
