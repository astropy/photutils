# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the _moments module.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_equal

from photutils.utils._moments import _moments, _moments_central


def test_moments():
    """
    Test _moments with a simple 2x2 array.
    """
    data = np.array([[0, 1], [0, 1]])
    moments = _moments(data, order=2)
    result = np.array([[2, 2, 2], [1, 1, 1], [1, 1, 1]])

    assert_equal(moments, result)
    assert_allclose(moments[0, 1] / moments[0, 0], 1.0)
    assert_allclose(moments[1, 0] / moments[0, 0], 0.5)


def test_moments_central():
    """
    Test _moments_central with a simple 2x2 array.
    """
    data = np.array([[0, 1], [0, 1]])
    moments = _moments_central(data, order=2)
    result = np.array([[2.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
    assert_allclose(moments, result)


def test_moments_central_nonsquare():
    """
    Test _moments_central with a non-square array.
    """
    data = np.array([[0, 1], [0, 1], [0, 1]])
    moments = _moments_central(data, order=2)
    result = np.array([[3.0, 0.0, 0.0], [0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    assert_allclose(moments, result)


def test_moments_central_invalid_dim():
    """
    Test that _moments_central with non-2D data raises ValueError.
    """
    data = np.arange(27).reshape(3, 3, 3)
    match = 'data must be a 2D array'
    with pytest.raises(ValueError, match=match):
        _moments_central(data, order=3)


def test_moments_central_negative_order():
    """
    Test that _moments_central with negative order raises ValueError.
    """
    data = np.array([[0, 1], [0, 1]])
    match = 'order must be non-negative'
    with pytest.raises(ValueError, match=match):
        _moments_central(data, order=-1)
