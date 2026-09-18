# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the _moments module.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_equal

from photutils.utils._moments import (PIXEL_VARIANCE, centroid_from_moments,
                                      covariance_from_moments, image_moments,
                                      inertia_tensor_from_moments)


@pytest.fixture
def moments_central():
    """
    Central moments for four sources with a leading source axis.

    Index ``[:, i, j]`` is the sum of ``y**i * x**j``. The sources are:

    0. An axis-aligned ellipse with x variance 4 and y variance 1.
    1. A unit-variance source with covariance 0.5 (45 degree tilt).
    2. A source with zero total flux (undefined covariance).
    3. A point-like source with zero second moments.
    """
    moments = np.zeros((4, 3, 3))
    moments[0, 0, 0] = 2.0
    moments[0, 0, 2] = 8.0
    moments[0, 2, 0] = 2.0
    moments[1, 0, 0] = 1.0
    moments[1, 0, 2] = 1.0
    moments[1, 2, 0] = 1.0
    moments[1, 1, 1] = 0.5
    moments[2, 0, 2] = 1.0
    moments[3, 0, 0] = 1.0
    return moments


def test_moments():
    """
    Test image_moments with a simple 2x2 array (raw moments).
    """
    data = np.array([[0, 1], [0, 1]])
    moments = image_moments(data, order=2)
    result = np.array([[2, 2, 2], [1, 1, 1], [1, 1, 1]])

    assert_equal(moments, result)
    assert_allclose(moments[0, 1] / moments[0, 0], 1.0)
    assert_allclose(moments[1, 0] / moments[0, 0], 0.5)


def test_moments_central():
    """
    Test image_moments with center=None (central moments).
    """
    data = np.array([[0, 1], [0, 1]])
    moments = image_moments(data, center=None, order=2)
    result = np.array([[2.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
    assert_allclose(moments, result)


def test_moments_central_nonsquare():
    """
    Test image_moments with center=None and a non-square array.
    """
    data = np.array([[0, 1], [0, 1], [0, 1]])
    moments = image_moments(data, center=None, order=2)
    result = np.array([[3.0, 0.0, 0.0], [0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    assert_allclose(moments, result)


def test_moments_central_invalid_dim():
    """
    Test that image_moments with non-2D data raises ValueError.
    """
    data = np.arange(27).reshape(3, 3, 3)
    match = 'data must be a 2D array'
    with pytest.raises(ValueError, match=match):
        image_moments(data, order=3)


def test_moments_central_negative_order():
    """
    Test that image_moments with negative order raises ValueError.
    """
    data = np.array([[0, 1], [0, 1]])
    match = 'order must be non-negative'
    with pytest.raises(ValueError, match=match):
        image_moments(data, order=-1)


def test_pixel_variance():
    """
    Test the single-pixel variance constant.
    """
    assert PIXEL_VARIANCE == 1.0 / 12.0


class TestCentroidFromMoments:
    """
    Tests for centroid_from_moments.
    """

    def test_values(self):
        """
        Test the (x, y) order against raw image moments.
        """
        data = np.zeros((5, 7))
        data[1, 4] = 1.0
        data[3, 4] = 3.0
        moments = np.array([image_moments(data, order=1)])
        centroid = centroid_from_moments(moments)
        assert centroid.shape == (1, 2)
        assert_allclose(centroid, [[4.0, 2.5]])

    def test_zero_flux(self):
        """
        Test that zero total flux gives NaN without a warning.
        """
        centroid = centroid_from_moments(np.zeros((2, 2, 2)))
        assert centroid.shape == (2, 2)
        assert np.all(np.isnan(centroid))


class TestInertiaTensorFromMoments:
    """
    Tests for inertia_tensor_from_moments.
    """

    def test_values(self, moments_central):
        """
        Test the element layout and the sign of the cross term.
        """
        tensor = inertia_tensor_from_moments(moments_central)
        assert tensor.shape == (4, 2, 2)
        assert_allclose(tensor[0], [[8.0, 0.0], [0.0, 2.0]])
        assert_allclose(tensor[1], [[1.0, -0.5], [-0.5, 1.0]])
        assert_allclose(tensor[2], [[1.0, 0.0], [0.0, 0.0]])
        assert_allclose(tensor[3], [[0.0, 0.0], [0.0, 0.0]])


class TestCovarianceFromMoments:
    """
    Tests for covariance_from_moments.
    """

    def test_values(self, moments_central):
        """
        Test the element layout and the flux normalization.
        """
        covar = covariance_from_moments(moments_central)
        assert covar.shape == (4, 2, 2)
        assert_allclose(covar[0], [[4.0, 0.0], [0.0, 1.0]])
        assert_allclose(covar[1], [[1.0, 0.5], [0.5, 1.0]])
        assert_allclose(covar[3], [[0.0, 0.0], [0.0, 0.0]])
        assert_equal(covar[:, 0, 1], covar[:, 1, 0])

    def test_zero_flux(self, moments_central):
        """
        Test that zero total flux is non-finite without a warning.
        """
        covar = covariance_from_moments(moments_central)
        assert not np.any(np.isfinite(covar[2]))

    def test_higher_order_moments_ignored(self, moments_central):
        """
        Test that third-order and higher moments are not used.
        """
        moments = np.full((4, 5, 5), 99.0)
        moments[:, :3, :3] = moments_central
        assert_equal(covariance_from_moments(moments),
                     covariance_from_moments(moments_central))
