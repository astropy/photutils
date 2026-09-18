# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the _moments module.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_equal

from photutils.utils._moments import (PIXEL_VARIANCE, centroid_from_moments,
                                      covariance_determinant,
                                      covariance_from_moments,
                                      covariance_min_eigval, image_moments,
                                      inertia_tensor_from_moments,
                                      is_singular_covariance)


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


@pytest.fixture
def covariances():
    """
    Raw ``(N, 2, 2)`` covariance matrices covering the shape cases.

    0. Resolved axis-aligned ellipse (det 4, min eigenvalue 1).
    1. Point-like, all zeros (det 0, min eigenvalue 0).
    2. Rank-1 degenerate along x (det 0, min eigenvalue 0).
    3. Elongated with one unresolved axis (det 0.2, min eigenvalue
       0.05, which is below 1/12 while det is above (1/12)**2).
    4. Not positive semidefinite (det -3, min eigenvalue -1).
    5. Partially non-finite.
    """
    return np.array([[[4.0, 0.0], [0.0, 1.0]],
                     [[0.0, 0.0], [0.0, 0.0]],
                     [[4.0, 0.0], [0.0, 0.0]],
                     [[4.0, 0.0], [0.0, 0.05]],
                     [[1.0, 2.0], [2.0, 1.0]],
                     [[np.nan, 0.0], [0.0, 1.0]]])


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


def test_covariance_determinant(covariances):
    """
    Test the determinants, including a NaN matrix without a warning.
    """
    det = covariance_determinant(covariances)
    assert det.shape == (6,)
    assert_allclose(det[:5], [4.0, 0.0, 0.0, 0.2, -3.0])
    assert np.isnan(det[5])


class TestCovarianceMinEigval:
    """
    Tests for covariance_min_eigval.
    """

    def test_values(self, covariances):
        """
        Test the closed-form smaller eigenvalue.
        """
        min_eig = covariance_min_eigval(covariances)
        assert min_eig.shape == (6,)
        assert_allclose(min_eig[:5], [1.0, 0.0, 0.0, 0.05, -1.0],
                        atol=1e-15)
        assert np.isnan(min_eig[5])

    def test_matches_eigvalsh(self):
        """
        Test the closed form against numpy for random matrices.
        """
        rng = np.random.default_rng(0)
        arr = rng.normal(size=(20, 2, 2))
        covar = arr @ arr.swapaxes(1, 2)  # symmetric
        expected = np.linalg.eigvalsh(covar)[:, 0]
        assert_allclose(covariance_min_eigval(covar), expected)

    def test_precomputed_determinant(self, covariances):
        """
        Test that a precomputed determinant gives the same result.
        """
        det = covariance_determinant(covariances)
        assert_equal(covariance_min_eigval(covariances, determinant=det),
                     covariance_min_eigval(covariances))


class TestIsSingularCovariance:
    """
    Tests for is_singular_covariance.
    """

    def test_determinant_only(self, covariances):
        """
        Test the determinant-only form.

        The elongated source 3 is not flagged. The source 4 with a
        negative determinant is flagged.
        """
        mask = is_singular_covariance(covariances,
                                      include_degenerate=False)
        assert mask.dtype == bool
        assert_equal(mask, [False, True, True, False, True, False])

    def test_include_degenerate(self, covariances):
        """
        Test that the elongated source 3 is also flagged.
        """
        mask = is_singular_covariance(covariances,
                                      include_degenerate=True)
        assert mask.dtype == bool
        assert_equal(mask, [False, True, True, True, True, False])

    @pytest.mark.parametrize('include_degenerate', [False, True])
    def test_threshold(self, include_degenerate):
        """
        Test that the threshold is the single-pixel variance.
        """
        covar = np.array([(PIXEL_VARIANCE - 1e-9) * np.eye(2),
                          (PIXEL_VARIANCE + 1e-9) * np.eye(2)])
        mask = is_singular_covariance(
            covar, include_degenerate=include_degenerate)
        assert_equal(mask, [True, False])
