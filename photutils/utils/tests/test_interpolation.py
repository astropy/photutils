# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the interpolation module.
"""

from unittest.mock import MagicMock

import numpy as np
import pytest
from numpy.testing import assert_allclose

from photutils.utils import ShepardIDWInterpolator as IDWInterp

SHAPE = (5, 5)
DATA = np.ones(SHAPE) * 2.0
MASK = np.zeros(DATA.shape, dtype=bool)
MASK[2, 2] = True
ERROR = np.ones(SHAPE)
BACKGROUND = np.ones(SHAPE)
WRONG_SHAPE = np.ones((2, 2))


class TestShepardIDWInterpolator:
    def setup_class(self):
        self.rng = np.random.default_rng(0)
        self.x = self.rng.random(100)
        self.y = np.sin(self.x)
        self.f = IDWInterp(self.x, self.y)

    @pytest.mark.parametrize('positions', [0.4, np.arange(2, 5) * 0.1])
    def test_idw_1d(self, positions):
        """
        Test 1D IDW interpolation.
        """
        f = IDWInterp(self.x, self.y)
        assert_allclose(f(positions), np.sin(positions), atol=1e-2)

    def test_idw_weights(self):
        """
        Test IDW interpolation with weights.
        """
        weights = self.y * 0.1
        f = IDWInterp(self.x, self.y, weights=weights)
        pos = 0.4
        assert_allclose(f(pos), np.sin(pos), atol=1e-2)

    def test_idw_2d(self):
        """
        Test 2D IDW interpolation.
        """
        pos = self.rng.random((1000, 2))
        val = np.sin(pos[:, 0] + pos[:, 1])
        f = IDWInterp(pos, val)
        x = 0.5
        y = 0.6
        assert_allclose(f([x, y]), np.sin(x + y), atol=1e-2)

    def test_idw_3d(self):
        """
        Test 3D IDW interpolation.
        """
        val = np.ones((3, 3, 3))
        pos = np.indices(val.shape)
        f = IDWInterp(pos, val)
        assert_allclose(f([0.5, 0.5, 0.5]), 1.0)

    def test_no_coordinates(self):
        """
        Test that empty coordinates raises ValueError.
        """
        match = 'coordinates must have at least one data point'
        with pytest.raises(ValueError, match=match):
            IDWInterp([], 0)

    def test_values_invalid_shape(self):
        """
        Test that mismatched values shape raises ValueError.
        """
        match = 'The number of values must match the number of coordinates'
        with pytest.raises(ValueError, match=match):
            IDWInterp(self.x, 0)

    def test_weights_invalid_shape(self):
        """
        Test that mismatched weights shape raises ValueError.
        """
        match = 'number of weights must match the number of coordinates'
        with pytest.raises(ValueError, match=match):
            IDWInterp(self.x, self.y, weights=10)

    def test_weights_negative(self):
        """
        Test that negative weights raises ValueError.
        """
        match = 'All weight values must be non-negative numbers'
        with pytest.raises(ValueError, match=match):
            IDWInterp(self.x, self.y, weights=-self.y)

    def test_n_neighbors_one(self):
        """
        Test IDW interpolation with n_neighbors=1.
        """
        result = self.f(0.5, n_neighbors=1)
        assert np.isscalar(result)
        assert_allclose(result, 0.479334, rtol=3e-7)

    def test_n_neighbors_negative(self):
        """
        Test that negative n_neighbors raises ValueError.
        """
        match = 'n_neighbors must be a positive integer'
        with pytest.raises(ValueError, match=match):
            self.f(0.5, n_neighbors=-1)

    def test_conf_dist_negative(self):
        """
        Test IDW interpolation with negative conf_dist.
        """
        assert_allclose(self.f(0.5, conf_dist=-1),
                        self.f(0.5, conf_dist=None))

    def test_dtype_none(self):
        """
        Test IDW interpolation with dtype=None.
        """
        result = self.f(0.5, dtype=None)
        assert result.dtype == float

    def test_positions_0d_nomatch(self):
        """
        Test that a 0D position with wrong dimensionality raises
        ValueError.
        """
        pos = self.rng.random((10, 2))
        val = np.sin(pos[:, 0] + pos[:, 1])
        f = IDWInterp(pos, val)
        match = 'position does not match the dimensionality'
        with pytest.raises(ValueError, match=match):
            f(0.5)

    def test_positions_1d_nomatch(self):
        """
        Test that a 1D position with wrong length raises ValueError.
        """
        pos = self.rng.random((10, 2))
        val = np.sin(pos[:, 0] + pos[:, 1])
        f = IDWInterp(pos, val)
        match = 'was provided as a 1D array, but its length does not match'
        with pytest.raises(ValueError, match=match):
            f([0.5])

    def test_positions_3d(self):
        """
        Test that a 3D positions array raises ValueError.
        """
        match = 'array_like object of dimensionality no larger than 2'
        with pytest.raises(ValueError, match=match):
            self.f(np.ones((3, 3, 3)))

    def test_scalar_values_1d(self):
        """
        Test IDW interpolation with a single 1D data point.
        """
        value = 10.0
        f = IDWInterp(2, value)
        assert_allclose(f(2), value)
        assert_allclose(f(-1), value)
        assert_allclose(f(0), value)
        assert_allclose(f(142), value)

    def test_scalar_values_2d(self):
        """
        Test IDW interpolation with a single 2D data point.
        """
        value = 10.0
        f = IDWInterp([[1, 2]], value)
        assert_allclose(f([1, 2]), value)
        assert_allclose(f([-1, 0]), value)
        assert_allclose(f([142, 213]), value)

    def test_scalar_values_3d(self):
        """
        Test IDW interpolation with a single 3D data point.
        """
        value = 10.0
        f = IDWInterp([[7, 4, 1]], value)
        assert_allclose(f([7, 4, 1]), value)
        assert_allclose(f([-1, 0, 7]), value)
        assert_allclose(f([142, 213, 5]), value)

    def test_no_valid_distances(self):
        """
        Test that NaN is returned when all distances are non-finite
        (dk.shape[0] == 0 after filtering).
        """
        coords = np.array([0.0, 1.0, 2.0])
        values = np.array([10.0, 20.0, 30.0])
        f = IDWInterp(coords, values)

        # Replace the kdtree with a mock that returns all-inf distances
        mock_kdtree = MagicMock()
        inf_distances = np.array([[np.inf, np.inf, np.inf]])
        inf_idx = np.array([[0, 1, 2]])
        mock_kdtree.query.return_value = (inf_distances, inf_idx)
        f.kdtree = mock_kdtree

        result = f(0.5, n_neighbors=3)
        assert np.isnan(result)

    def test_zero_weights(self):
        """
        Test that NaN is returned when all weights are zero, leading
        to wtot == 0.
        """
        coords = np.array([0.0, 1.0, 2.0])
        values = np.array([10.0, 20.0, 30.0])
        weights = np.array([0.0, 0.0, 0.0])
        f = IDWInterp(coords, values, weights=weights)
        result = f(0.5, n_neighbors=3)
        assert np.isnan(result)
