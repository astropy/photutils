# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the interpolation module.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from photutils.utils import ShepardIDWInterpolator as idw
from photutils.utils._optional_deps import HAS_SCIPY

SHAPE = (5, 5)
DATA = np.ones(SHAPE) * 2.0
MASK = np.zeros(DATA.shape, dtype=bool)
MASK[2, 2] = True
ERROR = np.ones(SHAPE)
BACKGROUND = np.ones(SHAPE)
WRONG_SHAPE = np.ones((2, 2))


@pytest.mark.skipif(not HAS_SCIPY, reason='scipy is required')
class TestShepardIDWInterpolator:
    def setup_class(self):
        self.rng = np.random.default_rng(0)
        self.x = self.rng.random(100)
        self.y = np.sin(self.x)
        self.f = idw(self.x, self.y)

    @pytest.mark.parametrize('positions', [0.4, np.arange(2, 5) * 0.1])
    def test_idw_1d(self, positions):
        f = idw(self.x, self.y)
        assert_allclose(f(positions), np.sin(positions), atol=1e-2)

    def test_idw_weights(self):
        weights = self.y * 0.1
        f = idw(self.x, self.y, weights=weights)
        pos = 0.4
        assert_allclose(f(pos), np.sin(pos), atol=1e-2)

    def test_idw_2d(self):
        pos = self.rng.random((1000, 2))
        val = np.sin(pos[:, 0] + pos[:, 1])
        f = idw(pos, val)
        x = 0.5
        y = 0.6
        assert_allclose(f([x, y]), np.sin(x + y), atol=1e-2)

    def test_idw_3d(self):
        val = np.ones((3, 3, 3))
        pos = np.indices(val.shape)
        f = idw(pos, val)
        assert_allclose(f([0.5, 0.5, 0.5]), 1.0)

    def test_no_coordinates(self):
        with pytest.raises(ValueError):
            idw([], 0)

    def test_values_invalid_shape(self):
        with pytest.raises(ValueError):
            idw(self.x, 0)

    def test_weights_invalid_shape(self):
        with pytest.raises(ValueError):
            idw(self.x, self.y, weights=10)

    def test_weights_negative(self):
        with pytest.raises(ValueError):
            idw(self.x, self.y, weights=-self.y)

    def test_n_neighbors_one(self):
        assert_allclose(self.f(0.5, n_neighbors=1), [0.479334], rtol=3e-7)

    def test_n_neighbors_negative(self):
        with pytest.raises(ValueError):
            self.f(0.5, n_neighbors=-1)

    def test_conf_dist_negative(self):
        assert_allclose(self.f(0.5, conf_dist=-1),
                        self.f(0.5, conf_dist=None))

    def test_dtype_none(self):
        result = self.f(0.5, dtype=None)
        assert result.dtype == float

    def test_positions_0d_nomatch(self):
        """test when position ndim doesn't match coordinates ndim"""
        pos = self.rng.random((10, 2))
        val = np.sin(pos[:, 0] + pos[:, 1])
        f = idw(pos, val)
        with pytest.raises(ValueError):
            f(0.5)

    def test_positions_1d_nomatch(self):
        """test when position ndim doesn't match coordinates ndim"""
        pos = self.rng.random((10, 2))
        val = np.sin(pos[:, 0] + pos[:, 1])
        f = idw(pos, val)
        with pytest.raises(ValueError):
            f([0.5])

    def test_positions_3d(self):
        with pytest.raises(ValueError):
            self.f(np.ones((3, 3, 3)))

    def test_scalar_values_1d(self):
        value = 10.0
        f = idw(2, value)
        assert_allclose(f(2), value)
        assert_allclose(f(-1), value)
        assert_allclose(f(0), value)
        assert_allclose(f(142), value)

    def test_scalar_values_2d(self):
        value = 10.0
        f = idw([[1, 2]], value)
        assert_allclose(f([1, 2]), value)
        assert_allclose(f([-1, 0]), value)
        assert_allclose(f([142, 213]), value)

    def test_scalar_values_3d(self):
        value = 10.0
        f = idw([[7, 4, 1]], value)
        assert_allclose(f([7, 4, 1]), value)
        assert_allclose(f([-1, 0, 7]), value)
        assert_allclose(f([142, 213, 5]), value)
