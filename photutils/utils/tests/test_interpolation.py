# Licensed under a 3-clause BSD style license - see LICENSE.rst

import warnings

import numpy as np
from numpy.testing import assert_allclose
import pytest
from astropy.utils.exceptions import AstropyDeprecationWarning

from .. import ShepardIDWInterpolator as idw
from .. import interpolate_masked_data, mask_to_mirrored_num

try:
    import scipy    # noqa
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


SHAPE = (5, 5)
DATA = np.ones(SHAPE) * 2.0
MASK = np.zeros_like(DATA, dtype=bool)
MASK[2, 2] = True
ERROR = np.ones(SHAPE)
BACKGROUND = np.ones(SHAPE)
WRONG_SHAPE = np.ones((2, 2))


@pytest.mark.skipif('not HAS_SCIPY')
class TestShepardIDWInterpolator:
    def setup_class(self):
        np.random.seed(123)
        self.x = np.random.random(100)
        self.y = np.sin(self.x)
        self.f = idw(self.x, self.y)

    @pytest.mark.parametrize('positions', [0.4, np.arange(2, 5)*0.1])
    def test_idw_1d(self, positions):
        f = idw(self.x, self.y)
        assert_allclose(f(positions), np.sin(positions), atol=1e-2)

    def test_idw_weights(self):
        weights = self.y * 0.1
        f = idw(self.x, self.y, weights=weights)
        pos = 0.4
        assert_allclose(f(pos), np.sin(pos), atol=1e-2)

    def test_idw_2d(self):
        pos = np.random.rand(1000, 2)
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
        assert_allclose(self.f(0.5, n_neighbors=1), 0.48103656)

    def test_n_neighbors_negative(self):
        with pytest.raises(ValueError):
            self.f(0.5, n_neighbors=-1)

    def test_conf_dist_negative(self):
        assert_allclose(self.f(0.5, conf_dist=-1),
                        self.f(0.5, conf_dist=None))

    def test_dtype_none(self):
        result = self.f(0.5, dtype=None)
        assert result.dtype.type == np.float64

    def test_positions_0d_nomatch(self):
        """test when position ndim doesn't match coordinates ndim"""
        pos = np.random.rand(10, 2)
        val = np.sin(pos[:, 0] + pos[:, 1])
        f = idw(pos, val)
        with pytest.raises(ValueError):
            f(0.5)

    def test_positions_1d_nomatch(self):
        """test when position ndim doesn't match coordinates ndim"""
        pos = np.random.rand(10, 2)
        val = np.sin(pos[:, 0] + pos[:, 1])
        f = idw(pos, val)
        with pytest.raises(ValueError):
            f([0.5])

    def test_positions_3d(self):
        with pytest.raises(ValueError):
            self.f(np.ones((3, 3, 3)))


class TestInterpolateMaskedData:
    def setup_class(cls):
        """Ignore all deprecation warnings here."""
        warnings.simplefilter('ignore', AstropyDeprecationWarning)

    def teardown_class(cls):
        warnings.resetwarnings()

    def test_mask_shape(self):
        with pytest.raises(ValueError):
            interpolate_masked_data(DATA, WRONG_SHAPE)

    def test_error_shape(self):
        with pytest.raises(ValueError):
            interpolate_masked_data(DATA, MASK, error=WRONG_SHAPE)

    def test_background_shape(self):
        with pytest.raises(ValueError):
            interpolate_masked_data(DATA, MASK, background=WRONG_SHAPE)

    def test_interpolation(self):
        data2 = DATA.copy()
        data2[2, 2] = 100.
        error2 = ERROR.copy()
        error2[2, 2] = 100.
        background2 = BACKGROUND.copy()
        background2[2, 2] = 100.
        data, error, background = interpolate_masked_data(
            data2, MASK, error=error2, background=background2)
        assert_allclose(data, DATA)
        assert_allclose(error, ERROR)
        assert_allclose(background, BACKGROUND)

    def test_interpolation_larger_mask(self):
        data2 = DATA.copy()
        data2[2, 2] = 100.
        error2 = ERROR.copy()
        error2[2, 2] = 100.
        background2 = BACKGROUND.copy()
        background2[2, 2] = 100.
        mask2 = MASK.copy()
        mask2[1:4, 1:4] = True
        data, error, background = interpolate_masked_data(
            data2, MASK, error=error2, background=background2)
        assert_allclose(data, DATA)
        assert_allclose(error, ERROR)
        assert_allclose(background, BACKGROUND)


class TestMaskToMirroredNum:
    def test_mask_to_mirrored_num(self):
        """
        Test mask_to_mirrored_num.
        """
        center = (1.5, 1.5)
        data = np.arange(16).reshape(4, 4)
        mask = np.zeros_like(data, dtype=bool)
        mask[0, 0] = True
        mask[1, 1] = True
        data_ref = data.copy()
        data_ref[0, 0] = data[3, 3]
        data_ref[1, 1] = data[2, 2]
        mirror_data = mask_to_mirrored_num(data, mask, center)
        assert_allclose(mirror_data, data_ref, rtol=0, atol=1.e-6)

    def test_mask_to_mirrored_num_range(self):
        """
        Test mask_to_mirrored_num when mirrored pixels are outside of the
        image.
        """
        center = (2.5, 2.5)
        data = np.arange(16).reshape(4, 4)
        mask = np.zeros_like(data, dtype=bool)
        mask[0, 0] = True
        mask[1, 1] = True
        data_ref = data.copy()
        data_ref[0, 0] = 0.
        data_ref[1, 1] = 0.
        mirror_data = mask_to_mirrored_num(data, mask, center)
        assert_allclose(mirror_data, data_ref, rtol=0, atol=1.e-6)

    def test_mask_to_mirrored_num_masked(self):
        """
        Test mask_to_mirrored_num when mirrored pixels are also masked.
        """
        center = (0.5, 0.5)
        data = np.arange(16).reshape(4, 4)
        data[0, 0] = 100
        mask = np.zeros_like(data, dtype=bool)
        mask[0, 0] = True
        mask[1, 1] = True
        data_ref = data.copy()
        data_ref[0, 0] = 0.
        data_ref[1, 1] = 0.
        mirror_data = mask_to_mirrored_num(data, mask, center)
        assert_allclose(mirror_data, data_ref, rtol=0, atol=1.e-6)

    def test_mask_to_mirrored_num_bbox(self):
        """
        Test mask_to_mirrored_num with a bounding box.
        """
        center = (1.5, 1.5)
        data = np.arange(16).reshape(4, 4)
        data[0, 0] = 100
        mask = np.zeros_like(data, dtype=bool)
        mask[0, 0] = True
        mask[1, 1] = True
        data_ref = data.copy()
        data_ref[1, 1] = data[2, 2]
        bbox = (1, 2, 1, 2)
        mirror_data = mask_to_mirrored_num(data, mask, center, bbox=bbox)
        assert_allclose(mirror_data, data_ref, rtol=0, atol=1.e-6)
