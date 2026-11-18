# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the morans_i module.
"""

import astropy.units as u
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_equal

from photutils.background import make_morans_i_map


def _slow_box_sum(array, size):
    """
    Sum an array in a square box centered on each pixel.
    """
    half = size // 2
    result = np.zeros(array.shape)
    for iy in range(array.shape[0]):
        for ix in range(array.shape[1]):
            result[iy, ix] = array[max(iy - half, 0):iy + half + 1,
                                   max(ix - half, 0):ix + half + 1].sum()
    return result


def _reference_morans_i(data, patch_size, *, mask=None, weights=None):
    """
    Slow pure-NumPy reference for the sliding Moran's I statistic.
    """
    data = np.asarray(data, dtype=float)
    valid = np.isfinite(data)
    if mask is not None:
        valid &= ~mask
    if weights is None:
        weights = np.ones(data.shape)
    with np.errstate(invalid='ignore'):
        valid &= np.isfinite(weights) & (weights > 0)
    wts = np.where(valid, weights, 0.0)
    values = np.where(valid, data, 0.0)
    ny, nx = data.shape

    neighbor_mean = np.zeros(data.shape)
    usable = np.zeros(data.shape, dtype=bool)
    for iy in range(ny):
        for ix in range(nx):
            if not valid[iy, ix]:
                continue
            slc = (slice(max(iy - 1, 0), iy + 2),
                   slice(max(ix - 1, 0), ix + 2))
            neighbor_wts = wts[slc].copy()
            neighbor_wts[iy - slc[0].start, ix - slc[1].start] = 0.0
            if neighbor_wts.sum() > 0:
                neighbor_mean[iy, ix] = ((neighbor_wts * values[slc]).sum()
                                         / neighbor_wts.sum())
                usable[iy, ix] = True
    wts[~usable] = 0.0

    sums = [usable.astype(float), wts, wts * values, wts * values**2,
            wts * neighbor_mean, wts * values * neighbor_mean]
    sums = [_slow_box_sum(array, patch_size) for array in sums]
    count, sum_w, sum_x, sum_xx, sum_n, sum_xn = sums

    result = np.full(data.shape, np.nan)
    good = usable & (count >= 0.25 * patch_size**2)
    with np.errstate(invalid='ignore', divide='ignore'):
        mean = sum_x / sum_w
        var = sum_xx / sum_w - mean**2
        good &= var > 1.0e-10 * sum_xx / sum_w
        stat = (sum_xn / sum_w - mean * sum_n / sum_w) / var
    result[good] = stat[good]
    return result


@pytest.fixture(name='noise')
def fixture_noise():
    rng = np.random.default_rng(0)
    return rng.normal(0.0, 1.0, (201, 251))


@pytest.fixture(name='small')
def fixture_small():
    rng = np.random.default_rng(1)
    data = rng.normal(0.0, 1.0, (40, 50))
    data[15:22, 20:30] += 4.0
    return data


class TestMakeMoransIMap:
    def test_reference(self, small):
        """
        Test against a slow reference, including the image edges.
        """
        result = make_morans_i_map(small, patch_size=7)
        assert result.shape == small.shape
        assert result.dtype == np.float32
        assert np.all(np.isfinite(result))
        assert_allclose(result, _reference_morans_i(small, 7), atol=1e-5)

    def test_reference_masked(self, small):
        """
        Test against a slow reference with masked and nonfinite pixels.
        """
        data = small.copy()
        data[5, 5] = np.nan
        data[30, 40] = np.inf
        mask = np.zeros(data.shape, dtype=bool)
        mask[10:14, 10:14] = True
        mask[0:8, 44:] = True  # most of a corner patch

        result = make_morans_i_map(data, patch_size=7, mask=mask)
        expected = _reference_morans_i(data, 7, mask=mask)
        assert_equal(np.isnan(result), np.isnan(expected))
        assert_allclose(result, expected, atol=1e-5)
        assert np.all(np.isnan(result[mask]))
        assert np.isnan(result[5, 5])
        assert np.isnan(result[30, 40])
        assert np.isnan(result[0, 49])

    def test_weights(self, small):
        """
        Test weighted data against a slow reference, including pixels
        that are excluded by their weights.
        """
        rng = np.random.default_rng(5)
        weights = rng.uniform(0.1, 3.0, small.shape)
        weights[3, 4] = 0.0
        weights[20, 30] = -1.0
        weights[25, 10] = np.nan
        weights[30:34, 20:24] = 0.0
        weights[31, 21] = 0.3  # no neighbors with a positive weight
        mask = np.zeros(small.shape, dtype=bool)
        mask[10:14, 10:14] = True

        result = make_morans_i_map(small, patch_size=7, mask=mask,
                                   weights=weights)
        expected = _reference_morans_i(small, 7, mask=mask, weights=weights)
        assert_equal(np.isnan(result), np.isnan(expected))
        assert_allclose(result, expected, atol=1e-5)
        for idx in ((3, 4), (20, 30), (25, 10), (31, 21), (11, 11)):
            assert np.isnan(result[idx])
        assert not np.allclose(result, make_morans_i_map(small, patch_size=7),
                               atol=1e-3, equal_nan=True)

    def test_weights_scale(self, small):
        """
        Test that the result does not depend on the weight scale and
        that equal weights match the unweighted result.
        """
        expected = make_morans_i_map(small, patch_size=7)
        for scale in (1.0, 1.0e-12, 3.0e12):
            weights = np.full(small.shape, scale)
            result = make_morans_i_map(small, patch_size=7, weights=weights)
            assert_allclose(result, expected, atol=1e-5)

    def test_patch_larger_than_image(self, small):
        data = small[:12, :9]
        result = make_morans_i_map(data, patch_size=21)
        assert_allclose(result, _reference_morans_i(data, 21), atol=1e-5)

    def test_noise_and_source(self, noise):
        """
        Test that uncorrelated noise gives values near zero and that a
        source gives a large positive value.
        """
        result = make_morans_i_map(noise)
        assert_allclose(np.mean(result), -1.0 / 440.0, atol=0.003)
        assert_allclose(np.std(result), 0.023, atol=0.004)

        data = noise.copy()
        yy, xx = np.mgrid[:201, :251]
        data += 10.0 * np.exp(-((xx - 120)**2 + (yy - 100)**2) / 18.0)
        result = make_morans_i_map(data)
        assert result[100, 120] > 0.5

    def test_shift_and_scale_invariance(self, noise):
        expected = make_morans_i_map(noise)
        assert_allclose(make_morans_i_map(noise + 1.0e6), expected, atol=1e-5)
        assert_allclose(make_morans_i_map(noise * 1.0e-8 - 3.0), expected,
                        atol=1e-5)

    def test_float32(self, noise):
        expected = make_morans_i_map(noise.astype(np.float32).astype(float))
        result = make_morans_i_map(noise.astype(np.float32))
        assert_allclose(result, expected, atol=1e-5)

    def test_non_contiguous_and_int(self, noise):
        expected = make_morans_i_map(np.ascontiguousarray(noise[::2, ::2]))
        assert_equal(make_morans_i_map(noise[::2, ::2]), expected)

        data = np.round(noise * 100).astype(int)
        assert_equal(make_morans_i_map(data),
                     make_morans_i_map(data.astype(float)))

    def test_quantity(self, noise):
        result = make_morans_i_map(noise << u.Jy)
        assert not isinstance(result, u.Quantity)
        assert_equal(result, make_morans_i_map(noise))

    def test_constant_data(self):
        """
        Test that patches with zero variance are NaN.
        """
        assert np.all(np.isnan(make_morans_i_map(np.full((30, 30), 5.0))))

        rng = np.random.default_rng(2)
        data = rng.normal(100.0, 1.0, (60, 60))
        data[:, 30:] = 0.0
        result = make_morans_i_map(data, patch_size=11)
        assert np.all(np.isnan(result[:, 36:]))
        assert np.all(np.isfinite(result[:, :30]))

    def test_all_masked(self, noise):
        assert np.all(np.isnan(make_morans_i_map(np.full((30, 30), np.nan))))
        mask = np.ones(noise.shape, dtype=bool)
        assert np.all(np.isnan(make_morans_i_map(noise, mask=mask)))

    def test_inputs(self, noise):
        match = 'data must be a 2D array'
        with pytest.raises(ValueError, match=match):
            make_morans_i_map(np.ones(10))

        match = 'patch_size must be an odd integer'
        for patch_size in (20, 1, 3, -5, 21.0, True):
            with pytest.raises(ValueError, match=match):
                make_morans_i_map(noise, patch_size=patch_size)

        match = 'mask must have the same shape as data'
        with pytest.raises(ValueError, match=match):
            make_morans_i_map(noise, mask=np.zeros((3, 3), dtype=bool))

        match = 'weights must have the same shape as data'
        with pytest.raises(ValueError, match=match):
            make_morans_i_map(noise, weights=np.ones((3, 3)))

    def test_masked_array(self, small):
        """
        Test that the mask of a MaskedArray input is combined with the
        input mask.
        """
        mask = np.zeros(small.shape, dtype=bool)
        mask[10:14, 10:14] = True
        other = np.zeros(small.shape, dtype=bool)
        other[30:33, 5:9] = True
        data = np.ma.MaskedArray(small, mask=mask)

        expected = make_morans_i_map(small, mask=mask)
        assert np.all(np.isnan(expected[mask]))
        assert_equal(make_morans_i_map(data), expected)
        assert_equal(make_morans_i_map(data, mask=other),
                     make_morans_i_map(small, mask=mask | other))
        assert_equal(make_morans_i_map(np.ma.MaskedArray(small)),
                     make_morans_i_map(small))

    def test_inputs_unchanged(self, noise):
        data = noise.copy()
        data[3, 3] = np.nan
        original = data.copy()
        make_morans_i_map(data)
        assert_equal(data, original)
