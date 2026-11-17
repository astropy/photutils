# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the morans_i module.
"""

import astropy.units as u
import numpy as np
import pytest
from astropy.utils.exceptions import AstropyUserWarning
from numpy.testing import assert_allclose, assert_equal
from scipy.ndimage import (binary_dilation, binary_erosion, binary_fill_holes,
                           uniform_filter)

from photutils.background import make_morans_i_map, make_morans_i_mask
from photutils.datasets import make_100gaussians_image


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


def _reference_opening(mask, iterations, stat):
    """
    Open a mask with SciPy, with flagged pixels outside of the image
    and where the statistic is NaN for the erosion.
    """
    if iterations == 0:
        return mask
    excluded = np.isnan(stat)
    eroded = binary_erosion(mask | excluded, iterations=iterations,
                            border_value=1)
    eroded &= ~excluded
    return mask & binary_dilation(eroded, iterations=iterations)


def _crowded_image(n_sources, *, seed=0):
    """
    Make a 256x256 image of Gaussian sources on uncorrelated noise.

    Returns the image and the noiseless source model.
    """
    rng = np.random.default_rng(seed)
    data = rng.normal(0.0, 1.0, (256, 256))
    yy, xx = np.mgrid[:256, :256]
    model = np.zeros(data.shape)
    for _ in range(n_sources):
        xcen, ycen = rng.uniform(0, 256, 2)
        amplitude = rng.uniform(2.0, 50.0)
        sigma = rng.uniform(1.5, 6.0)
        model += amplitude * np.exp(-((xx - xcen)**2 + (yy - ycen)**2)
                                    / (2.0 * sigma**2))
    return data + model, model


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

    def test_reference_wide(self):
        """
        Test a wide image and its transpose against a slow reference.

        The box sums of images that are at least 256 pixels wide add
        and subtract whole rows, and the box sums of narrower images
        filter along the columns.
        """
        rng = np.random.default_rng(7)
        data = rng.normal(0.0, 1.0, (30, 300))
        data[10:20, 100:130] += 4.0
        weights = rng.uniform(0.5, 2.0, data.shape)
        mask = rng.random(data.shape) < 0.01

        result = make_morans_i_map(data, patch_size=7, mask=mask,
                                   weights=weights)
        expected = _reference_morans_i(data, 7, mask=mask, weights=weights)
        assert np.any(np.isfinite(result))
        assert_equal(np.isnan(result), np.isnan(expected))
        assert_allclose(result, expected, atol=1e-5)

        result = make_morans_i_map(data.T, patch_size=7, mask=mask.T,
                                   weights=weights.T)
        assert_equal(np.isnan(result), np.isnan(expected.T))
        assert_allclose(result, expected.T, atol=1e-5)

    @pytest.mark.parametrize('strip_pixels', [1000, 25000])
    def test_strips(self, monkeypatch, strip_pixels):
        """
        Test that the map does not depend on the strips of rows used to
        calculate it.

        The strips are as short as their overlap or 50 rows tall,
        instead of the whole 300-row image.
        """
        rng = np.random.default_rng(8)
        data = make_100gaussians_image()
        data[100:120, 200:260] = np.nan
        weights = rng.uniform(0.5, 2.0, data.shape)
        mask = rng.random(data.shape) < 0.01
        expected = make_morans_i_map(data, mask=mask, weights=weights)

        monkeypatch.setattr('photutils.background.morans_i._STRIP_PIXELS',
                            strip_pixels)
        result = make_morans_i_map(data, mask=mask, weights=weights)
        assert_equal(np.isnan(result), np.isnan(expected))
        assert_allclose(result, expected, atol=1e-6)

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


class TestMakeMoransIMask:
    def test_noise(self, noise):
        """
        Test that almost none of a pure noise image is masked.
        """
        mask = make_morans_i_mask(noise)
        assert mask.shape == noise.shape
        assert mask.dtype == bool
        assert np.mean(mask) < 0.01
        assert not np.any(make_morans_i_mask(noise, n_sigma=5.0))

    def test_correlated_noise(self):
        """
        Test that correlated noise, which has a large positive Moran's
        I everywhere, is not masked with the default threshold.
        """
        rng = np.random.default_rng(3)
        data = uniform_filter(rng.normal(0.0, 1.0, (301, 301)), 2)
        assert np.median(make_morans_i_map(data)) > 0.3
        mask = make_morans_i_mask(data)
        assert np.mean(mask) < 0.01

    def test_sources(self):
        data = make_100gaussians_image()
        mask = make_morans_i_mask(data)
        assert mask.dtype == bool
        assert 0.05 < np.mean(mask) < 0.5

        # The brightest pixels are sources
        assert np.all(mask[data > np.percentile(data, 99.5)])

        # The unmasked pixels are consistent with the true background
        # (mean of 5 and standard deviation of 2)
        assert_allclose(np.mean(data[~mask]), 5.0, atol=0.2)
        assert np.mean(data[mask]) > 6.0

    def test_percent(self):
        data = make_100gaussians_image()
        mask = make_morans_i_mask(data, percent=35.0, opening_iterations=0)
        assert 0.34 < np.mean(mask) < 0.40

        # percent takes precedence over n_sigma
        mask2 = make_morans_i_mask(data, n_sigma=100.0, percent=35.0,
                                   opening_iterations=0)
        assert_equal(mask2, mask)

    @pytest.mark.parametrize('opening_iterations', [0, 1, 2, 3])
    @pytest.mark.parametrize('percent', [20.0, 50.0])
    @pytest.mark.parametrize('shape', [(300, 500), (30, 40)])
    def test_morphology(self, shape, percent, opening_iterations):
        """
        Test the hole filling and the opening against SciPy.
        """
        data = make_100gaussians_image()[:shape[0], :shape[1]]
        mask = make_morans_i_mask(data, percent=percent,
                                  opening_iterations=opening_iterations)

        stat = make_morans_i_map(data)
        threshold = np.percentile(stat[np.isfinite(stat)], 100.0 - percent)
        expected = _reference_opening(binary_fill_holes(stat > threshold),
                                      opening_iterations, stat)
        assert np.any(expected)
        assert_equal(mask, expected)

    def test_fill_holes(self):
        """
        Test that the hole of a ring-shaped source is filled.
        """
        rng = np.random.default_rng(4)
        data = rng.normal(0.0, 1.0, (301, 301))
        yy, xx = np.mgrid[:301, :301]
        radius = np.hypot(xx - 150, yy - 150)
        ring = (radius > 90) & (radius < 110)
        data[ring] += rng.normal(0.0, 30.0, (301, 301))[ring] + 50.0
        mask = make_morans_i_mask(data, patch_size=5, percent=15.0,
                                  opening_iterations=0)
        assert np.all(mask[radius < 100])
        assert not mask[0, 0]

        stat = make_morans_i_map(data, patch_size=5)
        threshold = np.percentile(stat[np.isfinite(stat)], 85.0)
        assert not np.all((stat > threshold)[radius < 100])
        assert_equal(mask, binary_fill_holes(stat > threshold))

    def test_weights(self):
        """
        Test that the weights are used for the statistic.
        """
        data = make_100gaussians_image()
        rng = np.random.default_rng(7)
        weights = rng.uniform(0.2, 5.0, data.shape)
        weights[100:140, 200:260] = 0.0
        mask = make_morans_i_mask(data, patch_size=11, percent=30.0,
                                  weights=weights)
        assert not np.any(mask[105:135, 205:255])

        stat = make_morans_i_map(data, patch_size=11, weights=weights)
        threshold = np.percentile(stat[np.isfinite(stat)], 70.0)
        expected = _reference_opening(binary_fill_holes(stat > threshold),
                                      3, stat)
        assert_equal(mask, expected)
        assert not np.array_equal(
            mask, make_morans_i_mask(data, patch_size=11, percent=30.0))

    def test_n_sigma(self):
        data = make_100gaussians_image()
        low = make_morans_i_mask(data, n_sigma=3.0, opening_iterations=0)
        high = make_morans_i_mask(data, n_sigma=20.0, opening_iterations=0)
        assert np.sum(high) < np.sum(low)
        assert np.all(low[high])

    def test_n_sigma_crowded(self):
        """
        Test that sources are masked in a field where most of the
        patches contain sources.

        The median of the statistic is then a source value, so it
        cannot be used as the background value.
        """
        data, model = _crowded_image(60)
        stat = make_morans_i_map(data)
        assert np.median(stat) > 0.3
        mask = make_morans_i_mask(data)
        assert np.mean(mask) > 0.5
        assert np.all(mask[model > 2.0])

    def test_n_sigma_no_background(self):
        """
        Test that a warning is raised if the background value of the
        statistic cannot be estimated.
        """
        data, _ = _crowded_image(150)
        match = 'background value of the Moran'
        with pytest.warns(AstropyUserWarning, match=match):
            make_morans_i_mask(data)

        # No warning is raised with a fixed percentage
        make_morans_i_mask(data, percent=50.0)

    def test_n_sigma_strong_gradient(self):
        """
        Test that a warning is raised for a background that changes
        strongly across each patch.
        """
        rng = np.random.default_rng(0)
        data = rng.normal(5.0, 2.0, (300, 500))
        yy, xx = np.mgrid[:300, :500]
        data += xx * yy / 1000.0
        match = 'background changes strongly'
        with pytest.warns(AstropyUserWarning, match=match):
            make_morans_i_mask(data)

    def test_n_sigma_hot_pixels(self):
        """
        Test that unmasked hot pixels do not prevent the sources away
        from them from being masked.

        The patches next to a hot pixel have extreme negative values of
        the statistic, which must not inflate the estimated scatter of
        the background values. A hot pixel still dominates the statistic
        of the patches that contain it.
        """
        data, model = _crowded_image(20)
        expected = make_morans_i_mask(data)
        rng = np.random.default_rng(13)
        hot = np.zeros(data.shape, dtype=bool)
        hot[tuple(rng.integers(0, 256, (2, 10)))] = True
        data[hot] = 1.0e5
        assert np.nanmin(make_morans_i_map(data)) < -10.0

        mask = make_morans_i_mask(data)
        far = ~binary_dilation(hot, np.ones((43, 43), dtype=bool))
        assert np.all(mask[far & (model > 2.0)])
        assert np.mean(mask[far] != expected[far]) < 0.02

    @pytest.mark.parametrize(('patch_size', 'iterations'),
                             [(5, 0), (11, 3), (21, 8)])
    def test_opening_iterations_default(self, patch_size, iterations):
        """
        Test that the default number of opening iterations is set by
        the patch size.
        """
        data = make_100gaussians_image()
        kwargs = {'patch_size': patch_size, 'percent': 30.0}
        mask = make_morans_i_mask(data, **kwargs)
        expected = make_morans_i_mask(data, opening_iterations=iterations,
                                      **kwargs)
        assert_equal(mask, expected)
        other = make_morans_i_mask(data, opening_iterations=iterations + 2,
                                   **kwargs)
        assert not np.array_equal(mask, other)

    def test_opening_at_edges(self):
        """
        Test that the opening does not remove the flagged regions of
        sources at the image edges.
        """
        rng = np.random.default_rng(10)
        data = rng.normal(0.0, 1.0, (151, 201))
        yy, xx = np.mgrid[:151, :201]
        for ycen, xcen in ((0, 60), (75, 200), (150, 0), (75, 100)):
            data += 50.0 * np.exp(-((xx - xcen)**2 + (yy - ycen)**2) / 8.0)
        mask = make_morans_i_mask(data)
        for ycen, xcen in ((0, 60), (75, 200), (150, 0), (75, 100)):
            assert mask[ycen, xcen]
        assert np.mean(mask) < 0.15

    @pytest.mark.parametrize('excluded_by', ['nan', 'mask', 'weights'])
    def test_opening_at_excluded_regions(self, excluded_by):
        """
        Test that the opening does not remove the flagged regions of
        sources next to excluded regions and that the excluded pixels
        are not flagged.
        """
        rng = np.random.default_rng(10)
        data = rng.normal(0.0, 1.0, (151, 301))
        yy, xx = np.mgrid[:151, :301]
        for ycen, xcen in ((75, 199), (75, 100)):
            data += 50.0 * np.exp(-((xx - xcen)**2 + (yy - ycen)**2) / 8.0)
        excluded = np.zeros(data.shape, dtype=bool)
        excluded[:, 200:] = True

        kwargs = {}
        if excluded_by == 'nan':
            data[excluded] = np.nan
        elif excluded_by == 'mask':
            kwargs['mask'] = excluded
        else:
            kwargs['weights'] = np.where(excluded, 0.0, 1.0)

        mask = make_morans_i_mask(data, **kwargs)
        assert mask[75, 199]
        assert mask[75, 100]
        assert not np.any(mask[excluded])
        assert np.mean(mask) < 0.15

    def test_opening_iterations(self):
        data = make_100gaussians_image()
        mask0 = make_morans_i_mask(data, n_sigma=2.0, opening_iterations=0)
        mask2 = make_morans_i_mask(data, n_sigma=2.0, opening_iterations=2)
        assert np.sum(mask2) < np.sum(mask0)
        assert np.all(mask0[mask2])

    def test_mask(self):
        """
        Test that the input mask is excluded from the statistic and is
        not itself flagged as a source.
        """
        data = make_100gaussians_image()
        bad = np.zeros(data.shape, dtype=bool)
        bad[:, 250:] = True
        data[bad] = 1.0e10
        mask = make_morans_i_mask(data, mask=bad)
        assert not np.any(mask[bad])
        assert np.any(mask[~bad])

    def test_masked_array(self):
        """
        Test that the mask of a MaskedArray input is excluded from the
        statistic.
        """
        data = make_100gaussians_image()
        bad = np.zeros(data.shape, dtype=bool)
        bad[:, 250:] = True
        data[bad] = 1.0e10
        expected = make_morans_i_mask(data, mask=bad)
        assert_equal(make_morans_i_mask(np.ma.MaskedArray(data, mask=bad)),
                     expected)

    def test_no_valid_pixels(self):
        mask = make_morans_i_mask(np.full((50, 50), np.nan))
        assert mask.dtype == bool
        assert not np.any(mask)

    def test_inputs(self, noise):
        match = 'n_sigma must be a positive number'
        for n_sigma in (0.0, -1.0, np.nan):
            with pytest.raises(ValueError, match=match):
                make_morans_i_mask(noise, n_sigma=n_sigma)

        match = 'percent must be between 0 and 100'
        for percent in (-1.0, 0.0, 100.0, 150.0):
            with pytest.raises(ValueError, match=match):
                make_morans_i_mask(noise, percent=percent)

        match = 'opening_iterations must be a nonnegative integer'
        for iterations in (-1, 1.5, True):
            with pytest.raises(ValueError, match=match):
                make_morans_i_mask(noise, opening_iterations=iterations)

        match = 'patch_size must be an odd integer'
        with pytest.raises(ValueError, match=match):
            make_morans_i_mask(noise, patch_size=20)
