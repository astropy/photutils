# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the batch minimum/maximum kernel and the sort-free
ApertureStats ``min`` and ``max`` path.
"""

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
from astropy.stats import SigmaClip
from numpy.testing import assert_allclose, assert_array_equal

from photutils.aperture import ApertureStats, CircularAperture
from photutils.aperture._batch_stats import batch_minmax


def _packed(rng, counts):
    starts = np.concatenate(([0], np.cumsum(counts)[:-1])).astype(np.intp)
    values = rng.normal(size=int(counts.sum()))
    return values, starts, counts.astype(np.intp)


def test_kernel_matches_numpy():
    rng = np.random.default_rng(3)
    counts = rng.integers(0, 40, 300)
    counts[::7] = 0
    counts[1::11] = 1
    values, starts, counts = _packed(rng, counts)
    vmin, vmax = batch_minmax(values, starts, counts)
    for k, (start, count) in enumerate(zip(starts, counts, strict=True)):
        if count == 0:
            assert np.isnan(vmin[k])
            assert np.isnan(vmax[k])
        else:
            chunk = values[start:start + count]
            assert vmin[k] == chunk.min()
            assert vmax[k] == chunk.max()


def test_kernel_single_value_and_duplicates():
    values = np.array([2.0, 5.0, 5.0, -1.0, -1.0, 7.0])
    starts = np.array([0, 1, 3], dtype=np.intp)
    counts = np.array([1, 2, 3], dtype=np.intp)
    vmin, vmax = batch_minmax(values, starts, counts)
    assert_array_equal(vmin, [2.0, 5.0, -1.0])
    assert_array_equal(vmax, [2.0, 5.0, 7.0])


def test_kernel_thread_safety():
    rng = np.random.default_rng(4)
    values, starts, counts = _packed(rng, rng.integers(1, 30, 200))
    expected = batch_minmax(values, starts, counts)

    def run(_):
        return batch_minmax(values, starts, counts)

    with ThreadPoolExecutor(max_workers=4) as pool:
        for result in pool.map(run, range(8)):
            for got, want in zip(result, expected, strict=True):
                assert_array_equal(got, want)


@pytest.fixture
def stats_inputs():
    rng = np.random.default_rng(5)
    data = rng.normal(size=(120, 120))
    positions = np.column_stack((rng.uniform(5, 115, 60),
                                 rng.uniform(5, 115, 60)))
    return data, CircularAperture(positions, r=4.0)


def _reference_extremes(data, aperture):
    vmin = []
    vmax = []
    for mask in aperture.to_mask(method='center'):
        cutout = mask.get_values(data)
        vmin.append(cutout.min())
        vmax.append(cutout.max())
    return np.array(vmin), np.array(vmax)


@pytest.mark.parametrize('n_threads', [1, 3])
def test_min_max_without_sort(stats_inputs, n_threads):
    """
    Test that min and max alone are computed without the per-source
    sort and match both the reference and the sorted path.
    """
    data, aperture = stats_inputs
    ref_min, ref_max = _reference_extremes(data, aperture)
    stats = ApertureStats(data, aperture, n_threads=n_threads)
    assert stats._fast_gather is not None
    assert_allclose(stats.min, ref_min)
    assert_allclose(stats.max, ref_max)
    assert '_sorted_values' not in stats.__dict__
    assert '_order_stats' not in stats.__dict__
    # The sorted path gives the same values
    vmin, vmax, _ = stats._order_stats
    assert_array_equal(vmin, stats.min)
    assert_array_equal(vmax, stats.max)


def test_min_max_after_sort(stats_inputs):
    """
    Test that min and max agree with the sorted path (and the
    reference) when the sorted buffer already exists.
    """
    data, aperture = stats_inputs
    stats = ApertureStats(data, aperture)
    median = stats.median
    assert '_sorted_values' in stats.__dict__
    assert_array_equal(stats._minmax[0], stats._order_stats[0])
    assert_array_equal(stats._minmax[1], stats._order_stats[1])
    ref_min, ref_max = _reference_extremes(data, aperture)
    assert_allclose(stats.min, ref_min)
    assert_allclose(stats.max, ref_max)
    assert np.all(stats.min <= median)
    assert np.all(median <= stats.max)


def test_min_max_with_sigma_clip(stats_inputs):
    """
    Test that the sigma-clipped min and max are the extremes of the
    surviving values (the clipping kernel's sorted buffer).
    """
    data, aperture = stats_inputs
    sigma_clip = SigmaClip(sigma=2.0, maxiters=5)
    stats = ApertureStats(data, aperture, sigma_clip=sigma_clip)
    assert stats._fast_gather.sorted_values is not None
    assert_array_equal(stats._minmax[0], stats._order_stats[0])
    assert_array_equal(stats._minmax[1], stats._order_stats[1])
    vmin = []
    vmax = []
    for mask in aperture.to_mask(method='center'):
        clipped = sigma_clip(mask.get_values(data), masked=False)
        vmin.append(clipped.min())
        vmax.append(clipped.max())
    assert_allclose(stats.min, vmin)
    assert_allclose(stats.max, vmax)


def test_min_max_scalar_and_unit():
    data = np.arange(400.0).reshape(20, 20)
    aperture = CircularAperture((10.0, 10.0), r=2.5)
    stats = ApertureStats(data, aperture)
    values = aperture.to_mask(method='center').get_values(data)
    assert stats.min == values.min()
    assert stats.max == values.max()
