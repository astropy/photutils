# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the callback-free sort of the batch statistics kernels.
"""

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from photutils.aperture._batch_stats import _heapsort_values, batch_sort_values


def _adversarial_arrays(rng, n):
    """
    Arrays of length ``n`` that exercise the partitioning, sentinel,
    and short-range paths of the introsort.
    """
    base = rng.normal(size=n)
    arrays = [base, np.sort(base), np.sort(base)[::-1].copy(),
              np.full(n, 3.5), rng.integers(0, 3, n).astype(float),
              rng.integers(0, max(n // 4, 1), n).astype(float)]
    # Organ pipe and sawtooth patterns
    half = np.arange(n // 2, dtype=float)
    arrays.append(np.concatenate((half, half[::-1], np.zeros(n % 2))))
    arrays.append((np.arange(n) % 7).astype(float))
    # Values with the same magnitude and both signs, and extremes
    arrays.append(np.where(rng.random(n) < 0.5, base, -base))
    arrays.append(np.where(rng.random(n) < 0.01, 1e300, base))
    return arrays


@pytest.mark.parametrize('n', [0, 1, 2, 3, 15, 16, 17, 31, 32, 33, 100,
                               1000, 100_000])
def test_sort_values_matches_numpy(n):
    rng = np.random.default_rng(n)
    for arr in _adversarial_arrays(rng, n):
        arr = np.ascontiguousarray(arr, dtype=np.float64)
        starts = np.array([0], dtype=np.intp)
        counts = np.array([n], dtype=np.intp)
        result = batch_sort_values(arr, starts, counts)
        assert_array_equal(result, np.sort(arr))


def test_sort_values_many_sources():
    """
    Test that each source's slice is sorted independently, including
    empty sources, and that the input is not modified.
    """
    rng = np.random.default_rng(7)
    counts = rng.integers(0, 60, 500).astype(np.intp)
    counts[::13] = 0
    counts[1::17] = 1
    counts[2::19] = 17
    starts = np.concatenate(([0], np.cumsum(counts)[:-1])).astype(np.intp)
    values = rng.normal(size=int(counts.sum()))
    original = values.copy()
    result = batch_sort_values(values, starts, counts)
    assert_array_equal(values, original)
    for start, count in zip(starts, counts, strict=True):
        assert_array_equal(result[start:start + count],
                           np.sort(values[start:start + count]))


@pytest.mark.parametrize('n', [0, 1, 2, 3, 16, 17, 100, 1000, 100_000])
def test_heapsort_fallback(n):
    """
    Test the heapsort that the introsort falls back to when its
    recursion budget is exhausted (not reachable with ordinary inputs).
    """
    rng = np.random.default_rng(n + 1)
    for arr in _adversarial_arrays(rng, n):
        values = np.ascontiguousarray(arr, dtype=np.float64).copy()
        expected = np.sort(values)
        _heapsort_values(values)
        assert_array_equal(values, expected)
