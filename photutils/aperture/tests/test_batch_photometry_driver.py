# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the low-level batch Cython aperture photometry driver.
"""

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from photutils.aperture._batch_photometry import (SHAPE_CIRCLE,
                                                  SHAPE_CIRCULAR_ANNULUS,
                                                  SHAPE_ELLIPSE,
                                                  SHAPE_ELLIPTICAL_ANNULUS,
                                                  SHAPE_RECTANGLE,
                                                  SHAPE_RECTANGULAR_ANNULUS,
                                                  batch_aperture_sums)

N_THREADS = 8
N_CALLS_PER_THREAD = 4


def _batch_inputs():
    """
    Build deterministic data, error, mask, and source positions for the
    batch-driver tests.
    """
    rng = np.random.default_rng(42)
    data = rng.random((80, 80))
    error = rng.random((80, 80)) + 0.1
    mask = np.zeros((80, 80), dtype=np.uint8)
    mask[::7, ::5] = 1
    positions = np.array([[20.0, 25.0], [40.0, 40.0], [55.0, 30.0],
                          [10.0, 60.0], [70.0, 70.0], [35.0, 15.0]])
    return data, error, mask, positions


# Aperture shape specs: (shape_code, params, ext_x, ext_y).
# The half-extents need only bound the aperture; they are identical
# across the baseline and concurrent calls, so the comparison is exact.
_BATCH_SPECS = [
    (SHAPE_CIRCLE, [8.0], 8.0, 8.0),
    (SHAPE_CIRCULAR_ANNULUS, [5.0, 8.0], 8.0, 8.0),
    (SHAPE_ELLIPSE, [8.0, 5.0, 0.7], 8.0, 8.0),
    (SHAPE_ELLIPTICAL_ANNULUS, [4.0, 2.0, 8.0, 5.0, 0.7], 8.0, 8.0),
    (SHAPE_RECTANGLE, [12.0, 7.0, 0.5], 7.0, 7.0),
    (SHAPE_RECTANGULAR_ANNULUS, [6.0, 4.0, 12.0, 7.0, 0.5], 7.0, 7.0),
]


@pytest.mark.parametrize('use_exact', [1, 0])
def test_readonly_arrays(use_exact):
    """
    Test that the batch driver accepts read-only (non-writeable) data,
    error, positions, and params arrays and returns results identical to
    writeable arrays.

    The data, error, positions, and params arguments are declared as
    ``const`` typed memoryviews so that read-only arrays do not raise a
    ``ValueError``.
    """
    data, error, mask, positions = _batch_inputs()
    params = np.array([8.0], dtype=np.float64)

    expected = batch_aperture_sums(data, error, mask, positions, SHAPE_CIRCLE,
                                   params, 8.0, 8.0, use_exact, 8)

    for arr in (data, error, positions, params):
        arr.setflags(write=False)
    result = batch_aperture_sums(data, error, mask, positions, SHAPE_CIRCLE,
                                 params, 8.0, 8.0, use_exact, 8)

    for res_arr, exp_arr in zip(result, expected, strict=True):
        assert_array_equal(res_arr, exp_arr)


@pytest.mark.parametrize(('shape_code', 'params', 'ext_x', 'ext_y'),
                         _BATCH_SPECS)
@pytest.mark.parametrize('use_exact', [1, 0])
def test_batch_aperture_sums_threadsafe(shape_code, params, ext_x, ext_y,
                                        use_exact):
    data, error, mask, positions = _batch_inputs()
    params = np.array(params, dtype=np.float64)

    def fn():
        return batch_aperture_sums(data, error, mask, positions, shape_code,
                                   params, ext_x, ext_y, use_exact, 8)

    expected = fn()
    with ThreadPoolExecutor(max_workers=N_THREADS) as ex:
        futures = [ex.submit(fn)
                   for _ in range(N_THREADS * N_CALLS_PER_THREAD)]
        for future in futures:
            result = future.result()
            for res_arr, exp_arr in zip(result, expected, strict=True):
                assert_array_equal(res_arr, exp_arr)


def test_batch_aperture_sums_mixed_concurrent():
    """
    Run every aperture shape through the batch driver concurrently.

    Mixing shapes within the thread pool surfaces interference between
    calls if any shared mutable state (e.g., a module-level scratch
    buffer) were introduced into the ``nogil`` source loop.
    """
    data, error, mask, positions = _batch_inputs()

    def task(spec):
        shape_code, params, ext_x, ext_y = spec
        params = np.array(params, dtype=np.float64)
        return batch_aperture_sums(data, error, mask, positions, shape_code,
                                   params, ext_x, ext_y, 1, 5)

    expected = {spec[0]: task(spec) for spec in _BATCH_SPECS}

    with ThreadPoolExecutor(max_workers=N_THREADS) as ex:
        futures = {ex.submit(task, spec): spec[0]
                   for spec in _BATCH_SPECS for _ in range(N_THREADS)}
        for fut, shape_code in futures.items():
            result = fut.result()
            for res_arr, exp_arr in zip(result, expected[shape_code],
                                        strict=True):
                assert_array_equal(res_arr, exp_arr)
