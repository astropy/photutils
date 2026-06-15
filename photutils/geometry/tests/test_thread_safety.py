# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Test that the Cython overlap functions return identical results when run
concurrently from multiple threads.

This catches data races and shared-state issues introduced by ``with
nogil:`` sections in the functions.
"""
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from photutils.geometry import (circular_overlap_grid, elliptical_overlap_grid,
                                rectangular_overlap_grid)
from photutils.geometry._batch_photometry import (SHAPE_CIRCLE,
                                                  SHAPE_CIRCULAR_ANNULUS,
                                                  SHAPE_ELLIPSE,
                                                  SHAPE_ELLIPTICAL_ANNULUS,
                                                  SHAPE_RECTANGLE,
                                                  SHAPE_RECTANGULAR_ANNULUS,
                                                  batch_aperture_sums)
from photutils.geometry._polygon_overlap import polygon_overlap_grid

N_THREADS = 8
N_CALLS_PER_THREAD = 4


def _run_concurrent(fn, *, n_threads=N_THREADS,
                    n_calls=N_CALLS_PER_THREAD):
    expected = fn()
    with ThreadPoolExecutor(max_workers=n_threads) as ex:
        futures = [ex.submit(fn)
                   for _ in range(n_threads * n_calls)]
        for fut in futures:
            np.testing.assert_array_equal(fut.result(), expected)


@pytest.mark.parametrize('use_exact', [1, 0])
def test_circular_overlap_grid_threadsafe(use_exact):
    def fn():
        return circular_overlap_grid(-15.0, 15.0, -15.0, 15.0, 200, 200,
                                     8.0, use_exact, 5)
    _run_concurrent(fn)


@pytest.mark.parametrize('use_exact', [1, 0])
def test_elliptical_overlap_grid_threadsafe(use_exact):
    def fn():
        return elliptical_overlap_grid(-15.0, 15.0, -15.0, 15.0, 200, 200,
                                       8.0, 5.0, 0.7, use_exact, 5)
    _run_concurrent(fn)


@pytest.mark.parametrize('use_exact', [1, 0])
def test_rectangular_overlap_grid_threadsafe(use_exact):
    def fn():
        return rectangular_overlap_grid(-15.0, 15.0, -15.0, 15.0, 200, 200,
                                        12.0, 7.0, 0.5, use_exact, 5)
    _run_concurrent(fn)


@pytest.mark.parametrize('use_exact', [1, 0])
def test_polygon_overlap_grid_threadsafe(use_exact):
    n_vertices = 32
    angles = np.linspace(0.0, 2.0 * np.pi, n_vertices, endpoint=False)
    radii = 8.0 + 2.0 * np.sin(5.0 * angles)
    vx = np.ascontiguousarray(radii * np.cos(angles))
    vy = np.ascontiguousarray(radii * np.sin(angles))

    def fn():
        return polygon_overlap_grid(-15.0, 15.0, -15.0, 15.0, 200, 200,
                                    vx, vy, use_exact, 8)
    _run_concurrent(fn)


def test_mixed_functions_concurrent():
    """
    Run all geometry function types concurrently from many threads.

    This is a more aggressive smoke test that mixes different functions
    so that if any of them shares mutable state (e.g., a module-level
    static buffer), interference between calls would surface.
    """
    n_vertices = 16
    angles = np.linspace(0.0, 2.0 * np.pi, n_vertices, endpoint=False)
    vx = np.ascontiguousarray(7.0 * np.cos(angles))
    vy = np.ascontiguousarray(5.0 * np.sin(angles))

    expected_circ = circular_overlap_grid(-10.0, 10.0, -10.0, 10.0,
                                          150, 150, 6.0, 1, 5)
    expected_ell = elliptical_overlap_grid(-10.0, 10.0, -10.0, 10.0,
                                           150, 150, 7.0, 4.0, 0.3, 1, 5)
    expected_rect = rectangular_overlap_grid(-10.0, 10.0, -10.0, 10.0,
                                             150, 150, 9.0, 5.0, 0.4, 1, 5)
    expected_poly = polygon_overlap_grid(-10.0, 10.0, -10.0, 10.0,
                                         150, 150, vx, vy, 1, 5)

    def task(which):
        if which == 0:
            return 'c', circular_overlap_grid(
                -10.0, 10.0, -10.0, 10.0, 150, 150, 6.0, 1, 5)
        if which == 1:
            return 'e', elliptical_overlap_grid(
                -10.0, 10.0, -10.0, 10.0, 150, 150, 7.0, 4.0, 0.3, 1, 5)
        if which == 2:
            return 'r', rectangular_overlap_grid(
                -10.0, 10.0, -10.0, 10.0, 150, 150, 9.0, 5.0, 0.4, 1, 5)
        return 'p', polygon_overlap_grid(
            -10.0, 10.0, -10.0, 10.0, 150, 150, vx, vy, 1, 5)

    expected_map = {'c': expected_circ, 'e': expected_ell,
                    'r': expected_rect, 'p': expected_poly}

    with ThreadPoolExecutor(max_workers=N_THREADS) as ex:
        futures = [ex.submit(task, i % 4)
                   for i in range(N_THREADS * 8)]
        for future in futures:
            tag, result = future.result()
            assert_array_equal(result, expected_map[tag])


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
            sums, errs, overlap = future.result()
            assert_array_equal(sums, expected[0])
            assert_array_equal(errs, expected[1])
            assert_array_equal(overlap, expected[2])


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
            sums, errs, overlap = fut.result()
            exp_sums, exp_errs, exp_overlap = expected[shape_code]
            assert_array_equal(sums, exp_sums)
            assert_array_equal(errs, exp_errs)
            assert_array_equal(overlap, exp_overlap)
