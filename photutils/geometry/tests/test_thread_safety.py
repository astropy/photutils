# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Test that the Cython overlap functions are free-threading compatible and
return identical results when run concurrently from multiple threads.

This catches data races and shared-state issues introduced by ``with
nogil:`` sections in the functions, and verifies that each compiled
module opts into free-threading (``freethreading_compatible=True``).
"""
import importlib
import sys
import sysconfig
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from photutils.geometry import (circular_overlap_grid, elliptical_overlap_grid,
                                rectangular_overlap_grid)
from photutils.geometry._polygon_overlap import polygon_overlap_grid

N_THREADS = 8
N_CALLS_PER_THREAD = 4

# The compiled geometry modules that carry the
# ``# cython: freethreading_compatible=True`` directive.
GEOMETRY_MODULES = ('core', 'circle_overlap', 'ellipse_overlap',
                    'rectangle_overlap', '_polygon_overlap')

GIL_DISABLED = bool(sysconfig.get_config_var('Py_GIL_DISABLED'))


@pytest.mark.parametrize('modname', GEOMETRY_MODULES)
def test_geometry_module_importable(modname):
    """
    Test that each compiled geometry module imports cleanly.

    On a free-threaded build, importing a module that was not compiled
    with ``freethreading_compatible=True`` emits a ``RuntimeWarning``
    and reenables the GIL. This import check is the entry point for the
    free-threading verification below.
    """
    assert importlib.import_module(f'photutils.geometry.{modname}') is not None


@pytest.mark.skipif(not GIL_DISABLED,
                    reason='requires a free-threaded (GIL-disabled) build')
def test_geometry_modules_do_not_reenable_gil():
    """
    Test that importing the geometry Cython modules does not reenable
    the GIL on a free-threaded build.

    Each module is compiled with the ``freethreading_compatible=True``
    directive, which marks it as free-threading compatible
    (``Py_MOD_GIL_NOT_USED``). A module lacking that directive would
    force the runtime to reenable the GIL on import, which this test
    detects.
    """
    for modname in GEOMETRY_MODULES:
        importlib.import_module(f'photutils.geometry.{modname}')
    assert sys._is_gil_enabled() is False


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
