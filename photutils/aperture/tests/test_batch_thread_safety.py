# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests that the aperture batch Cython drivers are free-threading
compatible and return identical results when run concurrently from
multiple threads.

This complements the low-level ``batch_aperture_sums`` concurrency tests
in ``test_batch_photometry_driver.py`` by covering the
``batch_aperture_gather`` gatherer, the higher-level `ApertureStats`
statistics pipeline (which drives the ``_batch_stats`` order/moment/
sigma-clip nogil kernels), and the free-threading opt-in of the compiled
modules.
"""
import importlib
import sys
import sysconfig
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
from astropy.stats import SigmaClip
from numpy.testing import assert_allclose, assert_array_equal

from photutils.aperture._batch_photometry import (SHAPE_CIRCLE,
                                                  SHAPE_CIRCULAR_ANNULUS,
                                                  SHAPE_ELLIPSE,
                                                  SHAPE_RECTANGLE)
from photutils.aperture._batch_stats import batch_aperture_gather
from photutils.aperture.circle import CircularAperture
from photutils.aperture.stats import ApertureStats
from photutils.datasets import make_100gaussians_image

N_THREADS = 8
N_CALLS_PER_THREAD = 4

# The compiled aperture modules that carry the
# ``# cython: freethreading_compatible=True`` directive.
APERTURE_MODULES = ('_batch_photometry', '_batch_stats')

GIL_DISABLED = bool(sysconfig.get_config_var('Py_GIL_DISABLED'))


@pytest.mark.parametrize('modname', APERTURE_MODULES)
def test_aperture_module_importable(modname):
    """
    Test that each compiled aperture batch module imports cleanly.

    On a free-threaded build, importing a module that was not compiled
    with ``freethreading_compatible=True`` emits a ``RuntimeWarning``
    and reenables the GIL. This import check is the entry point for the
    free-threading verification below.
    """
    assert importlib.import_module(f'photutils.aperture.{modname}') is not None


@pytest.mark.skipif(not GIL_DISABLED,
                    reason='requires a free-threaded (GIL-disabled) build')
def test_aperture_modules_do_not_reenable_gil():
    """
    Test that importing the aperture Cython batch modules does not
    reenable the GIL on a free-threaded build.

    Each module is compiled with the ``freethreading_compatible=True``
    directive, which marks it as free-threading compatible
    (``Py_MOD_GIL_NOT_USED``). A module lacking that directive would
    force the runtime to reenable the GIL on import, which this test
    detects.
    """
    for modname in APERTURE_MODULES:
        importlib.import_module(f'photutils.aperture.{modname}')
    assert sys._is_gil_enabled() is False


def _gather_inputs():
    """
    Build deterministic data, mask, and source positions for the
    ``batch_aperture_gather`` tests.
    """
    rng = np.random.default_rng(42)
    data = rng.random((80, 80))
    mask = np.zeros((80, 80), dtype=np.uint8)
    mask[::7, ::5] = 1
    positions = np.array([[20.0, 25.0], [40.0, 40.0], [55.0, 30.0],
                          [10.0, 60.0], [70.0, 70.0], [35.0, 15.0]])
    return data, mask, positions


# Aperture shape specs for the gatherer: (shape_code, params, ext_x, ext_y).
_GATHER_SPECS = [
    (SHAPE_CIRCLE, [8.0], 8.0, 8.0),
    (SHAPE_CIRCULAR_ANNULUS, [5.0, 8.0], 8.0, 8.0),
    (SHAPE_ELLIPSE, [8.0, 5.0, 0.7], 8.0, 8.0),
    (SHAPE_RECTANGLE, [12.0, 7.0, 0.5], 7.0, 7.0),
]


def _assert_gather_equal(result, expected):
    """
    Assert two ``batch_aperture_gather`` results are equal.

    The packed ``values``/``local_x``/``local_y`` buffers are allocated
    with an upper-bound length (the sum of the bounding-box areas), so
    only the valid per-source region ``[starts[k]:starts[k] + counts[k]]``
    is meaningful; the trailing entries are uninitialized and must not be
    compared.
    """
    values, lx, ly, starts, counts, overlap = result
    e_values, e_lx, e_ly, e_starts, e_counts, e_overlap = expected
    assert_array_equal(starts, e_starts)
    assert_array_equal(counts, e_counts)
    assert_array_equal(overlap, e_overlap)
    for start, count in zip(starts, counts, strict=True):
        sl = slice(int(start), int(start) + int(count))
        assert_array_equal(values[sl], e_values[sl])
        assert_array_equal(lx[sl], e_lx[sl])
        assert_array_equal(ly[sl], e_ly[sl])


def test_batch_aperture_gather_accepts_none_local_bkg():
    """
    Test that ``batch_aperture_gather`` accepts ``local_bkg=None`` and
    matches an all-zero local background.

    The per-pixel background subtraction is guarded by a ``has_bkg``
    check so that a `None` ``local_bkg`` is treated as no subtraction
    rather than dereferencing a NULL memoryview.
    """
    data, mask, positions = _gather_inputs()
    params = np.array([8.0], dtype=np.float64)
    zeros = np.zeros(positions.shape[0], dtype=np.float64)

    none_result = batch_aperture_gather(data, mask, positions, SHAPE_CIRCLE,
                                        params, 8.0, 8.0, None)
    zero_result = batch_aperture_gather(data, mask, positions, SHAPE_CIRCLE,
                                        params, 8.0, 8.0, zeros)

    _assert_gather_equal(none_result, zero_result)


@pytest.mark.parametrize(('shape_code', 'params', 'ext_x', 'ext_y'),
                         _GATHER_SPECS)
def test_batch_aperture_gather_threadsafe(shape_code, params, ext_x, ext_y):
    data, mask, positions = _gather_inputs()
    params = np.array(params, dtype=np.float64)
    local_bkg = np.linspace(0.0, 0.5, positions.shape[0])

    def fn():
        return batch_aperture_gather(data, mask, positions, shape_code,
                                     params, ext_x, ext_y, local_bkg)

    expected = fn()
    with ThreadPoolExecutor(max_workers=N_THREADS) as ex:
        futures = [ex.submit(fn)
                   for _ in range(N_THREADS * N_CALLS_PER_THREAD)]
        for future in futures:
            _assert_gather_equal(future.result(), expected)


def test_batch_aperture_gather_mixed_concurrent():
    """
    Run every aperture shape through the gatherer concurrently.

    Mixing shapes within the thread pool surfaces interference between
    calls if any shared mutable state (e.g., a module-level scratch
    buffer) were introduced into the ``nogil`` gather loop.
    """
    data, mask, positions = _gather_inputs()
    local_bkg = np.linspace(0.0, 0.5, positions.shape[0])

    def task(spec):
        shape_code, params, ext_x, ext_y = spec
        params = np.array(params, dtype=np.float64)
        return batch_aperture_gather(data, mask, positions, shape_code,
                                     params, ext_x, ext_y, local_bkg)

    expected = {spec[0]: task(spec) for spec in _GATHER_SPECS}

    with ThreadPoolExecutor(max_workers=N_THREADS) as ex:
        futures = {ex.submit(task, spec): spec[0]
                   for spec in _GATHER_SPECS for _ in range(N_THREADS)}
        for fut, shape_code in futures.items():
            _assert_gather_equal(fut.result(), expected[shape_code])


# Properties that exercise the ``_batch_stats`` order-statistic, moment,
# MAD, biweight, Gini, and mean/variance nogil kernels.
_STATS_PROPERTIES = ('mean', 'median', 'std', 'mad_std', 'biweight_location',
                     'biweight_midvariance', 'gini', 'min', 'max',
                     'sum', 'var', 'centroid')


@pytest.mark.parametrize('use_sigma_clip', [False, True])
def test_aperture_stats_threadsafe(use_sigma_clip):
    """
    Test that `ApertureStats` releases the GIL and is thread-safe by
    running many concurrent calls that drive the ``_batch_stats`` nogil
    kernels and checking that they all give identical results.
    """
    data = make_100gaussians_image()
    error = np.sqrt(np.abs(data))
    positions = ((145.1, 168.3), (84.7, 224.1), (48.3, 200.3), (200.0, 100.0))
    aperture = CircularAperture(positions, r=6.0)
    sigma_clip = SigmaClip(sigma=3.0, maxiters=10) if use_sigma_clip else None

    def compute():
        stats = ApertureStats(data, aperture, error=error,
                              sigma_clip=sigma_clip)
        return {name: np.asarray(getattr(stats, name))
                for name in _STATS_PROPERTIES}

    expected = compute()
    with ThreadPoolExecutor(max_workers=N_THREADS) as ex:
        futures = [ex.submit(compute)
                   for _ in range(N_THREADS * N_CALLS_PER_THREAD)]
        for future in futures:
            result = future.result()
            for name in _STATS_PROPERTIES:
                assert_allclose(result[name], expected[name],
                                rtol=0, atol=0, equal_nan=True)
