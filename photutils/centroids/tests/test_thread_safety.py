# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests that the centroid functions are thread-safe and return identical
results when run concurrently from multiple threads.

The centroid functions hold no shared mutable state. There are no
module-level caches, each call creates its own fitter instance,
and fit non-convergence is detected from the fitter's ``fit_info``
rather than by capturing warnings (``warnings.catch_warnings``
mutates process-global state). Because the test suite turns warnings
into errors, any spurious warning raised in a worker thread (e.g.,
a cross-thread "fit may not have converged" injection) fails the
corresponding future.
"""

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from photutils.centroids import (centroid_1dg, centroid_2dg, centroid_com,
                                 centroid_quadratic, centroid_sources)
from photutils.centroids.tests.helpers import make_gaussian_source

N_THREADS = 8
N_CALLS_PER_THREAD = 4

CENTROID_FUNCS = [centroid_com, centroid_quadratic, centroid_1dg,
                  centroid_2dg]

# Centroid functions that accept an error keyword
ERROR_FUNCS = (centroid_1dg, centroid_2dg)


@pytest.fixture(name='source_data')
def fixture_source_data():
    """
    Create a single-source cutout with a mask and an error array.
    """
    data = make_gaussian_source((51, 51), 10.0, 24.7, 25.2, 3.0, 3.0, 0)
    mask = np.zeros(data.shape, dtype=bool)
    mask[10, 10] = True
    error = np.sqrt(np.abs(data))
    return data, mask, error


def _make_task(func, data, mask, error):
    """
    Return a zero-argument callable that computes the centroid of the
    data with the given centroid function.
    """
    kwargs = {'mask': mask}
    if func in ERROR_FUNCS:
        kwargs['error'] = error

    def task():
        return func(data, **kwargs)

    return task


@pytest.mark.parametrize('centroid_func', CENTROID_FUNCS)
def test_concurrent_centroid_calls(source_data, centroid_func):
    """
    Test that concurrent calls to each centroid function on shared
    input arrays return results identical to a serial call.
    """
    task = _make_task(centroid_func, *source_data)
    expected = task()

    with ThreadPoolExecutor(max_workers=N_THREADS) as executor:
        futures = [executor.submit(task)
                   for _ in range(N_THREADS * N_CALLS_PER_THREAD)]
        for future in futures:
            assert_array_equal(future.result(), expected)


def test_mixed_concurrent_centroid_calls(source_data):
    """
    Test all of the centroid functions running interleaved in one
    thread pool.

    Mixing the functions surfaces cross-call interference (e.g., the
    previous warning-capture convergence detection in centroid_2dg could
    suppress another thread's warnings or inject a spurious convergence
    warning into it).
    """
    tasks = {func.__name__: _make_task(func, *source_data)
             for func in CENTROID_FUNCS}
    expected = {name: task() for name, task in tasks.items()}

    with ThreadPoolExecutor(max_workers=N_THREADS) as executor:
        futures = [(name, executor.submit(task))
                   for name, task in tasks.items()
                   for _ in range(N_THREADS)]
        for name, future in futures:
            assert_array_equal(future.result(), expected[name])


def test_concurrent_convergence_warning_isolation():
    """
    Test that converging centroid_2dg fits stay warning-free while
    non-converging fits warn concurrently in other threads.

    With the previous warning-capture convergence detection, a
    non-converging fit in one thread could inject its warning into
    a converging fit in another thread (or have its own warning
    suppressed). Warnings are errors in the test suite, so an injected
    warning would fail the converging futures below.
    """
    good_data = make_gaussian_source((51, 51), 10.0, 24.7, 25.2, 3.0,
                                     3.0, 0)
    # The large offset makes the fit hit the maximum number of function
    # evaluations without converging
    bad_data = good_data + 100000.0

    def good_task():
        return centroid_2dg(good_data)

    def bad_task():
        try:
            centroid_2dg(bad_data)
        except Warning as exc:
            return str(exc)
        return 'no warning'

    expected = good_task()
    with ThreadPoolExecutor(max_workers=N_THREADS) as executor:
        good_futures = [executor.submit(good_task)
                        for _ in range(N_THREADS * N_CALLS_PER_THREAD)]
        bad_futures = [executor.submit(bad_task)
                       for _ in range(N_THREADS * N_CALLS_PER_THREAD)]
        for future in good_futures:
            assert_array_equal(future.result(), expected)
        for future in bad_futures:
            assert 'fit may' in future.result()


def test_concurrent_centroid_sources():
    """
    Test that concurrent centroid_sources calls on a shared image
    return results identical to a serial call.
    """
    xpos = [25.0, 75.0]
    ypos = [30.0, 70.0]
    data = (make_gaussian_source((100, 100), 10.0, xpos[0], ypos[0],
                                 4.0, 4.0, 0)
            + make_gaussian_source((100, 100), 10.0, xpos[1], ypos[1],
                                   4.0, 4.0, 0))
    error = np.ones(data.shape)

    def task():
        return centroid_sources(data, xpos, ypos, box_size=21,
                                centroid_func=centroid_2dg, error=error)

    expected = task()
    with ThreadPoolExecutor(max_workers=N_THREADS) as executor:
        futures = [executor.submit(task)
                   for _ in range(N_THREADS * N_CALLS_PER_THREAD)]
        for future in futures:
            xcen, ycen = future.result()
            assert_array_equal(xcen, expected[0])
            assert_array_equal(ycen, expected[1])
