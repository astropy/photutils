# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests that the psf_matching tools are safe for concurrent use.

The kernel-making functions are pure (they copy their inputs and hold no
module-level mutable state) and the window classes are immutable after
construction, so concurrent calls must produce results identical to
serial calls and must not modify shared inputs.
"""

import threading
from concurrent.futures import ThreadPoolExecutor

from numpy.testing import assert_equal

from photutils.psf_matching import (SplitCosineBellWindow, make_kernel,
                                    make_wiener_kernel, resize_psf)

N_THREADS = 8
N_CALLS = 25


def _run_concurrently(task, n_threads=N_THREADS):
    """
    Run ``task`` in ``n_threads`` threads and return all results.

    A barrier maximizes the overlap of the concurrent calls.
    """
    barrier = threading.Barrier(n_threads)

    def worker():
        barrier.wait()
        return [task() for _ in range(N_CALLS)]

    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        futures = [executor.submit(worker) for _ in range(n_threads)]
        return [result for future in futures for result in future.result()]


def test_concurrent_make_kernel(psf1, psf2):
    """
    Test that concurrent make_kernel calls on shared input arrays
    return results identical to a serial call and do not modify the
    inputs.
    """
    window = SplitCosineBellWindow(alpha=0.2, beta=0.3)
    psf1_orig = psf1.copy()
    psf2_orig = psf2.copy()
    expected = make_kernel(psf1, psf2, window=window)

    def task():
        return make_kernel(psf1, psf2, window=window)

    for result in _run_concurrently(task):
        assert_equal(result, expected)

    assert_equal(psf1, psf1_orig)
    assert_equal(psf2, psf2_orig)


def test_concurrent_make_wiener_kernel(psf1, psf2):
    """
    Test that concurrent make_wiener_kernel calls with a penalty
    return results identical to a serial call.
    """
    expected = make_wiener_kernel(psf1, psf2, penalty='laplacian')

    def task():
        return make_wiener_kernel(psf1, psf2, penalty='laplacian')

    for result in _run_concurrently(task):
        assert_equal(result, expected)


def test_concurrent_resize_psf(psf1):
    """
    Test that concurrent resize_psf calls on a shared input array
    return results identical to a serial call and do not modify the
    input.
    """
    psf1_orig = psf1.copy()
    expected = resize_psf(psf1, 0.1, 0.05)

    def task():
        return resize_psf(psf1, 0.1, 0.05)

    for result in _run_concurrently(task):
        assert_equal(result, expected)

    assert_equal(psf1, psf1_orig)


def test_concurrent_shared_window_instance():
    """
    Test that a single window instance shared across threads returns
    identical results for concurrent calls with different shapes.
    """
    window = SplitCosineBellWindow(alpha=0.4, beta=0.3)
    shapes = [(25, 25), (51, 51), (25, 51), (51, 25)]
    expected = [window(shape) for shape in shapes]

    def task():
        return [window(shape) for shape in shapes]

    for results in _run_concurrently(task):
        for result, expect in zip(results, expected, strict=True):
            assert_equal(result, expect)
