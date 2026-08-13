# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests that the utils tools are safe for concurrent use.

``ImageDepth.__call__`` uses only local state (a fresh per-call random
generator and a local copy of the SigmaClip instance), so concurrent
calls on a shared, seeded instance must produce results identical to
serial calls. The interpolator and coordinate tools are read-only or
pure, so concurrent calls must also match serial results.
"""

import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from astropy.stats import SigmaClip
from numpy.testing import assert_equal

from photutils.utils import ImageDepth, ShepardIDWInterpolator
from photutils.utils._coords import make_random_xycoords

N_THREADS = 8
N_CALLS = 3


def _run_concurrently(task, *, n_threads=N_THREADS, n_calls=N_CALLS):
    """
    Run ``task`` in ``n_threads`` threads and return all results.

    A barrier maximizes the overlap of the concurrent calls.
    """
    barrier = threading.Barrier(n_threads)

    def worker():
        barrier.wait()
        return [task() for _ in range(n_calls)]

    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        futures = [executor.submit(worker) for _ in range(n_threads)]
        return [result for future in futures for result in future.result()]


def _make_depth_inputs():
    """
    Return a noise image and a source mask for the ImageDepth tests.
    """
    rng = np.random.default_rng(1)
    data = rng.normal(0.0, 1.0, (150, 150))
    mask = np.zeros(data.shape, dtype=bool)
    mask[60:80, 40:70] = True
    return data, mask


def test_concurrent_image_depth_shared_instance():
    """
    Test that concurrent calls on a shared, seeded ImageDepth instance
    return results identical to a serial call and leave the result
    attributes consistent.
    """
    data, mask = _make_depth_inputs()
    n_iters = 2
    depth = ImageDepth(3, n_sigma=5.0, n_apertures=50, n_iters=n_iters,
                       mask_pad=2, overlap=True, seed=123, zeropoint=23.9,
                       progress_bar=False)
    expected = depth(data, mask)
    expected_fluxes = depth.fluxes

    def task():
        return depth(data, mask)

    for result in _run_concurrently(task):
        assert_equal(result, expected)

    # The result attributes reflect a single completed call
    assert len(depth.fluxes) == n_iters
    assert len(depth.apertures) == n_iters
    assert len(depth.flux_limits) == n_iters
    for fluxes, fluxes_exp in zip(depth.fluxes, expected_fluxes, strict=True):
        assert_equal(fluxes, fluxes_exp)


def test_concurrent_image_depth_shared_sigma_clip():
    """
    Test that ImageDepth instances sharing a user-input SigmaClip
    instance can run concurrently, producing results identical to
    serial calls (each call must use a local copy of the SigmaClip
    instance).
    """
    data, mask = _make_depth_inputs()
    sigma_clip = SigmaClip(sigma=3.0, maxiters=5)

    def make_depth():
        return ImageDepth(3, n_sigma=5.0, n_apertures=50, n_iters=1,
                          mask_pad=2, overlap=True, seed=42,
                          sigma_clip=sigma_clip, progress_bar=False)

    expected = make_depth()(data, mask)

    def task():
        return make_depth()(data, mask)

    for result in _run_concurrently(task):
        assert_equal(result, expected)


def test_concurrent_idw_interpolator():
    """
    Test that concurrent queries of a shared ShepardIDWInterpolator
    return results identical to a serial call.
    """
    rng = np.random.default_rng(0)
    coords = rng.random((500, 2))
    values = np.sin(coords[:, 0] + coords[:, 1])
    interp = ShepardIDWInterpolator(coords, values)
    positions = rng.random((100, 2))
    expected = interp(positions)

    def task():
        return interp(positions)

    for result in _run_concurrently(task, n_calls=10):
        assert_equal(result, expected)


def test_concurrent_make_random_xycoords():
    """
    Test that concurrent seeded make_random_xycoords calls return
    results identical to a serial call.
    """
    kwargs = {'min_separation': 2.0, 'seed': 7}
    expected = make_random_xycoords(100, (0, 500), (0, 300), **kwargs)

    def task():
        return make_random_xycoords(100, (0, 500), (0, 300), **kwargs)

    for result in _run_concurrently(task, n_calls=10):
        assert_equal(result, expected)
