# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests that the datasets tools are safe for concurrent use.

The functions are pure (they copy the input model, use only local random
generators, and hold no module-level mutable state), so concurrent calls
must produce results identical to serial calls and must not modify
shared inputs.
"""

import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from astropy.modeling.models import Moffat2D
from astropy.table import QTable
from numpy.testing import assert_equal

from photutils.datasets import (make_model_image, make_model_params,
                                make_noise_image)

N_THREADS = 8
N_CALLS = 10


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


def test_concurrent_make_model_image():
    """
    Test that concurrent make_model_image calls sharing a single model
    instance and parameter table return results identical to a serial
    call and do not modify the shared model.
    """
    params = QTable()
    params['x_0'] = [25.0, 50.0, 75.0]
    params['y_0'] = [30.0, 50.0, 70.0]
    params['gamma'] = [1.7, 2.32, 5.8]
    params['alpha'] = [2.9, 5.7, 4.6]
    model = Moffat2D(amplitude=1)
    model_params_orig = model.parameters.copy()
    shape = (100, 100)
    model_shape = (11, 11)
    expected = make_model_image(shape, model, params,
                                model_shape=model_shape)

    def task():
        return make_model_image(shape, model, params,
                                model_shape=model_shape)

    for result in _run_concurrently(task):
        assert_equal(result, expected)

    assert_equal(model.parameters, model_params_orig)


def test_concurrent_make_model_params():
    """
    Test that concurrent seeded make_model_params calls return results
    identical to a serial call.
    """
    kwargs = {'flux': (100, 500), 'min_separation': 3, 'seed': 0}
    expected = make_model_params((100, 100), 5, **kwargs)

    def task():
        return make_model_params((100, 100), 5, **kwargs)

    for result in _run_concurrently(task):
        assert_equal(result['x_0'], expected['x_0'])
        assert_equal(result['y_0'], expected['y_0'])
        assert_equal(result['flux'], expected['flux'])


def test_concurrent_make_noise_image():
    """
    Test that concurrent seeded make_noise_image calls return results
    identical to a serial call.
    """
    shape = (100, 100)
    kwargs = {'distribution': 'gaussian', 'mean': 0.0, 'stddev': 2.0,
              'seed': 123}
    expected = make_noise_image(shape, **kwargs)

    def task():
        return make_noise_image(shape, **kwargs)

    for result in _run_concurrently(task):
        assert_equal(result, expected)


def test_concurrent_make_noise_image_unseeded_streams():
    """
    Test that concurrent unseeded make_noise_image calls use
    independent random streams (no shared global state).
    """
    shape = (10, 10)

    def task():
        return make_noise_image(shape, distribution='gaussian',
                                mean=0.0, stddev=2.0)

    results = _run_concurrently(task)
    stacked = np.array(results)
    # All results should be distinct (independent entropy per call)
    assert len(np.unique(stacked.sum(axis=(1, 2)))) == len(results)
