# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests that the morphology functions are thread-safe and return
identical results when run concurrently from multiple threads.

Both ``data_properties`` and ``gini`` are pure functions with no
module-level mutable state. Each call operates only on its inputs
and local variables, and ``data_properties`` constructs a new
``SourceCatalog`` per call.
"""

from concurrent.futures import ThreadPoolExecutor

import numpy as np
from numpy.testing import assert_array_equal

from photutils.morphology import data_properties, gini

N_THREADS = 8
N_CALLS_PER_THREAD = 4

DATA_PROPERTIES_COLUMNS = ['x_centroid', 'y_centroid', 'semimajor_axis',
                           'semiminor_axis', 'orientation', 'area']


def make_shared_inputs():
    """
    Return a shared image, mask, and background for the concurrent
    calls.

    Returns
    -------
    data : 2D `~numpy.ndarray`
        The image.

    mask : 2D `~numpy.ndarray`
        A boolean mask.

    background : 2D `~numpy.ndarray`
        A background image.
    """
    rng = np.random.default_rng(0)
    data = rng.random((51, 51))
    data[20:30, 20:30] += 50.0
    mask = np.zeros(data.shape, dtype=bool)
    mask[10, 10] = True
    background = np.full(data.shape, 0.5)
    return data, mask, background


def run_concurrently(task, expected):
    """
    Run ``task`` concurrently and check each result against
    ``expected``.

    Parameters
    ----------
    task : callable
        The zero-argument callable to run in the thread pool.

    expected : tuple
        The expected result from a serial call.
    """
    with ThreadPoolExecutor(max_workers=N_THREADS) as executor:
        futures = [executor.submit(task)
                   for _ in range(N_THREADS * N_CALLS_PER_THREAD)]
        for future in futures:
            assert_array_equal(future.result(), expected)


def test_concurrent_data_properties():
    """
    Test that concurrent data_properties calls on shared input arrays
    return results identical to a serial call.
    """
    data, mask, background = make_shared_inputs()

    def task():
        props = data_properties(data, mask=mask, background=background)
        values = [getattr(props, name).value
                  for name in DATA_PROPERTIES_COLUMNS
                  if name not in ('x_centroid', 'y_centroid')]
        values += [props.x_centroid, props.y_centroid,
                   props.background_sum]
        return values

    run_concurrently(task, task())


def test_concurrent_gini():
    """
    Test that concurrent gini calls on shared input arrays return
    results identical to a serial call.
    """
    data, mask, _ = make_shared_inputs()

    def task():
        return gini(data, mask=mask)

    run_concurrently(task, task())


def test_mixed_concurrent_calls():
    """
    Test data_properties and gini running interleaved in one thread
    pool.
    """
    data, mask, _ = make_shared_inputs()

    def gini_task():
        return gini(data, mask=mask)

    def props_task():
        props = data_properties(data, mask=mask)
        return [props.x_centroid, props.y_centroid, props.area.value]

    tasks = {'gini': gini_task, 'data_properties': props_task}
    expected = {name: task() for name, task in tasks.items()}

    with ThreadPoolExecutor(max_workers=N_THREADS) as executor:
        futures = [(name, executor.submit(task))
                   for name, task in tasks.items()
                   for _ in range(N_THREADS)]
        for name, future in futures:
            assert_array_equal(future.result(), expected[name])
