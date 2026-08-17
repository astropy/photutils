# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests that the background tools are safe for concurrent use.

The estimator classes are stateless after construction. With the
default string-based ``cenfunc`` and ``stdfunc`` options, sigma
clipping routes through astropy's fast C implementation, which computes
its clipping bounds in local variables, so concurrent calls on a
shared estimator instance produce results identical to serial calls.
A `~astropy.stats.SigmaClip` instance with callable or biweight
``cenfunc``/``stdfunc`` options stores its bounds on the instance during
calls and must not be shared between threads.

`~photutils.background.Background2D` copies the input data before
inserting NaN values for masked pixels, so concurrent constructions may
share input arrays. After construction, the instance is immutable except
for two idempotent cached properties, so concurrent property reads on a
shared instance are safe.

`~photutils.background.LocalBackground` holds no mutable state and
computes the standard estimators through batched compiled kernels.
"""

import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_equal

from photutils.background import (Background2D, BiweightLocationBackground,
                                  BiweightScaleBackgroundRMS, LocalBackground,
                                  MADStdBackgroundRMS, MeanBackground,
                                  MedianBackground, MMMBackground,
                                  ModeEstimatorBackground,
                                  SExtractorBackground, StdBackgroundRMS)

N_THREADS = 8
N_CALLS = 10


@pytest.fixture(name='image_data')
def fixture_image_data():
    """
    A shared image with sources, outliers, and a mask.
    """
    rng = np.random.default_rng(0)
    data = rng.normal(loc=5.0, scale=2.0, size=(90, 90))
    data[20:25, 30:35] += 100.0  # source
    data[60, 70] = 1000.0  # outlier for sigma clipping
    mask = np.zeros(data.shape, dtype=bool)
    mask[40:45, 40:45] = True
    return data, mask


def run_concurrently(task, n_threads=N_THREADS):
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


BKG_ESTIMATORS = [MeanBackground, MedianBackground,
                  ModeEstimatorBackground, MMMBackground,
                  SExtractorBackground, BiweightLocationBackground]
RMS_ESTIMATORS = [StdBackgroundRMS, MADStdBackgroundRMS,
                  BiweightScaleBackgroundRMS]


@pytest.mark.parametrize('estimator_cls',
                         BKG_ESTIMATORS + RMS_ESTIMATORS)
def test_concurrent_estimator_calls(estimator_cls, image_data):
    """
    Test that concurrent calls to a shared estimator instance with
    default sigma clipping return results identical to serial calls
    and do not modify the shared input data.
    """
    data, _ = image_data
    data_orig = data.copy()
    estimator = estimator_cls()
    expected_scalar = estimator(data)
    expected_rows = estimator(data, axis=1)

    def task():
        return estimator(data), estimator(data, axis=1)

    for scalar, rows in run_concurrently(task):
        assert_equal(scalar, expected_scalar)
        assert_equal(rows, expected_rows)

    assert_equal(data, data_orig)


@pytest.mark.parametrize('n_threads', [1, 2])
def test_concurrent_background2d_construction(n_threads, image_data):
    """
    Test that concurrent Background2D constructions sharing the input
    data and mask arrays return results identical to a serial
    construction and do not modify the shared inputs.

    With ``n_threads`` > 1, each construction also runs its own
    internal thread pool, exercising nested threading.
    """
    data, mask = image_data
    data_orig = data.copy()
    kwargs = {'mask': mask, 'exclude_percentile': 50.0,
              'n_threads': n_threads}
    # A box size that does not evenly divide the data shape
    # exercises the extra row, column, and corner box paths
    reference = Background2D(data, (25, 25), **kwargs)

    def task():
        bkg = Background2D(data, (25, 25), **kwargs)
        return (bkg.background_mesh, bkg.background_rms_mesh,
                bkg.background, bkg.background_rms)

    for result in run_concurrently(task, n_threads=4):
        assert_equal(result[0], reference.background_mesh)
        assert_equal(result[1], reference.background_rms_mesh)
        # The multithreaded full-size interpolation is identical up
        # to floating-point rounding
        assert_allclose(result[2], reference.background)
        assert_allclose(result[3], reference.background_rms)

    assert_equal(data, data_orig)


def test_concurrent_background2d_shared_instance(image_data):
    """
    Test that concurrent property reads on a shared Background2D
    instance, including the cold first accesses of the cached median
    properties, return results identical to serial reads.
    """
    data, mask = image_data
    bkg = Background2D(data, (30, 30), mask=mask,
                       exclude_percentile=50.0)
    reference = Background2D(data, (30, 30), mask=mask,
                             exclude_percentile=50.0)
    expected = (reference.background, reference.background_rms,
                reference.background_median,
                reference.background_rms_median,
                reference.n_pixels_map)

    def task():
        return (bkg.background, bkg.background_rms,
                bkg.background_median, bkg.background_rms_median,
                bkg.n_pixels_map)

    for result in run_concurrently(task):
        for value, expect in zip(result, expected, strict=True):
            assert_equal(value, expect)


def test_concurrent_local_background(image_data):
    """
    Test that concurrent calls to a shared LocalBackground instance
    return results identical to a serial call, for both a standard
    estimator (batched kernel path) and a custom estimator callable
    (per-aperture loop path).
    """
    data, mask = image_data
    x = np.array([20.0, 45.5, 70.0])
    y = np.array([25.0, 45.5, 65.0])
    local_bkgs = [LocalBackground(5, 10),
                  LocalBackground(5, 10, bkg_estimator=np.nanmedian)]

    for local_bkg in local_bkgs:
        expected = local_bkg(data, x, y, mask=mask)

        def task(local_bkg=local_bkg):
            return local_bkg(data, x, y, mask=mask)

        for result in run_concurrently(task):
            assert_equal(result, expected)
