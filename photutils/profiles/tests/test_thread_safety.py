# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests that the profile classes are safe for concurrent use.

All lazily-computed attributes cache raw (unnormalized) values
that are immutable once computed, and the only mutable state is
the scalar ``normalization_value`` attribute, which ``normalize``
and ``unnormalize`` update with a single atomic attribute store.
Normalization-dependent attributes are derived on access by dividing
the raw values by ``normalization_value``, so a concurrent reader sees
values consistent with either the old or the new normalization, never a
mixture within one returned array.
"""

import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from numpy.testing import assert_equal

from photutils.profiles import CurveOfGrowth, RadialProfile

N_THREADS = 8
N_TOGGLES = 200
N_READS = 500


def test_concurrent_cold_access(profile_data):
    """
    Test that concurrent first accesses of the lazy attributes on a
    single shared instance return results identical to a serial
    computation.
    """
    xycen, data, error, _ = profile_data

    edge_radii = np.arange(36)
    rp = RadialProfile(data, xycen, edge_radii, error=error)
    reference = RadialProfile(data, xycen, edge_radii, error=error)
    expected = (reference.profile, reference.profile_error,
                reference.area)

    barrier = threading.Barrier(N_THREADS)

    def task():
        barrier.wait()
        return rp.profile, rp.profile_error, rp.area

    with ThreadPoolExecutor(max_workers=N_THREADS) as executor:
        futures = [executor.submit(task) for _ in range(N_THREADS)]
        for future in futures:
            for result, expect in zip(future.result(), expected,
                                      strict=True):
                assert_equal(result, expect)


def test_concurrent_normalize_and_read(profile_data):
    """
    Test that profile reads concurrent with normalize/unnormalize
    toggling always return one of the two consistent states, never a
    mixed state.
    """
    xycen, data, _, _ = profile_data

    radii = np.arange(1, 36)
    cog = CurveOfGrowth(data, xycen, radii)
    raw = cog.profile.copy()
    cog.normalize()
    normalized = cog.profile.copy()
    cog.unnormalize()

    barrier = threading.Barrier(N_THREADS)

    def writer():
        barrier.wait()
        for _ in range(N_TOGGLES):
            cog.normalize()
            cog.unnormalize()

    def reader():
        barrier.wait()
        return [cog.profile for _ in range(N_READS)]

    with ThreadPoolExecutor(max_workers=N_THREADS) as executor:
        writer_future = executor.submit(writer)
        reader_futures = [executor.submit(reader)
                          for _ in range(N_THREADS - 1)]
        writer_future.result()
        for future in reader_futures:
            for snapshot in future.result():
                assert (np.array_equal(snapshot, raw)
                        or np.array_equal(snapshot, normalized))


def test_concurrent_normalize_methods(profile_data):
    """
    Test that concurrent normalize calls with different methods leave
    the instance in one of the valid states, with the profile consistent
    with the final normalization value.
    """
    xycen, data, _, _ = profile_data

    radii = np.arange(1, 36)
    cog = CurveOfGrowth(data, xycen, radii)
    raw = cog.profile.copy()

    expected_values = set()
    for method in ('max', 'sum'):
        twin = CurveOfGrowth(data, xycen, radii)
        twin.normalize(method=method)
        expected_values.add(twin.normalization_value)

    barrier = threading.Barrier(N_THREADS)

    def task(method):
        barrier.wait()
        for _ in range(N_TOGGLES):
            cog.normalize(method=method)

    methods = ['max' if i % 2 == 0 else 'sum' for i in range(N_THREADS)]
    with ThreadPoolExecutor(max_workers=N_THREADS) as executor:
        futures = [executor.submit(task, method) for method in methods]
        for future in futures:
            future.result()

    assert cog.normalization_value in expected_values
    assert_equal(cog.profile, raw / cog.normalization_value)


def test_concurrent_fit_access_during_normalize(profile_data):
    """
    Test that the fit properties read concurrently with
    normalize/unnormalize toggling are always consistent with one of the
    two normalization states, and that the FWHM is unaffected.
    """
    xycen, data, _, _ = profile_data

    edge_radii = np.arange(36)
    rp = RadialProfile(data, xycen, edge_radii)
    raw_amplitude = rp.gaussian_fit.amplitude.value
    raw_fwhm = rp.gaussian_fwhm
    rp.normalize()
    norm_amplitude = rp.gaussian_fit.amplitude.value
    rp.unnormalize()

    n_workers = 4
    barrier = threading.Barrier(n_workers)

    def writer():
        barrier.wait()
        for _ in range(N_TOGGLES):
            rp.normalize()
            rp.unnormalize()

    def reader():
        barrier.wait()
        return [(rp.gaussian_fit.amplitude.value, rp.gaussian_fwhm)
                for _ in range(100)]

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        writer_future = executor.submit(writer)
        reader_futures = [executor.submit(reader)
                          for _ in range(n_workers - 1)]
        writer_future.result()
        for future in reader_futures:
            for amplitude, fwhm in future.result():
                assert amplitude in (raw_amplitude, norm_amplitude)
                assert fwhm == raw_fwhm
