# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Configuration file for the pytest test suite.
"""

import pytest


def pytest_collection_modifyitems(items):
    """
    Run the tests that require matplotlib or rasterio in a single
    thread with pytest-run-parallel.

    Matplotlib is not thread-safe. The pyplot figure state is global,
    and the mathtext parser fails when several threads render text at
    once. Rasterio can crash on free-threaded builds when several
    threads call it at once. Every test that uses matplotlib or
    rasterio is skipped without it, so the skip reason identifies
    these tests.
    """
    for item in items:
        reasons = [marker.kwargs.get('reason', '')
                   for marker in item.iter_markers('skipif')]
        for dep in ('matplotlib', 'rasterio'):
            if any(dep in reason for reason in reasons):
                item.add_marker(pytest.mark.thread_unsafe(
                    reason=f'{dep} is not thread-safe'))
                break
