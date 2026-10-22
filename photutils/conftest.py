# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Configuration file for the pytest test suite.
"""

from importlib.metadata import packages_distributions, requires, version

import pytest
from packaging.requirements import Requirement

from photutils.utils._optional_deps import _DIST_TO_IMPORT

try:
    from pytest_astropy_header.display import (PYTEST_HEADER_MODULES,
                                               TESTED_VERSIONS)
except ImportError:
    PYTEST_HEADER_MODULES = {}
    TESTED_VERSIONS = {}


def pytest_configure():
    """
    Configure pytest settings.
    """
    # Resolve the distribution name for this package
    import_name = __package__
    dist_name = packages_distributions().get(import_name, [import_name])[0]

    # Collect all dependency import names (core + 'all' extra)
    dep_names = set()
    for req_str in (requires(dist_name) or []):
        req = Requirement(req_str)
        if not req.marker or req.marker.evaluate({'extra': 'all'}):
            dep_names.add(_DIST_TO_IMPORT.get(req.name, req.name))

    PYTEST_HEADER_MODULES.clear()
    for dep in sorted(dep_names):
        PYTEST_HEADER_MODULES[dep] = dep

    TESTED_VERSIONS[dist_name] = version(dist_name)


def pytest_collection_modifyitems(items):
    """
    Run the tests that require matplotlib in a single thread with
    pytest-run-parallel.

    Matplotlib is not thread-safe. The pyplot figure state is global,
    and the mathtext parser fails when several threads render text at
    once. Every test that uses matplotlib is skipped without it, so
    the skip reason identifies these tests.
    """
    reason = 'matplotlib is required'
    for item in items:
        if any(marker.kwargs.get('reason') == reason
               for marker in item.iter_markers('skipif')):
            item.add_marker(pytest.mark.thread_unsafe(
                reason='matplotlib is not thread-safe'))
