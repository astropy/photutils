# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Configuration file for the pytest test suite.
"""

from importlib.metadata import packages_distributions, requires, version

from packaging.requirements import Requirement

try:
    from pytest_astropy_header.display import (PYTEST_HEADER_MODULES,
                                               TESTED_VERSIONS)
except ImportError:
    PYTEST_HEADER_MODULES = {}
    TESTED_VERSIONS = {}


def _get_self_dependencies(extra='all'):
    """
    Get the package name and its dependencies.

    Parameters
    ----------
    extra : str, optional
        The extra requirements to consider (default is "all").

    Returns
    -------
    result : tuple
        A tuple containing the package name and a sorted list of its
        dependencies.

    Notes
    -----
    This function requires the package to be installed with valid
    metadata (e.g., via pip) in the current environment. Note that
    for namespace packages, a single import name may map to multiple
    distributions; this implementation identifies the first distribution
    found in the mapping.
    """
    pkg_dists = packages_distributions()

    # Build a reverse lookup map:
    # {'scikit-learn': 'sklearn', 'Pillow': 'PIL', ...}
    # packages_distributions() returns {'sklearn': ['scikit-learn'], ...}
    dist_to_import = {}
    for import_name, dist_names in pkg_dists.items():
        for dist in dist_names:
            dist_to_import[dist.lower()] = import_name

    # Determine which package this file belongs to
    import_name = __name__.split('.')[0]
    dist_names = pkg_dists.get(import_name)

    if not dist_names:
        msg = (f"Could not find the distribution for the package "
               f"containing '{import_name}'. This usually means that the "
               f"package is not installed in the current environment.")
        raise RuntimeError(msg)

    package_name = dist_names[0]
    raw_reqs = requires(package_name) or []

    dependencies = []
    for req_str in raw_reqs:
        req = Requirement(req_str)
        if not req.marker or req.marker.evaluate({'extra': extra}):
            dist_name = req.name.lower()
            dep_name = dist_to_import.get(dist_name, req.name)
            dependencies.append(dep_name)

    return package_name, sorted(set(dependencies))


def pytest_configure():
    """
    Configure pytest settings.
    """
    # Get dependencies from pyproject.toml
    project_name, deps = _get_self_dependencies(extra='all')

    PYTEST_HEADER_MODULES.clear()
    for dep in deps:
        PYTEST_HEADER_MODULES[dep] = dep

    TESTED_VERSIONS[project_name] = version(project_name)
