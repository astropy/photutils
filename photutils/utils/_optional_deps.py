# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tools for optional dependencies.

Attributes ``HAS_<PKG>`` (e.g., ``HAS_MATPLOTLIB``, ``HAS_SKIMAGE``)
are booleans that indicate whether the corresponding package can be
imported. The actual import is performed lazily on first attribute
access via :pep:`562`.

The ``HAS_*`` names are derived from the *import* name of each optional
dependency (uppercased, hyphens/dots replaced by underscores).
For the handful of packages whose import name differs from their
distribution (pip) name, a small translation dict (``_DIST_TO_IMPORT``)
is maintained.
"""

import importlib
from importlib.metadata import packages_distributions, requires

from packaging.requirements import Requirement

# Hardcoded translation for packages whose import name differs from
# their distribution (pip) name. All other packages are assumed to be
# importable using their distribution name directly. If a new optional
# dependency is added whose import name does not match its dist name,
# add a single entry below.
_DIST_TO_IMPORT = {
    'scikit-image': 'skimage',
}


def _get_optional_deps(dist_name, *, extra='all'):
    """
    Return the optional-dependency distribution names for ``dist_name``.

    Parameters
    ----------
    dist_name : str
        The distribution (pip) name of the package whose metadata is
        queried (e.g., ``'photutils'``).

    extra : str, optional
        The extras group to query (default ``'all'``).

    Returns
    -------
    deps : list of str
        Sorted distribution names of the optional dependencies.
    """
    deps = set()
    for req_str in (requires(dist_name) or []):
        req = Requirement(req_str)
        if req.marker and req.marker.evaluate({'extra': extra}):
            deps.add(req.name)
    return sorted(deps)


def _dist_to_has_key(dist_name):
    """
    Convert a distribution name to the corresponding ``HAS_*`` key.

    For example, ``'scikit-image'`` -> ``'SKIMAGE'`` and
    ``'matplotlib'`` -> ``'MATPLOTLIB'``.
    """
    import_name = _DIST_TO_IMPORT.get(dist_name, dist_name)

    return import_name.upper().replace('-', '_').replace('.', '_')


# Derive the distribution name of this package from its top-level import
# name
_pkg_import_name = __name__.split('.')[0]
_pkg_dist_name = packages_distributions().get(_pkg_import_name,
                                              [_pkg_import_name])[0]

# Build lookup: HAS_* suffix -> dist name.
_optional_deps = _get_optional_deps(_pkg_dist_name, extra='all')
_deps_by_key = {_dist_to_has_key(d): d for d in _optional_deps}

__all__ = [f'HAS_{key}' for key in sorted(_deps_by_key)]

_cache = {}


# Implemented as a module-level __getattr__ to allow for lazy imports on
# first access. See PEP 562 for details.
def __getattr__(name):
    if name.startswith('HAS_'):
        key = name[4:]
        if key in _deps_by_key:
            if name not in _cache:
                dist_name = _deps_by_key[key]
                import_name = _DIST_TO_IMPORT.get(dist_name, dist_name)
                try:
                    importlib.import_module(import_name)
                    _cache[name] = True
                except ImportError:
                    _cache[name] = False
            return _cache[name]

    msg = f'Module {__name__!r} has no attribute {name!r}'
    raise AttributeError(msg)
