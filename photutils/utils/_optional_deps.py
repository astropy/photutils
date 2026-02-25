# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tools for optional dependencies.

This module provides lazy checks for optional packages. Attributes
``HAS_<pkg>`` (e.g. ``HAS_MATPLOTLIB``, ``HAS_BOTTLENECK``) are booleans
indicating whether the corresponding package could be imported. Imports
are performed lazily on first attribute access.
"""

import importlib

# Check for optional dependencies using lazy import from `PEP 562
# <https://www.python.org/dev/peps/pep-0562/>`_.

# This list is a duplicate of the dependencies in pyproject.toml "all".
# Note that in some cases the package names are different from the
# pip-install name (e.g. scikit-image -> skimage).
optional_deps = ['matplotlib', 'regions', 'skimage', 'gwcs',
                 'bottleneck', 'tqdm', 'rasterio', 'shapely']
deps = {key.upper(): key for key in optional_deps}
__all__ = [f'HAS_{pkg}' for pkg in deps]

_cache = {}


def __getattr__(name):
    if name in __all__:
        if name not in _cache:
            try:
                importlib.import_module(deps[name[4:]])
            except ImportError:
                _cache[name] = False
            else:
                _cache[name] = True
        return _cache[name]

    msg = f'Module {__name__!r} has no attribute {name!r}'
    raise AttributeError(msg)
