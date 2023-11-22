# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Checks for optional dependencies using lazy import from `PEP 562
<https://www.python.org/dev/peps/pep-0562/>`_.
"""

import importlib

# This list is a duplicate of the dependencies in pyproject.toml "all".
# Note that in some cases the package names are different from the
# pip-install name (e.g.k scikit-image -> skimage).
optional_deps = ['scipy', 'matplotlib', 'skimage', 'sklearn', 'gwcs',
                 'bottleneck', 'tqdm', 'rasterio', 'shapely']
deps = {key.upper(): key for key in optional_deps}
__all__ = [f'HAS_{pkg}' for pkg in deps]


def __getattr__(name):
    if name in __all__:
        try:
            importlib.import_module(deps[name[4:]])
        except ImportError:
            return False
        return True

    raise AttributeError(f'Module {__name__!r} has no attribute {name!r}.')
