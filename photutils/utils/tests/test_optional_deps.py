# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the _optional_deps module.
"""

import builtins
import importlib
from unittest.mock import patch

import photutils.utils._optional_deps as od_mod


def test_has_missing_package():
    """
    Test that HAS_<pkg> returns False for a package that cannot be
    imported.
    """
    real_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == 'rasterio':
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    # Clear the cache so the lazy check runs again
    od_mod._cache.pop('HAS_RASTERIO', None)

    with patch.object(importlib, 'import_module', side_effect=mock_import):
        result = od_mod.HAS_RASTERIO

    assert result is False

    # Clear cache to restore normal state
    od_mod._cache.pop('HAS_RASTERIO', None)
