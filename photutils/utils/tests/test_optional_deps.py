# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the _optional_deps module.
"""

import importlib
from unittest.mock import patch

import pytest

import photutils.utils._optional_deps as od_mod
from photutils.utils._optional_deps import (_deps_by_key, _dist_to_has_key,
                                            _get_optional_deps, _pkg_dist_name)


def _clear_cache(*names):
    """
    Remove named entries from the module-level cache.

    This is necessary to ensure that tests of the caching behavior are
    valid and not affected by previous tests.
    """
    for name in names:
        od_mod._cache.pop(name, None)


class TestAll:
    def test_all_is_list(self):
        """
        Test that ``__all__`` is a list.
        """
        assert isinstance(od_mod.__all__, list)

    def test_all_entries_start_with_has(self):
        """
        Test that every entry in ``__all__`` starts with ``HAS_``.
        """
        for name in od_mod.__all__:
            assert name.startswith('HAS_')

    def test_all_is_sorted(self):
        """
        Test that ``__all__`` is sorted alphabetically.
        """
        assert od_mod.__all__ == sorted(od_mod.__all__)

    def test_skimage_always_in_all(self):
        """
        Test that ``HAS_SKIMAGE`` is always present in ``__all__``.
        """
        assert 'HAS_SKIMAGE' in od_mod.__all__

    def test_all_matches_deps_by_key(self):
        """
        Test that ``__all__`` matches the keys in ``_deps_by_key``.
        """
        expected = sorted(f'HAS_{k}' for k in _deps_by_key)
        assert od_mod.__all__ == expected


class TestHasAttributes:
    def test_installed_package_returns_true(self):
        """
        Test that a ``HAS_*`` attribute is `True` when the import
        succeeds.
        """
        _clear_cache('HAS_MATPLOTLIB')
        try:
            with patch.object(importlib, 'import_module'):
                assert od_mod.HAS_MATPLOTLIB is True
        finally:
            _clear_cache('HAS_MATPLOTLIB')

    def test_import_error_returns_false(self):
        """
        Test that a ``HAS_*`` attribute is `False` when the import
        raises `ImportError`.
        """
        _clear_cache('HAS_MATPLOTLIB')
        try:
            with patch.object(importlib, 'import_module',
                              side_effect=ImportError):
                result = od_mod.HAS_MATPLOTLIB
            assert result is False
        finally:
            _clear_cache('HAS_MATPLOTLIB')

    def test_result_is_bool(self):
        """
        Test that ``HAS_*`` attributes are `bool`.
        """
        _clear_cache('HAS_MATPLOTLIB')
        try:
            assert isinstance(od_mod.HAS_MATPLOTLIB, bool)
        finally:
            _clear_cache('HAS_MATPLOTLIB')


class TestCaching:
    def test_value_is_cached(self):
        """
        Test that the result of a ``HAS_*`` lookup is stored in the
        cache.
        """
        _clear_cache('HAS_MATPLOTLIB')
        try:
            _ = od_mod.HAS_MATPLOTLIB
            assert 'HAS_MATPLOTLIB' in od_mod._cache
        finally:
            _clear_cache('HAS_MATPLOTLIB')

    def test_cached_value_returned_without_reimport(self):
        """
        Test that a cached value is returned without calling
        ``import_module`` again.
        """
        _clear_cache('HAS_MATPLOTLIB')
        od_mod._cache['HAS_MATPLOTLIB'] = True
        try:
            with patch.object(
                importlib, 'import_module',
                side_effect=AssertionError('should not be called'),
            ):
                assert od_mod.HAS_MATPLOTLIB is True
        finally:
            _clear_cache('HAS_MATPLOTLIB')


class TestAttributeErrors:
    def test_typo_raises(self):
        """
        Test that a plausible typo like ``HAS_SKIIMAGE`` raises
        `AttributeError`.
        """
        match = 'HAS_SKIIMAGE'
        with pytest.raises(AttributeError, match=match):
            _ = od_mod.HAS_SKIIMAGE

    def test_non_dependency_raises(self):
        """
        Test that ``HAS_NUMPY`` raises `AttributeError` because numpy is
        not an optional dependency.
        """
        if 'NUMPY' not in od_mod._deps_by_key:
            match = 'HAS_NUMPY'
            with pytest.raises(AttributeError, match=match):
                _ = od_mod.HAS_NUMPY

    def test_arbitrary_name_raises(self):
        """
        Test that a completely unknown attribute raises
        `AttributeError`.
        """
        with pytest.raises(AttributeError):
            _ = od_mod.SOME_UNKNOWN_ATTRIBUTE

    def test_has_prefix_only_raises(self):
        """
        Test that ``HAS_`` with no suffix raises `AttributeError`.
        """
        with pytest.raises(AttributeError):
            _ = od_mod.HAS_


class TestDistToHasKey:
    def test_simple(self):
        """
        Test that a simple distribution name is uppercased.
        """
        assert _dist_to_has_key('matplotlib') == 'MATPLOTLIB'

    def test_hyphenated(self):
        """
        Test that a hyphenated distribution name is converted via the
        import-name lookup (e.g., ``scikit-image`` -> ``SKIMAGE``).
        """
        assert _dist_to_has_key('scikit-image') == 'SKIMAGE'

    def test_dotted(self):
        """
        Test that dots in a distribution name are replaced with
        underscores.
        """
        assert _dist_to_has_key('stsci.stimage') == 'STSCI_STIMAGE'

    def test_mixed_case(self):
        """
        Test that mixed-case distribution names are uppercased.
        """
        assert _dist_to_has_key('MyPackage') == 'MYPACKAGE'


class TestGetOptionalDeps:
    def test_returns_sorted_list(self):
        """
        Test that ``_get_optional_deps`` returns a sorted list.
        """
        result = _get_optional_deps(_pkg_dist_name, extra='all')
        assert isinstance(result, list)
        assert result == sorted(result)

    def test_scikit_image_present(self):
        """
        Test that ``scikit-image`` appears in the optional
        dependencies.
        """
        result = _get_optional_deps(_pkg_dist_name, extra='all')
        assert 'scikit-image' in result

    def test_returns_dist_names_not_import_names(self):
        """
        Test that the returned names are distribution names, not import
        names (e.g., ``scikit-image`` instead of ``skimage``).
        """
        result = _get_optional_deps(_pkg_dist_name, extra='all')
        assert 'scikit-image' in result
        assert 'skimage' not in result
