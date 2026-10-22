# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the _misc module.
"""

import builtins
import threading
from copy import deepcopy
from functools import cached_property
from unittest.mock import patch

import numpy as np
import pytest

from photutils.utils._misc import (_get_date, _get_meta, _get_version_info,
                                   deepcopy_instance,
                                   get_cached_property_names)


def test_get_cached_property_names():
    """
    Test that get_cached_property_names returns the sorted names of
    the cached properties of a class, including inherited ones, and
    not plain properties.
    """
    class Base:
        @cached_property
        def b_value(self):
            return 1

        @property
        def plain(self):
            return 2

    class Derived(Base):
        @cached_property
        def a_value(self):
            return 3

    assert get_cached_property_names(Base) == ('b_value',)
    assert get_cached_property_names(Derived) == ('a_value', 'b_value')
    assert 'plain' not in get_cached_property_names(Derived)


def test_deepcopy_instance():
    """
    Test that deepcopy_instance deep copies the instance dict and the
    named cache dicts, and keeps references back to the instance.
    """
    class Cached:
        def __init__(self):
            self.values = np.arange(3)
            self.cache = {'a': np.ones(2)}
            self.parent = self

        def __deepcopy__(self, memo):
            return deepcopy_instance(self, memo, ('cache',))

        @cached_property
        def total(self):
            return self.values.sum()

    obj = Cached()
    _ = obj.total
    result = deepcopy(obj)

    assert type(result) is Cached
    assert result.total == obj.total
    assert result.parent is result
    assert result.values is not obj.values
    assert result.cache is not obj.cache
    assert result.cache['a'] is not obj.cache['a']
    result.cache['b'] = 1
    assert 'b' not in obj.cache


class _DeepcopyInstance:
    """
    Base class whose deep copies are made with deepcopy_instance.
    """

    def __deepcopy__(self, memo):
        return deepcopy_instance(self, memo, getattr(self, 'cache_names',
                                                     ()))


def test_deepcopy_instance_shared_cache():
    """
    Test that a cache dict shared by several objects is also shared by
    their copies.
    """
    class Cached(_DeepcopyInstance):
        cache_names = ('cache',)

        def __init__(self, cache):
            self.cache = cache

    cache = {'a': np.ones(2)}
    obj1, obj2 = deepcopy([Cached(cache), Cached(cache)])
    assert obj1.cache is obj2.cache
    assert obj1.cache is not cache
    assert obj1.cache['a'] is not cache['a']


def test_deepcopy_instance_slots():
    """
    Test that deepcopy_instance copies the slot values of a subclass
    that defines __slots__.
    """
    class Slotted(_DeepcopyInstance):
        __slots__ = ('value',)

    obj = Slotted()
    obj.value = [1, 2]
    obj.other = 3
    result = deepcopy(obj)
    assert result.value == [1, 2]
    assert result.value is not obj.value
    assert result.other == 3


@pytest.mark.parametrize('define_setstate', [True, False])
def test_deepcopy_instance_custom_state(define_setstate):
    """
    Test that deepcopy_instance copies a subclass that defines its own
    state through its __getstate__ and __setstate__ methods.
    """
    class CustomState(_DeepcopyInstance):
        def __init__(self):
            self.values = [1, 2]
            self.lock = threading.Lock()

        def __getstate__(self):
            state = self.__dict__.copy()
            del state['lock']
            return state

    if define_setstate:
        def setstate(self, state):
            self.__dict__.update(state)
            self.lock = threading.Lock()

        CustomState.__setstate__ = setstate

    obj = CustomState()
    result = deepcopy(obj)
    assert result.values == [1, 2]
    assert result.values is not obj.values
    if define_setstate:
        assert result.lock is not obj.lock
    else:
        assert 'lock' not in result.__dict__


def test_deepcopy_instance_custom_state_instance_dict():
    """
    Test that deepcopy_instance copies a custom state that is the
    instance dict itself without changing the original instance.
    """
    class DictState(_DeepcopyInstance):
        def __init__(self):
            self.values = [1, 2]

        def __getstate__(self):
            return self.__dict__

        def __setstate__(self, state):
            state['restored'] = True
            self.__dict__ = state

    obj = DictState()
    result = deepcopy(obj)
    assert result.values == [1, 2]
    assert result.values is not obj.values
    assert result.restored
    assert 'restored' not in obj.__dict__


def test_get_version_info_import_error():
    """
    Test that _get_version_info returns None for packages that cannot
    be imported.
    """
    real_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == 'gwcs':
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    with patch('builtins.__import__', side_effect=mock_import):
        versions = _get_version_info()
    assert versions['gwcs'] is None


@pytest.mark.parametrize('utc', [False, True])
def test_get_date_oserror_fallback(utc):
    """
    Test that _get_date falls back when astimezone raises OSError.
    """
    with patch('photutils.utils._misc.datetime') as mock_dt:
        mock_now = mock_dt.now.return_value
        mock_now.astimezone.side_effect = OSError('no timezone')
        mock_now.strftime.return_value = '2025-01-01 00:00:00'
        mock_dt.now.return_value = mock_now
        from datetime import UTC
        mock_dt.UTC = UTC
        result = _get_date(utc=utc)
        assert isinstance(result, str)


@pytest.mark.parametrize('utc', [False, True])
def test_get_meta(utc):
    """
    Test _get_meta returns expected keys.
    """
    meta = _get_meta(utc=utc)
    keys = ('date', 'version')
    for key in keys:
        assert key in meta

    versions = meta['version']
    assert isinstance(versions, dict)
    keys = ('Python', 'photutils', 'astropy', 'numpy', 'scipy',
            'matplotlib', 'gwcs', 'bottleneck')
    for key in keys:
        assert key in versions
