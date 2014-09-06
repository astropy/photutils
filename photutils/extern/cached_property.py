# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
_cached_property function copied from scikit-image _regionprops.py.
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)


class _cached_property(object):
    """Decorator to use a function as a cached property.

    The function is only called the first time and each successive call returns
    the cached result of the first call.

        class Foo(object):

            @_cached_property
            def foo(self):
                return "Cached"

        class Foo(object):

            def __init__(self):
                self._cache_active = False

            @_cached_property
            def foo(self):
                return "Not cached"

    Adapted from <http://wiki.python.org/moin/PythonDecoratorLibrary>.
    """

    def __init__(self, func, name=None, doc=None):
        self.__name__ = name or func.__name__
        self.__module__ = func.__module__
        self.__doc__ = doc or func.__doc__
        self.func = func

    def __get__(self, obj, type=None):
        if obj is None:
            return self

        # call every time, if cache is not active
        if not obj.__dict__.get('_cache_active', True):
            return self.func(obj)

        # try to retrieve from cache or call and store result in cache
        try:
            value = obj.__dict__[self.__name__]
        except KeyError:
            value = self.func(obj)
            obj.__dict__[self.__name__] = value
        return value
