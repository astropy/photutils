# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tools for getting the installed package versions and the current date
and time for use in output metadata, for finding the cached properties
of a class, and for deep copying instances that other threads may
modify.
"""

import inspect
import sys
from copy import deepcopy
from datetime import UTC, datetime
from functools import cached_property


def get_cached_property_names(cls):
    """
    Return the names of the cached properties of a class, including
    those inherited from its superclasses.

    Classes store this result in a ``_cached_properties`` class
    attribute when they are created (see ``__init_subclass__``).

    The class is only read, never modified, so this function is safe
    to call concurrently from multiple threads, including on
    free-threaded Python builds.

    Parameters
    ----------
    cls : type
        The class to inspect.

    Returns
    -------
    result : tuple of str
        The sorted names of the `functools.cached_property` attributes
        of ``cls``.
    """
    members = inspect.getmembers(
        cls, predicate=lambda obj: isinstance(obj, cached_property))
    return tuple(name for name, _ in members)


def deepcopy_instance(obj, memo, cache_names=()):
    """
    Return a deep copy of an instance whose attributes other threads
    may add to during the copy.

    The default deep copy iterates the instance dict, which raises
    "dictionary changed size during iteration" if another thread
    stores a cached property at the same time. The instance dict and
    the dict attributes named in ``cache_names`` are each copied in one
    step before they are iterated.

    Otherwise the copy follows the `copy` module protocol. The slot
    values of a subclass that defines ``__slots__`` are copied, and a
    class that defines ``__getstate__`` or ``__setstate__`` is copied
    through those methods. In that case, only a state that is the
    instance dict itself is copied in one step, and ``cache_names``
    is not used. A cache dict that several objects share (e.g., a
    catalog and its slices) is also shared by their copies.

    Parameters
    ----------
    obj : object
        The instance to copy.

    memo : dict
        The `copy.deepcopy` memo dictionary.

    cache_names : tuple of str, optional
        The names of the dict attributes of ``obj`` that other threads
        may add to.

    Returns
    -------
    result : object
        A deep copy of ``obj``.
    """
    cls = obj.__class__
    result = cls.__new__(cls)
    memo[id(obj)] = result

    state = obj.__getstate__()
    if (cls.__getstate__ is not object.__getstate__
            or hasattr(cls, '__setstate__')):
        # A custom __getstate__ may also return the instance dict itself
        if state is obj.__dict__:
            state = dict(state)
        state = deepcopy(state, memo)
        if hasattr(result, '__setstate__'):
            result.__setstate__(state)
        else:
            result.__dict__.update(state)
        return result

    # The default state is the instance dict, or a (dict, slots) tuple
    # for a subclass that defines __slots__. Only the slot values are
    # taken from it. On a free-threaded build, the default state was
    # seen to be None (meaning an empty instance dict) for an instance
    # with attributes while other threads stored cached properties, so
    # the instance dict is read directly.
    slots = state[1] if isinstance(state, tuple) else {}
    attrs = dict(obj.__dict__)
    for name in cache_names:
        cache = attrs.pop(name)
        if id(cache) not in memo:
            memo[id(cache)] = deepcopy(dict(cache), memo)
            # Keep the cache alive until the copy is done so that its
            # id cannot be reused, as the copy module does
            memo.setdefault(id(memo), []).append(cache)
        result.__dict__[name] = memo[id(cache)]
    result.__dict__.update(deepcopy(attrs, memo))
    for name, value in deepcopy(slots, memo).items():
        setattr(result, name, value)
    return result


def _get_version_info():
    """
    Return a dictionary of the installed version numbers for photutils
    and its dependencies.

    Returns
    -------
    result : dict
        A dictionary containing the version numbers for photutils and
        its dependencies.
    """
    versions = {'Python': sys.version.split()[0]}

    packages = ('photutils', 'astropy', 'numpy', 'scipy', 'matplotlib',
                'gwcs', 'bottleneck')
    for package in packages:
        try:
            pkg = __import__(package)
            version = pkg.__version__
        except ImportError:
            version = None

        versions[package] = version

    return versions


def _get_date(*, utc=False):
    """
    Return a string of the current date/time.

    Parameters
    ----------
    utc : bool, optional
        Whether to use the UTC timezone instead of the local timezone.

    Returns
    -------
    result : str
        The current date/time.
    """
    try:
        now = datetime.now().astimezone() if not utc else datetime.now(UTC)
    except OSError:
        # System timezone may be unavailable on some configurations,
        # so fall back to UTC
        now = datetime.now(UTC)
    return now.strftime('%Y-%m-%d %H:%M:%S %Z')


def _get_meta(*, utc=False):
    """
    Return a metadata dictionary with the package versions and current
    date/time.

    Parameters
    ----------
    utc : bool, optional
        Whether to use the UTC timezone instead of the local timezone.

    Returns
    -------
    result : dict
        A dictionary containing package versions and the current
        date/time.
    """
    return {'date': _get_date(utc=utc),
            'version': _get_version_info()}
