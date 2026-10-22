# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Thread-safety checks shared by the tests of several subpackages.

The helpers are kept in a plain module so that test files can import
them without importing a ``conftest.py``, which pytest also loads as a
plugin.
"""

import threading
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy

from photutils.utils._misc import get_cached_property_names


def check_concurrent_deepcopy(make_obj, *, names=None, n_threads=8,
                              n_rounds=30):
    """
    Check that a shared object can be deep copied while other threads
    compute its cached properties.

    Each round makes a new object. Every thread computes the cached
    properties in a different order and deep copies the object before
    each one. The default deep copy iterates the instance dict, which
    raises "dictionary changed size during iteration" if another thread
    stores a cached property at the same time.

    Parameters
    ----------
    make_obj : callable
        A function that takes no arguments and returns a new object.

    names : tuple of str, optional
        The names of the cached properties to compute. If `None`, all
        the cached properties of the class of the object are used.

    n_threads : int, optional
        The number of threads.

    n_rounds : int, optional
        The number of rounds, each with a new object.
    """
    for _ in range(n_rounds):
        obj = make_obj()
        if names is None:
            names = get_cached_property_names(type(obj))
        barrier = threading.Barrier(n_threads)

        def worker(start, obj=obj, names=names, barrier=barrier):
            barrier.wait()
            for name in names[start:] + names[:start]:
                deepcopy(obj)
                getattr(obj, name)

        starts = [i % len(names) for i in range(n_threads)]
        with ThreadPoolExecutor(max_workers=n_threads) as executor:
            list(executor.map(worker, starts))
