# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tools for getting the installed package versions and the current date
and time for use in output metadata, and for finding the cached
properties of a class.
"""

import inspect
import sys
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
