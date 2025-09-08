# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Define tools to get the installed astropy and photutils versions.
"""

import sys
from datetime import UTC, datetime


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

    packages = ('photutils', 'astropy', 'numpy', 'scipy', 'skimage',
                'matplotlib', 'gwcs', 'bottleneck')
    for package in packages:
        try:
            pkg = __import__(package)
            version = pkg.__version__
        except ImportError:
            version = None

        versions[package] = version

    return versions


def _get_date(utc=False):
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
    now = datetime.now().astimezone() if not utc else datetime.now(UTC)
    return now.strftime('%Y-%m-%d %H:%M:%S %Z')


def _get_meta(utc=False):
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
