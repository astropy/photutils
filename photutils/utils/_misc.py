# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
This module provides tools to return the installed astropy and photutils
versions.
"""


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
    versions = {}
    packages = ('photutils', 'astropy', 'numpy', 'scipy', 'skimage')
    for package in packages:
        try:
            pkg = __import__(package)
            version = pkg.__version__
        except ImportError:
            version = None

        versions[package] = version

    return versions
