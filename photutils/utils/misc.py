# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
This module provides tools to return the installed astropy and photutils
versions.
"""

from astropy.utils import deprecated

__all__ = ['get_version_info']


@deprecated('0.7')
def get_version_info():
    """
    Return astropy and photutils versions.

    Returns
    -------
    result : str
        The astropy and photutils versions.
    """

    return _get_version_info()


def _get_version_info():
    """
    Return astropy and photutils versions.

    Returns
    -------
    result : str
        The astropy and photutils versions.
    """

    from astropy import __version__
    astropy_version = __version__

    from photutils import __version__
    photutils_version = __version__

    return 'astropy: {0}, photutils: {1}'.format(astropy_version,
                                                 photutils_version)
