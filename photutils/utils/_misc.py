# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
This module provides tools to return the installed astropy and photutils
versions.
"""


def _get_version_info():
    """
    Return astropy and photutils versions.

    Returns
    -------
    result : str
        The astropy and photutils versions.
    """

    from astropy import __version__ as astropy_version
    from photutils import __version__ as photutils_version

    return 'astropy: {0}, photutils: {1}'.format(astropy_version,
                                                 photutils_version)
