# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
This module provides custom exceptions.
"""

from astropy.utils.exceptions import AstropyWarning

__all__ = ['NoDetectionsWarning']


class NoDetectionsWarning(AstropyWarning):
    """
    A warning class to indicate no sources were detected.
    """
