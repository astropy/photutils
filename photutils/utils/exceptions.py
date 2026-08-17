# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Custom exceptions.
"""

from astropy.utils.exceptions import AstropyUserWarning, AstropyWarning

__all__ = ['DeblendWarning', 'NoDetectionsWarning']


class NoDetectionsWarning(AstropyWarning):
    """
    A warning class to indicate no sources were detected.
    """


class DeblendWarning(AstropyUserWarning):
    """
    A warning class to indicate issues encountered while deblending
    sources.
    """
