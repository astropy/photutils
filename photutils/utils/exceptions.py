# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Custom exceptions.
"""

from astropy.utils.exceptions import (AstropyDeprecationWarning,
                                      AstropyUserWarning, AstropyWarning)

__all__ = ['DeblendWarning', 'NoDetectionsWarning',
           'PhotutilsDeprecationWarning']


class NoDetectionsWarning(AstropyWarning):
    """
    A warning class to indicate no sources were detected.
    """


class DeblendWarning(AstropyUserWarning):
    """
    A warning class to indicate issues encountered while deblending
    sources.
    """


class PhotutilsDeprecationWarning(AstropyDeprecationWarning):
    """
    A warning class to indicate deprecated Photutils features.

    This is a subclass of
    `~astropy.utils.exceptions.AstropyDeprecationWarning`, so existing
    warning filters for that class continue to match. The class name
    identifies Photutils as the source of the warning.
    """
