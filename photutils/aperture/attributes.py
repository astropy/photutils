# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Descriptor class(es) for aperture attribute validation.
"""

import weakref

from astropy.coordinates import SkyCoord
import astropy.units as u
import numpy as np


class ApertureAttribute:
    """
    Base descriptor class for aperture attribute validation.
    """

    def __init__(self, name):
        self.name = name
        self.values = weakref.WeakKeyDictionary()

    def __get__(self, instance, owner):
        if instance is None:
            return self
        return self.values.get(instance, None)

    def __set__(self, instance, value):
        self._validate(value)
        if isinstance(value, (u.Quantity, SkyCoord)):
            self.values[instance] = value
        else:
            self.values[instance] = float(value)

    def _validate(self, value):
        """
        Validate the attribute value.

        An exception is raised if the value is invalid.
        """

        raise NotImplementedError


class Scalar(ApertureAttribute):
    """
    Check that value is a scalar.
    """

    def _validate(self, value):
        if not np.isscalar(value):
            raise ValueError(f'{self.name} must be a scalar')


class PositiveScalar(ApertureAttribute):
    """
    Check that value is a stricly positive (>= 0) scalar.
    """

    def _validate(self, value):
        if not np.isscalar(value) or value <= 0:
            raise ValueError(f'{self.name} must be a positive scalar')
