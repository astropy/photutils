# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Descriptor class(es) for aperture attribute validation.
"""

import weakref

from astropy.coordinates import SkyCoord
import astropy.units as u
import numpy as np


__all__ = ['ApertureAttribute', 'PixelPositions', 'SkyCoordPositions',
           'Scalar', 'PositiveScalar', 'AngleOrPixelScalarQuantity']


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


class PixelPositions(ApertureAttribute):
    """
    Validate and set positions for pixel-based apertures.

    In all cases, pixel positions are converted to a 2D `~numpy.ndarray`
    (without units).
    """

    def __set__(self, instance, value):
        # This is needed for zip to work seamlessly in Python 3
        # (e.g. positions = zip(xpos, ypos))
        if isinstance(value, zip):
            value = tuple(value)

        value = np.atleast_2d(value).astype(float)  # np.ndarray
        self._validate(value)

        if isinstance(value, u.Quantity):
            value = value.value

        if value.shape[1] != 2 and value.shape[0] == 2:
            value = np.transpose(value)

        self.values[instance] = value

    def _validate(self, value):
        if isinstance(value, u.Quantity) and value.unit != u.pixel:
            raise u.UnitsError(f'{self.name} must be in pixel units')

        if (value.shape[1] != 2 and value.shape[0] != 2) or value.ndim > 2:
            raise TypeError(f'{self.name} must be an (x, y) pixel position '
                            'or a list or array of (x, y) pixel positions.')

        if np.any(~np.isfinite(value)):
            raise ValueError(f'{self.name} must not contain any non-finite '
                             '(e.g. NaN or inf) positions')


class SkyCoordPositions(ApertureAttribute):
    """
    Check that value is a `~astropy.coordinates.SkyCoord`.
    """

    def _validate(self, value):
        if not isinstance(value, SkyCoord):
            raise ValueError(f'{self.name} must be a SkyCoord instance')


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


class AngleOrPixelScalarQuantity(ApertureAttribute):
    """
    Check that value is either an angular or a pixel scalar
    `~astropy.units.Quantity`.
    """

    def _validate(self, value):
        if isinstance(value, u.Quantity):
            if not value.isscalar:
                raise ValueError(f'{self.name} must be a scalar')

            if not (value.unit.physical_type == 'angle' or
                    value.unit == u.pixel):
                raise ValueError(f'{self.name} must have angular or pixel '
                                 'units')
        else:
            raise TypeError(f'{self.name} must be an astropy Quantity '
                            'instance')
