# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
This module defines descriptor classes for aperture attribute
validation.
"""

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord

__all__ = ['ApertureAttribute', 'PixelPositions', 'SkyCoordPositions',
           'PositiveScalar', 'ScalarAngle', 'ScalarAngleOrValue']


class ApertureAttribute:
    """
    Base descriptor class for aperture attribute validation.

    Parameters
    ----------
    doc : str, optional
        The description string for the attribute.
    """

    def __init__(self, doc=''):
        self.__doc__ = doc
        self.name = ''

    def __set_name__(self, owner, name):
        self.name = name

    def __get__(self, instance, owner):
        if instance is None:
            return self  # pragma: no cover
        return instance.__dict__[self.name]

    def __set__(self, instance, value):
        self._validate(value)
        if not isinstance(value, (u.Quantity, SkyCoord)):
            value = float(value)
        # no need to reset if not already in the instance dict
        if self.name in instance.__dict__:
            self._reset_lazyproperties(instance)
        instance.__dict__[self.name] = value

    def _reset_lazyproperties(self, instance):
        # reset lazyproperties (if they exist) for aperture
        # parameter changes
        try:
            for key in instance._lazyproperties:
                instance.__dict__.pop(key, None)
        except AttributeError:
            pass

    def __delete__(self, instance):
        del instance.__dict__[self.name]  # pragma: no cover

    def _validate(self, value):
        """
        Validate the attribute value.

        An exception is raised if the value is invalid.
        """
        raise NotImplementedError  # pragma: no cover


class PixelPositions(ApertureAttribute):
    """
    Validate and set positions for pixel-based apertures.

    Pixel positions are converted to a 2D `~numpy.ndarray`.
    """

    def __set__(self, instance, value):
        # this is needed for zip (e.g., positions = zip(xpos, ypos))
        if isinstance(value, zip):
            value = tuple(value)

        value = self._validate(value)  # np.ndarray
        # no need to reset if not already in the instance dict
        if self.name in instance.__dict__:
            self._reset_lazyproperties(instance)
        instance.__dict__[self.name] = value

    def _validate(self, value):
        try:
            value = np.asanyarray(value).astype(float)  # np.ndarray
        except TypeError:
            # value is a zip object containing Quantity objects
            raise TypeError(f'{self.name!r} must not be a Quantity')

        if isinstance(value, u.Quantity):
            raise TypeError(f'{self.name!r} must not be a Quantity')

        if np.any(~np.isfinite(value)):
            raise ValueError(f'{self.name!r} must not contain any non-finite '
                             '(e.g., NaN or inf) positions')

        value_2d = np.atleast_2d(value)
        if value_2d.ndim > 2 or value_2d.shape[1] != 2:
            raise ValueError(f'{self.name!r} must be a (x, y) pixel position '
                             'or a list or array of (x, y) pixel positions, '
                             'e.g., [(x1, y1), (x2, y2), (x3, y3)]')

        return value


class SkyCoordPositions(ApertureAttribute):
    """
    Check that value is a `~astropy.coordinates.SkyCoord`.
    """

    def _validate(self, value):
        if not isinstance(value, SkyCoord):
            raise ValueError(f'{self.name!r} must be a SkyCoord instance')


class PositiveScalar(ApertureAttribute):
    """
    Check that value is a strictly positive (> 0) scalar.
    """

    def _validate(self, value):
        if not np.isscalar(value) or value <= 0:
            raise ValueError(f'{self.name!r} must be a positive scalar')


class ScalarAngle(ApertureAttribute):
    """
    Check that value is a scalar angle, either as a
    `~astropy.coordinates.Angle` or `~astropy.units.Quantity` with
    angular units.
    """

    def _validate(self, value):
        if isinstance(value, u.Quantity):
            if not value.isscalar:
                raise ValueError(f'{self.name!r} must be a scalar')

            if not value.unit.physical_type == 'angle':
                raise ValueError(f'{self.name!r} must have angular units')
        else:
            raise ValueError(f'{self.name!r} must be a scalar angle')


class PositiveScalarAngle(ApertureAttribute):
    """
    Check that value is a positive scalar angle, either as a
    `~astropy.coordinates.Angle` or `~astropy.units.Quantity` with
    angular units.
    """

    def _validate(self, value):
        if value <= 0:
            raise ValueError(f'{self.name!r} must be greater than zero')

        if isinstance(value, u.Quantity):
            if not value.isscalar:
                raise ValueError(f'{self.name!r} must be a scalar')

            if not value.unit.physical_type == 'angle':
                raise ValueError(f'{self.name!r} must have angular units')
        else:
            raise ValueError(f'{self.name!r} must be a scalar angle')


class ScalarAngleOrValue(ApertureAttribute):
    """
    Check that value is a scalar angle, either as a
    `~astropy.coordinates.Angle` or `~astropy.units.Quantity` with
    angular units, or a scalar float.
    """

    def __set__(self, instance, value):
        self._validate(value)
        # no need to reset if not already in the instance dict
        if self.name in instance.__dict__:
            self._reset_lazyproperties(instance)
        instance.__dict__[self.name] = value

        # also store the angle in radians as a float
        if isinstance(value, u.Quantity):
            value = value.to(u.radian).value
        name = f'_{self.name}_radians'
        instance.__dict__[name] = value

    def _validate(self, value):
        if isinstance(value, u.Quantity):
            if not value.isscalar:
                raise ValueError(f'{self.name!r} must be a scalar')

            if not value.unit.physical_type == 'angle':
                raise ValueError(f'{self.name!r} must have angular units')
        else:
            if not np.isscalar(value):
                raise ValueError(f'If not an angle Quantity, {self.name!r} '
                                 'must be a scalar float in radians')
