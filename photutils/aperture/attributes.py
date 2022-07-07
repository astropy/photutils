# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
This module defines descriptor classes for aperture attribute
validation.
"""

import warnings

from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.utils.exceptions import AstropyDeprecationWarning
import numpy as np

__all__ = ['ApertureAttribute', 'PixelPositions', 'SkyCoordPositions',
           'PositiveScalar', 'ScalarAngle', 'ScalarAngleOrValue',
           'ScalarAngleOrPixel']


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
        instance.__dict__[self.name] = value

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

    In all cases, pixel positions are converted to a 2D `~numpy.ndarray`
    (without units).
    """

    def __set__(self, instance, value):
        # This is needed for zip to work seamlessly in Python 3
        # (e.g., positions = zip(xpos, ypos))
        if isinstance(value, zip):
            value = tuple(value)

        value = np.asanyarray(value).astype(float)  # np.ndarray
        self._validate(value)

        if isinstance(value, u.Quantity):
            # deprecated in version 1.4.0
            warnings.warn('Inputing positions as a Quantity is deprecated '
                          'and will be removed in a future version.',
                          AstropyDeprecationWarning)

            if value.unit != u.pixel:
                raise ValueError('Input positions must have pixel units')
            value = value.value

        if value.ndim == 2 and value.shape[1] != 2 and value.shape[0] == 2:
            raise ValueError('Input positions must be an (x, y) pixel '
                             'position or a list or array of (x, y) pixel '
                             'positions, e.g., [(x1, y1), (x2, y2), '
                             '(x3, y3)].')

        instance.__dict__[self.name] = value

    def _validate(self, value):
        if isinstance(value, u.Quantity) and value.unit != u.pixel:
            raise u.UnitsError(f'{self.name} must be in pixel units')

        if np.any(~np.isfinite(value)):
            raise ValueError(f'{self.name} must not contain any non-finite '
                             '(e.g., NaN or inf) positions')

        value = np.atleast_2d(value)
        if (value.shape[1] != 2 and value.shape[0] != 2) or value.ndim > 2:
            raise TypeError(f'{self.name!r} must be a (x, y) pixel position '
                            'or a list or array of (x, y) pixel positions.')


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
    Check that value is a scalar angle, either as an astropy Angle or
    Quantity with angular units.
    """

    def _validate(self, value):
        if isinstance(value, u.Quantity):
            if not value.isscalar:
                raise ValueError(f'{self.name!r} must be a scalar')

            if not value.unit.physical_type == 'angle':
                raise ValueError(f'{self.name!r} must have angular units')
        else:
            raise TypeError(f'{self.name!r} must be a scalar angle')


class ScalarAngleOrValue(ApertureAttribute):
    """
    Check that value is a scalar angle, either as a
    `~astropy.coordinates.Angle` or `~astropy.units.Quantity` with
    angular units, or a scalar float.
    """

    def __set__(self, instance, value):
        self._validate(value)
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

            if not (value.unit.physical_type == 'angle'
                    or value.unit == u.pixel):
                raise ValueError(f'{self.name!r} must have angular or pixel '
                                 'units')
        else:
            if not np.isscalar(value):
                raise TypeError(f'{self.name!r} must be a scalar float in '
                                'radians')


class ScalarAngleOrPixel(ApertureAttribute):
    """
    Check that value is a scalar angle, either as a
    `~astropy.coordinates.Angle` or `~astropy.units.Quantity` with
    angular units, or a scalar `~astropy.units.Quantity` in pixel units.

    The value must be strictly positive (> 0).
    """

    def _validate(self, value):
        if isinstance(value, u.Quantity):
            if not value.isscalar:
                raise ValueError(f'{self.name!r} must be a scalar')

            if not (value.unit.physical_type == 'angle'
                    or value.unit == u.pixel):
                raise ValueError(f'{self.name!r} must have angular or pixel '
                                 'units')

            if value.unit == u.pixel:
                warnings.warn('Inputing sky aperture quantities in pixel '
                              'units is deprecated and will be removed in '
                              'a future version.', AstropyDeprecationWarning)

            if not value > 0:
                raise ValueError(f'{self.name!r} must be strictly positive')
        else:
            raise TypeError(f'{self.name!r} must be a scalar angle or pixel '
                            'Quantity')
