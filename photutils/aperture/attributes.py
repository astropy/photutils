# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
This module defines descriptor classes for aperture attribute
validation.
"""

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

    def __get__(self, instance, owner):
        if instance is None:
            return self
        return instance.__dict__[self.name]

    def __set__(self, instance, value):
        self._validate(value)
        if not isinstance(value, (u.Quantity, SkyCoord)):
            value = float(value)
        instance.__dict__[self.name] = value

    def __delete__(self, instance):
        del instance.__dict__[self.name]

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
        # (e.g., positions = zip(xpos, ypos))
        if isinstance(value, zip):
            value = tuple(value)

        value = np.asanyarray(value).astype(float)  # np.ndarray
        self._validate(value)

        if isinstance(value, u.Quantity):
            value = value.value

        if value.ndim == 2 and value.shape[1] != 2 and value.shape[0] == 2:
            raise ValueError('Input positions must be an (x, y) pixel '
                             'position or a list or array of (x, y) pixel '
                             'positions, e.g., [(x1, y1), (x2, y2), '
                             '(x3, y3)].')

        instance.__dict__[self.name] = value

    def _validate(self, value):
        if isinstance(value, u.Quantity) and value.unit != u.pixel:
            raise u.UnitsError('{} must be in pixel units'.format(self.name))

        if np.any(~np.isfinite(value)):
            raise ValueError('{} must not contain any non-finite (e.g., NaN '
                             'or inf) positions'.format(self.name))

        value = np.atleast_2d(value)
        if (value.shape[1] != 2 and value.shape[0] != 2) or value.ndim > 2:
            raise TypeError('{} must be an (x, y) pixel position or a list '
                            'or array of (x, y) pixel positions.'
                            .format(self.name))


class SkyCoordPositions(ApertureAttribute):
    """
    Check that value is a `~astropy.coordinates.SkyCoord`.
    """

    def _validate(self, value):
        if not isinstance(value, SkyCoord):
            raise ValueError('{} must be a SkyCoord instance'
                             .format(self.name))


class Scalar(ApertureAttribute):
    """
    Check that value is a scalar.
    """

    def _validate(self, value):
        if not np.isscalar(value):
            raise ValueError('{} must be a scalar'.format(self.name))


class PositiveScalar(ApertureAttribute):
    """
    Check that value is a strictly positive (> 0) scalar.
    """

    def _validate(self, value):
        if not np.isscalar(value) or value <= 0:
            raise ValueError('{} must be a positive scalar'.format(self.name))


class AngleScalarQuantity(ApertureAttribute):
    """
    Check that value is either an angular scalar
    `~astropy.units.Quantity`.
    """

    def _validate(self, value):
        if isinstance(value, u.Quantity):
            if not value.isscalar:
                raise ValueError('{} must be a scalar'.format(self.name))

            if not value.unit.physical_type == 'angle':
                raise ValueError('{} must have angular units'
                                 .format(self.name))
        else:
            raise TypeError('{} must be an astropy Quantity instance'.
                            format(self.name))


class AngleOrPixelScalarQuantity(ApertureAttribute):
    """
    Check that value is either an angular or a pixel scalar
    `~astropy.units.Quantity`.
    """

    def _validate(self, value):
        if isinstance(value, u.Quantity):
            if not value.isscalar:
                raise ValueError('{} must be a scalar'.format(self.name))

            if not (value.unit.physical_type == 'angle' or
                    value.unit == u.pixel):
                raise ValueError('{} must have angular or pixel units'
                                 .format(self.name))
        else:
            raise TypeError('{} must be an astropy Quantity instance'
                            .format(self.name))
