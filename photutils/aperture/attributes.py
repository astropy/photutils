# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Descriptor classes for aperture attribute validation.
"""

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord

__all__ = [
    'ApertureAttribute',
    'PixelPositions',
    'PositiveScalar',
    'PositiveScalarAngle',
    'ScalarAngle',
    'ScalarAngleOrValue',
    'SkyCoordPositions',
]


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
            return self
        return instance.__dict__[self.name]

    def __set__(self, instance, value):
        self._validate(value)
        if not isinstance(value, (u.Quantity, SkyCoord)):
            value = float(value)
        self._validate_ordered_pairs(instance, value)
        # No need to reset if not already in the instance dict
        if self.name in instance.__dict__:
            self._reset_cached_properties(instance)
        instance.__dict__[self.name] = value

    def _validate_ordered_pairs(self, instance, value):
        """
        Validate the ordering invariants between related parameters
        declared by the owner class in ``_ordered_pairs``.

        ``_ordered_pairs`` is a tuple of ``(inner, outer)`` parameter
        name pairs, where the inner value must be strictly less than the
        outer value (e.g., an annulus inner and outer radius). The check
        runs whenever either parameter of a pair is assigned, comparing
        the new value against the other parameter's current value. It is
        skipped while the other parameter is not yet set (e.g., partway
        through ``__init__``). The check runs before the new value is
        stored, so a failed assignment leaves the instance unchanged.
        """
        for inner, outer in getattr(instance, '_ordered_pairs', ()):
            if self.name == inner:
                other = instance.__dict__.get(outer)
                if other is not None and not other > value:
                    msg = f'{outer!r} must be greater than {inner!r}'
                    raise ValueError(msg)
            elif self.name == outer:
                other = instance.__dict__.get(inner)
                if other is not None and not value > other:
                    msg = f'{outer!r} must be greater than {inner!r}'
                    raise ValueError(msg)

    def _reset_cached_properties(self, instance):
        # Reset cached properties (if they exist) for aperture parameter
        # changes
        try:
            for key in instance._cached_properties:
                instance.__dict__.pop(key, None)
        except AttributeError:
            pass

    def __delete__(self, instance):
        del instance.__dict__[self.name]

    def _validate(self, value):
        """
        Validate the attribute value.

        An exception is raised if the value is invalid.
        """


class PixelPositions(ApertureAttribute):
    """
    Validate and set positions for pixel-based apertures.

    Pixel positions are converted to a 2D `~numpy.ndarray`.
    """

    def __set__(self, instance, value):
        # Needed for zip positions (e.g., positions = zip(xpos, ypos))
        if isinstance(value, zip):
            value = tuple(value)

        value = self._validate(value)  # np.ndarray
        # No need to reset if not already in the instance dict
        if self.name in instance.__dict__:
            self._reset_cached_properties(instance)
        instance.__dict__[self.name] = value

    def _validate(self, value):
        try:
            value = np.asanyarray(value).astype(float)  # np.ndarray
        except TypeError as exc:
            # Value is a zip object containing Quantity objects
            msg = f'{self.name!r} must not be a Quantity'
            raise TypeError(msg) from exc

        if isinstance(value, u.Quantity):
            msg = f'{self.name!r} must not be a Quantity'
            raise TypeError(msg)

        if np.any(~np.isfinite(value)):
            msg = (f'{self.name!r} must not contain any non-finite '
                   '(e.g., NaN or inf) positions')
            raise ValueError(msg)

        value_2d = np.atleast_2d(value)
        if value_2d.ndim > 2 or value_2d.shape[1] != 2:
            msg = (f'{self.name!r} must be a (x, y) pixel position '
                   'or a list or array of (x, y) pixel positions, '
                   'e.g., [(x1, y1), (x2, y2), (x3, y3)]')
            raise ValueError(msg)

        return value


class SkyCoordPositions(ApertureAttribute):
    """
    Check that value is a `~astropy.coordinates.SkyCoord`.
    """

    def _validate(self, value):
        if not isinstance(value, SkyCoord):
            msg = f'{self.name!r} must be a SkyCoord instance'
            raise TypeError(msg)


class PositiveScalar(ApertureAttribute):
    """
    Check that value is a strictly positive (> 0) scalar.
    """

    def _validate(self, value):
        msg = f'{self.name!r} must be a positive scalar'
        if not np.isscalar(value):
            raise ValueError(msg)
        try:
            # NaN compares False, so it is also rejected here
            positive = value > 0
        except TypeError:
            # Non-numeric scalars (e.g., strings) do not support
            # comparison with 0
            raise TypeError(msg) from None
        if not positive:
            raise ValueError(msg)


class ScalarAngle(ApertureAttribute):
    """
    Check that value is a scalar angle, either as a
    `~astropy.coordinates.Angle` or `~astropy.units.Quantity` with
    angular units.
    """

    def _validate(self, value):
        if isinstance(value, u.Quantity):
            if not value.isscalar:
                msg = f'{self.name!r} must be a scalar'
                raise ValueError(msg)

            if value.unit.physical_type != 'angle':
                msg = f'{self.name!r} must have angular units'
                raise ValueError(msg)
        else:
            msg = f'{self.name!r} must be a scalar angle'
            raise TypeError(msg)


class PositiveScalarAngle(ScalarAngle):
    """
    Check that value is a strictly positive (> 0) scalar angle, either
    as a `~astropy.coordinates.Angle` or `~astropy.units.Quantity` with
    angular units.
    """

    def _validate(self, value):
        super()._validate(value)

        # NaN compares False, so it is also rejected here
        if not value > 0:
            msg = f'{self.name!r} must be greater than zero'
            raise ValueError(msg)


class ScalarAngleOrValue(ApertureAttribute):
    """
    Check that value is a scalar angle, either as a
    `~astropy.coordinates.Angle` or `~astropy.units.Quantity` with
    angular units, or a scalar float.

    The value is always output as a `~astropy.units.Quantity` with
    angular units. If the value is not a `~astropy.units.Quantity`, it
    is assumed to be in radians.
    """

    def __set__(self, instance, value):
        self._validate(value)
        # No need to reset if not already in the instance dict
        if self.name in instance.__dict__:
            self._reset_cached_properties(instance)

        # If theta is not a Quantity, it is assumed to be in radians
        if not isinstance(value, u.Quantity):
            value <<= u.radian
        instance.__dict__[self.name] = value

    def _validate(self, value):
        if isinstance(value, u.Quantity):
            if not value.isscalar:
                msg = f'{self.name!r} must be a scalar'
                raise ValueError(msg)

            if value.unit.physical_type != 'angle':
                msg = f'{self.name!r} must have angular units'
                raise ValueError(msg)
        elif not np.isscalar(value):
            msg = (f'If not an angle Quantity, {self.name!r} must be a '
                   'scalar float in radians')
            raise ValueError(msg)
