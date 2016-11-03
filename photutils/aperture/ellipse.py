# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import math

import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.wcs.utils import skycoord_to_pixel

from .core import (PixelAperture, SkyAperture, ApertureMask,
                   _sanitize_pixel_positions, _translate_mask_method,
                   _make_annulus_path)
from ..geometry import elliptical_overlap_grid
from ..utils.wcs_helpers import (skycoord_to_pixel_scale_angle, assert_angle,
                                 assert_angle_or_pixel)


skycoord_to_pixel_mode = 'all'


__all__ = ['EllipticalMaskMixin', 'EllipticalAperture', 'EllipticalAnnulus',
           'SkyEllipticalAperture', 'SkyEllipticalAnnulus']


class EllipticalMaskMixin(object):
    """
    Mixin class to create masks for elliptical and elliptical-annulus
    aperture objects.
    """

    def to_mask(self, method='exact', subpixels=5):
        """
        Returns
        -------
        mask : list of `~photutils.ApertureMask`
            A list of aperture mask objects.
        """

        if method not in ('center', 'subpixel', 'exact'):
            raise ValueError('"{0}" method is not available for this '
                             'aperture.'.format(method))

        use_exact, subpixels = _translate_mask_method(method, subpixels)

        if hasattr(self, 'a'):
            a = self.a
            b = self.b
        elif hasattr(self, 'a_in'):    # annulus
            a = self.a_out
            b = self.b_out
            b_in = self.a_in * self.b_out / self.a_out
        else:
            raise ValueError('Cannot determine the aperture shape.')

        masks = []
        for position, _slice, _geom_slice in zip(self.positions, self._slices,
                                                 self._geom_slices):
            px_min, px_max = _geom_slice[1].start, _geom_slice[1].stop
            py_min, py_max = _geom_slice[0].start, _geom_slice[0].stop
            dx = px_max - px_min
            dy = py_max - py_min

            mask = elliptical_overlap_grid(px_min, px_max, py_min, py_max,
                                           dx, dy, a, b, self.theta,
                                           use_exact, subpixels)

            if hasattr(self, 'a_in'):    # annulus
                mask -= elliptical_overlap_grid(px_min, px_max, py_min,
                                                py_max, dx, dy, self.a_in,
                                                b_in, self.theta, use_exact,
                                                subpixels)

            masks.append(ApertureMask(position, mask, _slice, _geom_slice))

        return masks


class EllipticalAperture(EllipticalMaskMixin, PixelAperture):
    """
    Elliptical aperture(s), defined in pixel coordinates.

    Parameters
    ----------
    positions : tuple, list, array, or `~astropy.units.Quantity`
        Pixel coordinates of the aperture center(s), either as a single
        ``(x, y)`` tuple, a list of ``(x, y)`` tuples, an ``Nx2`` or
        ``2xN`` `~numpy.ndarray`, or an ``Nx2`` or ``2xN``
        `~astropy.units.Quantity` in units of pixels.  A ``2x2``
        `~numpy.ndarray` or `~astropy.units.Quantity` is interpreted as
        ``Nx2``, i.e. two rows of (x, y) coordinates.
    a : float
        The semimajor axis.
    b : float
        The semiminor axis.
    theta : float
        The position angle of the semimajor axis in radians
        (counterclockwise).

    Raises
    ------
    ValueError : `ValueError`
        If either axis (``a`` or ``b``) is negative.
    """

    def __init__(self, positions, a, b, theta):
        try:
            self.a = float(a)
            self.b = float(b)
            self.theta = float(theta)
        except TypeError:
            raise TypeError("'a' and 'b' and 'theta' must be numeric, "
                            "received {0} and {1} and {2}."
                            .format((type(a), type(b), type(theta))))

        if a < 0 or b < 0:
            raise ValueError("'a' and 'b' must be non-negative.")

        self.positions = _sanitize_pixel_positions(positions)

    @property
    def _slices(self):
        # TODO:  use an actual minimal bounding box
        radius = max(self.a, self.b)
        x_min = np.floor(self.positions[:, 0] - radius + 0.5).astype(int)
        x_max = np.floor(self.positions[:, 0] + radius + 1.5).astype(int)
        y_min = np.floor(self.positions[:, 1] - radius + 0.5).astype(int)
        y_max = np.floor(self.positions[:, 1] + radius + 1.5).astype(int)

        return [(slice(ymin, ymax), slice(xmin, xmax))
                for xmin, xmax, ymin, ymax in zip(x_min, x_max, y_min, y_max)]

    def area(self):
        return math.pi * self.a * self.b

    def plot(self, origin=(0, 0), source_id=None, ax=None, fill=False,
             **kwargs):
        import matplotlib.patches as mpatches

        plot_positions, ax, kwargs = self._prepare_plot(
            origin, source_id, ax, fill, **kwargs)

        theta_deg = self.theta * 180. / np.pi
        for position in plot_positions:
            patch = mpatches.Ellipse(position, 2.*self.a, 2.*self.b,
                                     theta_deg, **kwargs)
            ax.add_patch(patch)


class EllipticalAnnulus(EllipticalMaskMixin, PixelAperture):
    """
    Elliptical annulus aperture(s), defined in pixel coordinates.

    Parameters
    ----------
    positions : tuple, list, array, or `~astropy.units.Quantity`
        Pixel coordinates of the aperture center(s), either as a single
        ``(x, y)`` tuple, a list of ``(x, y)`` tuples, an ``Nx2`` or
        ``2xN`` `~numpy.ndarray`, or an ``Nx2`` or ``2xN``
        `~astropy.units.Quantity` in units of pixels.  A ``2x2``
        `~numpy.ndarray` or `~astropy.units.Quantity` is interpreted as
        ``Nx2``, i.e. two rows of (x, y) coordinates.
    a_in : float
        The inner semimajor axis.
    a_out : float
        The outer semimajor axis.
    b_out : float
        The outer semiminor axis. (The inner semiminor axis is determined
        by scaling by a_in/a_out.)
    theta : float
        The position angle of the semimajor axis in radians.
        (counterclockwise).

    Raises
    ------
    ValueError : `ValueError`
        If inner semimajor axis (``a_in``) is greater than outer semimajor
        axis (``a_out``).
    ValueError : `ValueError`
        If either the inner semimajor axis (``a_in``) or the outer semiminor
        axis (``b_out``) is negative.
    """

    def __init__(self, positions, a_in, a_out, b_out, theta):
        try:
            self.a_in = float(a_in)
            self.a_out = float(a_out)
            self.b_out = float(b_out)
            self.theta = float(theta)
        except TypeError:
            raise TypeError("'a_in' and 'a_out' and 'b_out' and 'theta' must "
                            "be numeric, received {0} and {1} and {2} and "
                            "{3}.".format((type(a_in), type(a_out),
                                           type(b_out), type(theta))))

        if not (a_out > a_in):
            raise ValueError("'a_out' must be greater than 'a_in'")
        if a_in < 0 or b_out < 0:
            raise ValueError("'a_in' and 'b_out' must be non-negative")

        self.b_in = a_in * b_out / a_out

        self.positions = _sanitize_pixel_positions(positions)

    @property
    def _slices(self):
        # TODO:  use an actual minimal bounding box
        radius = max(self.a_out, self.b_out)
        x_min = np.floor(self.positions[:, 0] - radius + 0.5).astype(int)
        x_max = np.floor(self.positions[:, 0] + radius + 1.5).astype(int)
        y_min = np.floor(self.positions[:, 1] - radius + 0.5).astype(int)
        y_max = np.floor(self.positions[:, 1] + radius + 1.5).astype(int)

        return [(slice(ymin, ymax), slice(xmin, xmax))
                for xmin, xmax, ymin, ymax in zip(x_min, x_max, y_min, y_max)]

    def area(self):
        return math.pi * (self.a_out * self.b_out - self.a_in * self.b_in)

    def plot(self, origin=(0, 0), source_id=None, ax=None, fill=False,
             **kwargs):
        import matplotlib.patches as mpatches

        plot_positions, ax, kwargs = self._prepare_plot(
            origin, source_id, ax, fill, **kwargs)

        theta_deg = self.theta * 180. / np.pi
        for position in plot_positions:
            patch_inner = mpatches.Ellipse(position, 2.*self.a_in,
                                           2.*self.b_in, theta_deg, **kwargs)
            patch_outer = mpatches.Ellipse(position, 2.*self.a_out,
                                           2.*self.b_out, theta_deg, **kwargs)
            path = _make_annulus_path(patch_inner, patch_outer)
            patch = mpatches.PathPatch(path, **kwargs)
            ax.add_patch(patch)


class SkyEllipticalAperture(SkyAperture):
    """
    Elliptical aperture(s), defined in sky coordinates.

    Parameters
    ----------
    positions : `~astropy.coordinates.SkyCoord`
        Celestial coordinates of the aperture center(s). This can be either
        scalar coordinates or an array of coordinates.
    a : `~astropy.units.Quantity`
        The semimajor axis, either in angular or pixel units.
    b : `~astropy.units.Quantity`
        The semiminor axis, either in angular or pixel units.
    theta : `~astropy.units.Quantity`
        The position angle of the semimajor axis (counterclockwise), either
        in angular or pixel units.
    """

    def __init__(self, positions, a, b, theta):
        if isinstance(positions, SkyCoord):
            self.positions = positions
        else:
            raise TypeError("positions should be a SkyCoord instance")

        assert_angle_or_pixel('a', a)
        assert_angle_or_pixel('b', b)
        assert_angle('theta', theta)

        if a.unit.physical_type != b.unit.physical_type:
            raise ValueError("a and b should either both be angles "
                             "or in pixels")

        self.a = a
        self.b = b
        self.theta = theta

    def to_pixel(self, wcs):
        """
        Return a EllipticalAperture instance in pixel coordinates.
        """

        x, y = skycoord_to_pixel(self.positions, wcs,
                                 mode=skycoord_to_pixel_mode)
        central_pos = SkyCoord([wcs.wcs.crval], frame=self.positions.name,
                               unit=wcs.wcs.cunit)
        xc, yc, scale, angle = skycoord_to_pixel_scale_angle(central_pos, wcs)

        if self.a.unit.physical_type == 'angle':
            a = (scale * self.a).to(u.pixel).value
            b = (scale * self.b).to(u.pixel).value
        else:  # pixel
            a = self.a.value
            b = self.b.value

        theta = (angle + self.theta).to(u.radian).value

        pixel_positions = np.array([x, y]).transpose()

        return EllipticalAperture(pixel_positions, a, b, theta)


class SkyEllipticalAnnulus(SkyAperture):
    """
    Elliptical annulus aperture(s), defined in sky coordinates.

    Parameters
    ----------
    positions : `~astropy.coordinates.SkyCoord`
        Celestial coordinates of the aperture center(s). This can be either
        scalar coordinates or an array of coordinates.
    a_in : `~astropy.units.Quantity`
        The inner semimajor axis, either in angular or pixel units.
    a_out : `~astropy.units.Quantity`
        The outer semimajor axis, either in angular or pixel units.
    b_out : `~astropy.units.Quantity`
        The outer semiminor axis, either in angular or pixel units. The inner
        semiminor axis is determined by scaling by a_in/a_out.
    theta : `~astropy.units.Quantity`
        The position angle of the semimajor axis (counterclockwise), either
        in angular or pixel units.
    """

    def __init__(self, positions, a_in, a_out, b_out, theta):
        if isinstance(positions, SkyCoord):
            self.positions = positions
        else:
            raise TypeError("positions should be a SkyCoord instance")

        assert_angle_or_pixel('a_in', a_in)
        assert_angle_or_pixel('a_out', a_out)
        assert_angle_or_pixel('b_out', b_out)
        assert_angle('theta', theta)

        if a_in.unit.physical_type != a_out.unit.physical_type:
            raise ValueError("a_in and a_out should either both be angles "
                             "or in pixels")

        if a_out.unit.physical_type != b_out.unit.physical_type:
            raise ValueError("a_out and b_out should either both be angles "
                             "or in pixels")

        self.a_in = a_in
        self.a_out = a_out
        self.b_out = b_out
        self.theta = theta

    def to_pixel(self, wcs):
        """
        Return a EllipticalAnnulus instance in pixel coordinates.
        """

        x, y = skycoord_to_pixel(self.positions, wcs,
                                 mode=skycoord_to_pixel_mode)
        central_pos = SkyCoord([wcs.wcs.crval], frame=self.positions.name,
                               unit=wcs.wcs.cunit)
        xc, yc, scale, angle = skycoord_to_pixel_scale_angle(central_pos, wcs)

        if self.a_in.unit.physical_type == 'angle':
            a_in = (scale * self.a_in).to(u.pixel).value
            a_out = (scale * self.a_out).to(u.pixel).value
            b_out = (scale * self.b_out).to(u.pixel).value
        else:
            a_in = self.a_in.value
            a_out = self.a_out.value
            b_out = self.b_out.value

        theta = (angle + self.theta).to(u.radian).value

        pixel_positions = np.array([x, y]).transpose()

        return EllipticalAnnulus(pixel_positions, a_in, a_out, b_out, theta)
