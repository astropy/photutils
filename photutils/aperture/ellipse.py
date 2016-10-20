# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import math
import warnings

import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.wcs.utils import skycoord_to_pixel
from astropy.utils.exceptions import AstropyUserWarning

from .core import (SkyAperture, PixelAperture, _sanitize_pixel_positions,
                   _make_annulus_path, _get_phot_extents, find_fluxvar)
from ..utils.wcs_helpers import (skycoord_to_pixel_scale_angle, assert_angle,
                                 assert_angle_or_pixel)


skycoord_to_pixel_mode = 'all'


__all__ = ['SkyEllipticalAperture', 'EllipticalAperture',
           'SkyEllipticalAnnulus', 'EllipticalAnnulus']


def do_elliptical_photometry(data, positions, a, b, theta, error,
                             pixelwise_error, method, subpixels, a_in=None):

    extents = np.zeros((len(positions), 4), dtype=int)

    # TODO: we can be more efficient in terms of bounding box
    radius = max(a, b)

    extents[:, 0] = positions[:, 0] - radius + 0.5
    extents[:, 1] = positions[:, 0] + radius + 1.5
    extents[:, 2] = positions[:, 1] - radius + 0.5
    extents[:, 3] = positions[:, 1] + radius + 1.5

    ood_filter, extent, phot_extent = _get_phot_extents(data, positions,
                                                        extents)

    flux = u.Quantity(np.zeros(len(positions), dtype=np.float), unit=data.unit)

    if error is not None:
        fluxvar = u.Quantity(np.zeros(len(positions), dtype=np.float),
                             unit=error.unit ** 2)

    # TODO: flag these objects
    if np.sum(ood_filter):
        flux[ood_filter] = np.nan
        warnings.warn("The aperture at position {0} does not have any "
                      "overlap with the data"
                      .format(positions[ood_filter]),
                      AstropyUserWarning)
        if np.sum(ood_filter) == len(positions):
            return (flux, )

    x_min, x_max, y_min, y_max = extent
    x_pmin, x_pmax, y_pmin, y_pmax = phot_extent

    if method == 'center':
        use_exact = 0
        subpixels = 1
    elif method == 'subpixel':
        use_exact = 0
    else:
        use_exact = 1
        subpixels = 1

    from ..geometry import elliptical_overlap_grid

    for i in range(len(flux)):

        if not np.isnan(flux[i]):

            fraction = elliptical_overlap_grid(x_pmin[i], x_pmax[i],
                                               y_pmin[i], y_pmax[i],
                                               x_max[i] - x_min[i],
                                               y_max[i] - y_min[i],
                                               a, b, theta, use_exact,
                                               subpixels)

            if a_in is not None:
                b_in = a_in * b / a
                fraction -= elliptical_overlap_grid(x_pmin[i], x_pmax[i],
                                                    y_pmin[i], y_pmax[i],
                                                    x_max[i] - x_min[i],
                                                    y_max[i] - y_min[i],
                                                    a_in, b_in, theta,
                                                    use_exact, subpixels)

            flux[i] = np.sum(data[y_min[i]:y_max[i],
                                  x_min[i]:x_max[i]] * fraction)

            if error is not None:
                fluxvar[i] = find_fluxvar(data, fraction, error, flux[i],
                                          x_min[i], x_max[i], y_min[i],
                                          y_max[i], pixelwise_error)

    if error is None:
        return (flux, )
    else:
        return (flux, np.sqrt(fluxvar))


def get_elliptical_fractions(data, positions, a, b, theta,
                             method, subpixels, a_in=None):

    extents = np.zeros((len(positions), 4), dtype=int)

    # TODO: we can be more efficient in terms of bounding box
    radius = max(a, b)

    extents[:, 0] = positions[:, 0] - radius + 0.5
    extents[:, 1] = positions[:, 0] + radius + 1.5
    extents[:, 2] = positions[:, 1] - radius + 0.5
    extents[:, 3] = positions[:, 1] + radius + 1.5

    ood_filter, extent, phot_extent = _get_phot_extents(data, positions,
                                                        extents)

    fractions = np.zeros((data.shape[0], data.shape[1], len(positions)),
                         dtype=np.float)

    # TODO: flag these objects
    if np.sum(ood_filter):
        warnings.warn("The aperture at position {0} does not have any "
                      "overlap with the data"
                      .format(positions[ood_filter]),
                      AstropyUserWarning)
        if np.sum(ood_filter) == len(positions):
            return np.squeeze(fractions)

    x_min, x_max, y_min, y_max = extent
    x_pmin, x_pmax, y_pmin, y_pmax = phot_extent

    if method == 'center':
        use_exact = 0
        subpixels = 1
    elif method == 'subpixel':
        use_exact = 0
    else:
        use_exact = 1
        subpixels = 1

    from ..geometry import elliptical_overlap_grid

    for i in range(len(positions)):

        if ood_filter[i] is not True:

            fractions[y_min[i]: y_max[i], x_min[i]: x_max[i], i] = \
                elliptical_overlap_grid(x_pmin[i], x_pmax[i],
                                        y_pmin[i], y_pmax[i],
                                        x_max[i] - x_min[i],
                                        y_max[i] - y_min[i],
                                        a, b, theta, use_exact,
                                        subpixels)

            if a_in is not None:
                b_in = a_in * b / a
                fractions[y_min[i]: y_max[i], x_min[i]: x_max[i], i] -= \
                    elliptical_overlap_grid(x_pmin[i], x_pmax[i],
                                            y_pmin[i], y_pmax[i],
                                            x_max[i] - x_min[i],
                                            y_max[i] - y_min[i],
                                            a_in, b_in, theta,
                                            use_exact, subpixels)

    return np.squeeze(fractions)


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


class EllipticalAperture(PixelAperture):
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

    def area(self):
        return math.pi * self.a * self.b

    def do_photometry(self, data, error=None, pixelwise_error=True,
                      method='exact', subpixels=5):

        if method not in ('center', 'subpixel', 'exact'):
            raise ValueError('{0} method not supported for aperture class '
                             '{1}'.format(method, self.__class__.__name__))

        flux = do_elliptical_photometry(data, self.positions,
                                        self.a, self.b, self.theta,
                                        error=error,
                                        pixelwise_error=pixelwise_error,
                                        method=method,
                                        subpixels=subpixels)
        return flux

    def get_fractions(self, data, method='exact', subpixels=5):

        if method not in ('center', 'subpixel', 'exact'):
            raise ValueError('{0} method not supported for aperture class '
                             '{1}'.format(method, self.__class__.__name__))

        fractions = get_elliptical_fractions(data, self.positions,
                                             self.a, self.b, self.theta,
                                             method=method,
                                             subpixels=subpixels)

        return fractions

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


class EllipticalAnnulus(PixelAperture):
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

    def do_photometry(self, data, error=None, pixelwise_error=True,
                      method='exact', subpixels=5):

        if method not in ('center', 'subpixel', 'exact'):
            raise ValueError('{0} method not supported for aperture class '
                             '{1}'.format(method, self.__class__.__name__))

        flux = do_elliptical_photometry(data, self.positions,
                                        self.a_out, self.b_out, self.theta,
                                        error=error,
                                        pixelwise_error=pixelwise_error,
                                        method=method,
                                        subpixels=subpixels,
                                        a_in=self.a_in)

        return flux

    def get_fractions(self, data, method='exact', subpixels=5):

        if method not in ('center', 'subpixel', 'exact'):
            raise ValueError('{0} method not supported for aperture class '
                             '{1}'.format(method, self.__class__.__name__))

        fractions = get_elliptical_fractions(
            data, self.positions, self.a_out, self.b_out, self.theta,
            method=method, subpixels=subpixels, a_in=self.a_in)

        return fractions
