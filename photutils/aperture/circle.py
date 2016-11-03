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
from ..geometry import circular_overlap_grid
from ..utils.wcs_helpers import (skycoord_to_pixel_scale_angle,
                                 assert_angle_or_pixel)


skycoord_to_pixel_mode = 'all'


__all__ = ['CircularMaskMixin', 'CircularAperture', 'CircularAnnulus',
           'SkyCircularAperture', 'SkyCircularAnnulus']


class CircularMaskMixin(object):
    """
    Mixin class to create masks for circular and circular-annulus
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

        if hasattr(self, 'r'):
            radius = self.r
        elif hasattr(self, 'r_out'):    # annulus
            radius = self.r_out
        else:
            raise ValueError('Cannot determine the aperture radius.')

        masks = []
        for position, _slice, _geom_slice in zip(self.positions, self._slices,
                                                 self._geom_slices):
            px_min, px_max = _geom_slice[1].start, _geom_slice[1].stop
            py_min, py_max = _geom_slice[0].start, _geom_slice[0].stop
            dx = px_max - px_min
            dy = py_max - py_min

            mask = circular_overlap_grid(px_min, px_max, py_min, py_max,
                                         dx, dy, radius, use_exact, subpixels)

            if hasattr(self, 'r_in'):    # annulus
                mask -= circular_overlap_grid(px_min, px_max, py_min, py_max,
                                              dx, dy, self.r_in, use_exact,
                                              subpixels)

            masks.append(ApertureMask(mask, _slice))

        return masks


class CircularAperture(CircularMaskMixin, PixelAperture):
    """
    Circular aperture(s), defined in pixel coordinates.

    Parameters
    ----------
    positions : array-like or `~astropy.units.Quantity`
        Pixel coordinates of the aperture center(s) in one of the
        following formats:

          * single ``(x, y)`` tuple
          * list of ``(x, y)`` tuples
          * ``Nx2`` or ``2xN`` `~numpy.ndarray`
          * ``Nx2`` or ``2xN`` `~astropy.units.Quantity` in pixel units

        Note that a ``2x2`` `~numpy.ndarray` or
        `~astropy.units.Quantity` is interpreted as ``Nx2``, i.e. two
        rows of (x, y) coordinates.

    r : float
        The radius of the aperture(s), in pixels.

    Raises
    ------
    ValueError : `ValueError`
        If the input radius, ``r``, is negative.
    """

    def __init__(self, positions, r):
        try:
            self.r = float(r)
        except TypeError:
            raise TypeError('r must be numeric, received {0}'.format(type(r)))

        if r < 0:
            raise ValueError('r must be non-negative')

        self.positions = _sanitize_pixel_positions(positions)

    # TODO: make lazyproperty?, but update if positions or radius change
    @property
    def _slices(self):
        x_min = np.floor(self.positions[:, 0] - self.r + 0.5).astype(int)
        x_max = np.floor(self.positions[:, 0] + self.r + 1.5).astype(int)
        y_min = np.floor(self.positions[:, 1] - self.r + 0.5).astype(int)
        y_max = np.floor(self.positions[:, 1] + self.r + 1.5).astype(int)

        return [(slice(ymin, ymax), slice(xmin, xmax))
                for xmin, xmax, ymin, ymax in zip(x_min, x_max, y_min, y_max)]

    def area(self):
        return math.pi * self.r ** 2

    def plot(self, origin=(0, 0), source_id=None, ax=None, fill=False,
             **kwargs):
        import matplotlib.patches as mpatches

        plot_positions, ax, kwargs = self._prepare_plot(
            origin, source_id, ax, fill, **kwargs)

        for position in plot_positions:
            patch = mpatches.Circle(position, self.r, **kwargs)
            ax.add_patch(patch)


class CircularAnnulus(CircularMaskMixin, PixelAperture):
    """
    Circular annulus aperture(s), defined in pixel coordinates.

    Parameters
    ----------
    positions : array-like or `~astropy.units.Quantity`
        Pixel coordinates of the aperture center(s) in one of the
        following formats:

        * single ``(x, y)`` tuple
        * list of ``(x, y)`` tuples
        * ``Nx2`` or ``2xN`` `~numpy.ndarray`
        * ``Nx2`` or ``2xN`` `~astropy.units.Quantity` in pixel units

        A ``2x2`` `~numpy.ndarray` or `~astropy.units.Quantity` is
        interpreted as ``Nx2``, i.e. two rows of (x, y) coordinates.
    r_in : float
        The inner radius of the annulus.
    r_out : float
        The outer radius of the annulus.

    Raises
    ------
    ValueError : `ValueError`
        If inner radius (``r_in``) is greater than outer radius (``r_out``).
    ValueError : `ValueError`
        If inner radius is negative.
    """

    def __init__(self, positions, r_in, r_out):
        try:
            self.r_in = r_in
            self.r_out = r_out
        except TypeError:
            raise TypeError("'r_in' and 'r_out' must be numeric, received "
                            "{0} and {1}".format((type(r_in), type(r_out))))

        if not (r_out > r_in):
            raise ValueError('r_out must be greater than r_in')
        if r_in < 0:
            raise ValueError('r_in must be non-negative')

        self.positions = _sanitize_pixel_positions(positions)

    @property
    def _slices(self):
        x_min = np.floor(self.positions[:, 0] - self.r_out + 0.5).astype(int)
        x_max = np.floor(self.positions[:, 0] + self.r_out + 1.5).astype(int)
        y_min = np.floor(self.positions[:, 1] - self.r_out + 0.5).astype(int)
        y_max = np.floor(self.positions[:, 1] + self.r_out + 1.5).astype(int)

        return [(slice(ymin, ymax), slice(xmin, xmax))
                for xmin, xmax, ymin, ymax in zip(x_min, x_max, y_min, y_max)]

    def area(self):
        return math.pi * (self.r_out ** 2 - self.r_in ** 2)

    def plot(self, origin=(0, 0), source_id=None, ax=None, fill=False,
             **kwargs):

        import matplotlib.patches as mpatches

        plot_positions, ax, kwargs = self._prepare_plot(
            origin, source_id, ax, fill, **kwargs)

        resolution = 20
        for position in plot_positions:
            patch_inner = mpatches.CirclePolygon(position, self.r_in,
                                                 resolution=resolution)
            patch_outer = mpatches.CirclePolygon(position, self.r_out,
                                                 resolution=resolution)
            path = _make_annulus_path(patch_inner, patch_outer)
            patch = mpatches.PathPatch(path, **kwargs)
            ax.add_patch(patch)


class SkyCircularAperture(SkyAperture):
    """
    Circular aperture(s), defined in sky coordinates.

    Parameters
    ----------
    positions : `~astropy.coordinates.SkyCoord`
        Celestial coordinates of the aperture center(s). This can be either
        scalar coordinates or an array of coordinates.
    r : `~astropy.units.Quantity`
        The radius of the aperture(s), either in angular or pixel units.
    """

    def __init__(self, positions, r):
        if isinstance(positions, SkyCoord):
            self.positions = positions
        else:
            raise TypeError('positions must be a SkyCoord object.')

        assert_angle_or_pixel('r', r)
        self.r = r

    def to_pixel(self, wcs):
        """
        Convert the aperture to a `CircularAperture` instance in
        pixel coordinates.
        """

        x, y = skycoord_to_pixel(self.positions, wcs,
                                 mode=skycoord_to_pixel_mode)

        if self.r.unit.physical_type == 'angle':
            central_pos = SkyCoord([wcs.wcs.crval], frame=self.positions.name,
                                   unit=wcs.wcs.cunit)
            xc, yc, scale, angle = skycoord_to_pixel_scale_angle(central_pos,
                                                                 wcs)
            r = (scale * self.r).to(u.pixel).value
        else:
            r = self.r.value    # pixels

        pixel_positions = np.array([x, y]).transpose()

        return CircularAperture(pixel_positions, r)


class SkyCircularAnnulus(SkyAperture):
    """
    Circular annulus aperture(s), defined in sky coordinates.

    Parameters
    ----------

    positions : `~astropy.coordinates.SkyCoord`
        Celestial coordinates of the aperture center(s). This can be either
        scalar coordinates or an array of coordinates.

    r_in : `~astropy.units.Quantity`
        The inner radius of the annulus, either in angular or pixel units.

    r_out : `~astropy.units.Quantity`
        The outer radius of the annulus, either in angular or pixel units.
    """

    def __init__(self, positions, r_in, r_out):
        if isinstance(positions, SkyCoord):
            self.positions = positions
        else:
            raise TypeError("positions should be a SkyCoord instance")

        assert_angle_or_pixel('r_in', r_in)
        assert_angle_or_pixel('r_out', r_out)

        if r_in.unit.physical_type != r_out.unit.physical_type:
            raise ValueError("r_in and r_out should either both be angles "
                             "or in pixels")

        self.r_in = r_in
        self.r_out = r_out

    def to_pixel(self, wcs):
        """
        Return a CircularAnnulus instance in pixel coordinates.
        """

        x, y = skycoord_to_pixel(self.positions, wcs,
                                 mode=skycoord_to_pixel_mode)
        if self.r_in.unit.physical_type == 'angle':
            central_pos = SkyCoord([wcs.wcs.crval], frame=self.positions.name,
                                   unit=wcs.wcs.cunit)
            xc, yc, scale, angle = skycoord_to_pixel_scale_angle(central_pos,
                                                                 wcs)
            r_in = (scale * self.r_in).to(u.pixel).value
            r_out = (scale * self.r_out).to(u.pixel).value
        else:  # pixel
            r_in = self.r_in.value
            r_out = self.r_out.value

        pixel_positions = np.array([x, y]).transpose()

        return CircularAnnulus(pixel_positions, r_in, r_out)
