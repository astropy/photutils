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
                   _make_annulus_path, get_phot_extents, find_fluxvar)
from ..utils.wcs_helpers import (skycoord_to_pixel_scale_angle,
                                 assert_angle_or_pixel)


skycoord_to_pixel_mode = 'all'


__all__ = ['SkyCircularAperture', 'CircularAperture',
           'SkyCircularAnnulus', 'CircularAnnulus']


def do_circular_photometry(data, positions, radius, error, pixelwise_error,
                           method, subpixels, r_in=None):

    extents = np.zeros((len(positions), 4), dtype=int)

    extents[:, 0] = positions[:, 0] - radius + 0.5
    extents[:, 1] = positions[:, 0] + radius + 1.5
    extents[:, 2] = positions[:, 1] - radius + 0.5
    extents[:, 3] = positions[:, 1] + radius + 1.5

    ood_filter, extent, phot_extent = get_phot_extents(data, positions,
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

    from ..geometry import circular_overlap_grid

    for i in range(len(flux)):

        if not np.isnan(flux[i]):

            fraction = circular_overlap_grid(x_pmin[i], x_pmax[i],
                                             y_pmin[i], y_pmax[i],
                                             x_max[i] - x_min[i],
                                             y_max[i] - y_min[i],
                                             radius, use_exact, subpixels)

            if r_in is not None:
                fraction -= circular_overlap_grid(x_pmin[i], x_pmax[i],
                                                  y_pmin[i], y_pmax[i],
                                                  x_max[i] - x_min[i],
                                                  y_max[i] - y_min[i],
                                                  r_in, use_exact, subpixels)

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


def get_circular_fractions(data, positions, radius, method, subpixels,
                           r_in=None):

    extents = np.zeros((len(positions), 4), dtype=int)

    extents[:, 0] = positions[:, 0] - radius + 0.5
    extents[:, 1] = positions[:, 0] + radius + 1.5
    extents[:, 2] = positions[:, 1] - radius + 0.5
    extents[:, 3] = positions[:, 1] + radius + 1.5

    ood_filter, extent, phot_extent = get_phot_extents(data, positions,
                                                       extents)

    fractions = np.zeros((data.shape[0], data.shape[1], len(positions)),
                         dtype=np.float)

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

    from ..geometry import circular_overlap_grid

    for i in range(len(positions)):

        if ood_filter[i] is not True:

            fractions[y_min[i]: y_max[i], x_min[i]: x_max[i], i] = \
                circular_overlap_grid(x_pmin[i], x_pmax[i],
                                      y_pmin[i], y_pmax[i],
                                      x_max[i] - x_min[i],
                                      y_max[i] - y_min[i],
                                      radius, use_exact, subpixels)

            if r_in is not None:
                fractions[y_min[i]: y_max[i], x_min[i]: x_max[i], i] -= \
                    circular_overlap_grid(x_pmin[i], x_pmax[i],
                                          y_pmin[i], y_pmax[i],
                                          x_max[i] - x_min[i],
                                          y_max[i] - y_min[i],
                                          r_in, use_exact, subpixels)

    return np.squeeze(fractions)


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
            raise TypeError("positions should be a SkyCoord instance")

        assert_angle_or_pixel('r', r)
        self.r = r

    def to_pixel(self, wcs):
        """
        Return a CircularAperture instance in pixel coordinates.
        """

        x, y = skycoord_to_pixel(self.positions, wcs,
                                 mode=skycoord_to_pixel_mode)

        if self.r.unit.physical_type == 'angle':
            central_pos = SkyCoord([wcs.wcs.crval], frame=self.positions.name,
                                   unit=wcs.wcs.cunit)
            xc, yc, scale, angle = skycoord_to_pixel_scale_angle(central_pos,
                                                                 wcs)
            r = (scale * self.r).to(u.pixel).value
        else:  # pixel
            r = self.r.value

        pixel_positions = np.array([x, y]).transpose()

        return CircularAperture(pixel_positions, r)


class CircularAperture(PixelAperture):
    """
    Circular aperture(s), defined in pixel coordinates.

    Parameters
    ----------
    positions : tuple, list, array, or `~astropy.units.Quantity`
        Pixel coordinates of the aperture center(s), either as a single
        ``(x, y)`` tuple, a list of ``(x, y)`` tuples, an ``Nx2`` or
        ``2xN`` `~numpy.ndarray`, or an ``Nx2`` or ``2xN``
        `~astropy.units.Quantity` in units of pixels.  A ``2x2``
        `~numpy.ndarray` or `~astropy.units.Quantity` is interpreted as
        ``Nx2``, i.e. two rows of (x, y) coordinates.
    r : float
        The radius of the aperture(s), in pixels.

    Raises
    ------
    ValueError : `ValueError`
        If the radius is negative.
    """

    def __init__(self, positions, r):

        try:
            self.r = float(r)
        except TypeError:
            raise TypeError('r must be numeric, received {0}'.format(type(r)))

        if r < 0:
            raise ValueError('r must be non-negative')

        self.positions = _sanitize_pixel_positions(positions)

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

    def do_photometry(self, data, error=None, pixelwise_error=True,
                      method='exact', subpixels=5):

        if method not in ('center', 'subpixel', 'exact'):
            raise ValueError('{0} method not supported for aperture class '
                             '{1}'.format(method, self.__class__.__name__))

        flux = do_circular_photometry(data, self.positions,
                                      self.r, error=error,
                                      pixelwise_error=pixelwise_error,
                                      method=method,
                                      subpixels=subpixels)
        return flux

    def get_fractions(self, data, method='exact', subpixels=5):

        if method not in ('center', 'subpixel', 'exact'):
            raise ValueError('{0} method not supported for aperture class '
                             '{1}'.format(method, self.__class__.__name__))

        fractions = get_circular_fractions(data, self.positions,
                                           self.r, method=method,
                                           subpixels=subpixels)
        return fractions


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


class CircularAnnulus(PixelAperture):
    """
    Circular annulus aperture(s), defined in pixel coordinates.

    Parameters
    ----------
    positions : tuple, list, array, or `~astropy.units.Quantity`
        Pixel coordinates of the aperture center(s), either as a single
        ``(x, y)`` tuple, a list of ``(x, y)`` tuples, an ``Nx2`` or
        ``2xN`` `~numpy.ndarray`, or an ``Nx2`` or ``2xN``
        `~astropy.units.Quantity` in units of pixels.  A ``2x2``
        `~numpy.ndarray` or `~astropy.units.Quantity` is interpreted as
        ``Nx2``, i.e. two rows of (x, y) coordinates.
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

    def area(self):
        return math.pi * (self.r_out ** 2 - self.r_in ** 2)

    def do_photometry(self, data, error=None, pixelwise_error=True,
                      method='exact', subpixels=5):

        if method not in ('center', 'subpixel', 'exact'):
            raise ValueError('{0} method not supported for aperture class '
                             '{1}'.format(method, self.__class__.__name__))

        flux = do_circular_photometry(data, self.positions,
                                      self.r_out, error=error,
                                      pixelwise_error=pixelwise_error,
                                      method=method,
                                      subpixels=subpixels,
                                      r_in=self.r_in)

        return flux

    def get_fractions(self, data, method='exact', subpixels=5):

        if method not in ('center', 'subpixel', 'exact'):
            raise ValueError('{0} method not supported for aperture class '
                             '{1}'.format(method, self.__class__.__name__))

        fractions = get_circular_fractions(data, self.positions,
                                           self.r_out, method=method,
                                           subpixels=subpixels,
                                           r_in=self.r_in)

        return fractions

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
