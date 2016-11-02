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

from .core import (Mask, SkyAperture, PixelAperture,
                   _sanitize_pixel_positions, _make_annulus_path,
                   _get_phot_extents, _calc_aperture_var,
                   _translate_mask_method)
from ..geometry import circular_overlap_grid
from ..utils.wcs_helpers import (skycoord_to_pixel_scale_angle,
                                 assert_angle_or_pixel)


skycoord_to_pixel_mode = 'all'


__all__ = ['CircularMixin', 'SkyCircularAperture', 'CircularAperture',
           'SkyCircularAnnulus', 'CircularAnnulus']


class CircularMixin(object):
    """
    Mixin class for circular apertures.
    """

    def to_mask(self, method='exact', subpixels=5):
        """
        Returns
        -------
        mask : list of `~photutils.Mask`
            A list of Mask objects.
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
            raise ValueError('Cannot determine aperture radius.')

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
                                              dx, dy, radius, use_exact,
                                              subpixels)

            masks.append(Mask(position, mask, _slice, _geom_slice))

        return masks

    def old_photometry(self, data, error=None, pixelwise_error=True,
                       method='exact', subpixels=5):
        """
        Perform circular photometry.
        """

        if method not in ('center', 'subpixel', 'exact'):
            raise ValueError('{0} method not supported for aperture class '
                             '{1}'.format(method, self.__class__.__name__))

        if hasattr(self, 'r'):
            radius = self.r
        elif hasattr(self, 'r_out'):    # annulus
            radius = self.r_out
        else:
            raise ValueError('cannot determine aperture radius.')

        extents = np.zeros((len(self), 4), dtype=int)
        extents[:, 0] = self.positions[:, 0] - radius + 0.5
        extents[:, 1] = self.positions[:, 0] + radius + 1.5
        extents[:, 2] = self.positions[:, 1] - radius + 0.5
        extents[:, 3] = self.positions[:, 1] + radius + 1.5

        ood_filter, extent, phot_extent = _get_phot_extents(
            data, self.positions, extents)

        flux = u.Quantity(np.zeros(len(self), dtype=np.float), unit=data.unit)

        if error is not None:
            fluxvar = u.Quantity(np.zeros(len(self), dtype=np.float),
                                 unit=error.unit ** 2)

        # TODO: flag these objects
        if np.any(ood_filter):
            flux[ood_filter] = np.nan
            warnings.warn('The aperture at position {0} does not have any '
                          'overlap with the data'.format(
                              self.positions[ood_filter]), AstropyUserWarning)
            if np.all(ood_filter):
                return flux

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

        for i in range(len(self)):
            if not np.isnan(flux[i]):
                fraction = circular_overlap_grid(x_pmin[i], x_pmax[i],
                                                 y_pmin[i], y_pmax[i],
                                                 x_max[i] - x_min[i],
                                                 y_max[i] - y_min[i],
                                                 radius, use_exact, subpixels)

                if hasattr(self, 'r_in'):
                    fraction -= circular_overlap_grid(x_pmin[i], x_pmax[i],
                                                      y_pmin[i], y_pmax[i],
                                                      x_max[i] - x_min[i],
                                                      y_max[i] - y_min[i],
                                                      self.r_in, use_exact,
                                                      subpixels)

                flux[i] = np.sum(data[y_min[i]:y_max[i],
                                      x_min[i]:x_max[i]] * fraction)

                if error is not None:
                    fluxvar[i] = _calc_aperture_var(
                        data, fraction, error, flux[i], x_min[i], x_max[i],
                        y_min[i], y_max[i], pixelwise_error)

        if error is None:
            return flux
        else:
            return flux, np.sqrt(fluxvar)

    def get_mask(self, data, method='exact', subpixels=5):
        """
        Define aperture mask(s).
        """

        if method not in ('center', 'subpixel', 'exact'):
            raise ValueError('{0} method not supported for aperture class '
                             '{1}'.format(method, self.__class__.__name__))

        if hasattr(self, 'r'):
            radius = self.r
        elif hasattr(self, 'r_out'):    # annulus
            radius = self.r_out
        else:
            raise ValueError('cannot determine aperture radius.')

        extents = np.zeros((len(self), 4), dtype=int)
        extents[:, 0] = self.positions[:, 0] - radius + 0.5
        extents[:, 1] = self.positions[:, 0] + radius + 1.5
        extents[:, 2] = self.positions[:, 1] - radius + 0.5
        extents[:, 3] = self.positions[:, 1] + radius + 1.5

        ood_filter, extent, phot_extent = _get_phot_extents(
            data, self.positions, extents)

        fractions = np.zeros((data.shape[0], data.shape[1], len(self)),
                             dtype=np.float)

        if np.sum(ood_filter):
            warnings.warn('The aperture at position {0} does not have any '
                          'overlap with the data'.format(
                              self.positions[ood_filter]), AstropyUserWarning)
            if np.sum(ood_filter) == len(self):
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

        for i in range(len(self)):
            if ood_filter[i] is not True:
                fractions[y_min[i]: y_max[i], x_min[i]: x_max[i], i] = \
                    circular_overlap_grid(x_pmin[i], x_pmax[i],
                                          y_pmin[i], y_pmax[i],
                                          x_max[i] - x_min[i],
                                          y_max[i] - y_min[i],
                                          radius, use_exact, subpixels)

                if hasattr(self, 'r_in'):
                    fractions[y_min[i]: y_max[i], x_min[i]: x_max[i], i] -= \
                        circular_overlap_grid(x_pmin[i], x_pmax[i],
                                              y_pmin[i], y_pmax[i],
                                              x_max[i] - x_min[i],
                                              y_max[i] - y_min[i],
                                              self.r_in, use_exact, subpixels)

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


class CircularAperture(CircularMixin, PixelAperture):
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


class CircularAnnulus(CircularMixin, PixelAperture):
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
