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

from .core import (PixelAperture, SkyAperture, ApertureMask,
                   _sanitize_pixel_positions, _translate_mask_method,
                   _make_annulus_path)
from ..geometry import rectangular_overlap_grid
from ..utils.wcs_helpers import (skycoord_to_pixel_scale_angle, assert_angle,
                                 assert_angle_or_pixel)


__all__ = ['RectangularMaskMixin', 'RectangularAperture',
           'RectangularAnnulus', 'SkyRectangularAperture',
           'SkyRectangularAnnulus']


class RectangularMaskMixin(object):
    """
    Mixin class to create masks for rectangular or rectangular-annulus
    aperture objects.
    """

    def to_mask(self, method='exact', subpixels=5):
        """
        Return a list of `ApertureMask` objects, one for each aperture
        position.

        Parameters
        ----------
        method : {'exact', 'center', 'subpixel'}, optional
            The method used to determine the overlap of the aperture on
            the pixel grid.  Not all options are available for all
            aperture types.  Note that the more precise methods are
            generally slower.  The following methods are available:

                * ``'exact'`` (default):
                  The the exact fractional overlap of the aperture and
                  each pixel is calculated.  The returned mask will
                  contain values between 0 and 1.

                * ``'center'``:
                  A pixel is considered to be entirely in or out of the
                  aperture depending on whether its center is in or out
                  of the aperture.  The returned mask will contain
                  values only of 0 (out) and 1 (in).

                * ``'subpixel'``:
                  A pixel is divided into subpixels (see the
                  ``subpixels`` keyword), each of which are considered
                  to be entirely in or out of the aperture depending on
                  whether its center is in or out of the aperture.  If
                  ``subpixels=1``, this method is equivalent to
                  ``'center'``.  The returned mask will contain values
                  between 0 and 1.

        subpixels : int, optional
            For the ``'subpixel'`` method, resample pixels by this factor
            in each dimension.  That is, each pixel is divided into
            ``subpixels ** 2`` subpixels.

        Returns
        -------
        mask : list of `~photutils.ApertureMask`
            A list of aperture mask objects.
        """

        if method == 'exact':
            warnings.warn("'exact' method is not implemented for rectangular "
                          "apertures -- instead using 'subpixel' method with "
                          "subpixels=32", AstropyUserWarning)
            method = 'subpixel'
            subpixels = 32
        elif method not in ('center', 'subpixel'):
            raise ValueError('"{0}" method is not available for this '
                             'aperture.'.format(method))

        use_exact, subpixels = _translate_mask_method(method, subpixels)

        if hasattr(self, 'w'):
            w = self.w
            h = self.h
        elif hasattr(self, 'w_out'):    # annulus
            w = self.w_out
            h = self.h_out
            h_in = self.w_in * self.h_out / self.w_out
        else:
            raise ValueError('Cannot determine the aperture radius.')

        masks = []
        for position, _slice, _geom_slice in zip(self.positions, self._slices,
                                                 self._geom_slices):
            px_min, px_max = _geom_slice[1].start, _geom_slice[1].stop
            py_min, py_max = _geom_slice[0].start, _geom_slice[0].stop
            dx = px_max - px_min
            dy = py_max - py_min

            mask = rectangular_overlap_grid(px_min, px_max, py_min, py_max,
                                            dx, dy, w, h, self.theta, 0,
                                            subpixels)

            if hasattr(self, 'w_in'):    # annulus
                mask -= rectangular_overlap_grid(px_min, px_max, py_min,
                                                 py_max, dx, dy,
                                                 self.w_in, h_in, self.theta,
                                                 0, subpixels)

            masks.append(ApertureMask(mask, _slice))

        return masks


class RectangularAperture(RectangularMaskMixin, PixelAperture):
    """
    Rectangular aperture(s), defined in pixel coordinates.

    Parameters
    ----------
    positions : array_like or `~astropy.units.Quantity`
        Pixel coordinates of the aperture center(s) in one of the
        following formats:

            * single ``(x, y)`` tuple
            * list of ``(x, y)`` tuples
            * ``Nx2`` or ``2xN`` `~numpy.ndarray`
            * ``Nx2`` or ``2xN`` `~astropy.units.Quantity` in pixel units

        Note that a ``2x2`` `~numpy.ndarray` or
        `~astropy.units.Quantity` is interpreted as ``Nx2``, i.e. two
        rows of (x, y) coordinates.

    w : float
        The full width of the aperture.  For ``theta=0`` the width side
        is along the ``x`` axis.

    h : float
        The full height of the aperture.  For ``theta=0`` the height
        side is along the ``y`` axis.

    theta : float
        The rotation angle in radians of the width (``w``) side from the
        positive ``x`` axis.  The rotation angle increases
        counterclockwise.

    Raises
    ------
    ValueError : `ValueError`
        If either width (``w``) or height (``h``) is negative.
    """

    def __init__(self, positions, w, h, theta):
        try:
            self.w = float(w)
            self.h = float(h)
            self.theta = float(theta)
        except TypeError:
            raise TypeError("'w' and 'h' and 'theta' must "
                            "be numeric, received {0} and {1} and {2}."
                            .format((type(w), type(h), type(theta))))
        if w < 0 or h < 0:
            raise ValueError("'w' and 'h' must be nonnegative.")

        self.positions = _sanitize_pixel_positions(positions)

    @property
    def _slices(self):
        # TODO:  use an actual minimal bounding box
        radius = max(self.h, self.w) * (2. ** -0.5)
        x_min = np.floor(self.positions[:, 0] - radius + 0.5).astype(int)
        x_max = np.floor(self.positions[:, 0] + radius + 1.5).astype(int)
        y_min = np.floor(self.positions[:, 1] - radius + 0.5).astype(int)
        y_max = np.floor(self.positions[:, 1] + radius + 1.5).astype(int)

        return [(slice(ymin, ymax), slice(xmin, xmax))
                for xmin, xmax, ymin, ymax in zip(x_min, x_max, y_min, y_max)]

    def area(self):
        return self.w * self.h

    def plot(self, origin=(0, 0), indices=None, ax=None, fill=False,
             **kwargs):
        import matplotlib.patches as mpatches

        plot_positions, ax, kwargs = self._prepare_plot(
            origin, indices, ax, fill, **kwargs)

        hw = self.w / 2.
        hh = self.h / 2.
        sint = math.sin(self.theta)
        cost = math.cos(self.theta)
        dx = (hh * sint) - (hw * cost)
        dy = -(hh * cost) - (hw * sint)
        plot_positions = plot_positions + np.array([dx, dy])
        theta_deg = self.theta * 180. / np.pi
        for position in plot_positions:
            patch = mpatches.Rectangle(position, self.w, self.h, theta_deg,
                                       **kwargs)
            ax.add_patch(patch)


class RectangularAnnulus(RectangularMaskMixin, PixelAperture):
    """
    Rectangular annulus aperture(s), defined in pixel coordinates.

    Parameters
    ----------
    positions : array_like or `~astropy.units.Quantity`
        Pixel coordinates of the aperture center(s) in one of the
        following formats:

            * single ``(x, y)`` tuple
            * list of ``(x, y)`` tuples
            * ``Nx2`` or ``2xN`` `~numpy.ndarray`
            * ``Nx2`` or ``2xN`` `~astropy.units.Quantity` in pixel units

        Note that a ``2x2`` `~numpy.ndarray` or
        `~astropy.units.Quantity` is interpreted as ``Nx2``, i.e. two
        rows of (x, y) coordinates.

    w_in : float
        The inner full width of the aperture.  For ``theta=0`` the width
        side is along the ``x`` axis.

    w_out : float
        The outer full width of the aperture.  For ``theta=0`` the width
        side is along the ``x`` axis.

    h_out : float
        The outer full height of the aperture.  The inner full height is
        calculated as:

            .. math:: h_{in} = h_{out}
                \\left(\\frac{w_{in}}{w_{out}}\\right)

        For ``theta=0`` the height side is along the ``y`` axis.

    theta : float
        The rotation angle in radians of the width side from the
        positive ``x`` axis.  The rotation angle increases
        counterclockwise.

    Raises
    ------
    ValueError : `ValueError`
        If inner width (``w_in``) is greater than outer width
        (``w_out``).

    ValueError : `ValueError`
        If either the inner width (``w_in``) or the outer height
        (``h_out``) is negative.
    """

    def __init__(self, positions, w_in, w_out, h_out, theta):
        try:
            self.w_in = float(w_in)
            self.w_out = float(w_out)
            self.h_out = float(h_out)
            self.theta = float(theta)
        except TypeError:
            raise TypeError("'w_in' and 'w_out' and 'h_out' and 'theta' must "
                            "be numeric, received {0} and {1} and {2} and "
                            "{3}.".format((type(w_in), type(w_out),
                                           type(h_out), type(theta))))

        if not (w_out > w_in):
            raise ValueError("'w_out' must be greater than 'w_in'")
        if w_in < 0 or h_out < 0:
            raise ValueError("'w_in' and 'h_out' must be non-negative")

        self.h_in = w_in * h_out / w_out

        self.positions = _sanitize_pixel_positions(positions)

    @property
    def _slices(self):
        # TODO:  use an actual minimal bounding box
        radius = max(self.h_out, self.w_out) * (2. ** -0.5)
        x_min = np.floor(self.positions[:, 0] - radius + 0.5).astype(int)
        x_max = np.floor(self.positions[:, 0] + radius + 1.5).astype(int)
        y_min = np.floor(self.positions[:, 1] - radius + 0.5).astype(int)
        y_max = np.floor(self.positions[:, 1] + radius + 1.5).astype(int)

        return [(slice(ymin, ymax), slice(xmin, xmax))
                for xmin, xmax, ymin, ymax in zip(x_min, x_max, y_min, y_max)]

    def area(self):
        return self.w_out * self.h_out - self.w_in * self.h_in

    def plot(self, origin=(0, 0), indices=None, ax=None, fill=False,
             **kwargs):
        import matplotlib.patches as mpatches

        plot_positions, ax, kwargs = self._prepare_plot(
            origin, indices, ax, fill, **kwargs)

        sint = math.sin(self.theta)
        cost = math.cos(self.theta)
        theta_deg = self.theta * 180. / np.pi

        hw_inner = self.w_in / 2.
        hh_inner = self.h_in / 2.
        dx_inner = (hh_inner * sint) - (hw_inner * cost)
        dy_inner = -(hh_inner * cost) - (hw_inner * sint)
        positions_inner = plot_positions + np.array([dx_inner, dy_inner])
        hw_outer = self.w_out / 2.
        hh_outer = self.h_out / 2.
        dx_outer = (hh_outer * sint) - (hw_outer * cost)
        dy_outer = -(hh_outer * cost) - (hw_outer * sint)
        positions_outer = plot_positions + np.array([dx_outer, dy_outer])

        for i, position_inner in enumerate(positions_inner):
            patch_inner = mpatches.Rectangle(position_inner,
                                             self.w_in, self.h_in,
                                             theta_deg, **kwargs)
            patch_outer = mpatches.Rectangle(positions_outer[i],
                                             self.w_out, self.h_out,
                                             theta_deg, **kwargs)
            path = _make_annulus_path(patch_inner, patch_outer)
            patch = mpatches.PathPatch(path, **kwargs)
            ax.add_patch(patch)


class SkyRectangularAperture(SkyAperture):
    """
    Rectangular aperture(s), defined in sky coordinates.

    Parameters
    ----------
    positions : `~astropy.coordinates.SkyCoord`
        Celestial coordinates of the aperture center(s). This can be
        either scalar coordinates or an array of coordinates.

    w : `~astropy.units.Quantity`
        The full width of the aperture, either in angular or pixel
        units.  For ``theta=0`` the width side is along the North-South
        axis.

    h :  `~astropy.units.Quantity`
        The full height of the aperture, either in angular or pixel
        units.  For ``theta=0`` the height side is along the East-West
        axis.

    theta : `~astropy.units.Quantity`
        The position angle (in angular units) of the width side.  For a
        right-handed world coordinate system, the position angle
        increases counterclockwise from North (PA=0).
    """

    def __init__(self, positions, w, h, theta):
        if isinstance(positions, SkyCoord):
            self.positions = positions
        else:
            raise TypeError("positions should be a SkyCoord instance")

        assert_angle_or_pixel('w', w)
        assert_angle_or_pixel('h', h)
        assert_angle('theta', theta)

        if w.unit.physical_type != h.unit.physical_type:
            raise ValueError("'w' and 'h' should either both be angles or "
                             "in pixels")

        self.w = w
        self.h = h
        self.theta = theta

    def to_pixel(self, wcs, mode='all'):
        """
        Convert the aperture to a `RectangularAperture` instance in
        pixel coordinates.

        Parameters
        ----------
        wcs : `~astropy.wcs.WCS`
            The WCS transformation to use.

        mode : {'all', 'wcs'}, optional
            Whether to do the transformation including distortions
            (``'all'``; default) or only including only the core WCS
            transformation (``'wcs'``).

        Returns
        -------
        aperture : `RectangularAperture` object
            A `RectangularAperture` object.
        """

        x, y = skycoord_to_pixel(self.positions, wcs, mode=mode)
        central_pos = SkyCoord([wcs.wcs.crval], frame=self.positions.name,
                               unit=wcs.wcs.cunit)
        xc, yc, scale, angle = skycoord_to_pixel_scale_angle(central_pos, wcs)

        if self.w.unit.physical_type == 'angle':
            w = (scale * self.w).to(u.pixel).value
            h = (scale * self.h).to(u.pixel).value
        else:
            # pixels
            w = self.w.value
            h = self.h.value

        theta = (angle + self.theta).to(u.radian).value

        pixel_positions = np.array([x, y]).transpose()

        return RectangularAperture(pixel_positions, w, h, theta)


class SkyRectangularAnnulus(SkyAperture):
    """
    Rectangular annulus aperture(s), defined in sky coordinates.

    Parameters
    ----------
    positions : `~astropy.coordinates.SkyCoord`
        Celestial coordinates of the aperture center(s). This can be
        either scalar coordinates or an array of coordinates.

    w_in : `~astropy.units.Quantity`
        The inner full width of the aperture, either in angular or pixel
        units.  For ``theta=0`` the width side is along the North-South
        axis.

    w_out : `~astropy.units.Quantity`
        The outer full width of the aperture, either in angular or pixel
        units.  For ``theta=0`` the width side is along the North-South
        axis.

    h_out : `~astropy.units.Quantity`
        The outer full height of the aperture, either in angular or
        pixel units.  The inner full height is calculated as:

            .. math:: h_{in} = h_{out}
                \\left(\\frac{w_{in}}{w_{out}}\\right)

        For ``theta=0`` the height side is along the East-West axis.

    theta : `~astropy.units.Quantity`
        The position angle (in angular units) of the width side.  For a
        right-handed world coordinate system, the position angle
        increases counterclockwise from North (PA=0).
    """

    def __init__(self, positions, w_in, w_out, h_out, theta):
        if isinstance(positions, SkyCoord):
            self.positions = positions
        else:
            raise TypeError("positions should be a SkyCoord instance")
        assert_angle_or_pixel('w_in', w_in)
        assert_angle_or_pixel('w_out', w_out)
        assert_angle_or_pixel('h_out', h_out)
        assert_angle('theta', theta)

        if w_in.unit.physical_type != w_out.unit.physical_type:
            raise ValueError("w_in and w_out should either both be angles or "
                             "in pixels")

        if w_out.unit.physical_type != h_out.unit.physical_type:
            raise ValueError("w_out and h_out should either both be angles "
                             "or in pixels")

        self.w_in = w_in
        self.w_out = w_out
        self.h_out = h_out
        self.theta = theta

    def to_pixel(self, wcs, mode='all'):
        """
        Convert the aperture to a `RectangularAnnulus` instance in pixel
        coordinates.

        Parameters
        ----------
        wcs : `~astropy.wcs.WCS`
            The WCS transformation to use.

        mode : {'all', 'wcs'}, optional
            Whether to do the transformation including distortions
            (``'all'``; default) or only including only the core WCS
            transformation (``'wcs'``).

        Returns
        -------
        aperture : `RectangularAnnulus` object
            A `RectangularAnnulus` object.
        """

        x, y = skycoord_to_pixel(self.positions, wcs, mode=mode)
        central_pos = SkyCoord([wcs.wcs.crval], frame=self.positions.name,
                               unit=wcs.wcs.cunit)
        xc, yc, scale, angle = skycoord_to_pixel_scale_angle(central_pos, wcs)

        if self.w_in.unit.physical_type == 'angle':
            w_in = (scale * self.w_in).to(u.pixel).value
            w_out = (scale * self.w_out).to(u.pixel).value
            h_out = (scale * self.h_out).to(u.pixel).value
        else:
            # pixels
            w_in = self.w_in.value
            w_out = self.w_out.value
            h_out = self.h_out.value

        theta = (angle + self.theta).to(u.radian).value

        pixel_positions = np.array([x, y]).transpose()

        return RectangularAnnulus(pixel_positions, w_in, w_out, h_out, theta)
