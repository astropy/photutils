# Licensed under a 3-clause BSD style license - see LICENSE.rst

import math

import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u

from .core import PixelAperture, SkyAperture
from .bounding_box import BoundingBox
from .mask import ApertureMask
from ..geometry import elliptical_overlap_grid
from ..utils.wcs_helpers import assert_angle, assert_angle_or_pixel


__all__ = ['EllipticalMaskMixin', 'EllipticalAperture', 'EllipticalAnnulus',
           'SkyEllipticalAperture', 'SkyEllipticalAnnulus']


class EllipticalMaskMixin:
    """
    Mixin class to create masks for elliptical and elliptical-annulus
    aperture objects.
    """

    def to_mask(self, method='exact', subpixels=5):
        """
        Return a list of `~photutils.ApertureMask` objects, one for each
        aperture position.

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

        use_exact, subpixels = self._translate_mask_mode(method, subpixels)

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
        for bbox, edges in zip(self.bounding_boxes, self._centered_edges):
            ny, nx = bbox.shape
            mask = elliptical_overlap_grid(edges[0], edges[1], edges[2],
                                           edges[3], nx, ny, a, b, self.theta,
                                           use_exact, subpixels)

            # subtract the inner ellipse for an annulus
            if hasattr(self, 'a_in'):
                mask -= elliptical_overlap_grid(edges[0], edges[1], edges[2],
                                                edges[3], nx, ny, self.a_in,
                                                b_in, self.theta, use_exact,
                                                subpixels)

            masks.append(ApertureMask(mask, bbox))

        return masks


class EllipticalAperture(EllipticalMaskMixin, PixelAperture):
    """
    Elliptical aperture(s), defined in pixel coordinates.

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

    a : float
        The semimajor axis.

    b : float
        The semiminor axis.

    theta : float, optional
        The rotation angle in radians of the semimajor axis from the
        positive ``x`` axis.  The rotation angle increases
        counterclockwise.  The default is 0.

    Raises
    ------
    ValueError : `ValueError`
        If either axis (``a`` or ``b``) is negative.
    """

    def __init__(self, positions, a, b, theta=0.):
        if a < 0 or b < 0:
            raise ValueError("'a' and 'b' must be non-negative.")

        self.positions = self._sanitize_positions(positions)
        self.a = float(a)
        self.b = float(b)
        self.theta = float(theta)
        self._params = ['a', 'b', 'theta']

    @property
    def bounding_boxes(self):
        """
        A list of minimal bounding boxes (`~photutils.BoundingBox`), one
        for each position, enclosing the exact elliptical apertures.
        """

        cos_theta = np.cos(self.theta)
        sin_theta = np.sin(self.theta)
        ax = self.a * cos_theta
        ay = self.a * sin_theta
        bx = self.b * -sin_theta
        by = self.b * cos_theta
        dx = np.sqrt(ax*ax + bx*bx)
        dy = np.sqrt(ay*ay + by*by)

        xmin = self.positions[:, 0] - dx
        xmax = self.positions[:, 0] + dx
        ymin = self.positions[:, 1] - dy
        ymax = self.positions[:, 1] + dy

        return [BoundingBox._from_float(x0, x1, y0, y1)
                for x0, x1, y0, y1 in zip(xmin, xmax, ymin, ymax)]

    def area(self):
        return math.pi * self.a * self.b

    def plot(self, origin=(0, 0), indices=None, ax=None, fill=False,
             **kwargs):
        import matplotlib.patches as mpatches

        plot_positions, ax, kwargs = self._prepare_plot(
            origin, indices, ax, fill, **kwargs)

        theta_deg = self.theta * 180. / np.pi
        for position in plot_positions:
            patch = mpatches.Ellipse(position, 2.*self.a, 2.*self.b,
                                     theta_deg, **kwargs)
            ax.add_patch(patch)

    def to_sky(self, wcs, mode='all'):
        """
        Convert the aperture to a `SkyEllipticalAperture` object defined
        in celestial coordinates.

        Parameters
        ----------
        wcs : `~astropy.wcs.WCS`
            The world coordinate system (WCS) transformation to use.

        mode : {'all', 'wcs'}, optional
            Whether to do the transformation including distortions
            (``'all'``; default) or only including only the core WCS
            transformation (``'wcs'``).

        Returns
        -------
        aperture : `SkyEllipticalAperture` object
            A `SkyEllipticalAperture` object.
        """

        sky_params = self._to_sky_params(wcs, mode=mode)
        return SkyEllipticalAperture(**sky_params)


class EllipticalAnnulus(EllipticalMaskMixin, PixelAperture):
    """
    Elliptical annulus aperture(s), defined in pixel coordinates.

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

    a_in : float
        The inner semimajor axis.

    a_out : float
        The outer semimajor axis.

    b_out : float
        The outer semiminor axis.  The inner semiminor axis is
        calculated as:

            .. math:: b_{in} = b_{out}
                \\left(\\frac{a_{in}}{a_{out}}\\right)

    theta : float, optional
        The rotation angle in radians of the semimajor axis from the
        positive ``x`` axis.  The rotation angle increases
        counterclockwise.  The default is 0.

    Raises
    ------
    ValueError : `ValueError`
        If inner semimajor axis (``a_in``) is greater than outer semimajor
        axis (``a_out``).

    ValueError : `ValueError`
        If either the inner semimajor axis (``a_in``) or the outer semiminor
        axis (``b_out``) is negative.
    """

    def __init__(self, positions, a_in, a_out, b_out, theta=0.):
        if not (a_out > a_in):
            raise ValueError('"a_out" must be greater than "a_in".')
        if a_in < 0 or b_out < 0:
            raise ValueError('"a_in" and "b_out" must be non-negative.')

        self.positions = self._sanitize_positions(positions)
        self.a_in = float(a_in)
        self.a_out = float(a_out)
        self.b_out = float(b_out)
        self.b_in = self.b_out * self.a_in / self.a_out
        self.theta = float(theta)
        self._params = ['a_in', 'a_out', 'b_out', 'theta']

    @property
    def bounding_boxes(self):
        """
        A list of minimal bounding boxes (`~photutils.BoundingBox`), one
        for each position, enclosing the exact elliptical apertures.
        """

        cos_theta = np.cos(self.theta)
        sin_theta = np.sin(self.theta)
        ax = self.a_out * cos_theta
        ay = self.a_out * sin_theta
        bx = self.b_out * -sin_theta
        by = self.b_out * cos_theta
        dx = np.sqrt(ax*ax + bx*bx)
        dy = np.sqrt(ay*ay + by*by)

        xmin = self.positions[:, 0] - dx
        xmax = self.positions[:, 0] + dx
        ymin = self.positions[:, 1] - dy
        ymax = self.positions[:, 1] + dy

        return [BoundingBox._from_float(x0, x1, y0, y1)
                for x0, x1, y0, y1 in zip(xmin, xmax, ymin, ymax)]

    def area(self):
        return math.pi * (self.a_out * self.b_out - self.a_in * self.b_in)

    def plot(self, origin=(0, 0), indices=None, ax=None, fill=False,
             **kwargs):
        import matplotlib.patches as mpatches

        plot_positions, ax, kwargs = self._prepare_plot(
            origin, indices, ax, fill, **kwargs)

        theta_deg = self.theta * 180. / np.pi
        for position in plot_positions:
            patch_inner = mpatches.Ellipse(position, 2.*self.a_in,
                                           2.*self.b_in, theta_deg, **kwargs)
            patch_outer = mpatches.Ellipse(position, 2.*self.a_out,
                                           2.*self.b_out, theta_deg, **kwargs)
            path = self._make_annulus_path(patch_inner, patch_outer)
            patch = mpatches.PathPatch(path, **kwargs)
            ax.add_patch(patch)

    def to_sky(self, wcs, mode='all'):
        """
        Convert the aperture to a `SkyEllipticalAnnulus` object defined
        in celestial coordinates.

        Parameters
        ----------
        wcs : `~astropy.wcs.WCS`
            The world coordinate system (WCS) transformation to use.

        mode : {'all', 'wcs'}, optional
            Whether to do the transformation including distortions
            (``'all'``; default) or only including only the core WCS
            transformation (``'wcs'``).

        Returns
        -------
        aperture : `SkyEllipticalAnnulus` object
            A `SkyEllipticalAnnulus` object.
        """

        sky_params = self._to_sky_params(wcs, mode=mode)
        return SkyEllipticalAnnulus(**sky_params)


class SkyEllipticalAperture(SkyAperture):
    """
    Elliptical aperture(s), defined in sky coordinates.

    Parameters
    ----------
    positions : `~astropy.coordinates.SkyCoord`
        Celestial coordinates of the aperture center(s). This can be
        either scalar coordinates or an array of coordinates.

    a : `~astropy.units.Quantity`
        The semimajor axis, either in angular or pixel units.

    b : `~astropy.units.Quantity`
        The semiminor axis, either in angular or pixel units.

    theta : `~astropy.units.Quantity`, optional
        The position angle (in angular units) of the semimajor axis.
        For a right-handed world coordinate system, the position angle
        increases counterclockwise from North (PA=0).  The default is 0
        degrees.
    """

    def __init__(self, positions, a, b, theta=0.*u.deg):
        if isinstance(positions, SkyCoord):
            self.positions = positions
        else:
            raise TypeError('positions must be a SkyCoord instance')

        assert_angle_or_pixel('a', a)
        assert_angle_or_pixel('b', b)
        assert_angle('theta', theta)

        if a.unit.physical_type != b.unit.physical_type:
            raise ValueError("a and b should either both be angles "
                             "or in pixels")

        self.a = a
        self.b = b
        self.theta = theta
        self._params = ['a', 'b', 'theta']

    def to_pixel(self, wcs, mode='all'):
        """
        Convert the aperture to an `EllipticalAperture` object defined
        in pixel coordinates.

        Parameters
        ----------
        wcs : `~astropy.wcs.WCS`
            The world coordinate system (WCS) transformation to use.

        mode : {'all', 'wcs'}, optional
            Whether to do the transformation including distortions
            (``'all'``; default) or only including only the core WCS
            transformation (``'wcs'``).

        Returns
        -------
        aperture : `EllipticalAperture` object
            An `EllipticalAperture` object.
        """

        pixel_params = self._to_pixel_params(wcs, mode=mode)
        return EllipticalAperture(**pixel_params)


class SkyEllipticalAnnulus(SkyAperture):
    """
    Elliptical annulus aperture(s), defined in sky coordinates.

    Parameters
    ----------
    positions : `~astropy.coordinates.SkyCoord`
        Celestial coordinates of the aperture center(s). This can be
        either scalar coordinates or an array of coordinates.

    a_in : `~astropy.units.Quantity`
        The inner semimajor axis, either in angular or pixel units.

    a_out : `~astropy.units.Quantity`
        The outer semimajor axis, either in angular or pixel units.

    b_out : float
        The outer semiminor axis, either in angular or pixel units.  The
        inner semiminor axis is calculated as:

            .. math:: b_{in} = b_{out}
                \\left(\\frac{a_{in}}{a_{out}}\\right)

    theta : `~astropy.units.Quantity`, optional
        The position angle (in angular units) of the semimajor axis.
        For a right-handed world coordinate system, the position angle
        increases counterclockwise from North (PA=0).  The default is 0
        degrees.
    """

    def __init__(self, positions, a_in, a_out, b_out, theta=0.*u.deg):
        if isinstance(positions, SkyCoord):
            self.positions = positions
        else:
            raise TypeError('positions must be a SkyCoord instance')

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
        self._params = ['a_in', 'a_out', 'b_out', 'theta']

    def to_pixel(self, wcs, mode='all'):
        """
        Convert the aperture to an `EllipticalAnnulus` object defined in
        pixel coordinates.

        Parameters
        ----------
        wcs : `~astropy.wcs.WCS`
            The world coordinate system (WCS) transformation to use.

        mode : {'all', 'wcs'}, optional
            Whether to do the transformation including distortions
            (``'all'``; default) or only including only the core WCS
            transformation (``'wcs'``).

        Returns
        -------
        aperture : `EllipticalAnnulus` object
            An `EllipticalAnnulus` object.
        """

        pixel_params = self._to_pixel_params(wcs, mode=mode)
        return EllipticalAnnulus(**pixel_params)
