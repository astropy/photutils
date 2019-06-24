# Licensed under a 3-clause BSD style license - see LICENSE.rst

import math

import numpy as np
import astropy.units as u

from .attributes import (PixelPositions, SkyCoordPositions, Scalar,
                         PositiveScalar, AngleScalarQuantity,
                         AngleOrPixelScalarQuantity)
from .core import PixelAperture, SkyAperture
from .bounding_box import BoundingBox
from .mask import ApertureMask
from ..geometry import elliptical_overlap_grid


__all__ = ['EllipticalMaskMixin', 'EllipticalAperture', 'EllipticalAnnulus',
           'SkyEllipticalAperture', 'SkyEllipticalAnnulus']


class EllipticalMaskMixin:
    """
    Mixin class to create masks for elliptical and elliptical-annulus
    aperture objects.
    """

    def to_mask(self, method='exact', subpixels=5):
        """
        Return a mask for the aperture.

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
        mask : `~photutils.ApertureMask` or list of `~photutils.ApertureMask`
            A mask for the aperture.  If the aperture is scalar then a
            single `~photutils.ApertureMask` is returned, otherwise a
            list of `~photutils.ApertureMask` is returned.
        """

        use_exact, subpixels = self._translate_mask_mode(method, subpixels)

        if hasattr(self, 'a'):
            a = self.a
            b = self.b
        elif hasattr(self, 'a_in'):    # annulus
            a = self.a_out
            b = self.b_out
        else:
            raise ValueError('Cannot determine the aperture shape.')

        masks = []
        for bbox, edges in zip(np.atleast_1d(self.bounding_boxes),
                               self._centered_edges):
            ny, nx = bbox.shape
            mask = elliptical_overlap_grid(edges[0], edges[1], edges[2],
                                           edges[3], nx, ny, a, b, self.theta,
                                           use_exact, subpixels)

            # subtract the inner ellipse for an annulus
            if hasattr(self, 'a_in'):
                mask -= elliptical_overlap_grid(edges[0], edges[1], edges[2],
                                                edges[3], nx, ny, self.a_in,
                                                self.b_in, self.theta,
                                                use_exact, subpixels)

            masks.append(ApertureMask(mask, bbox))

        if self.isscalar:
            return masks[0]
        else:
            return masks

    @staticmethod
    def _extents(semimajor_axis, semiminor_axis, theta):
        """
        Calculate half of the bounding box extents of an ellipse.
        """

        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        semimajor_x = semimajor_axis * cos_theta
        semimajor_y = semimajor_axis * sin_theta
        semiminor_x = semiminor_axis * -sin_theta
        semiminor_y = semiminor_axis * cos_theta
        x_extent = np.sqrt(semimajor_x**2 + semiminor_x**2)
        y_extent = np.sqrt(semimajor_y**2 + semiminor_y**2)

        return x_extent, y_extent


class EllipticalAperture(EllipticalMaskMixin, PixelAperture):
    """
    An elliptical aperture defined in pixel coordinates.

    The aperture has a single fixed size/shape, but it can have multiple
    positions (see the ``positions`` input).

    Parameters
    ----------
    positions : array_like or `~astropy.units.Quantity`
        The pixel coordinates of the aperture center(s) in one of the
        following formats:

            * single ``(x, y)`` pair as a tuple, list, or `~numpy.ndarray`
            * tuple, list, or `~numpy.ndarray` of ``(x, y)`` pairs
            * `~astropy.units.Quantity` instance of ``(x, y)`` pairs in
              pixel units

    a : float
        The semimajor axis of the ellipse in pixels.

    b : float
        The semiminor axis of the ellipse in pixels.

    theta : float, optional
        The rotation angle in radians of the ellipse semimajor axis from
        the positive ``x`` axis.  The rotation angle increases
        counterclockwise.  The default is 0.

    Raises
    ------
    ValueError : `ValueError`
        If either axis (``a`` or ``b``) is negative.

    Examples
    --------
    >>> from photutils import EllipticalAperture
    >>> aper = EllipticalAperture([10., 20.], 5., 3.)
    >>> aper = EllipticalAperture((10., 20.), 5., 3., theta=np.pi)

    >>> pos1 = (10., 20.)  # (x, y)
    >>> pos2 = (30., 40.)
    >>> pos3 = (50., 60.)
    >>> aper = EllipticalAperture([pos1, pos2, pos3], 5., 3.)
    >>> aper = EllipticalAperture((pos1, pos2, pos3), 5., 3., theta=np.pi)
    """

    positions = PixelPositions('positions')
    a = PositiveScalar('a')
    b = PositiveScalar('b')
    theta = Scalar('theta')

    def __init__(self, positions, a, b, theta=0.):
        self.positions = positions
        self.a = a
        self.b = b
        self.theta = theta
        self._params = ['a', 'b', 'theta']

    @property
    def bounding_boxes(self):
        """
        A list of minimal bounding boxes (`~photutils.BoundingBox`), one
        for each position, enclosing the exact elliptical apertures.
        """

        positions = np.atleast_2d(self.positions)
        x_delta, y_delta = self._extents(self.a, self.b, self.theta)
        xmin = positions[:, 0] - x_delta
        xmax = positions[:, 0] + x_delta
        ymin = positions[:, 1] - y_delta
        ymax = positions[:, 1] + y_delta

        bboxes = [BoundingBox.from_float(x0, x1, y0, y1)
                  for x0, x1, y0, y1 in zip(xmin, xmax, ymin, ymax)]

        if self.isscalar:
            return bboxes[0]
        else:
            return bboxes

    @property
    def area(self):
        return math.pi * self.a * self.b

    def _to_patch(self, origin=(0, 0), indices=None, **kwargs):
        """
        Return a `~matplotlib.patches.patch` for the aperture.

        Parameters
        ----------
        origin : array_like, optional
            The ``(x, y)`` position of the origin of the displayed
            image.

        indices : int or array of int, optional
            The indices of the aperture positions to plot.

        kwargs : `dict`
            Any keyword arguments accepted by
            `matplotlib.patches.Patch`.

        Returns
        -------
        patch : `~matplotlib.patches.patch` or list of `~matplotlib.patches.patch`
            A patch for the aperture.  If the aperture is scalar then a
            single `~matplotlib.patches.patch` is returned, otherwise a
            list of `~matplotlib.patches.patch` is returned.
        """

        import matplotlib.patches as mpatches

        xy_positions, patch_kwargs = self._define_patch_params(
            origin=origin, indices=indices, **kwargs)

        patches = []
        theta_deg = self.theta * 180. / np.pi
        for xy_position in xy_positions:
            patches.append(mpatches.Ellipse(xy_position, 2.*self.a, 2.*self.b,
                                            theta_deg, **patch_kwargs))

        if self.isscalar:
            return patches[0]
        else:
            return patches

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
    An elliptical annulus aperture defined in pixel coordinates.

    The aperture has a single fixed size/shape, but it can have multiple
    positions (see the ``positions`` input).

    Parameters
    ----------
    positions : array_like or `~astropy.units.Quantity`
        The pixel coordinates of the aperture center(s) in one of the
        following formats:

            * single ``(x, y)`` pair as a tuple, list, or `~numpy.ndarray`
            * tuple, list, or `~numpy.ndarray` of ``(x, y)`` pairs
            * `~astropy.units.Quantity` instance of ``(x, y)`` pairs in
              pixel units

    a_in : float
        The inner semimajor axis of the elliptical annulus in pixels.

    a_out : float
        The outer semimajor axis of the elliptical annulus in pixels.

    b_out : float
        The outer semiminor axis of the elliptical annulus in pixels.
        The inner semiminor axis is calculated as:

            .. math:: b_{in} = b_{out}
                \\left(\\frac{a_{in}}{a_{out}}\\right)

    theta : float, optional
        The rotation angle in radians of the ellipse semimajor axis from
        the positive ``x`` axis.  The rotation angle increases
        counterclockwise.  The default is 0.

    Raises
    ------
    ValueError : `ValueError`
        If inner semimajor axis (``a_in``) is greater than outer semimajor
        axis (``a_out``).

    ValueError : `ValueError`
        If either the inner semimajor axis (``a_in``) or the outer semiminor
        axis (``b_out``) is negative.

    Examples
    --------
    >>> from photutils import EllipticalAnnulus
    >>> aper = EllipticalAnnulus([10., 20.], 3., 8., 5.)
    >>> aper = EllipticalAnnulus((10., 20.), 3., 8., 5., theta=np.pi)

    >>> pos1 = (10., 20.)  # (x, y)
    >>> pos2 = (30., 40.)
    >>> pos3 = (50., 60.)
    >>> aper = EllipticalAnnulus([pos1, pos2, pos3], 3., 8., 5.)
    >>> aper = EllipticalAnnulus((pos1, pos2, pos3), 3., 8., 5., theta=np.pi)
    """

    positions = PixelPositions('positions')
    a_in = PositiveScalar('a_in')
    a_out = PositiveScalar('a_out')
    b_out = PositiveScalar('b_out')
    theta = Scalar('theta')

    def __init__(self, positions, a_in, a_out, b_out, theta=0.):
        if not a_out > a_in:
            raise ValueError('"a_out" must be greater than "a_in".')

        self.positions = positions
        self.a_in = a_in
        self.a_out = a_out
        self.b_out = b_out
        self.b_in = self.b_out * self.a_in / self.a_out
        self.theta = theta
        self._params = ['a_in', 'a_out', 'b_out', 'theta']

    @property
    def bounding_boxes(self):
        """
        A list of minimal bounding boxes (`~photutils.BoundingBox`), one
        for each position, enclosing the exact elliptical apertures.
        """

        positions = np.atleast_2d(self.positions)
        x_delta, y_delta = self._extents(self.a_out, self.b_out, self.theta)
        xmin = positions[:, 0] - x_delta
        xmax = positions[:, 0] + x_delta
        ymin = positions[:, 1] - y_delta
        ymax = positions[:, 1] + y_delta

        bboxes = [BoundingBox.from_float(x0, x1, y0, y1)
                  for x0, x1, y0, y1 in zip(xmin, xmax, ymin, ymax)]

        if self.isscalar:
            return bboxes[0]
        else:
            return bboxes

    @property
    def area(self):
        return math.pi * (self.a_out * self.b_out - self.a_in * self.b_in)

    def _to_patch(self, origin=(0, 0), indices=None, **kwargs):
        """
        Return a `~matplotlib.patches.patch` for the aperture.

        Parameters
        ----------
        origin : array_like, optional
            The ``(x, y)`` position of the origin of the displayed
            image.

        indices : int or array of int, optional
            The indices of the aperture positions to plot.

        kwargs : `dict`
            Any keyword arguments accepted by
            `matplotlib.patches.Patch`.

        Returns
        -------
        patch : `~matplotlib.patches.patch` or list of `~matplotlib.patches.patch`
            A patch for the aperture.  If the aperture is scalar then a
            single `~matplotlib.patches.patch` is returned, otherwise a
            list of `~matplotlib.patches.patch` is returned.
        """

        import matplotlib.patches as mpatches

        xy_positions, patch_kwargs = self._define_patch_params(
            origin=origin, indices=indices, **kwargs)

        patches = []
        theta_deg = self.theta * 180. / np.pi
        for xy_position in xy_positions:
            patch_inner = mpatches.Ellipse(xy_position, 2.*self.a_in,
                                           2.*self.b_in, theta_deg)
            patch_outer = mpatches.Ellipse(xy_position, 2.*self.a_out,
                                           2.*self.b_out, theta_deg)
            path = self._make_annulus_path(patch_inner, patch_outer)
            patches.append(mpatches.PathPatch(path, **patch_kwargs))

        if self.isscalar:
            return patches[0]
        else:
            return patches

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
    An elliptical aperture defined in sky coordinates.

    The aperture has a single fixed size/shape, but it can have multiple
    positions (see the ``positions`` input).

    Parameters
    ----------
    positions : `~astropy.coordinates.SkyCoord`
        The celestial coordinates of the aperture center(s). This can be
        either scalar coordinates or an array of coordinates.

    a : scalar `~astropy.units.Quantity`
        The semimajor axis of the ellipse, either in angular or pixel
        units.

    b : scalar `~astropy.units.Quantity`
        The semiminor axis of the ellipse, either in angular or pixel
        units.

    theta : scalar `~astropy.units.Quantity`, optional
        The position angle (in angular units) of the ellipse semimajor
        axis.  For a right-handed world coordinate system, the position
        angle increases counterclockwise from North (PA=0).  The default
        is 0 degrees.

    Examples
    --------
    >>> from astropy.coordinates import SkyCoord
    >>> import astropy.units as u
    >>> from photutils import SkyEllipticalAperture
    >>> positions = SkyCoord(ra=[10., 20.], dec=[30., 40.], unit='deg')
    >>> aper = SkyEllipticalAperture(positions, 1.0*u.arcsec, 0.5*u.arcsec)
    """

    positions = SkyCoordPositions('positions')
    a = AngleOrPixelScalarQuantity('a')
    b = AngleOrPixelScalarQuantity('b')
    theta = AngleScalarQuantity('theta')

    def __init__(self, positions, a, b, theta=0.*u.deg):
        if a.unit.physical_type != b.unit.physical_type:
            raise ValueError("a and b should either both be angles "
                             "or in pixels")

        self.positions = positions
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
    An elliptical annulus aperture defined in sky coordinates.

    The aperture has a single fixed size/shape, but it can have multiple
    positions (see the ``positions`` input).

    Parameters
    ----------
    positions : `~astropy.coordinates.SkyCoord`
        The celestial coordinates of the aperture center(s). This can be
        either scalar coordinates or an array of coordinates.

    a_in : scalar `~astropy.units.Quantity`
        The inner semimajor axis, either in angular or pixel units.

    a_out : scalar `~astropy.units.Quantity`
        The outer semimajor axis, either in angular or pixel units.

    b_out : scalar `~astropy.units.Quantity`
        The outer semiminor axis, either in angular or pixel units.  The
        inner semiminor axis is calculated as:

            .. math:: b_{in} = b_{out}
                \\left(\\frac{a_{in}}{a_{out}}\\right)

    theta : scalar `~astropy.units.Quantity`, optional
        The position angle (in angular units) of the ellipse semimajor
        axis.  For a right-handed world coordinate system, the position
        angle increases counterclockwise from North (PA=0).  The default
        is 0 degrees.

    Examples
    --------
    >>> from astropy.coordinates import SkyCoord
    >>> import astropy.units as u
    >>> from photutils import SkyEllipticalAnnulus
    >>> positions = SkyCoord(ra=[10., 20.], dec=[30., 40.], unit='deg')
    >>> aper = SkyEllipticalAnnulus(positions, 0.5*u.arcsec, 2.0*u.arcsec,
    ...                             1.0*u.arcsec)
    """

    positions = SkyCoordPositions('positions')
    a_in = AngleOrPixelScalarQuantity('a_in')
    a_out = AngleOrPixelScalarQuantity('a_out')
    b_out = AngleOrPixelScalarQuantity('b_out')
    theta = AngleScalarQuantity('theta')

    def __init__(self, positions, a_in, a_out, b_out, theta=0.*u.deg):
        if a_in.unit.physical_type != a_out.unit.physical_type:
            raise ValueError("a_in and a_out should either both be angles "
                             "or in pixels")

        if a_out.unit.physical_type != b_out.unit.physical_type:
            raise ValueError("a_out and b_out should either both be angles "
                             "or in pixels")

        self.positions = positions
        self.a_in = a_in
        self.a_out = a_out
        self.b_out = b_out
        self.b_in = self.b_out * self.a_in / self.a_out
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
