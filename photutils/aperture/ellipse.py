# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
This module defines elliptical and elliptical-annulus apertures in both
pixel and sky coordinates.
"""

import math

import astropy.units as u
import numpy as np

from photutils.aperture.attributes import (PixelPositions, PositiveScalar,
                                           PositiveScalarAngle, ScalarAngle,
                                           ScalarAngleOrValue,
                                           SkyCoordPositions)
from photutils.aperture.core import PixelAperture, SkyAperture
from photutils.aperture.mask import ApertureMask
from photutils.geometry import elliptical_overlap_grid

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
                  each pixel is calculated. The aperture weights will
                  contain values between 0 and 1.

                * ``'center'``:
                  A pixel is considered to be entirely in or out of the
                  aperture depending on whether its center is in or out
                  of the aperture. The aperture weights will contain
                  values only of 0 (out) and 1 (in).

                * ``'subpixel'``:
                  A pixel is divided into subpixels (see the
                  ``subpixels`` keyword), each of which are considered
                  to be entirely in or out of the aperture depending
                  on whether its center is in or out of the aperture.
                  If ``subpixels=1``, this method is equivalent to
                  ``'center'``. The aperture weights will contain values
                  between 0 and 1.

        subpixels : int, optional
            For the ``'subpixel'`` method, resample pixels by this
            factor in each dimension. That is, each pixel is divided
            into ``subpixels**2`` subpixels. This keyword is ignored
            unless ``method='subpixel'``.

        Returns
        -------
        mask : `~photutils.aperture.ApertureMask` or list of `~photutils.aperture.ApertureMask`
            A mask for the aperture.  If the aperture is scalar then a
            single `~photutils.aperture.ApertureMask` is returned,
            otherwise a list of `~photutils.aperture.ApertureMask` is
            returned.
        """
        use_exact, subpixels = self._translate_mask_mode(method, subpixels)

        if hasattr(self, 'a'):
            a = self.a
            b = self.b
        elif hasattr(self, 'a_in'):  # annulus
            a = self.a_out
            b = self.b_out
        else:
            raise ValueError('Cannot determine the aperture shape.')

        masks = []
        for bbox, edges in zip(self._bbox, self._centered_edges):
            ny, nx = bbox.shape
            mask = elliptical_overlap_grid(edges[0], edges[1], edges[2],
                                           edges[3], nx, ny, a, b,
                                           self._theta_radians,
                                           use_exact, subpixels)

            # subtract the inner ellipse for an annulus
            if hasattr(self, 'a_in'):
                mask -= elliptical_overlap_grid(edges[0], edges[1], edges[2],
                                                edges[3], nx, ny, self.a_in,
                                                self.b_in, self._theta_radians,
                                                use_exact, subpixels)

            masks.append(ApertureMask(mask, bbox))

        if self.isscalar:
            return masks[0]
        else:
            return masks

    @staticmethod
    def _calc_extents(semimajor_axis, semiminor_axis, theta):
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
    positions : array_like
        The pixel coordinates of the aperture center(s) in one of the
        following formats:

            * single ``(x, y)`` pair as a tuple, list, or `~numpy.ndarray`
            * tuple, list, or `~numpy.ndarray` of ``(x, y)`` pairs

    a : float
        The semimajor axis of the ellipse in pixels.

    b : float
        The semiminor axis of the ellipse in pixels.

    theta : float or `~astropy.units.Quantity`, optional
        The rotation angle as an angular quantity
        (`~astropy.units.Quantity` or `~astropy.coordinates.Angle`) or
        value in radians (as a float) from the positive ``x`` axis. The
        rotation angle increases counterclockwise.

    Raises
    ------
    ValueError : `ValueError`
        If either axis (``a`` or ``b``) is negative.

    Examples
    --------
    >>> from astropy.coordinates import Angle
    >>> from photutils.aperture import EllipticalAperture

    >>> theta = Angle(80, 'deg')
    >>> aper = EllipticalAperture([10.0, 20.0], 5.0, 3.0)
    >>> aper = EllipticalAperture((10.0, 20.0), 5.0, 3.0, theta=theta)

    >>> pos1 = (10.0, 20.0)  # (x, y)
    >>> pos2 = (30.0, 40.0)
    >>> pos3 = (50.0, 60.0)
    >>> aper = EllipticalAperture([pos1, pos2, pos3], 5.0, 3.0)
    >>> aper = EllipticalAperture((pos1, pos2, pos3), 5.0, 3.0, theta=theta)
    """

    _params = ('positions', 'a', 'b', 'theta')
    positions = PixelPositions('The center pixel position(s).')
    a = PositiveScalar('The semimajor axis in pixels.')
    b = PositiveScalar('The semiminor axis in pixels.')
    theta = ScalarAngleOrValue('The counterclockwise rotation angle as an '
                               'angular Quantity or value in radians from '
                               'the positive x axis.')

    def __init__(self, positions, a, b, theta=0.0):
        self.positions = positions
        self.a = a
        self.b = b
        self._theta_radians = 0.0  # defined by theta setter
        self.theta = theta

    @property
    def _xy_extents(self):
        return self._calc_extents(self.a, self.b, self._theta_radians)

    @property
    def area(self):
        return math.pi * self.a * self.b

    def _to_patch(self, origin=(0, 0), **kwargs):
        """
        Return a `~matplotlib.patches.patch` for the aperture.

        Parameters
        ----------
        origin : array_like, optional
            The ``(x, y)`` position of the origin of the displayed
            image.

        **kwargs : `dict`
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

        xy_positions, patch_kwargs = self._define_patch_params(origin=origin,
                                                               **kwargs)

        patches = []
        theta_deg = self._theta_radians * 180.0 / np.pi
        for xy_position in xy_positions:
            patches.append(mpatches.Ellipse(xy_position, 2.0 * self.a,
                                            2.0 * self.b, angle=theta_deg,
                                            **patch_kwargs))

        if self.isscalar:
            return patches[0]
        else:
            return patches

    def to_mask(self, method='exact', subpixels=5):
        return EllipticalMaskMixin.to_mask(self, method=method,
                                           subpixels=subpixels)

    def to_sky(self, wcs):
        """
        Convert the aperture to a `SkyEllipticalAperture` object defined
        in celestial coordinates.

        Parameters
        ----------
        wcs : WCS object
            A world coordinate system (WCS) transformation that
            supports the `astropy shared interface for WCS
            <https://docs.astropy.org/en/stable/wcs/wcsapi.html>`_
            (e.g., `astropy.wcs.WCS`, `gwcs.wcs.WCS`).

        Returns
        -------
        aperture : `SkyEllipticalAperture` object
            A `SkyEllipticalAperture` object.
        """
        return SkyEllipticalAperture(**self._to_sky_params(wcs))


class EllipticalAnnulus(EllipticalMaskMixin, PixelAperture):
    r"""
    An elliptical annulus aperture defined in pixel coordinates.

    The aperture has a single fixed size/shape, but it can have multiple
    positions (see the ``positions`` input).

    Parameters
    ----------
    positions : array_like
        The pixel coordinates of the aperture center(s) in one of the
        following formats:

            * single ``(x, y)`` pair as a tuple, list, or `~numpy.ndarray`
            * tuple, list, or `~numpy.ndarray` of ``(x, y)`` pairs

    a_in : float
        The inner semimajor axis of the elliptical annulus in pixels.

    a_out : float
        The outer semimajor axis of the elliptical annulus in pixels.

    b_out : float
        The outer semiminor axis of the elliptical annulus in pixels.

    b_in : `None` or float, optional
        The inner semiminor axis of the elliptical annulus in pixels.
        If `None`, then the the inner semiminor axis is calculated as:

            .. math:: b_{in} = b_{out}
                \left(\frac{a_{in}}{a_{out}}\right)

    theta : float or `~astropy.units.Quantity`, optional
        The rotation angle as an angular quantity
        (`~astropy.units.Quantity` or `~astropy.coordinates.Angle`) or
        value in radians (as a float) from the positive ``x`` axis. The
        rotation angle increases counterclockwise.

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
    >>> from astropy.coordinates import Angle
    >>> from photutils.aperture import EllipticalAnnulus

    >>> theta = Angle(80, 'deg')
    >>> aper = EllipticalAnnulus([10.0, 20.0], 3.0, 8.0, 5.0)
    >>> aper = EllipticalAnnulus((10.0, 20.0), 3.0, 8.0, 5.0, theta=theta)

    >>> pos1 = (10.0, 20.0)  # (x, y)
    >>> pos2 = (30.0, 40.0)
    >>> pos3 = (50.0, 60.0)
    >>> aper = EllipticalAnnulus([pos1, pos2, pos3], 3.0, 8.0, 5.0)
    >>> aper = EllipticalAnnulus((pos1, pos2, pos3), 3.0, 8.0, 5.0,
    ...                          theta=theta)
    """

    _params = ('positions', 'a_in', 'a_out', 'b_in', 'b_out', 'theta')
    positions = PixelPositions('The center pixel position(s).')
    a_in = PositiveScalar('The inner semimajor axis in pixels.')
    a_out = PositiveScalar('The outer semimajor axis in pixels.')
    b_in = PositiveScalar('The inner semiminor axis in pixels.')
    b_out = PositiveScalar('The outer semiminor axis in pixels.')
    theta = ScalarAngleOrValue('The counterclockwise rotation angle as an '
                               'angular Quantity or value in radians from '
                               'the positive x axis.')

    def __init__(self, positions, a_in, a_out, b_out, b_in=None, theta=0.0):
        if not a_out > a_in:
            raise ValueError('"a_out" must be greater than "a_in".')

        self.positions = positions
        self.a_in = a_in
        self.a_out = a_out
        self.b_out = b_out

        if b_in is None:
            b_in = self.b_out * self.a_in / self.a_out
        else:
            if not b_out > b_in:
                raise ValueError('"b_out" must be greater than "b_in".')
        self.b_in = b_in

        self._theta_radians = 0.0  # defined by theta setter
        self.theta = theta

    @property
    def _xy_extents(self):
        return self._calc_extents(self.a_out, self.b_out, self._theta_radians)

    @property
    def area(self):
        return math.pi * (self.a_out * self.b_out - self.a_in * self.b_in)

    def _to_patch(self, origin=(0, 0), **kwargs):
        """
        Return a `~matplotlib.patches.patch` for the aperture.

        Parameters
        ----------
        origin : array_like, optional
            The ``(x, y)`` position of the origin of the displayed
            image.

        **kwargs : `dict`
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

        xy_positions, patch_kwargs = self._define_patch_params(origin=origin,
                                                               **kwargs)

        patches = []
        theta_deg = self._theta_radians * 180.0 / np.pi
        for xy_position in xy_positions:
            patch_inner = mpatches.Ellipse(xy_position, 2.0 * self.a_in,
                                           2.0 * self.b_in, angle=theta_deg)
            patch_outer = mpatches.Ellipse(xy_position, 2.0 * self.a_out,
                                           2.0 * self.b_out, angle=theta_deg)
            path = self._make_annulus_path(patch_inner, patch_outer)
            patches.append(mpatches.PathPatch(path, **patch_kwargs))

        if self.isscalar:
            return patches[0]
        else:
            return patches

    def to_mask(self, method='exact', subpixels=5):
        return EllipticalMaskMixin.to_mask(self, method=method,
                                           subpixels=subpixels)

    def to_sky(self, wcs):
        """
        Convert the aperture to a `SkyEllipticalAnnulus` object defined
        in celestial coordinates.

        Parameters
        ----------
        wcs : WCS object
            A world coordinate system (WCS) transformation that
            supports the `astropy shared interface for WCS
            <https://docs.astropy.org/en/stable/wcs/wcsapi.html>`_
            (e.g., `astropy.wcs.WCS`, `gwcs.wcs.WCS`).

        Returns
        -------
        aperture : `SkyEllipticalAnnulus` object
            A `SkyEllipticalAnnulus` object.
        """
        return SkyEllipticalAnnulus(**self._to_sky_params(wcs))


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
        The semimajor axis of the ellipse in angular units.

    b : scalar `~astropy.units.Quantity`
        The semiminor axis of the ellipse in angular units.

    theta : scalar `~astropy.units.Quantity`, optional
        The position angle (in angular units) of the ellipse semimajor
        axis.  For a right-handed world coordinate system, the position
        angle increases counterclockwise from North (PA=0).

    Examples
    --------
    >>> from astropy.coordinates import SkyCoord
    >>> import astropy.units as u
    >>> from photutils.aperture import SkyEllipticalAperture
    >>> positions = SkyCoord(ra=[10.0, 20.0], dec=[30.0, 40.0], unit='deg')
    >>> aper = SkyEllipticalAperture(positions, 1.0*u.arcsec, 0.5*u.arcsec)
    """

    _params = ('positions', 'a', 'b', 'theta')
    positions = SkyCoordPositions('The center position(s) in sky coordinates.')
    a = PositiveScalarAngle('The semimajor axis in angular units.')
    b = PositiveScalarAngle('The semiminor axis in angular units.')
    theta = ScalarAngle('The position angle in angular units of the ellipse '
                        'semimajor axis.')

    def __init__(self, positions, a, b, theta=0.0 * u.deg):
        self.positions = positions
        self.a = a
        self.b = b
        self.theta = theta

    def to_pixel(self, wcs):
        """
        Convert the aperture to an `EllipticalAperture` object defined
        in pixel coordinates.

        Parameters
        ----------
        wcs : WCS object
            A world coordinate system (WCS) transformation that
            supports the `astropy shared interface for WCS
            <https://docs.astropy.org/en/stable/wcs/wcsapi.html>`_
            (e.g., `astropy.wcs.WCS`, `gwcs.wcs.WCS`).

        Returns
        -------
        aperture : `EllipticalAperture` object
            An `EllipticalAperture` object.
        """
        return EllipticalAperture(**self._to_pixel_params(wcs))


class SkyEllipticalAnnulus(SkyAperture):
    r"""
    An elliptical annulus aperture defined in sky coordinates.

    The aperture has a single fixed size/shape, but it can have multiple
    positions (see the ``positions`` input).

    Parameters
    ----------
    positions : `~astropy.coordinates.SkyCoord`
        The celestial coordinates of the aperture center(s). This can be
        either scalar coordinates or an array of coordinates.

    a_in : scalar `~astropy.units.Quantity`
        The inner semimajor axis in angular units.

    a_out : scalar `~astropy.units.Quantity`
        The outer semimajor axis in angular units.

    b_out : scalar `~astropy.units.Quantity`
        The outer semiminor axis in angular units.

    b_in : `None` or scalar `~astropy.units.Quantity`
        The inner semiminor axis in angular units. If `None`, then the
        inner semiminor axis is calculated as:

            .. math:: b_{in} = b_{out}
                \left(\frac{a_{in}}{a_{out}}\right)

    theta : scalar `~astropy.units.Quantity`, optional
        The position angle (in angular units) of the ellipse semimajor
        axis.  For a right-handed world coordinate system, the position
        angle increases counterclockwise from North (PA=0).

    Examples
    --------
    >>> from astropy.coordinates import SkyCoord
    >>> import astropy.units as u
    >>> from photutils.aperture import SkyEllipticalAnnulus
    >>> positions = SkyCoord(ra=[10.0, 20.0], dec=[30.0, 40.0], unit='deg')
    >>> aper = SkyEllipticalAnnulus(positions, 0.5*u.arcsec, 2.0*u.arcsec,
    ...                             1.0*u.arcsec)
    """

    _params = ('positions', 'a_in', 'a_out', 'b_in', 'b_out', 'theta')
    positions = SkyCoordPositions('The center position(s) in sky coordinates.')
    a_in = PositiveScalarAngle('The inner semimajor axis in angular units.')
    a_out = PositiveScalarAngle('The outer semimajor axis in angular units.')
    b_in = PositiveScalarAngle('The inner semiminor axis in angular units.')
    b_out = PositiveScalarAngle('The outer semiminor axis in angular units.')
    theta = ScalarAngle('The position angle in angular units of the ellipse '
                        'semimajor axis.')

    def __init__(self, positions, a_in, a_out, b_out, b_in=None,
                 theta=0.0 * u.deg):
        if not a_out > a_in:
            raise ValueError('"a_out" must be greater than "a_in".')

        self.positions = positions
        self.a_in = a_in
        self.a_out = a_out
        self.b_out = b_out

        if b_in is None:
            b_in = self.b_out * self.a_in / self.a_out
        else:
            if not b_out > b_in:
                raise ValueError('"b_out" must be greater than "b_in".')
        self.b_in = b_in

        self.theta = theta

    def to_pixel(self, wcs):
        """
        Convert the aperture to an `EllipticalAnnulus` object defined in
        pixel coordinates.

        Parameters
        ----------
        wcs : WCS object
            A world coordinate system (WCS) transformation that
            supports the `astropy shared interface for WCS
            <https://docs.astropy.org/en/stable/wcs/wcsapi.html>`_
            (e.g., `astropy.wcs.WCS`, `gwcs.wcs.WCS`).

        Returns
        -------
        aperture : `EllipticalAnnulus` object
            An `EllipticalAnnulus` object.
        """
        return EllipticalAnnulus(**self._to_pixel_params(wcs))
