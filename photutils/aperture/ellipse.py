# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Elliptical and elliptical-annulus apertures in both pixel and sky
coordinates.
"""

import math

import astropy.units as u
import numpy as np
from astropy.coordinates import Angle
from astropy.utils import lazyproperty

from photutils.aperture.attributes import (PixelPositions, PositiveScalar,
                                           PositiveScalarAngle, ScalarAngle,
                                           ScalarAngleOrValue,
                                           SkyCoordPositions)
from photutils.aperture.core import PixelAperture, SkyAperture
from photutils.aperture.mask import ApertureMask
from photutils.geometry import elliptical_overlap_grid
from photutils.geometry._batch_photometry import (SHAPE_ELLIPSE,
                                                  SHAPE_ELLIPTICAL_ANNULUS)
from photutils.utils._deprecation import (deprecated,
                                          deprecated_positional_kwargs)
from photutils.utils._wcs_helpers import (pixel_shape_to_sky_svd,
                                          sky_shape_to_pixel_svd)

__all__ = [
    'EllipticalAnnulus',
    'EllipticalAperture',
    'EllipticalMaskMixin',
    'SkyEllipticalAnnulus',
    'SkyEllipticalAperture',
]


@deprecated('3.0', until='4.0')
class EllipticalMaskMixin:  # pragma: no cover
    """
    Mixin class to create masks for elliptical and elliptical-annulus
    aperture objects.

    .. deprecated:: 3.0
    """

    def to_mask(self, method='exact', subpixels=5):
        """
        Return a mask for the aperture.

        Parameters
        ----------
        method : {'exact', 'center', 'subpixel'}, optional
            The method used to determine the pixel weights (the fraction
            of the pixel area covered by the aperture):

            * ``'exact'`` (default):
              Calculates the exact geometric overlap area. Weights are
              continuous in the range [0, 1].
            * ``'center'``:
              Binary weighting based on the pixel center. Weights are
              either 0 or 1. A pixel is included only if its center lies
              strictly inside the aperture; pixel centers lying exactly
              on the aperture boundary are excluded (weight 0).
            * ``'subpixel'``:
              Approximates the overlap by averaging binary samples on a
              subgrid. The number of samples is set by the ``subpixels``
              parameter. Weights are discrete in the range [0, 1]. A
              subpixel is included only if its center lies strictly
              inside the aperture; subpixel centers lying exactly on the
              aperture boundary are excluded (weight 0).

        subpixels : int, optional
            The subsampling factor per axis used when
            ``method='subpixel'``. Each pixel is divided into a grid of
            ``subpixels**2`` subpixels to approximate the overlap. This
            parameter is ignored for other methods.

        Returns
        -------
        mask : `~photutils.aperture.ApertureMask` or list of \
                `~photutils.aperture.ApertureMask`
            A mask for the aperture. If the aperture is scalar then
            a single `~photutils.aperture.ApertureMask` is returned,
            otherwise a list of `~photutils.aperture.ApertureMask` is
            returned.
        """
        use_exact, subpixels = self._translate_mask_method(method, subpixels)

        if hasattr(self, 'a'):
            a = self.a
            b = self.b
        elif hasattr(self, 'a_in'):  # annulus
            a = self.a_out
            b = self.b_out
        else:
            msg = 'Cannot determine the aperture shape'
            raise ValueError(msg)

        masks = []
        for bbox, edges in zip(self._bbox, self._centered_edges, strict=True):
            ny, nx = bbox.shape
            mask = elliptical_overlap_grid(edges[0], edges[1], edges[2],
                                           edges[3], nx, ny, a, b,
                                           self._theta_rad, use_exact,
                                           subpixels)

            # Subtract the inner ellipse for an annulus
            if hasattr(self, 'a_in'):
                mask -= elliptical_overlap_grid(edges[0], edges[1], edges[2],
                                                edges[3], nx, ny, self.a_in,
                                                self.b_in, self._theta_rad,
                                                use_exact, subpixels)

            masks.append(ApertureMask(mask, bbox))

        if self.isscalar:
            return masks[0]

        return masks

    @staticmethod
    def _calc_extents(semimajor_axis, semiminor_axis, theta_rad):
        """
        Calculate half of the bounding box extents of an ellipse.
        """
        return _calc_ellipse_extents(semimajor_axis, semiminor_axis, theta_rad)


def _calc_ellipse_extents(semimajor_axis, semiminor_axis, theta_rad):
    """
    Calculate half of the bounding box extents of an ellipse.
    """
    cos_theta = np.cos(theta_rad)
    sin_theta = np.sin(theta_rad)
    semimajor_x = semimajor_axis * cos_theta
    semimajor_y = semimajor_axis * sin_theta
    semiminor_x = semiminor_axis * -sin_theta
    semiminor_y = semiminor_axis * cos_theta
    x_extent = np.sqrt(semimajor_x**2 + semiminor_x**2)
    y_extent = np.sqrt(semimajor_y**2 + semiminor_y**2)

    return x_extent, y_extent


class EllipticalAperture(PixelAperture):
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

    @deprecated_positional_kwargs(since='3.0', until='4.0')
    def __init__(self, positions, a, b, theta=0.0):
        self.positions = positions
        self.a = a
        self.b = b
        self.theta = theta
        self._theta_rad = self.theta.to_value(u.radian)

    @lazyproperty
    def _xy_extents(self):
        """
        The half of the bounding box extents of the ellipse in the x and
        y directions.
        """
        return _calc_ellipse_extents(self.a, self.b, self._theta_rad)

    def _batch_shape_params(self):
        return SHAPE_ELLIPSE, (self.a, self.b, self._theta_rad)

    @lazyproperty
    def area(self):
        """
        The exact geometric area of the aperture shape.
        """
        return math.pi * self.a * self.b

    def _to_patch(self, *, origin=(0, 0), **kwargs):
        """
        Return a `~matplotlib.patches.Patch` for the aperture.

        Parameters
        ----------
        origin : array_like, optional
            The ``(x, y)`` position of the origin of the displayed
            image.

        **kwargs : dict, optional
            Any keyword arguments accepted by
            `matplotlib.patches.Patch`.

        Returns
        -------
        patch : `~matplotlib.patches.Patch` or list of \
                `~matplotlib.patches.Patch`
            A patch for the aperture. If the aperture is scalar then a
            single `~matplotlib.patches.Patch` is returned, otherwise a
            list of `~matplotlib.patches.Patch` is returned.
        """
        import matplotlib.patches as mpatches

        xy_positions, patch_kwargs = self._define_patch_params(origin=origin,
                                                               **kwargs)

        angle = self.theta.to_value(u.deg)
        patches = [mpatches.Ellipse(xy_position, 2.0 * self.a, 2.0 * self.b,
                                    angle=angle, **patch_kwargs)
                   for xy_position in xy_positions]

        if self.isscalar:
            return patches[0]

        return patches

    def _compute_overlap(self, edges, nx, ny, use_exact, subpixels):
        """
        Compute the overlap of the aperture on the pixel grid.

        Parameters
        ----------
        edges : list of 4 1D `~numpy.ndarray`
            The edges of the pixel grid in the form of
            ``[x_edges, y_edges, x_centers, y_centers]``.

        nx, ny : int
            The number of pixels in the x and y directions.

        use_exact : bool
            Whether to use the exact method for calculating the overlap.

        subpixels : int
            The number of subpixels to use in each dimension for the
            subpixel method.

        Returns
        -------
        overlap : 2D `~numpy.ndarray`
            The overlap of the aperture on the pixel grid. The values
            will be between 0 and 1, where 0 means no overlap and 1
            means full overlap.
        """
        return elliptical_overlap_grid(edges[0], edges[1], edges[2],
                                       edges[3], nx, ny, self.a, self.b,
                                       self._theta_rad, use_exact, subpixels)

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

        Notes
        -----
        The aperture shape parameters are converted using the local WCS
        properties (pixel scale, rotation angle) evaluated at the first
        aperture position. Because aperture objects require scalar shape
        parameters, only a single reference position is used for the
        conversion. For apertures with multiple positions used with a
        WCS that has spatially-varying distortions, this may produce
        inaccurate results for positions far from the first position.
        """
        xpos, ypos = np.transpose(self.positions)
        positions = wcs.pixel_to_world(xpos, ypos)

        first_pos = np.atleast_2d(self.positions)[0]
        pixcoord = (float(first_pos[0]), float(first_pos[1]))
        _, sky_width, sky_height, sky_angle = pixel_shape_to_sky_svd(
            pixcoord, wcs, 2 * self.a, 2 * self.b, self._theta_rad)

        a = Angle(sky_width / 2, 'arcsec')
        b = Angle(sky_height / 2, 'arcsec')
        return SkyEllipticalAperture(positions=positions, a=a, b=b,
                                     theta=sky_angle)


class EllipticalAnnulus(PixelAperture):
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
        If `None`, then the inner semiminor axis is calculated as:

        .. math::

            b_{in} = b_{out} \left(\frac{a_{in}}{a_{out}}\right)

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

    @deprecated_positional_kwargs(since='3.0', until='4.0')
    def __init__(self, positions, a_in, a_out, b_out, b_in=None, theta=0.0):
        if not a_out > a_in:
            msg = "'a_out' must be greater than 'a_in'"
            raise ValueError(msg)

        self.positions = positions
        self.a_in = a_in
        self.a_out = a_out
        self.b_out = b_out

        if b_in is None:
            b_in = self.b_out * self.a_in / self.a_out
        elif not b_out > b_in:
            msg = "'b_out' must be greater than 'b_in'"
            raise ValueError(msg)
        self.b_in = b_in

        self.theta = theta
        self._theta_rad = self.theta.to_value(u.radian)

    @lazyproperty
    def _xy_extents(self):
        """
        The half of the bounding box extents of the outer ellipse in the
        x and y directions.
        """
        return _calc_ellipse_extents(self.a_out, self.b_out, self._theta_rad)

    def _batch_shape_params(self):
        return SHAPE_ELLIPTICAL_ANNULUS, (self.a_in, self.b_in, self.a_out,
                                          self.b_out, self._theta_rad)

    @lazyproperty
    def area(self):
        """
        The exact geometric area of the aperture shape.
        """
        return math.pi * (self.a_out * self.b_out - self.a_in * self.b_in)

    def _to_patch(self, *, origin=(0, 0), **kwargs):
        """
        Return a `~matplotlib.patches.Patch` for the aperture.

        Parameters
        ----------
        origin : array_like, optional
            The ``(x, y)`` position of the origin of the displayed
            image.

        **kwargs : dict, optional
            Any keyword arguments accepted by
            `matplotlib.patches.Patch`.

        Returns
        -------
        patch : `~matplotlib.patches.Patch` or list of \
                `~matplotlib.patches.Patch`
            A patch for the aperture. If the aperture is scalar then a
            single `~matplotlib.patches.Patch` is returned, otherwise a
            list of `~matplotlib.patches.Patch` is returned.
        """
        import matplotlib.patches as mpatches

        xy_positions, patch_kwargs = self._define_patch_params(origin=origin,
                                                               **kwargs)

        patches = []
        angle = self.theta.to_value(u.deg)
        for xy_position in xy_positions:
            patch_inner = mpatches.Ellipse(xy_position, 2.0 * self.a_in,
                                           2.0 * self.b_in, angle=angle)
            patch_outer = mpatches.Ellipse(xy_position, 2.0 * self.a_out,
                                           2.0 * self.b_out, angle=angle)
            path = self._make_annulus_path(patch_inner, patch_outer)
            patches.append(mpatches.PathPatch(path, **patch_kwargs))

        if self.isscalar:
            return patches[0]

        return patches

    def _compute_overlap(self, edges, nx, ny, use_exact, subpixels):
        """
        Compute the overlap of the aperture on the pixel grid.

        Parameters
        ----------
        edges : list of 4 1D `~numpy.ndarray`
            The edges of the pixel grid in the form of
            ``[x_edges, y_edges, x_centers, y_centers]``.

        nx, ny : int
            The number of pixels in the x and y directions.

        use_exact : bool
            Whether to use the exact method for calculating the overlap.

        subpixels : int
            The number of subpixels to use in each dimension for the
            subpixel method.

        Returns
        -------
        overlap : 2D `~numpy.ndarray`
            The overlap of the aperture on the pixel grid. The values
            will be between 0 and 1, where 0 means no overlap and 1
            means full overlap.
        """
        overlap = elliptical_overlap_grid(edges[0], edges[1], edges[2],
                                          edges[3], nx, ny, self.a_out,
                                          self.b_out, self._theta_rad,
                                          use_exact, subpixels)
        overlap -= elliptical_overlap_grid(edges[0], edges[1], edges[2],
                                           edges[3], nx, ny, self.a_in,
                                           self.b_in, self._theta_rad,
                                           use_exact, subpixels)
        return overlap

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

        Notes
        -----
        The aperture shape parameters are converted using the local WCS
        properties (pixel scale, rotation angle) evaluated at the first
        aperture position. Because aperture objects require scalar shape
        parameters, only a single reference position is used for the
        conversion. For apertures with multiple positions used with a
        WCS that has spatially-varying distortions, this may produce
        inaccurate results for positions far from the first position.
        """
        xpos, ypos = np.transpose(self.positions)
        positions = wcs.pixel_to_world(xpos, ypos)

        first_pos = np.atleast_2d(self.positions)[0]
        pixcoord = (float(first_pos[0]), float(first_pos[1]))

        _, sky_w_out, sky_h_out, sky_angle = pixel_shape_to_sky_svd(
            pixcoord, wcs, 2 * self.a_out, 2 * self.b_out, self._theta_rad)
        _, sky_w_in, sky_h_in, _ = pixel_shape_to_sky_svd(
            pixcoord, wcs, 2 * self.a_in, 2 * self.b_in, self._theta_rad)

        a_out = Angle(sky_w_out / 2, 'arcsec')
        b_out = Angle(sky_h_out / 2, 'arcsec')
        a_in = Angle(sky_w_in / 2, 'arcsec')
        b_in = Angle(sky_h_in / 2, 'arcsec')
        return SkyEllipticalAnnulus(positions=positions, a_in=a_in,
                                    a_out=a_out, b_out=b_out,
                                    b_in=b_in, theta=sky_angle)


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
        axis. For a right-handed world coordinate system, the position
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

    @deprecated_positional_kwargs(since='3.0', until='4.0')
    def __init__(self, positions, a, b, theta=0.0 * u.deg):
        self.positions = positions
        self.a = a
        self.b = b
        self.theta = theta
        self._theta_rad = self.theta.to_value(u.radian)

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

        Notes
        -----
        The aperture shape parameters are converted using the local WCS
        properties (pixel scale, rotation angle) evaluated at the first
        aperture position. Because aperture objects require scalar shape
        parameters, only a single reference position is used for the
        conversion. For apertures with multiple positions used with a
        WCS that has spatially-varying distortions, this may produce
        inaccurate results for positions far from the first position.
        """
        xpos, ypos = wcs.world_to_pixel(self.positions)
        positions = np.transpose((xpos, ypos))

        skypos = self.positions if self.isscalar else self.positions[0]
        _, pix_width, pix_height, pix_angle = sky_shape_to_pixel_svd(
            skypos, wcs,
            2 * self.a.to_value(u.arcsec),
            2 * self.b.to_value(u.arcsec),
            self._theta_rad)

        a = pix_width / 2
        b = pix_height / 2
        return EllipticalAperture(positions=positions, a=a, b=b,
                                  theta=pix_angle)


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

        .. math::

            b_{in} = b_{out} \left(\frac{a_{in}}{a_{out}}\right)

    theta : scalar `~astropy.units.Quantity`, optional
        The position angle (in angular units) of the ellipse semimajor
        axis. For a right-handed world coordinate system, the position
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

    @deprecated_positional_kwargs(since='3.0', until='4.0')
    def __init__(self, positions, a_in, a_out, b_out, b_in=None,
                 theta=0.0 * u.deg):
        if not a_out > a_in:
            msg = "'a_out' must be greater than 'a_in'"
            raise ValueError(msg)

        self.positions = positions
        self.a_in = a_in
        self.a_out = a_out
        self.b_out = b_out

        if b_in is None:
            b_in = self.b_out * self.a_in / self.a_out
        elif not b_out > b_in:
            msg = "'b_out' must be greater than 'b_in'"
            raise ValueError(msg)
        self.b_in = b_in

        self.theta = theta
        self._theta_rad = self.theta.to_value(u.radian)

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

        Notes
        -----
        The aperture shape parameters are converted using the local WCS
        properties (pixel scale, rotation angle) evaluated at the first
        aperture position. Because aperture objects require scalar shape
        parameters, only a single reference position is used for the
        conversion. For apertures with multiple positions used with a
        WCS that has spatially-varying distortions, this may produce
        inaccurate results for positions far from the first position.
        """
        xpos, ypos = wcs.world_to_pixel(self.positions)
        positions = np.transpose((xpos, ypos))

        skypos = self.positions if self.isscalar else self.positions[0]

        _, pix_w_out, pix_h_out, pix_angle = sky_shape_to_pixel_svd(
            skypos, wcs,
            2 * self.a_out.to_value(u.arcsec),
            2 * self.b_out.to_value(u.arcsec),
            self._theta_rad)
        _, pix_w_in, pix_h_in, _ = sky_shape_to_pixel_svd(
            skypos, wcs,
            2 * self.a_in.to_value(u.arcsec),
            2 * self.b_in.to_value(u.arcsec),
            self._theta_rad)

        a_out = pix_w_out / 2
        b_out = pix_h_out / 2
        a_in = pix_w_in / 2
        b_in = pix_h_in / 2
        return EllipticalAnnulus(positions=positions, a_in=a_in,
                                 a_out=a_out, b_out=b_out,
                                 b_in=b_in, theta=pix_angle)
