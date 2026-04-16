# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Circular and circular-annulus apertures in both pixel and sky
coordinates.
"""

import math

import astropy.units as u
import numpy as np
from astropy.coordinates import Angle
from astropy.utils import lazyproperty

from photutils.aperture.attributes import (PixelPositions, PositiveScalar,
                                           PositiveScalarAngle,
                                           SkyCoordPositions)
from photutils.aperture.core import PixelAperture, SkyAperture
from photutils.aperture.mask import ApertureMask
from photutils.geometry import circular_overlap_grid
from photutils.utils._deprecation import deprecated
from photutils.utils._wcs_helpers import (pixel_to_sky_mean_scale,
                                          sky_to_pixel_mean_scale)

__all__ = [
    'CircularAnnulus',
    'CircularAperture',
    'CircularMaskMixin',
    'SkyCircularAnnulus',
    'SkyCircularAperture',
]


@deprecated('3.0')
class CircularMaskMixin:  # pragma: no cover
    """
    Mixin class to create masks for circular and circular-annulus
    aperture objects.

    .. deprecated:: 3.0
    """

    def to_mask(self, method='exact', subpixels=5):
        """
        Return a mask for the aperture.

        Parameters
        ----------
        method : {'exact', 'center', 'subpixel'}, optional
            The method used to determine the overlap of the aperture
            on the pixel grid. Not all options are available for all
            aperture types. Note that the more precise methods are
            generally slower. The following methods are available:

            * ``'exact'`` (default):
              The exact fractional overlap of the aperture and each
              pixel is calculated. The aperture weights will contain
              values between 0 and 1.

            * ``'center'``:
              A pixel is considered to be entirely in or out of the
              aperture depending on whether its center is in or out of
              the aperture. The aperture weights will contain values
              only of 0 (out) and 1 (in).

            * ``'subpixel'``:
              A pixel is divided into subpixels (see the ``subpixels``
              keyword), each of which are considered to be entirely in
              or out of the aperture depending on whether its center is
              in or out of the aperture. If ``subpixels=1``, this method
              is equivalent to ``'center'``. The aperture weights will
              contain values between 0 and 1.

        subpixels : int, optional
            For the ``'subpixel'`` method, resample pixels by this
            factor in each dimension. That is, each pixel is divided
            into ``subpixels**2`` subpixels. This keyword is ignored
            unless ``method='subpixel'``.

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

        if hasattr(self, 'r'):
            radius = self.r
        elif hasattr(self, 'r_out'):  # annulus
            radius = self.r_out
        else:
            msg = 'Cannot determine the aperture radius'
            raise ValueError(msg)

        masks = []
        for bbox, edges in zip(self._bbox, self._centered_edges, strict=True):
            ny, nx = bbox.shape
            mask = circular_overlap_grid(edges[0], edges[1], edges[2],
                                         edges[3], nx, ny, radius, use_exact,
                                         subpixels)

            # Subtract the inner circle for an annulus
            if hasattr(self, 'r_in'):
                mask -= circular_overlap_grid(edges[0], edges[1], edges[2],
                                              edges[3], nx, ny, self.r_in,
                                              use_exact, subpixels)

            masks.append(ApertureMask(mask, bbox))

        if self.isscalar:
            return masks[0]

        return masks


class CircularAperture(PixelAperture):
    """
    A circular aperture defined in pixel coordinates.

    The aperture has a single fixed size/shape, but it can have multiple
    positions (see the ``positions`` input).

    Parameters
    ----------
    positions : array_like
        The pixel coordinates of the aperture center(s) in one of the
        following formats:

        * single ``(x, y)`` pair as a tuple, list, or `~numpy.ndarray`
        * tuple, list, or `~numpy.ndarray` of ``(x, y)`` pairs

    r : float
        The radius of the circle in pixels.

    Raises
    ------
    ValueError : `ValueError`
        If the input radius, ``r``, is negative.

    Examples
    --------
    >>> from photutils.aperture import CircularAperture
    >>> aper = CircularAperture([10.0, 20.0], 3.0)
    >>> aper = CircularAperture((10.0, 20.0), 3.0)

    >>> pos1 = (10.0, 20.0)  # (x, y)
    >>> pos2 = (30.0, 40.0)
    >>> pos3 = (50.0, 60.0)
    >>> aper = CircularAperture([pos1, pos2, pos3], 3.0)
    >>> aper = CircularAperture((pos1, pos2, pos3), 3.0)
    """

    _params = ('positions', 'r')
    positions = PixelPositions('The center pixel position(s).')
    r = PositiveScalar('The radius in pixels.')

    def __init__(self, positions, r):
        self.positions = positions
        self.r = r

    @lazyproperty
    def _xy_extents(self):
        return self.r, self.r

    @lazyproperty
    def area(self):
        """
        The exact geometric area of the aperture shape.
        """
        return math.pi * self.r**2

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

        patches = [mpatches.Circle(xy_position, self.r, **patch_kwargs)
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
        return circular_overlap_grid(edges[0], edges[1], edges[2],
                                     edges[3], nx, ny, self.r,
                                     use_exact, subpixels)

    def to_sky(self, wcs):
        """
        Convert the aperture to a `SkyCircularAperture` object defined
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
        aperture : `SkyCircularAperture` object
            A `SkyCircularAperture` object.

        Notes
        -----
        The aperture shape parameters are converted using the local
        WCS pixel scale evaluated at the first aperture position.
        Because aperture objects require scalar shape parameters, only
        a single reference position is used for the conversion. For
        apertures with multiple positions used with a WCS that has
        spatially-varying distortions, this may produce inaccurate
        results for positions far from the first position.
        """
        xpos, ypos = np.transpose(self.positions)
        positions = wcs.pixel_to_world(xpos, ypos)

        first_pos = np.atleast_2d(self.positions)[0]
        _, mean_scale = pixel_to_sky_mean_scale(
            (float(first_pos[0]), float(first_pos[1])), wcs)

        r = Angle(self.r * mean_scale, 'arcsec')
        return SkyCircularAperture(positions=positions, r=r)


class CircularAnnulus(PixelAperture):
    """
    A circular annulus aperture defined in pixel coordinates.

    The aperture has a single fixed size/shape, but it can have multiple
    positions (see the ``positions`` input).

    Parameters
    ----------
    positions : array_like
        The pixel coordinates of the aperture center(s) in one of the
        following formats:

        * single ``(x, y)`` pair as a tuple, list, or `~numpy.ndarray`
        * tuple, list, or `~numpy.ndarray` of ``(x, y)`` pairs

    r_in : float
        The inner radius of the circular annulus in pixels.

    r_out : float
        The outer radius of the circular annulus in pixels.

    Raises
    ------
    ValueError : `ValueError`
        If inner radius (``r_in``) is greater than outer radius
        (``r_out``).

    ValueError : `ValueError`
        If inner radius (``r_in``) is negative.

    Examples
    --------
    >>> from photutils.aperture import CircularAnnulus
    >>> aper = CircularAnnulus([10.0, 20.0], 3.0, 5.0)
    >>> aper = CircularAnnulus((10.0, 20.0), 3.0, 5.0)

    >>> pos1 = (10.0, 20.0)  # (x, y)
    >>> pos2 = (30.0, 40.0)
    >>> pos3 = (50.0, 60.0)
    >>> aper = CircularAnnulus([pos1, pos2, pos3], 3.0, 5.0)
    >>> aper = CircularAnnulus((pos1, pos2, pos3), 3.0, 5.0)
    """

    _params = ('positions', 'r_in', 'r_out')
    positions = PixelPositions('The center pixel position(s).')
    r_in = PositiveScalar('The inner radius in pixels.')
    r_out = PositiveScalar('The outer radius in pixels.')

    def __init__(self, positions, r_in, r_out):
        if not r_out > r_in:
            msg = "'r_out' must be greater than 'r_in'"
            raise ValueError(msg)

        self.positions = positions
        self.r_in = r_in
        self.r_out = r_out

    @lazyproperty
    def _xy_extents(self):
        return self.r_out, self.r_out

    @lazyproperty
    def area(self):
        """
        The exact geometric area of the aperture shape.
        """
        return math.pi * (self.r_out**2 - self.r_in**2)

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
        for xy_position in xy_positions:
            patch_inner = mpatches.Circle(xy_position, self.r_in)
            patch_outer = mpatches.Circle(xy_position, self.r_out)
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
        overlap = circular_overlap_grid(edges[0], edges[1], edges[2],
                                        edges[3], nx, ny, self.r_out,
                                        use_exact, subpixels)
        overlap -= circular_overlap_grid(edges[0], edges[1], edges[2],
                                         edges[3], nx, ny, self.r_in,
                                         use_exact, subpixels)
        return overlap

    def to_sky(self, wcs):
        """
        Convert the aperture to a `SkyCircularAnnulus` object defined in
        celestial coordinates.

        Parameters
        ----------
        wcs : WCS object
            A world coordinate system (WCS) transformation that
            supports the `astropy shared interface for WCS
            <https://docs.astropy.org/en/stable/wcs/wcsapi.html>`_
            (e.g., `astropy.wcs.WCS`, `gwcs.wcs.WCS`).

        Returns
        -------
        aperture : `SkyCircularAnnulus` object
            A `SkyCircularAnnulus` object.

        Notes
        -----
        The aperture shape parameters are converted using the local
        WCS pixel scale evaluated at the first aperture position.
        Because aperture objects require scalar shape parameters, only
        a single reference position is used for the conversion. For
        apertures with multiple positions used with a WCS that has
        spatially-varying distortions, this may produce inaccurate
        results for positions far from the first position.
        """
        xpos, ypos = np.transpose(self.positions)
        positions = wcs.pixel_to_world(xpos, ypos)

        first_pos = np.atleast_2d(self.positions)[0]
        _, mean_scale = pixel_to_sky_mean_scale(
            (float(first_pos[0]), float(first_pos[1])), wcs)

        r_in = Angle(self.r_in * mean_scale, 'arcsec')
        r_out = Angle(self.r_out * mean_scale, 'arcsec')
        return SkyCircularAnnulus(positions=positions, r_in=r_in, r_out=r_out)


class SkyCircularAperture(SkyAperture):
    """
    A circular aperture defined in sky coordinates.

    The aperture has a single fixed size/shape, but it can have multiple
    positions (see the ``positions`` input).

    Parameters
    ----------
    positions : `~astropy.coordinates.SkyCoord`
        The celestial coordinates of the aperture center(s). This can be
        either scalar coordinates or an array of coordinates.

    r : scalar `~astropy.units.Quantity`
        The radius of the circle in angular units.

    Examples
    --------
    >>> from astropy.coordinates import SkyCoord
    >>> import astropy.units as u
    >>> from photutils.aperture import SkyCircularAperture
    >>> positions = SkyCoord(ra=[10.0, 20.0], dec=[30.0, 40.0], unit='deg')
    >>> aper = SkyCircularAperture(positions, 0.5*u.arcsec)
    """

    _params = ('positions', 'r')
    positions = SkyCoordPositions('The center position(s) in sky coordinates.')
    r = PositiveScalarAngle('The radius in angular units.')

    def __init__(self, positions, r):
        self.positions = positions
        self.r = r

    def to_pixel(self, wcs):
        """
        Convert the aperture to a `CircularAperture` object defined in
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
        aperture : `CircularAperture` object
            A `CircularAperture` object.

        Notes
        -----
        The aperture shape parameters are converted using the local
        WCS pixel scale evaluated at the first aperture position.
        Because aperture objects require scalar shape parameters, only
        a single reference position is used for the conversion. For
        apertures with multiple positions used with a WCS that has
        spatially-varying distortions, this may produce inaccurate
        results for positions far from the first position.
        """
        xpos, ypos = wcs.world_to_pixel(self.positions)
        positions = np.transpose((xpos, ypos))

        skypos = self.positions if self.isscalar else self.positions[0]
        _, mean_scale = sky_to_pixel_mean_scale(skypos, wcs)

        r = self.r.to(u.arcsec).value * mean_scale
        return CircularAperture(positions=positions, r=r)


class SkyCircularAnnulus(SkyAperture):
    """
    A circular annulus aperture defined in sky coordinates.

    The aperture has a single fixed size/shape, but it can have multiple
    positions (see the ``positions`` input).

    Parameters
    ----------
    positions : `~astropy.coordinates.SkyCoord`
        The celestial coordinates of the aperture center(s). This can be
        either scalar coordinates or an array of coordinates.

    r_in : scalar `~astropy.units.Quantity`
        The inner radius of the circular annulus in angular units.

    r_out : scalar `~astropy.units.Quantity`
        The outer radius of the circular annulus in angular units.

    Examples
    --------
    >>> from astropy.coordinates import SkyCoord
    >>> import astropy.units as u
    >>> from photutils.aperture import SkyCircularAnnulus
    >>> positions = SkyCoord(ra=[10.0, 20.0], dec=[30.0, 40.0], unit='deg')
    >>> aper = SkyCircularAnnulus(positions, 0.5*u.arcsec, 1.0*u.arcsec)
    """

    _params = ('positions', 'r_in', 'r_out')
    positions = SkyCoordPositions('The center position(s) in sky coordinates.')
    r_in = PositiveScalarAngle('The inner radius in angular units.')
    r_out = PositiveScalarAngle('The outer radius in angular units.')

    def __init__(self, positions, r_in, r_out):
        if not r_out > r_in:
            msg = "'r_out' must be greater than 'r_in'"
            raise ValueError(msg)

        self.positions = positions
        self.r_in = r_in
        self.r_out = r_out

    def to_pixel(self, wcs):
        """
        Convert the aperture to a `CircularAnnulus` object defined in
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
        aperture : `CircularAnnulus` object
            A `CircularAnnulus` object.

        Notes
        -----
        The aperture shape parameters are converted using the local
        WCS pixel scale evaluated at the first aperture position.
        Because aperture objects require scalar shape parameters, only
        a single reference position is used for the conversion. For
        apertures with multiple positions used with a WCS that has
        spatially-varying distortions, this may produce inaccurate
        results for positions far from the first position.
        """
        xpos, ypos = wcs.world_to_pixel(self.positions)
        positions = np.transpose((xpos, ypos))

        skypos = self.positions if self.isscalar else self.positions[0]
        _, mean_scale = sky_to_pixel_mean_scale(skypos, wcs)

        r_in = self.r_in.to(u.arcsec).value * mean_scale
        r_out = self.r_out.to(u.arcsec).value * mean_scale
        return CircularAnnulus(positions=positions, r_in=r_in, r_out=r_out)
