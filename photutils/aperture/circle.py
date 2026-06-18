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
from photutils.aperture.polygon import PolygonAperture, SkyPolygonAperture
from photutils.geometry import circular_overlap_grid
from photutils.geometry._batch_photometry import (SHAPE_CIRCLE,
                                                  SHAPE_CIRCULAR_ANNULUS)
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


def _circular_polygon_offsets(r, n_vertices):
    """
    Compute the vertex offsets that approximate a circle of radius ``r``
    using ``n_vertices`` equally spaced vertices.

    ``r`` may be a plain number (pixel offsets) or a
    `~astropy.units.Quantity` (angular offsets); the returned offsets
    carry the same type.

    Parameters
    ----------
    r : float or `~astropy.units.Quantity`
        The radius of the circle.

    n_vertices : int
        The number of polygon vertices used to approximate the circle.

    Returns
    -------
    offsets : 2D `~numpy.ndarray`
        The vertex offsets that approximate the circle. The shape is
        ``(n_vertices, 2)``, where the second axis corresponds to the
        ``(x, y)`` offsets. The offsets are in the same units as the
        input ``r``.
    """
    n_vertices = int(n_vertices)
    if n_vertices < 3:
        msg = f'n_vertices must be at least 3, got {n_vertices}'
        raise ValueError(msg)

    theta = np.linspace(0.0, 2.0 * np.pi, n_vertices, endpoint=False)
    return np.column_stack([np.cos(theta), np.sin(theta)]) * r


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

    def _batch_shape_params(self):
        return SHAPE_CIRCLE, (self.r,)

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

    def to_polygon(self, n_vertices=100):
        """
        Return a `~photutils.aperture.PolygonAperture` that
        approximates this circular aperture.

        Parameters
        ----------
        n_vertices : int, optional
            The number of polygon vertices used to approximate the
            circle. Must be at least 3. Default is 100.

        Returns
        -------
        aperture : `~photutils.aperture.PolygonAperture`
            A polygon aperture that approximates the circle.
        """
        offsets = _circular_polygon_offsets(self.r, n_vertices)
        return PolygonAperture._from_convex_offsets(self.positions, offsets)


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

    def _batch_shape_params(self):
        return SHAPE_CIRCULAR_ANNULUS, (self.r_in, self.r_out)

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

        r = self.r.to_value(u.arcsec) * mean_scale
        return CircularAperture(positions=positions, r=r)

    def to_polygon(self, n_vertices=100):
        """
        Return a `~photutils.aperture.SkyPolygonAperture` that
        approximates this circular aperture.

        Parameters
        ----------
        n_vertices : int, optional
            The number of polygon vertices used to approximate the
            circle. Must be at least 3. Default is 100.

        Returns
        -------
        aperture : `~photutils.aperture.SkyPolygonAperture`
            A sky polygon aperture that approximates the circle.
        """
        offsets = _circular_polygon_offsets(self.r, n_vertices)
        return SkyPolygonAperture._from_convex_offsets(self.positions,
                                                       offsets)


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

        r_in = self.r_in.to_value(u.arcsec) * mean_scale
        r_out = self.r_out.to_value(u.arcsec) * mean_scale
        return CircularAnnulus(positions=positions, r_in=r_in, r_out=r_out)
