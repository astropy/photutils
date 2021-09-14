# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
This module defines circular and circular-annulus apertures in both
pixel and sky coordinates.
"""

import math

import numpy as np

from .attributes import (AngleOrPixelScalarQuantity, PixelPositions,
                         PositiveScalar, SkyCoordPositions)
from .core import PixelAperture, SkyAperture
from .mask import ApertureMask
from ..geometry import circular_overlap_grid

__all__ = ['CircularMaskMixin', 'CircularAperture', 'CircularAnnulus',
           'SkyCircularAperture', 'SkyCircularAnnulus']


class CircularMaskMixin:
    """
    Mixin class to create masks for circular and circular-annulus
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
        mask : `~photutils.aperture.ApertureMask` or list of `~photutils.aperture.ApertureMask`
            A mask for the aperture.  If the aperture is scalar then a
            single `~photutils.aperture.ApertureMask` is returned,
            otherwise a list of `~photutils.aperture.ApertureMask` is
            returned.
        """
        use_exact, subpixels = self._translate_mask_mode(method, subpixels)

        if hasattr(self, 'r'):
            radius = self.r
        elif hasattr(self, 'r_out'):  # annulus
            radius = self.r_out
        else:
            raise ValueError('Cannot determine the aperture radius.')

        masks = []
        for bbox, edges in zip(np.atleast_1d(self.bbox),
                               self._centered_edges):
            ny, nx = bbox.shape
            mask = circular_overlap_grid(edges[0], edges[1], edges[2],
                                         edges[3], nx, ny, radius, use_exact,
                                         subpixels)

            # subtract the inner circle for an annulus
            if hasattr(self, 'r_in'):
                mask -= circular_overlap_grid(edges[0], edges[1], edges[2],
                                              edges[3], nx, ny, self.r_in,
                                              use_exact, subpixels)

            masks.append(ApertureMask(mask, bbox))

        if self.isscalar:
            return masks[0]
        else:
            return masks


class CircularAperture(CircularMaskMixin, PixelAperture):
    """
    A circular aperture defined in pixel coordinates.

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

    r : float
        The radius of the circle in pixels.

    Raises
    ------
    ValueError : `ValueError`
        If the input radius, ``r``, is negative.

    Examples
    --------
    >>> from photutils.aperture import CircularAperture
    >>> aper = CircularAperture([10., 20.], 3.)
    >>> aper = CircularAperture((10., 20.), 3.)

    >>> pos1 = (10., 20.)  # (x, y)
    >>> pos2 = (30., 40.)
    >>> pos3 = (50., 60.)
    >>> aper = CircularAperture([pos1, pos2, pos3], 3.)
    >>> aper = CircularAperture((pos1, pos2, pos3), 3.)
    """

    _shape_params = ('r',)
    positions = PixelPositions('positions',
                               description='The center pixel position(s).')
    r = PositiveScalar('r', description='The radius in pixels.')

    def __init__(self, positions, r):
        self.positions = positions
        self.r = r

    @property
    def _xy_extents(self):
        return self.r, self.r

    @property
    def area(self):
        return math.pi * self.r ** 2

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
        for xy_position in xy_positions:
            patches.append(mpatches.Circle(xy_position, self.r,
                                           **patch_kwargs))

        if self.isscalar:
            return patches[0]
        else:
            return patches

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
        """
        return SkyCircularAperture(**self._to_sky_params(wcs))


class CircularAnnulus(CircularMaskMixin, PixelAperture):
    """
    A circular annulus aperture defined in pixel coordinates.

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

    r_in : float
        The inner radius of the circular annulus in pixels.

    r_out : float
        The outer radius of the circular annulus in pixels.

    Raises
    ------
    ValueError : `ValueError`
        If inner radius (``r_in``) is greater than outer radius (``r_out``).

    ValueError : `ValueError`
        If inner radius (``r_in``) is negative.

    Examples
    --------
    >>> from photutils.aperture import CircularAnnulus
    >>> aper = CircularAnnulus([10., 20.], 3., 5.)
    >>> aper = CircularAnnulus((10., 20.), 3., 5.)

    >>> pos1 = (10., 20.)  # (x, y)
    >>> pos2 = (30., 40.)
    >>> pos3 = (50., 60.)
    >>> aper = CircularAnnulus([pos1, pos2, pos3], 3., 5.)
    >>> aper = CircularAnnulus((pos1, pos2, pos3), 3., 5.)
    """

    _shape_params = ('r_in', 'r_out')
    positions = PixelPositions('positions',
                               description='The center pixel position(s).')
    r_in = PositiveScalar('r_in',
                          description='The inner radius in pixels.')
    r_out = PositiveScalar('r_out',
                           description='The outer radius in pixels.')

    def __init__(self, positions, r_in, r_out):
        if not r_out > r_in:
            raise ValueError('r_out must be greater than r_in')

        self.positions = positions
        self.r_in = r_in
        self.r_out = r_out

    @property
    def _xy_extents(self):
        return self.r_out, self.r_out

    @property
    def area(self):
        return math.pi * (self.r_out ** 2 - self.r_in ** 2)

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
        for xy_position in xy_positions:
            patch_inner = mpatches.Circle(xy_position, self.r_in)
            patch_outer = mpatches.Circle(xy_position, self.r_out)
            path = self._make_annulus_path(patch_inner, patch_outer)
            patches.append(mpatches.PathPatch(path, **patch_kwargs))

        if self.isscalar:
            return patches[0]
        else:
            return patches

    def to_sky(self, wcs):
        """
        Convert the aperture to a `SkyCircularAnnulus` object defined
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
        aperture : `SkyCircularAnnulus` object
            A `SkyCircularAnnulus` object.
        """
        return SkyCircularAnnulus(**self._to_sky_params(wcs))


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
        The radius of the circle, either in angular or pixel units.

    Examples
    --------
    >>> from astropy.coordinates import SkyCoord
    >>> import astropy.units as u
    >>> from photutils.aperture import SkyCircularAperture
    >>> positions = SkyCoord(ra=[10., 20.], dec=[30., 40.], unit='deg')
    >>> aper = SkyCircularAperture(positions, 0.5*u.arcsec)
    """

    _shape_params = ('r',)
    positions = SkyCoordPositions(
        'positions',
        description='The center position(s) in sky coordinates.')
    r = AngleOrPixelScalarQuantity(
        'r',
        description='The radius, in angular or pixel units.')

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
        """
        return CircularAperture(**self._to_pixel_params(wcs))


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
        The inner radius of the circular annulus, either in angular or
        pixel units.

    r_out : scalar `~astropy.units.Quantity`
        The outer radius of the circular annulus, either in angular or
        pixel units.

    Examples
    --------
    >>> from astropy.coordinates import SkyCoord
    >>> import astropy.units as u
    >>> from photutils.aperture import SkyCircularAnnulus
    >>> positions = SkyCoord(ra=[10., 20.], dec=[30., 40.], unit='deg')
    >>> aper = SkyCircularAnnulus(positions, 0.5*u.arcsec, 1.0*u.arcsec)
    """

    _shape_params = ('r_in', 'r_out')
    positions = SkyCoordPositions(
        'positions',
        description='The center position(s) in sky coordinates.')
    r_in = AngleOrPixelScalarQuantity(
        'r_in',
        description='The inner radius, in angular or pixel units.')
    r_out = AngleOrPixelScalarQuantity(
        'r_out',
        description='The outer radius, in angular or pixel units.')

    def __init__(self, positions, r_in, r_out):
        if r_in.unit.physical_type != r_out.unit.physical_type:
            raise ValueError("r_in and r_out should either both be angles "
                             "or in pixels.")

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
        """
        return CircularAnnulus(**self._to_pixel_params(wcs))
