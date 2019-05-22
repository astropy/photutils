# Licensed under a 3-clause BSD style license - see LICENSE.rst

import math

import numpy as np

from .attributes import (PixelPositions, SkyCoordPositions, PositiveScalar,
                         AngleOrPixelScalarQuantity)
from .core import PixelAperture, SkyAperture
from .bounding_box import BoundingBox
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

        if hasattr(self, 'r'):
            radius = self.r
        elif hasattr(self, 'r_out'):    # annulus
            radius = self.r_out
        else:
            raise ValueError('Cannot determine the aperture radius.')

        masks = []
        for bbox, edges in zip(self.bounding_boxes, self._centered_edges):
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
    >>> from photutils import CircularAperture
    >>> aper = CircularAperture([10., 20.], 3.)
    >>> aper = CircularAperture((10., 20.), 3.)

    >>> pos1 = (10., 20.)  # (x, y)
    >>> pos2 = (30., 40.)
    >>> pos3 = (50., 60.)
    >>> aper = CircularAperture([pos1, pos2, pos3], 3.)
    >>> aper = CircularAperture((pos1, pos2, pos3), 3.)
    """

    positions = PixelPositions('positions')
    r = PositiveScalar('r')

    def __init__(self, positions, r):
        self.positions = positions
        self.r = r
        self._params = ['r']

    @property
    def bounding_boxes(self):
        positions = np.atleast_2d(self.positions)
        xmin = positions[:, 0] - self.r
        xmax = positions[:, 0] + self.r
        ymin = positions[:, 1] - self.r
        ymax = positions[:, 1] + self.r

        return [BoundingBox.from_float(x0, x1, y0, y1)
                for x0, x1, y0, y1 in zip(xmin, xmax, ymin, ymax)]

    def area(self):
        return math.pi * self.r ** 2

    def plot(self, origin=(0, 0), indices=None, ax=None, fill=False,
             **kwargs):
        import matplotlib.patches as mpatches

        plot_positions, ax, kwargs = self._prepare_plot(
            origin, indices, ax, fill, **kwargs)

        for position in plot_positions:
            patch = mpatches.Circle(position, self.r, **kwargs)
            ax.add_patch(patch)

    def to_sky(self, wcs, mode='all'):
        """
        Convert the aperture to a `SkyCircularAperture` object defined
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
        aperture : `SkyCircularAperture` object
            A `SkyCircularAperture` object.
        """

        sky_params = self._to_sky_params(wcs, mode=mode)
        return SkyCircularAperture(**sky_params)


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
    >>> from photutils import CircularAnnulus
    >>> aper = CircularAnnulus([10., 20.], 3., 5.)
    >>> aper = CircularAnnulus((10., 20.), 3., 5.)

    >>> pos1 = (10., 20.)  # (x, y)
    >>> pos2 = (30., 40.)
    >>> pos3 = (50., 60.)
    >>> aper = CircularAnnulus([pos1, pos2, pos3], 3., 5.)
    >>> aper = CircularAnnulus((pos1, pos2, pos3), 3., 5.)
    """

    positions = PixelPositions('positions')
    r_in = PositiveScalar('r_in')
    r_out = PositiveScalar('r_out')

    def __init__(self, positions, r_in, r_out):
        if not r_out > r_in:
            raise ValueError('r_out must be greater than r_in')

        self.positions = positions
        self.r_in = r_in
        self.r_out = r_out
        self._params = ['r_in', 'r_out']

    @property
    def bounding_boxes(self):
        positions = np.atleast_2d(self.positions)
        xmin = positions[:, 0] - self.r_out
        xmax = positions[:, 0] + self.r_out
        ymin = positions[:, 1] - self.r_out
        ymax = positions[:, 1] + self.r_out

        return [BoundingBox.from_float(x0, x1, y0, y1)
                for x0, x1, y0, y1 in zip(xmin, xmax, ymin, ymax)]

    def area(self):
        return math.pi * (self.r_out ** 2 - self.r_in ** 2)

    def plot(self, origin=(0, 0), indices=None, ax=None, fill=False,
             **kwargs):
        import matplotlib.patches as mpatches

        plot_positions, ax, kwargs = self._prepare_plot(
            origin, indices, ax, fill, **kwargs)

        for position in plot_positions:
            patch_inner = mpatches.Circle(position, self.r_in)
            patch_outer = mpatches.Circle(position, self.r_out)
            path = self._make_annulus_path(patch_inner, patch_outer)
            patch = mpatches.PathPatch(path, **kwargs)
            ax.add_patch(patch)

    def to_sky(self, wcs, mode='all'):
        """
        Convert the aperture to a `SkyCircularAnnulus` object defined
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
        aperture : `SkyCircularAnnulus` object
            A `SkyCircularAnnulus` object.
        """

        sky_params = self._to_sky_params(wcs, mode=mode)
        return SkyCircularAnnulus(**sky_params)


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
    >>> from photutils import SkyCircularAperture
    >>> positions = SkyCoord(ra=[10., 20.], dec=[30., 40.], unit='deg')
    >>> aper = SkyCircularAperture(positions, 0.5*u.arcsec)
    """

    positions = SkyCoordPositions('positions')
    r = AngleOrPixelScalarQuantity('r')

    def __init__(self, positions, r):
        self.positions = positions
        self.r = r
        self._params = ['r']

    def to_pixel(self, wcs, mode='all'):
        """
        Convert the aperture to a `CircularAperture` object defined in
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
        aperture : `CircularAperture` object
            A `CircularAperture` object.
        """

        pixel_params = self._to_pixel_params(wcs, mode=mode)
        return CircularAperture(**pixel_params)


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
    >>> from photutils import SkyCircularAnnulus
    >>> positions = SkyCoord(ra=[10., 20.], dec=[30., 40.], unit='deg')
    >>> aper = SkyCircularAnnulus(positions, 0.5*u.arcsec, 1.0*u.arcsec)
    """

    positions = SkyCoordPositions('positions')
    r_in = AngleOrPixelScalarQuantity('r_in')
    r_out = AngleOrPixelScalarQuantity('r_out')

    def __init__(self, positions, r_in, r_out):
        if r_in.unit.physical_type != r_out.unit.physical_type:
            raise ValueError("r_in and r_out should either both be angles "
                             "or in pixels.")

        self.positions = positions
        self.r_in = r_in
        self.r_out = r_out
        self._params = ['r_in', 'r_out']

    def to_pixel(self, wcs, mode='all'):
        """
        Convert the aperture to a `CircularAnnulus` object defined in
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
        aperture : `CircularAnnulus` object
            A `CircularAnnulus` object.
        """

        pixel_params = self._to_pixel_params(wcs, mode=mode)
        return CircularAnnulus(**pixel_params)
