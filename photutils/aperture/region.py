# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
This module defines arbitrary region apertures in both
pixel and sky coordinates.
"""

import numpy as np
from astropy.utils import lazyproperty

from photutils.aperture.attributes import (PixelPositions, SkyCoordPositions,
                                           WrappedRegion, WrappedSkyRegion)
from photutils.aperture.bounding_box import BoundingBox
from photutils.aperture.core import PixelAperture, SkyAperture
from photutils.aperture.mask import ApertureMask

__all__ = ['RegionalMaskMixin', 'RegionalAperture', 'SkyRegionalAperture']


class RegionalMaskMixin:
    """
    Mixin class to create masks for regional apertures.
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

        subpixels: int, optional
            For the ``'subpixel'`` method, resample pixels by this
            factor in each dimension. That is, each pixel is divided
            into ``subpixels**2`` subpixels. This keyword is ignored
            unless ``method='subpixel'``.

        Returns
        -------
        mask: `~photutils.aperture.ApertureMask`
            A mask for the aperture.
        """
        region_mask = self.region.to_mask(mode=method, subpixels=subpixels)
        return ApertureMask(region_mask.data, region_mask.bbox)


class RegionalAperture(RegionalMaskMixin, PixelAperture):
    """
    A region aperture defined in pixel coordinates.

    The aperture has a single fixed size/shape and a single position.

    Parameters
    ----------
    positions : array_like
        The pixel coordinates of the aperture center in one of the
        following formats:

            * single ``(x, y)`` pair as a tuple, list, or `~numpy.ndarray`
            * tuple, list, or `~numpy.ndarray` containing a single
              ``(x, y)`` pair

    region : `~regions.Region`
        The region representing the area to be used as an aperture.

    Raises
    ------
    ValueError : `ValueError`
        If ``positions`` contains more than one ``(x, y)`` pair.

    Examples
    --------
    >>> from regions import CirclePixelRegion, PixCoord
    >>> from photutils.aperture import RegionalAperture

    >>> x, y = 10.0, 20.0
    >>> region = CirclePixelRegion(PixCoord(x, y), 3.0)
    >>> aper = RegionalAperture((x, y), region)
    """

    _params = ['region', 'positions']
    positions = PixelPositions('The center pixel position.')
    region = WrappedRegion('The region of the aperture.')

    def __init__(self, positions, region):
        # TODO: use ApertureAttribute to do the validation here
        if np.array(positions).shape not in ((2,), (1, 2)):
            raise ValueError(
                'A region aperture can only be associated with one position.'
            )
        self.positions = positions
        self.region = region

    @lazyproperty
    def _bbox(self):
        region_bbox = self.region.bounding_box

        return [
            BoundingBox.from_float(
                region_bbox.ixmin,
                region_bbox.ixmax,
                region_bbox.iymin,
                region_bbox.iymax,
            )
        ]

    @lazyproperty
    def bbox(self):
        return self._bbox[0]

    def _xy_extents(self):
        x_extents = (
            self.positions[0, 0] - self.bbox.ixmin,
            self.bbox.ixmax - self.positions[0, 0],
        )
        y_extents = (
            self.positions[0, 1] - self.bbox.iymin,
            self.bbox.iymax - self.positions[0, 1],
        )
        return max(x_extents), max(y_extents)

    def area(self):
        return self.region.area

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
            A patch for the aperture. If the aperture is scalar then a
            single `~matplotlib.patches.patch` is returned, otherwise a
            list of `~matplotlib.patches.patch` is returned.
        """
        return self.region.as_artist(origin=origin, **kwargs)

    def to_mask(self, method='exact', subpixels=5):
        return RegionalMaskMixin.to_mask(self, method=method, subpixels=subpixels)

    def to_sky(self, wcs):
        """
        Convert the aperture to a `SkyRegionalAperture` object defined
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
        aperture : `SkyRegionalAperture` object
            A `SkyRegionalAperture` object.
        """
        return SkyRegionalAperture(**self._to_sky_params(wcs))


class SkyRegionalAperture(SkyAperture):
    """
    An aperture defined in sky coordinates from a `~regions.SkyRegion`.

    The aperture has a single fixed size/shape and a single position.

    Parameters
    ----------
    positions : `~astropy.coordinates.SkyCoord`
        The celestial coordinates of the aperture center. This can be
        either scalar coordinates or an array of coordinates.

    region : `~regions.SkyRegion`
        The region representing the area to be used as an aperture.

    Examples
    --------
    >>> from astropy.coordinates import SkyCoord
    >>> import astropy.units as u
    >>> from regions import CircleSkyRegion
    >>> from photutils.aperture import SkyRegionalAperture

    >>> pos = SkyCoord(ra=10.0, dec=20.0, unit='deg')
    >>> region = CircleSkyRegion(pos, 0.5*u.arcsec)
    >>> aper = SkyRegionalAperture(pos, region)
    """

    _params = ('positions', 'region')
    positions = SkyCoordPositions('The center position in sky coordinates.')
    region = WrappedSkyRegion('The region of the aperture.')

    def __init__(self, positions, region):
        # TODO: use ApertureAttribute to do the validation here
        if np.array(positions).shape != ():
            raise ValueError(
                'A sky region aperture can only be associated with one position.'
            )
        self.positions = positions
        self.region = region

    def to_pixel(self, wcs):
        """
        Convert the aperture to a `RegionalAperture` object defined in
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
        aperture : `RegionalAperture` object
            A `RegionalAperture` object.
        """
        return RegionalAperture(**self._to_pixel_params(wcs))
