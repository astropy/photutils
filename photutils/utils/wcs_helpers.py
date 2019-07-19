# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
This module provides WCS helper tools.
"""

from astropy import units as u
from astropy.utils import deprecated

from ._wcs_helpers import _pixel_to_world, _pixel_scale_angle_at_skycoord


@deprecated('0.7', 'The pixel_scale_angle_at_skycoord function is deprecated '
            'and will be removed in v0.8')
def pixel_scale_angle_at_skycoord(skycoord, wcs, offset=1. * u.arcsec):
    """
    Calculate the pixel scale and WCS rotation angle at the position of
    a SkyCoord coordinate.

    Parameters
    ----------
    skycoord : `~astropy.coordinates.SkyCoord`
        The SkyCoord coordinate.

    wcs : WCS object
        A world coordinate system (WCS) transformation that supports the
        `astropy shared interface for WCS
        <https://docs.astropy.org/en/stable/wcs/wcsapi.html>`_ (e.g.
        `astropy.wcs.WCS`, `gwcs.wcs.WCS`).

    offset : `~astropy.units.Quantity`
        A small angular offset to use to compute the pixel scale and
        position angle.

    Returns
    -------
    scale : `~astropy.units.Quantity`
        The pixel scale in arcsec/pixel.

    angle : `~astropy.units.Quantity`
        The angle (in degrees) measured counterclockwise from the
        positive x axis to the "North" axis of the celestial coordinate
        system.

    Notes
    -----
    If distortions are present in the image, the x and y pixel scales
    likely differ.  This function computes a single pixel scale along
    the North/South axis.
    """

    return _pixel_scale_angle_at_skycoord(skycoord, wcs, offset=offset)


@deprecated('0.7', 'The assert_angle_or_pixel function is deprecated and '
            'will be removed in v0.8')
def assert_angle_or_pixel(name, q):
    """
    Check that ``q`` is either an angular or a pixel
    :class:`~astropy.units.Quantity`.
    """

    if isinstance(q, u.Quantity):
        if q.unit.physical_type == 'angle' or q.unit == u.pixel:
            pass
        else:
            raise ValueError("{0} should have angular or pixel "
                             "units".format(name))
    else:
        raise TypeError("{0} should be a Quantity instance".format(name))


@deprecated('0.7', 'The assert_angle function is deprecated and will be '
            'removed in v0.8')
def assert_angle(name, q):
    """
    Check that ``q`` is an angular :class:`~astropy.units.Quantity`.
    """

    if isinstance(q, u.Quantity):
        if q.unit.physical_type == 'angle':
            pass
        else:
            raise ValueError("{0} should have angular units".format(name))
    else:
        raise TypeError("{0} should be a Quantity instance".format(name))


@deprecated('0.7', 'The pixel_to_icrs_coords function is deprecated and will '
            'be removed in v0.8')
def pixel_to_icrs_coords(x, y, wcs):
    """
    Convert pixel coordinates to ICRS Right Ascension and Declination.

    This is merely a convenience function to extract RA and Dec. from a
    `~astropy.coordinates.SkyCoord` instance so they can be put in
    separate columns in a `~astropy.table.Table`.

    Parameters
    ----------
    x : float or array-like
        The x pixel coordinate.

    y : float or array-like
        The y pixel coordinate.

    wcs : WCS object
        A world coordinate system (WCS) transformation that supports the
        `astropy shared interface for WCS
        <https://docs.astropy.org/en/stable/wcs/wcsapi.html>`_ (e.g.
        `astropy.wcs.WCS`, `gwcs.wcs.WCS`).

    Returns
    -------
    ra : `~astropy.units.Quantity`
        The ICRS Right Ascension in degrees.

    dec : `~astropy.units.Quantity`
        The ICRS Declination in degrees.
    """

    icrs_coords = (_pixel_to_world(x, y, wcs)).icrs
    icrs_ra = icrs_coords.ra.degree * u.deg
    icrs_dec = icrs_coords.dec.degree * u.deg

    return icrs_ra, icrs_dec
