# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
from astropy import units as u
from astropy.coordinates import UnitSphericalRepresentation
from astropy.wcs.utils import skycoord_to_pixel, pixel_to_skycoord


def pixel_scale_angle_at_skycoord(skycoord, wcs, offset=1. * u.arcsec):
    """
    Calculate the pixel scale and WCS rotation angle at the position of
    a SkyCoord coordinate.

    Parameters
    ----------
    skycoord : `~astropy.coordinates.SkyCoord`
        The SkyCoord coordinate.
    wcs : `~astropy.wcs.WCS`
        The world coordinate system (WCS) transformation to use.
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

    # We take a point directly "above" (in latitude) the input position
    # and convert it to pixel coordinates, then we use the pixel deltas
    # between the input and offset point to calculate the pixel scale and
    # angle.

    # Find the coordinates as a representation object
    coord = skycoord.represent_as('unitspherical')

    # Add a a small perturbation in the latitude direction (since longitude
    # is more difficult because it is not directly an angle)
    coord_new = UnitSphericalRepresentation(coord.lon, coord.lat + offset)
    coord_offset = skycoord.realize_frame(coord_new)

    # Find pixel coordinates of offset coordinates and pixel deltas
    x_offset, y_offset = skycoord_to_pixel(coord_offset, wcs, mode='all')
    x, y = skycoord_to_pixel(skycoord, wcs, mode='all')
    dx = x_offset - x
    dy = y_offset - y

    scale = offset.to(u.arcsec) / (np.hypot(dx, dy) * u.pixel)
    angle = (np.arctan2(dy, dx) * u.radian).to(u.deg)

    return scale, angle


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

    wcs : `~astropy.wcs.WCS`
        The WCS transformation to use to convert from pixel coordinates
        to ICRS world coordinates.
        `~astropy.table.Table`.

    Returns
    -------
    ra : `~astropy.units.Quantity`
        The ICRS Right Ascension in degrees.

    dec : `~astropy.units.Quantity`
        The ICRS Declination in degrees.
    """

    icrs_coords = pixel_to_skycoord(x, y, wcs).icrs
    icrs_ra = icrs_coords.ra.degree * u.deg
    icrs_dec = icrs_coords.dec.degree * u.deg

    return icrs_ra, icrs_dec
