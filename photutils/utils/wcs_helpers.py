# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
from astropy import units as u
from astropy.coordinates import UnitSphericalRepresentation
from astropy.wcs.utils import skycoord_to_pixel, pixel_to_skycoord

skycoord_to_pixel_mode = 'all'


def skycoord_to_pixel_scale_angle(coords, wcs):
    """
    Convert a set of SkyCoord coordinates into pixel coordinates, pixel
    scales, and position angles.

    Parameters
    ----------
    coords : `~astropy.coordinates.SkyCoord`
        The coordinates to convert
    wcs : `~astropy.wcs.WCS`
        The WCS transformation to use

    Returns
    -------
    x, y : `~numpy.ndarray`
        The x and y pixel coordinates corresponding to the input coordinates
    scale : `~astropy.units.Quantity`
        The pixel scale at each location, in degrees/pixel
    angle : `~astropy.units.Quantity`
        The position angle of the celestial coordinate system in pixel space.
    """

    # Convert to pixel coordinates
    x, y = skycoord_to_pixel(coords, wcs, mode=skycoord_to_pixel_mode)

    # We take a point directly 'above' (in latitude) the position requested
    # and convert it to pixel coordinates, then we use that to figure out the
    # scale and position angle of the coordinate system at the location of
    # the points.

    # Find the coordinates as a representation object
    r_old = coords.represent_as('unitspherical')

    # Add a a small perturbation in the latitude direction (since longitude
    # is more difficult because it is not directly an angle).
    dlat = 1 * u.arcsec
    r_new = UnitSphericalRepresentation(r_old.lon, r_old.lat + dlat)
    coords_offset = coords.realize_frame(r_new)

    # Find pixel coordinates of offset coordinates
    x_offset, y_offset = skycoord_to_pixel(coords_offset, wcs,
                                           mode=skycoord_to_pixel_mode)

    # Find vector
    dx = x_offset - x
    dy = y_offset - y

    # Find the length of the vector
    scale = np.hypot(dx, dy) * u.pixel / dlat

    # Find the position angle
    angle = np.arctan2(dy, dx) * u.radian

    return x, y, scale, angle


def assert_angle_or_pixel(name, q):
    """
    Check that ``q`` is either an angular or a pixel
    :class:`~astropy.units.Quantity`.
    """
    if isinstance(q, u.Quantity):
        if q.unit.physical_type == 'angle' or q.unit is u.pixel:
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
