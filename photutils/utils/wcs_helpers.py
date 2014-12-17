# Licensed under a 3-clause BSD style license - see LICENSE.rst

# These are functions specific to photutils that do not already exist in
# astropy.wcs or photutils.extern.wcs_utils

import numpy as np
from astropy import units as u
from astropy.coordinates import UnitSphericalRepresentation

from ..extern.wcs_utils import skycoord_to_pixel


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
    x, y = skycoord_to_pixel(coords, wcs)

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
    x_offset, y_offset = skycoord_to_pixel(coords_offset, wcs)

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
