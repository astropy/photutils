# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
This module provides WCS helper tools.
"""

from astropy import units as u
from astropy.coordinates import UnitSphericalRepresentation
from astropy.wcs import WCS
from astropy.wcs.utils import pixel_to_skycoord, skycoord_to_pixel
import numpy as np


def _pixel_to_world(xpos, ypos, wcs):
    """
    Calculate the sky coordinates at the input pixel positions.

    Parameters
    ----------
    xpos, ypos : float or array-like
        The x and y pixel position(s).

    wcs : WCS object or `None`
        A world coordinate system (WCS) transformation that
        supports the `astropy shared interface for WCS
        <https://docs.astropy.org/en/stable/wcs/wcsapi.html>`_ (e.g.,
        `astropy.wcs.WCS`, `gwcs.wcs.WCS`).

    Returns
    -------
    skycoord : `~astropy.coordinates.SkyCoord`
        The sky coordinate(s) at the input position(s).
    """

    if wcs is None:
        return None

    # NOTE: unitless gwcs objects fail with Quantity inputs
    if isinstance(xpos, u.Quantity) or isinstance(ypos, u.Quantity):
        raise TypeError('xpos and ypos must not be Quantity objects')

    try:
        return wcs.pixel_to_world(xpos, ypos)
    except AttributeError:
        if isinstance(wcs, WCS):
            # for Astropy < 3.1 WCS support
            return pixel_to_skycoord(xpos, ypos, wcs, origin=0)
        else:
            raise ValueError('Input wcs does not support the shared WCS '
                             'interface.')


def _world_to_pixel(skycoord, wcs):
    """
    Calculate the sky coordinates at the input pixel positions.

    Parameters
    ----------
    skycoord : `~astropy.coordinates.SkyCoord`
        The sky coordinate(s).

    wcs : WCS object or `None`
        A world coordinate system (WCS) transformation that
        supports the `astropy shared interface for WCS
        <https://docs.astropy.org/en/stable/wcs/wcsapi.html>`_ (e.g.,
        `astropy.wcs.WCS`, `gwcs.wcs.WCS`).

    Returns
    -------
    xpos, ypos : float or array-like
        The x and y pixel position(s) at the input sky coordinate(s).
    """

    if wcs is None:
        return None

    try:
        return wcs.world_to_pixel(skycoord)
    except AttributeError:
        if isinstance(wcs, WCS):
            # for Astropy < 3.1 WCS support
            return skycoord_to_pixel(skycoord, wcs, origin=0)
        else:
            raise ValueError('Input wcs does not support the shared WCS '
                             'interface.')


def _pixel_scale_angle_at_skycoord(skycoord, wcs, offset=1.0*u.arcsec):
    """
    Calculate the pixel scale and WCS rotation angle at the position of
    a SkyCoord coordinate.

    Parameters
    ----------
    skycoord : `~astropy.coordinates.SkyCoord`
        The SkyCoord coordinate.

    wcs : WCS object
        A world coordinate system (WCS) transformation that
        supports the `astropy shared interface for WCS
        <https://docs.astropy.org/en/stable/wcs/wcsapi.html>`_ (e.g.,
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

    try:
        x, y = wcs.world_to_pixel(skycoord)

        # We take a point directly North (i.e., latitude offset) the
        # input sky coordinate and convert it to pixel coordinates,
        # then we use the pixel deltas between the input and offset sky
        # coordinate to calculate the pixel scale and angle.
        skycoord_offset = skycoord.directional_offset_by(0.0, offset)
        x_offset, y_offset = wcs.world_to_pixel(skycoord_offset)
    except AttributeError:
        # for Astropy < 3.1 WCS support
        x, y = skycoord_to_pixel(skycoord, wcs)
        coord = skycoord.represent_as('unitspherical')
        coord_new = UnitSphericalRepresentation(coord.lon, coord.lat + offset)
        coord_offset = skycoord.realize_frame(coord_new)
        x_offset, y_offset = skycoord_to_pixel(coord_offset, wcs)

    dx = x_offset - x
    dy = y_offset - y
    scale = offset.to(u.arcsec) / (np.hypot(dx, dy) * u.pixel)
    angle = (np.arctan2(dy, dx) * u.radian).to(u.deg)

    return scale, angle
