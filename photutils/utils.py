import numpy as np

from astropy import units as u
from astropy.wcs import WCSSUB_CELESTIAL


def skycoord_to_pixel(coords, wcs):
    """
    Convert a set of SkyCoord coordinates into pixel coordinates.

    This function assumes that the coordinates are celestial.

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
    """

    # TODO this should be simplified once wcs_world2pix() supports
    # SkyCoord objects as input

    # TODO: remove local wcs_to_celestial_frame once Astropy 1.0 is out
    try:
        from astropy.wcs.utils import wcs_to_celestial_frame
    except ImportError:  # Astropy < 1.0
        from .extern.wcs_utils import wcs_to_celestial_frame

    # Keep only the celestial part of the axes, also re-orders lon/lat
    wcs = wcs.sub([WCSSUB_CELESTIAL])

    if wcs.naxis != 2:
        raise ValueError("WCS should contain celestial component")

    # Check which frame the WCS uses
    frame = wcs_to_celestial_frame(wcs)

    # Check what unit the WCS needs
    xw_unit = u.Unit(wcs.wcs.cunit[0])
    yw_unit = u.Unit(wcs.wcs.cunit[1])

    # Convert positions to frame
    coords = coords.transform_to(frame)

    # Extract longitude and latitude
    lon = coords.spherical.lon.to(xw_unit)
    lat = coords.spherical.lat.to(yw_unit)

    # Convert to pixel coordinates
    xp, yp = wcs.wcs_world2pix(lon, lat, 0)

    # Make quantity with explicit pixel units
    pixel = u.Quantity(np.array([xp, yp]).transpose(),
                       unit=u.pixel, copy=False)

    return pixel
