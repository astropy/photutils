import numpy as np

from astropy import units as u
from astropy.wcs import WCSSUB_CELESTIAL
from astropy.coordinates import SkyCoord, UnitSphericalRepresentation


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

    return xp, yp


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
    if isinstance(q, u.Quantity):
        if q.unit.physical_type == 'angle' or q.unit is u.pixel:
            pass
        else:
            raise ValueError("{0} should have angular or pixel units".format(name))
    else:
        raise TypeError("{0} should be a Quantity instance".format(name))


def assert_angle(name, q):
    if isinstance(q, u.Quantity):
        if q.unit.physical_type == 'angle':
            pass
        else:
            raise ValueError("{0} should have angular units".format(name))
    else:
        raise TypeError("{0} should be a Quantity instance".format(name))


def pixel_to_skycoord(xp, yp, wcs, origin=0, mode='all'):
    """
    Convert a set of pixel coordinates into a SkyCoord coordinate.

    Parameters
    ----------
    xp, yp : float or `numpy.ndarray`
        The coordinates to convert.
    wcs : `~astropy.wcs.WCS`
        The WCS transformation to use.
    origin : int
        Whether to return 0 or 1-based pixel coordinates.
    mode : 'all' or 'wcs'
        Whether to do the transformation including distortions (``'all'``) or
        only including only the core WCS transformation (``'wcs'``).

    Returns
    -------
    coords : `~astropy.coordinates.SkyCoord`
        The celestial coordinates
    """

    # temporary workaround
    has_distortion = any(getattr(wcs, dist_attr) is not None
                         for dist_attr in ['cpdis1', 'cpdis2', 'det2im1', 'det2im2', 'sip'])

    if has_distortion and wcs.naxis != 2:
        raise ValueError("Can only handle WCS with distortions for 2-dimensional WCS")

    # Keep only the celestial part of the axes, also re-orders lon/lat
    wcs = wcs.sub([WCSSUB_CELESTIAL])

    if wcs.naxis != 2:
        raise ValueError("WCS should contain celestial component")

    # TODO: remove local wcs_to_celestial_frame once Astropy 1.0 is out
    try:
        from astropy.wcs.utils import wcs_to_celestial_frame
    except ImportError:  # Astropy < 1.0
        from .extern.wcs_utils import wcs_to_celestial_frame

    # Check which frame the WCS uses
    frame = wcs_to_celestial_frame(wcs)

    # Check what unit the WCS gives
    lon_unit = u.Unit(wcs.wcs.cunit[0])
    lat_unit = u.Unit(wcs.wcs.cunit[1])

    # Convert pixel coordinates to celestial coordinates
    if mode == 'all':
        lon, lat = wcs.all_pix2world(xp, yp, origin)
    elif mode == 'wcs':
        lon, lat = wcs.wcs_pix2world(xp, yp, origin)
    else:
        raise ValueError("mode should be either 'all' or 'wcs'")

    # Add units to longitude/latitude
    lon = lon * lon_unit
    lat = lat * lat_unit

    # Create SkyCoord object
    data = UnitSphericalRepresentation(lon=lon, lat=lat)
    coords = SkyCoord(frame.realize_frame(data))

    return coords
