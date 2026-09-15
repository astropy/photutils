# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Pytest configuration and WCS test fixtures for photutils.utils tests.
"""

import astropy.units as u
import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from astropy.io.fits import Header
from astropy.wcs import WCS

# WCS test constants
WCS_CENTER = SkyCoord(100 * u.deg, 30 * u.deg)
WCS_CDELT_ARCSEC = 0.1


def _make_simple_wcs(skycoord, resolution, size, *, rotation_deg=0.0):
    """
    Create a simple TAN WCS with optional rotation.

    Parameters
    ----------
    skycoord : `~astropy.coordinates.SkyCoord`
        The center sky coordinate (CRVAL).

    resolution : `~astropy.units.Quantity`
        The pixel scale (CDELT) as an angular quantity.

    size : int
        Number of pixels along each axis.

    rotation_deg : float, optional
        Rotation angle in degrees (default: 0).

    Returns
    -------
    wcs : `~astropy.wcs.WCS`
        The WCS object.
    """
    cdelt_deg = resolution.to_value(u.deg)
    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [size / 2 + 0.5, size / 2 + 0.5]
    wcs.wcs.crval = [skycoord.ra.deg, skycoord.dec.deg]
    wcs.wcs.ctype = ['RA---TAN', 'DEC--TAN']

    if rotation_deg == 0.0:
        wcs.wcs.cdelt = [-cdelt_deg, cdelt_deg]
    else:
        angle_rad = np.radians(rotation_deg)
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        wcs.wcs.cd = [[-cdelt_deg * cos_a, cdelt_deg * sin_a],
                      [cdelt_deg * sin_a, cdelt_deg * cos_a]]

    return wcs


def make_sip_wcs(shape=(20, 20), *, center=WCS_CENTER, coeffs=None):
    """
    Build a TAN-SIP WCS with the standard test pixel scale.

    Parameters
    ----------
    shape : tuple of int, optional
        The ``(ny, nx)`` image shape. CRPIX is at the image center.

    center : `~astropy.coordinates.SkyCoord`, optional
        The sky position of CRPIX.

    coeffs : dict, optional
        The SIP coefficients, e.g. ``{'A_2_0': 1e-6, 'B_0_2': 1e-6}``.
        The polynomial orders are set from the highest given indices.

    Returns
    -------
    wcs : `~astropy.wcs.WCS`
        The WCS object with SIP distortion.
    """
    header = Header()
    header['NAXIS'] = 2
    header['NAXIS1'] = shape[1]
    header['NAXIS2'] = shape[0]
    header['CRPIX1'] = shape[1] / 2 + 0.5
    header['CRPIX2'] = shape[0] / 2 + 0.5
    header['CRVAL1'] = center.ra.deg
    header['CRVAL2'] = center.dec.deg
    header['CTYPE1'] = 'RA---TAN-SIP'
    header['CTYPE2'] = 'DEC--TAN-SIP'
    cdelt = WCS_CDELT_ARCSEC / 3600.0
    header['CD1_1'] = -cdelt
    header['CD1_2'] = 0.0
    header['CD2_1'] = 0.0
    header['CD2_2'] = cdelt

    coeffs = coeffs or {}
    for prefix in ('A', 'B'):
        orders = [int(key[2]) + int(key[4]) for key in coeffs
                  if key.startswith(prefix)]
        header[f'{prefix}_ORDER'] = max(orders, default=2)
    for key, value in coeffs.items():
        header[key] = value

    return WCS(header)


class CountingWCS:
    """
    Wrapper that counts the calls to the WCS transform methods.
    """

    def __init__(self, real_wcs):
        self._wcs = real_wcs
        self.n_pixel_to_world = 0
        self.n_pixel_to_world_values = 0
        self.n_world_to_pixel = 0

    @property
    def world_axis_units(self):
        return self._wcs.world_axis_units

    @property
    def world_axis_object_components(self):
        return self._wcs.world_axis_object_components

    def pixel_to_world(self, *args, **kwargs):
        self.n_pixel_to_world += 1
        return self._wcs.pixel_to_world(*args, **kwargs)

    def pixel_to_world_values(self, *args, **kwargs):
        self.n_pixel_to_world_values += 1
        return self._wcs.pixel_to_world_values(*args, **kwargs)

    def world_to_pixel(self, *args, **kwargs):
        self.n_world_to_pixel += 1
        return self._wcs.world_to_pixel(*args, **kwargs)


@pytest.fixture
def simple_wcs():
    """
    Non-distorted TAN WCS aligned with the celestial axes.
    """
    return _make_simple_wcs(WCS_CENTER, WCS_CDELT_ARCSEC * u.arcsec, 20)


@pytest.fixture
def rotated_wcs():
    """
    Non-distorted TAN WCS with a 25-degree rotation (CD matrix).
    """
    return _make_simple_wcs(WCS_CENTER, WCS_CDELT_ARCSEC * u.arcsec, 20,
                            rotation_deg=25.0)


@pytest.fixture
def sip_wcs():
    """
    TAN WCS with small SIP distortion terms.
    """
    coeffs = {'A_2_0': 1e-7, 'A_0_2': 1e-7, 'B_2_0': 1e-7, 'B_0_2': 1e-7}
    return make_sip_wcs(coeffs=coeffs)


@pytest.fixture
def nonsquare_wcs():
    """
    Non-distorted TAN WCS with non-square pixels (0.03 x 0.05 deg).
    """
    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [10.5, 10.5]
    wcs.wcs.crval = [WCS_CENTER.ra.deg, WCS_CENTER.dec.deg]
    wcs.wcs.cdelt = [-0.03, 0.05]
    wcs.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    return wcs


@pytest.fixture
def swapped_wcs():
    """
    Non-distorted TAN WCS with the latitude axis first.

    The CD matrix maps the same sky footprint as ``simple_wcs`` (RA
    increasing to the left along x, Dec increasing along y), but the
    world axes are ordered (Dec, RA). The Jacobians and conversions must
    not depend on the world axis order.
    """
    cdelt = WCS_CDELT_ARCSEC / 3600
    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [10.5, 10.5]
    wcs.wcs.crval = [WCS_CENTER.dec.deg, WCS_CENTER.ra.deg]
    wcs.wcs.cd = [[0.0, cdelt], [-cdelt, 0.0]]
    wcs.wcs.ctype = ['DEC--TAN', 'RA---TAN']
    return wcs


@pytest.fixture
def flipped_wcs():
    """
    Non-distorted TAN WCS with a flipped parity (North down, East left).

    Both CDELT values are negative, so the determinant of the pixel
    scale matrix is positive (parity = +1). This is the opposite parity
    from the standard astronomical convention (North up, East left) and
    reproduces the mirrored-aperture bug reported for such WCS.
    """
    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [10.5, 10.5]
    wcs.wcs.crval = [WCS_CENTER.ra.deg, WCS_CENTER.dec.deg]
    wcs.wcs.cdelt = [-WCS_CDELT_ARCSEC / 3600, -WCS_CDELT_ARCSEC / 3600]
    wcs.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    return wcs


@pytest.fixture
def center_xy_coord(simple_wcs):
    """
    Return the center (x, y) tuple at CRPIX of the simple WCS.
    """
    x, y = simple_wcs.world_to_pixel(WCS_CENTER)
    return (x, y)
