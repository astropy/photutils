# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Pytest configuration and WCS test fixtures for photutils.utils tests.
"""

import astropy.units as u
import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS

# WCS test constants
WCS_CENTER = SkyCoord(100 * u.deg, 30 * u.deg)
WCS_CDELT_ARCSEC = 0.1


def _make_simple_wcs(skycoord, resolution, size, rotation_deg=0.0):
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
    cdelt_deg = resolution.to(u.deg).value
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


def _make_sip_wcs():
    """
    Create a TAN WCS with small SIP distortion terms.

    Returns
    -------
    wcs : `~astropy.wcs.WCS`
        The WCS object with SIP distortion.
    """
    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [10.5, 10.5]
    wcs.wcs.crval = [WCS_CENTER.ra.deg, WCS_CENTER.dec.deg]
    wcs.wcs.cdelt = [-WCS_CDELT_ARCSEC / 3600, WCS_CDELT_ARCSEC / 3600]
    wcs.wcs.ctype = ['RA---TAN-SIP', 'DEC--TAN-SIP']

    # Small SIP distortion coefficients
    m = 2  # A/B order
    wcs.sip = None  # will be populated from the header
    sip_header = wcs.to_header()
    sip_header['CTYPE1'] = 'RA---TAN-SIP'
    sip_header['CTYPE2'] = 'DEC--TAN-SIP'
    sip_header['A_ORDER'] = m
    sip_header['B_ORDER'] = m
    sip_header['A_2_0'] = 1e-7
    sip_header['A_0_2'] = 1e-7
    sip_header['B_2_0'] = 1e-7
    sip_header['B_0_2'] = 1e-7

    return WCS(sip_header)


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
    return _make_sip_wcs()


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
