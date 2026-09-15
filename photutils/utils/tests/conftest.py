# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Pytest configuration and WCS test fixtures for photutils.utils tests.
"""

import astropy.units as u
import pytest
from astropy.wcs import WCS

from photutils.utils.tests.wcs_test_helpers import (WCS_CDELT_ARCSEC,
                                                    WCS_CENTER,
                                                    make_simple_wcs,
                                                    make_sip_wcs)


@pytest.fixture
def simple_wcs():
    """
    Non-distorted TAN WCS aligned with the celestial axes.
    """
    return make_simple_wcs(WCS_CENTER, WCS_CDELT_ARCSEC * u.arcsec, 20)


@pytest.fixture
def rotated_wcs():
    """
    Non-distorted TAN WCS with a 25-degree rotation (CD matrix).
    """
    return make_simple_wcs(WCS_CENTER, WCS_CDELT_ARCSEC * u.arcsec, 20,
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
