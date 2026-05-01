# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the rotation-angle convention in to_sky/to_pixel conversions
of elliptical and rectangular apertures (and their annulus variants).

For a directed photutils sky aperture, the ``theta`` parameter is
the position angle (PA) of the major (width) axis measured from
North, counterclockwise. For a directed pixel aperture, ``theta`` is
measured counterclockwise from the positive ``x`` axis. For a standard
axis-aligned WCS (RA increasing to the left, equal isotropic pixel
scale, with North = +y), these two conventions differ by exactly 90 deg:

    sky.to_pixel(wcs).theta == sky.theta + 90 deg   (mod 360 deg)
    pixel.to_sky(wcs).theta == pixel.theta - 90 deg (mod 360 deg)

These tests pin that convention down so that any future refactor of
the internal _wcs_helpers cannot silently re-introduce a symmetric 90
deg offset bug, which is invisible to roundtrip-only tests because the
offset cancels in sky -> pixel -> sky (and vice versa).
"""

import astropy.units as u
import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS

from photutils.aperture import (EllipticalAnnulus, EllipticalAperture,
                                RectangularAnnulus, RectangularAperture,
                                SkyEllipticalAnnulus, SkyEllipticalAperture,
                                SkyRectangularAnnulus, SkyRectangularAperture)

CENTER = SkyCoord(100 * u.deg, 30 * u.deg)
PIX_CENTER = (10.5, 10.5)


@pytest.fixture
def axis_aligned_wcs():
    """
    A simple axis-aligned TAN WCS (RA increases to the left, equal pixel
    scales, North = +y).
    """
    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [10.5, 10.5]
    wcs.wcs.crval = [CENTER.ra.deg, CENTER.dec.deg]
    wcs.wcs.cdelt = [-0.1 / 3600, 0.1 / 3600]
    wcs.wcs.ctype = ['RA---TAN', 'DEC--TAN']

    return wcs


@pytest.fixture
def rotated_wcs():
    """
    A 25 deg rotated TAN WCS.
    """
    rotation_deg = 25.0
    cdelt = 0.1 / 3600
    rad = np.radians(rotation_deg)
    cos_a = np.cos(rad)
    sin_a = np.sin(rad)
    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [10.5, 10.5]
    wcs.wcs.crval = [CENTER.ra.deg, CENTER.dec.deg]
    wcs.wcs.cd = [[-cdelt * cos_a, cdelt * sin_a],
                  [cdelt * sin_a, cdelt * cos_a]]
    wcs.wcs.ctype = ['RA---TAN', 'DEC--TAN']

    return wcs


def _angle_diff_deg(actual, desired):
    """
    Signed angular difference in degrees, wrapped to (-180, 180].
    """
    diff = (actual.to(u.deg).value - desired.to(u.deg).value + 180) % 360
    return diff - 180


def _make_sky_pix_pair(sky_cls, pix_cls, theta):
    """
    Construct a (sky_aperture, pixel_aperture) pair of the given classes
    with the given theta.

    A small dispatch on class is needed because the directed aperture
    classes have different shape parameter names (a/b vs w/h, plus the
    annulus variants).
    """
    if sky_cls is SkyEllipticalAperture:
        sky = SkyEllipticalAperture(CENTER, a=2 * u.arcsec,
                                    b=1 * u.arcsec, theta=theta)
        pix = EllipticalAperture(PIX_CENTER, a=4, b=2, theta=theta)
    elif sky_cls is SkyRectangularAperture:
        sky = SkyRectangularAperture(CENTER, w=2 * u.arcsec,
                                     h=1 * u.arcsec, theta=theta)
        pix = RectangularAperture(PIX_CENTER, w=4, h=2, theta=theta)
    elif sky_cls is SkyEllipticalAnnulus:
        sky = SkyEllipticalAnnulus(CENTER, a_in=1 * u.arcsec,
                                   a_out=2 * u.arcsec,
                                   b_out=1 * u.arcsec, theta=theta)
        pix = EllipticalAnnulus(PIX_CENTER, a_in=2, a_out=4, b_out=2,
                                theta=theta)
    elif sky_cls is SkyRectangularAnnulus:
        sky = SkyRectangularAnnulus(CENTER, w_in=1 * u.arcsec,
                                    w_out=2 * u.arcsec,
                                    h_out=1 * u.arcsec, theta=theta)
        pix = RectangularAnnulus(PIX_CENTER, w_in=2, w_out=4, h_out=2,
                                 theta=theta)
    else:
        msg = f'unsupported aperture class: {sky_cls}'
        raise ValueError(msg)

    assert isinstance(pix, pix_cls)

    return sky, pix


DIRECTED_APERTURE_PAIRS = [
    pytest.param(SkyEllipticalAperture, EllipticalAperture, id='ellipse'),
    pytest.param(SkyRectangularAperture, RectangularAperture, id='rectangle'),
    pytest.param(SkyEllipticalAnnulus, EllipticalAnnulus,
                 id='ellipse_annulus'),
    pytest.param(SkyRectangularAnnulus, RectangularAnnulus,
                 id='rectangle_annulus'),
]

ANGLES_DEG = [0, 30, 60, 135, 200, 315]


class TestSkyPixelAngleConvention:
    """
    Verify the photutils sky/pixel angle conventions on an axis-aligned
    WCS where North = +y.
    """

    @pytest.mark.parametrize(('sky_cls', 'pix_cls'), DIRECTED_APERTURE_PAIRS)
    @pytest.mark.parametrize('theta_deg', ANGLES_DEG)
    def test_sky_to_pixel_theta(self, axis_aligned_wcs, sky_cls, pix_cls,
                                theta_deg):
        """
        Verify ``sky.to_pixel`` returns ``theta + 90 deg`` (mod 360).
        """
        sky, _ = _make_sky_pix_pair(sky_cls, pix_cls, theta_deg * u.deg)
        pix = sky.to_pixel(axis_aligned_wcs)
        diff = _angle_diff_deg(pix.theta, (theta_deg + 90) * u.deg)
        assert abs(diff) < 1e-3

    @pytest.mark.parametrize(('sky_cls', 'pix_cls'), DIRECTED_APERTURE_PAIRS)
    @pytest.mark.parametrize('theta_deg', ANGLES_DEG)
    def test_pixel_to_sky_theta(self, axis_aligned_wcs, sky_cls, pix_cls,
                                theta_deg):
        """
        Verify ``pixel.to_sky`` returns ``theta - 90 deg`` (mod 360).
        """
        _, pix = _make_sky_pix_pair(sky_cls, pix_cls, theta_deg * u.deg)
        sky = pix.to_sky(axis_aligned_wcs)
        diff = _angle_diff_deg(sky.theta, (theta_deg - 90) * u.deg)
        assert abs(diff) < 1e-3

    @pytest.mark.parametrize(('sky_cls', 'pix_cls'), DIRECTED_APERTURE_PAIRS)
    @pytest.mark.parametrize('theta_deg', ANGLES_DEG)
    def test_sky_pixel_sky_roundtrip_theta(self, axis_aligned_wcs, sky_cls,
                                           pix_cls, theta_deg):
        """
        Verify ``sky -> pixel -> sky`` preserves the original theta.
        """
        sky, _ = _make_sky_pix_pair(sky_cls, pix_cls, theta_deg * u.deg)
        sky_rt = sky.to_pixel(axis_aligned_wcs).to_sky(axis_aligned_wcs)
        diff = _angle_diff_deg(sky_rt.theta, theta_deg * u.deg)
        assert abs(diff) < 1e-3

    @pytest.mark.parametrize(('sky_cls', 'pix_cls'), DIRECTED_APERTURE_PAIRS)
    @pytest.mark.parametrize('theta_deg', ANGLES_DEG)
    def test_pixel_sky_pixel_roundtrip_theta(self, axis_aligned_wcs, sky_cls,
                                             pix_cls, theta_deg):
        """
        Verify ``pixel -> sky -> pixel`` preserves the original theta.
        """
        _, pix = _make_sky_pix_pair(sky_cls, pix_cls, theta_deg * u.deg)
        pix_rt = pix.to_sky(axis_aligned_wcs).to_pixel(axis_aligned_wcs)
        diff = _angle_diff_deg(pix_rt.theta, theta_deg * u.deg)
        assert abs(diff) < 1e-3

    def test_north_is_plus_y(self, axis_aligned_wcs):
        """
        Test that a sky aperture at PA=0 (North) maps to a pixel
        aperture pointing along +y, i.e., pixel theta = 90 deg.
        """
        sky = SkyEllipticalAperture(CENTER, a=2 * u.arcsec,
                                    b=1 * u.arcsec, theta=0 * u.deg)
        pix = sky.to_pixel(axis_aligned_wcs)
        diff = _angle_diff_deg(pix.theta, 90 * u.deg)
        assert abs(diff) < 1e-3

    def test_east_is_minus_x(self, axis_aligned_wcs):
        """
        Test that a sky aperture at PA=90 deg (East) maps to a pixel
        aperture pointing along -x, i.e., pixel theta = 180 deg.
        """
        sky = SkyEllipticalAperture(CENTER, a=2 * u.arcsec,
                                    b=1 * u.arcsec, theta=90 * u.deg)
        pix = sky.to_pixel(axis_aligned_wcs)
        diff = _angle_diff_deg(pix.theta, 180 * u.deg)
        assert abs(diff) < 1e-3


class TestRotatedWCSRoundtrip:
    """
    For a non-axis-aligned WCS we cannot pin the absolute sky-pixel
    angle relation without re-deriving the WCS rotation, but the
    roundtrips must still preserve theta.
    """

    @pytest.mark.parametrize(('sky_cls', 'pix_cls'), DIRECTED_APERTURE_PAIRS)
    @pytest.mark.parametrize('theta_deg', ANGLES_DEG)
    def test_sky_pixel_sky_roundtrip(self, rotated_wcs, sky_cls, pix_cls,
                                     theta_deg):
        sky, _ = _make_sky_pix_pair(sky_cls, pix_cls, theta_deg * u.deg)
        sky_rt = sky.to_pixel(rotated_wcs).to_sky(rotated_wcs)
        diff = _angle_diff_deg(sky_rt.theta, theta_deg * u.deg)
        assert abs(diff) < 1e-3

    @pytest.mark.parametrize(('sky_cls', 'pix_cls'), DIRECTED_APERTURE_PAIRS)
    @pytest.mark.parametrize('theta_deg', ANGLES_DEG)
    def test_pixel_sky_pixel_roundtrip(self, rotated_wcs, sky_cls, pix_cls,
                                       theta_deg):
        _, pix = _make_sky_pix_pair(sky_cls, pix_cls, theta_deg * u.deg)
        pix_rt = pix.to_sky(rotated_wcs).to_pixel(rotated_wcs)
        diff = _angle_diff_deg(pix_rt.theta, theta_deg * u.deg)
        assert abs(diff) < 1e-3
