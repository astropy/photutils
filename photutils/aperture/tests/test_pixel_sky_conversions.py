# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for aperture pixel-to-sky and sky-to-pixel coordinate
conversions.

This module covers:

* roundtrip conversions for all pixel and sky aperture classes through
  both FITS (`~astropy.wcs.WCS`) and generalized (`gwcs.wcs.WCS`) WCS
  objects;

* the rotation-angle convention for directed (elliptical and
  rectangular) apertures, which pins down the 90 deg offset between the
  sky position angle (measured from North) and the pixel angle
  (measured from the positive ``x`` axis); and

* region-style parametrized conversions that exercise the SVD/Jacobian
  shape conversion across simple, rotated, distorted (SIP), and sheared
  WCS for every aperture type.
"""

import astropy.units as u
import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from numpy.testing import assert_allclose

from photutils.aperture import (CircularAnnulus, CircularAperture,
                                EllipticalAnnulus, EllipticalAperture,
                                RectangularAnnulus, RectangularAperture,
                                SkyEllipticalAnnulus, SkyEllipticalAperture,
                                SkyRectangularAnnulus, SkyRectangularAperture)
from photutils.datasets import make_4gaussians_image, make_gwcs, make_wcs
from photutils.utils._optional_deps import HAS_GWCS


@pytest.mark.skipif(not HAS_GWCS, reason='gwcs is required')
@pytest.mark.parametrize('wcs_type', ['wcs', 'gwcs'])
def test_to_sky_pixel(wcs_type):
    data = make_4gaussians_image()

    if wcs_type == 'wcs':
        wcs = make_wcs(data.shape)
    elif wcs_type == 'gwcs':
        wcs = make_gwcs(data.shape)

    ap = CircularAperture(((12.3, 15.7), (48.19, 98.14)), r=3.14)
    ap2 = ap.to_sky(wcs).to_pixel(wcs)
    assert_allclose(ap.positions, ap2.positions)
    assert_allclose(ap.r, ap2.r)

    ap = CircularAnnulus(((12.3, 15.7), (48.19, 98.14)), r_in=3.14,
                         r_out=5.32)
    ap2 = ap.to_sky(wcs).to_pixel(wcs)
    assert_allclose(ap.positions, ap2.positions)
    assert_allclose(ap.r_in, ap2.r_in)
    assert_allclose(ap.r_out, ap2.r_out)

    ap = EllipticalAperture(((12.3, 15.7), (48.19, 98.14)), a=3.14, b=5.32,
                            theta=np.deg2rad(103.0))
    ap2 = ap.to_sky(wcs).to_pixel(wcs)
    assert_allclose(ap.positions, ap2.positions)
    assert_allclose(ap.a, ap2.a)
    assert_allclose(ap.b, ap2.b)
    assert_allclose(ap.theta, ap2.theta)

    ap = EllipticalAnnulus(((12.3, 15.7), (48.19, 98.14)), a_in=3.14,
                           a_out=15.32, b_out=4.89,
                           theta=np.deg2rad(103.0))
    ap2 = ap.to_sky(wcs).to_pixel(wcs)
    assert_allclose(ap.positions, ap2.positions)
    assert_allclose(ap.a_in, ap2.a_in)
    assert_allclose(ap.a_out, ap2.a_out)
    assert_allclose(ap.b_out, ap2.b_out)
    assert_allclose(ap.theta, ap2.theta)

    ap = RectangularAperture(((12.3, 15.7), (48.19, 98.14)), w=3.14, h=5.32,
                             theta=np.deg2rad(103.0))
    ap2 = ap.to_sky(wcs).to_pixel(wcs)
    assert_allclose(ap.positions, ap2.positions)
    assert_allclose(ap.w, ap2.w)
    assert_allclose(ap.h, ap2.h)
    assert_allclose(ap.theta, ap2.theta)

    ap = RectangularAnnulus(((12.3, 15.7), (48.19, 98.14)), w_in=3.14,
                            w_out=15.32, h_out=4.89,
                            theta=np.deg2rad(103.0))
    ap2 = ap.to_sky(wcs).to_pixel(wcs)
    assert_allclose(ap.positions, ap2.positions)
    assert_allclose(ap.w_in, ap2.w_in)
    assert_allclose(ap.w_out, ap2.w_out)
    assert_allclose(ap.h_out, ap2.h_out)
    assert_allclose(ap.theta, ap2.theta)


# ---------------------------------------------------------------------------
# Rotation-angle convention tests for directed (elliptical and
# rectangular) apertures.
#
# For a directed photutils sky aperture, the ``theta`` parameter is the
# position angle (PA) of the major (width) axis measured from North,
# counterclockwise. For a directed pixel aperture, ``theta`` is measured
# counterclockwise from the positive ``x`` axis. For a standard
# axis-aligned WCS (RA increasing to the left, equal isotropic pixel
# scale, with North = +y), these two conventions differ by exactly 90
# deg:
#
#     sky.to_pixel(wcs).theta == sky.theta + 90 deg   (mod 360 deg)
#     pixel.to_sky(wcs).theta == pixel.theta - 90 deg (mod 360 deg)
#
# These tests pin that convention down so that any future refactor of
# the internal _wcs_helpers cannot silently re-introduce a symmetric 90
# deg offset bug, which is invisible to roundtrip-only tests because the
# offset cancels in sky -> pixel -> sky (and vice versa).
# ---------------------------------------------------------------------------

CENTER = SkyCoord(100 * u.deg, 30 * u.deg)
PIX_CENTER = (10.5, 10.5)

DIRECTED_APERTURE_PAIRS = [
    pytest.param(SkyEllipticalAperture, EllipticalAperture, id='ellipse'),
    pytest.param(SkyRectangularAperture, RectangularAperture, id='rectangle'),
    pytest.param(SkyEllipticalAnnulus, EllipticalAnnulus,
                 id='ellipse_annulus'),
    pytest.param(SkyRectangularAnnulus, RectangularAnnulus,
                 id='rectangle_annulus'),
]

SHEAR_DIRECTED_PAIRS = [
    (SkyRectangularAperture, RectangularAperture),
    (SkyRectangularAnnulus, RectangularAnnulus),
    (SkyEllipticalAperture, EllipticalAperture),
    (SkyEllipticalAnnulus, EllipticalAnnulus),
]

ANGLES_DEG = [0, 30, 60, 135, 200, 315]


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
def sheared_wcs():
    """
    A sheared TAN WCS where the pixel x and y axes are 80 deg apart on
    the sky (10 deg of shear from a perpendicular axis-aligned grid).
    """
    cdelt = 0.1 / 3600
    a = np.radians(80.0)
    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [10.5, 10.5]
    wcs.wcs.crval = [CENTER.ra.deg, CENTER.dec.deg]
    wcs.wcs.cd = [[-cdelt, cdelt * np.cos(a)],
                  [0.0, cdelt * np.sin(a)]]
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


class TestSkyPixelAngleConvention:
    """
    Verify the sky/pixel angle conventions on an axis-aligned WCS where
    North = +y.
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
        assert abs(diff) < 2e-5

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
        assert abs(diff) < 2e-5

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
        assert abs(diff) < 2e-5

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
        assert abs(diff) < 2e-5

    def test_north_is_plus_y(self, axis_aligned_wcs):
        """
        Test that a sky aperture at PA=0 (North) maps to a pixel
        aperture pointing along +y, i.e., pixel theta = 90 deg.
        """
        sky = SkyEllipticalAperture(CENTER, a=2 * u.arcsec,
                                    b=1 * u.arcsec, theta=0 * u.deg)
        pix = sky.to_pixel(axis_aligned_wcs)
        diff = _angle_diff_deg(pix.theta, 90 * u.deg)
        assert abs(diff) < 2e-5

    def test_east_is_minus_x(self, axis_aligned_wcs):
        """
        Test that a sky aperture at PA=90 deg (East) maps to a pixel
        aperture pointing along -x, i.e., pixel theta = 180 deg.
        """
        sky = SkyEllipticalAperture(CENTER, a=2 * u.arcsec,
                                    b=1 * u.arcsec, theta=90 * u.deg)
        pix = sky.to_pixel(axis_aligned_wcs)
        diff = _angle_diff_deg(pix.theta, 180 * u.deg)
        assert abs(diff) < 2e-5


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
        assert abs(diff) < 2e-5

    @pytest.mark.parametrize(('sky_cls', 'pix_cls'), DIRECTED_APERTURE_PAIRS)
    @pytest.mark.parametrize('theta_deg', ANGLES_DEG)
    def test_pixel_sky_pixel_roundtrip(self, rotated_wcs, sky_cls, pix_cls,
                                       theta_deg):
        _, pix = _make_sky_pix_pair(sky_cls, pix_cls, theta_deg * u.deg)
        pix_rt = pix.to_sky(rotated_wcs).to_pixel(rotated_wcs)
        diff = _angle_diff_deg(pix_rt.theta, theta_deg * u.deg)
        assert abs(diff) < 2e-5


class TestShearedWCS:
    """
    Verify that directed apertures round-trip exactly through a sheared
    WCS, where the pixel x and y axes are not perpendicular on the sky.
    """

    @pytest.mark.parametrize(('sky_cls', 'pix_cls'), SHEAR_DIRECTED_PAIRS)
    @pytest.mark.parametrize('theta_deg', [0.0, 30.0, 75.0])
    def test_sky_pixel_sky_roundtrip_under_shear(self, sheared_wcs,
                                                 sky_cls, pix_cls,
                                                 theta_deg):
        sky, _ = _make_sky_pix_pair(sky_cls, pix_cls, theta_deg * u.deg)
        sky_rt = sky.to_pixel(sheared_wcs).to_sky(sheared_wcs)
        diff = _angle_diff_deg(sky_rt.theta, theta_deg * u.deg)
        assert abs(diff) < 1e-6

        # Shape parameters must round-trip too
        for attr in ('a', 'b', 'w', 'h', 'a_in', 'a_out', 'b_in', 'b_out',
                     'w_in', 'w_out', 'h_in', 'h_out'):
            if hasattr(sky, attr):
                assert u.allclose(getattr(sky_rt, attr),
                                  getattr(sky, attr), rtol=1e-6)

    @pytest.mark.parametrize(('sky_cls', 'pix_cls'), SHEAR_DIRECTED_PAIRS)
    @pytest.mark.parametrize('theta_deg', [0.0, 30.0, 75.0])
    def test_pixel_sky_pixel_roundtrip_under_shear(self, sheared_wcs,
                                                   sky_cls, pix_cls,
                                                   theta_deg):
        _, pix = _make_sky_pix_pair(sky_cls, pix_cls, theta_deg * u.deg)
        pix_rt = pix.to_sky(sheared_wcs).to_pixel(sheared_wcs)
        diff = _angle_diff_deg(pix_rt.theta, theta_deg * u.deg)
        assert abs(diff) < 1e-6


class TestCircularInputAnglePreserved:
    """
    Verify that the SVD path preserves the input rotation angle when the
    input rectangular/elliptical aperture is shape-degenerate (width ==
    height): the converted theta must round-trip exactly.

    For a circular shape the SVD principal axis is otherwise arbitrary,
    so the helper falls back to the mapped width semi-axis direction to
    preserve orientation.
    """

    @pytest.mark.parametrize('theta_deg', [0.0, 30.0, 75.0])
    @pytest.mark.parametrize(
        'sky_cls',
        [SkyRectangularAperture, SkyEllipticalAperture])
    def test_square_sky_roundtrip(self, rotated_wcs, sky_cls, theta_deg):
        if sky_cls is SkyRectangularAperture:
            sky = SkyRectangularAperture(CENTER, w=2 * u.arcsec,
                                         h=2 * u.arcsec,
                                         theta=theta_deg * u.deg)
        else:
            sky = SkyEllipticalAperture(CENTER, a=2 * u.arcsec,
                                        b=2 * u.arcsec,
                                        theta=theta_deg * u.deg)
        sky_rt = sky.to_pixel(rotated_wcs).to_sky(rotated_wcs)
        diff = _angle_diff_deg(sky_rt.theta, theta_deg * u.deg)
        assert abs(diff) < 1e-6

    @pytest.mark.parametrize('theta_deg', [0.0, 30.0, 75.0])
    @pytest.mark.parametrize(
        'pix_cls',
        [RectangularAperture, EllipticalAperture])
    def test_square_pixel_roundtrip(self, rotated_wcs, pix_cls, theta_deg):
        if pix_cls is RectangularAperture:
            pix = RectangularAperture(PIX_CENTER, w=4, h=4,
                                      theta=theta_deg * u.deg)
        else:
            pix = EllipticalAperture(PIX_CENTER, a=4, b=4,
                                     theta=theta_deg * u.deg)
        pix_rt = pix.to_sky(rotated_wcs).to_pixel(rotated_wcs)
        diff = _angle_diff_deg(pix_rt.theta, theta_deg * u.deg)
        assert abs(diff) < 1e-6
