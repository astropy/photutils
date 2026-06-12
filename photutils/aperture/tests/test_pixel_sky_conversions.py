# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for aperture pixel-to-sky and sky-to-pixel coordinate conversions.

This module covers:

* roundtrip conversions for all pixel and sky aperture classes through
  simple, rotated, distorted (SIP), sheared, and generalized
  (`gwcs.wcs.WCS`) WCS objects, exercising the SVD/Jacobian shape
  conversion (the SIP and gWCS cases in particular force the distortion
  (Jacobian) code path of the internal ``_wcs_helpers``)

* the rotation-angle convention for directed (elliptical and
  rectangular) apertures, which pins down the 90 deg offset between the
  sky position angle (measured from North) and the pixel angle (measured
  from the positive ``x`` axis).
"""

import astropy.units as u
import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from astropy.io.fits import Header
from astropy.tests.helper import assert_quantity_allclose
from astropy.wcs import WCS
from numpy.testing import assert_allclose

from photutils.aperture import (CircularAnnulus, CircularAperture,
                                EllipticalAnnulus, EllipticalAperture,
                                RectangularAnnulus, RectangularAperture,
                                SkyCircularAnnulus, SkyCircularAperture,
                                SkyEllipticalAnnulus, SkyEllipticalAperture,
                                SkyRectangularAnnulus, SkyRectangularAperture)
from photutils.datasets import make_gwcs
from photutils.utils._optional_deps import HAS_GWCS

# Module constants
CENTER = SkyCoord(100 * u.deg, 30 * u.deg)
PIX_CENTER = (10.5, 10.5)

# Position angles (deg) used to pin the sky/pixel angle convention.
ANGLES_DEG = [0, 30, 60, 135, 200, 315]

# Position angles (deg) used for the sheared-WCS and degenerate-shape
# roundtrip tests.
SHEAR_ANGLES_DEG = [0.0, 30.0, 75.0]

# Each case bundles a matching (sky, pixel) aperture class pair with
# their construction kwargs. ``size_attrs`` lists the shape attributes
# compared in roundtrip assertions and ``has_angle`` flags whether the
# aperture carries a ``theta`` rotation angle.
_APERTURE_CASES = [
    {'id': 'ellipse',
     'sky_cls': SkyEllipticalAperture,
     'sky_kw': {'a': 2 * u.arcsec, 'b': 1 * u.arcsec, 'theta': 30 * u.deg},
     'pix_cls': EllipticalAperture,
     'pix_kw': {'a': 4, 'b': 2, 'theta': 45 * u.deg},
     'size_attrs': ('a', 'b'),
     'has_angle': True},
    {'id': 'rectangle',
     'sky_cls': SkyRectangularAperture,
     'sky_kw': {'w': 2 * u.arcsec, 'h': 1 * u.arcsec, 'theta': 30 * u.deg},
     'pix_cls': RectangularAperture,
     'pix_kw': {'w': 4, 'h': 2, 'theta': 45 * u.deg},
     'size_attrs': ('w', 'h'),
     'has_angle': True},
    {'id': 'ellipse_annulus',
     'sky_cls': SkyEllipticalAnnulus,
     'sky_kw': {'a_in': 1 * u.arcsec, 'a_out': 2 * u.arcsec,
                'b_out': 1 * u.arcsec, 'theta': 30 * u.deg},
     'pix_cls': EllipticalAnnulus,
     'pix_kw': {'a_in': 2, 'a_out': 4, 'b_out': 2, 'theta': 45 * u.deg},
     'size_attrs': ('a_in', 'a_out', 'b_out'),
     'has_angle': True},
    {'id': 'rectangle_annulus',
     'sky_cls': SkyRectangularAnnulus,
     'sky_kw': {'w_in': 1 * u.arcsec, 'w_out': 2 * u.arcsec,
                'h_out': 1 * u.arcsec, 'theta': 30 * u.deg},
     'pix_cls': RectangularAnnulus,
     'pix_kw': {'w_in': 2, 'w_out': 4, 'h_out': 2, 'theta': 45 * u.deg},
     'size_attrs': ('w_in', 'w_out', 'h_out'),
     'has_angle': True},
    {'id': 'circle',
     'sky_cls': SkyCircularAperture,
     'sky_kw': {'r': 2 * u.arcsec},
     'pix_cls': CircularAperture,
     'pix_kw': {'r': 4},
     'size_attrs': ('r',),
     'has_angle': False},
    {'id': 'circle_annulus',
     'sky_cls': SkyCircularAnnulus,
     'sky_kw': {'r_in': 1 * u.arcsec, 'r_out': 2 * u.arcsec},
     'pix_cls': CircularAnnulus,
     'pix_kw': {'r_in': 2, 'r_out': 4},
     'size_attrs': ('r_in', 'r_out'),
     'has_angle': False},
]

# All aperture cases and the directed (rotatable) subset, wrapped as
# pytest params with readable ids.
APERTURE_CASES = [pytest.param(case, id=case['id'])
                  for case in _APERTURE_CASES]
DIRECTED_CASES = [pytest.param(case, id=case['id'])
                  for case in _APERTURE_CASES if case['has_angle']]


def _angle_diff_deg(actual, desired):
    """
    Signed angular difference in degrees, wrapped to (-180, 180].
    """
    diff = (actual.to_value(u.deg) - desired.to_value(u.deg) + 180) % 360
    return diff - 180


def _build_pair(case, theta):
    """
    Build a matching (sky_aperture, pixel_aperture) pair from a case,
    overriding the rotation angle ``theta`` of both apertures.
    """
    sky = case['sky_cls'](CENTER, **{**case['sky_kw'], 'theta': theta})
    pix = case['pix_cls'](PIX_CENTER, **{**case['pix_kw'], 'theta': theta})
    return sky, pix


def _make_sip_wcs(ra_deg=100.0, dec_deg=30.0):
    """
    Build a small TAN-SIP WCS centered at the given (RA, Dec).

    The SIP terms are tiny but nonzero, which forces the Jacobian
    (distortion) code path to be exercised by ``to_pixel``/``to_sky``.
    """
    header = Header()
    header['NAXIS'] = 2
    header['NAXIS1'] = 21
    header['NAXIS2'] = 21
    header['CRPIX1'] = 10.5
    header['CRPIX2'] = 10.5
    header['CRVAL1'] = ra_deg
    header['CRVAL2'] = dec_deg
    header['CTYPE1'] = 'RA---TAN-SIP'
    header['CTYPE2'] = 'DEC--TAN-SIP'
    cdelt = 0.1 / 3600.0
    header['CD1_1'] = -cdelt
    header['CD1_2'] = 0.0
    header['CD2_1'] = 0.0
    header['CD2_2'] = cdelt
    header['A_ORDER'] = 2
    header['A_2_0'] = 1e-7
    header['A_0_2'] = 1e-7
    header['B_ORDER'] = 2
    header['B_2_0'] = 1e-7
    header['B_0_2'] = 1e-7

    return WCS(header)


@pytest.fixture
def simple_wcs():
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
def sip_wcs():
    """
    A TAN-SIP WCS with small distortion terms centered at CENTER.
    """
    return _make_sip_wcs()


@pytest.fixture
def flipped_wcs():
    """
    A flipped-parity TAN WCS (North down, East left).

    Both CDELT values are negative, so the pixel scale matrix has
    a positive determinant (parity = +1), opposite the standard
    astronomical convention. North (increasing Dec) points along -y and
    East (increasing RA) points along -x.
    """
    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [10.5, 10.5]
    wcs.wcs.crval = [CENTER.ra.deg, CENTER.dec.deg]
    wcs.wcs.cdelt = [-0.1 / 3600, -0.1 / 3600]
    wcs.wcs.ctype = ['RA---TAN', 'DEC--TAN']

    return wcs


class TestSkyToPixel:
    """
    Converting a sky aperture to a pixel aperture must return the
    matching pixel class with positive shape parameters.
    """

    @pytest.mark.parametrize('case', APERTURE_CASES)
    def test_simple_wcs(self, simple_wcs, case):
        pix = case['sky_cls'](CENTER, **case['sky_kw']).to_pixel(
            simple_wcs)
        assert isinstance(pix, case['pix_cls'])
        for attr in case['size_attrs']:
            assert getattr(pix, attr) > 0

    @pytest.mark.parametrize('case', APERTURE_CASES)
    def test_rotated_wcs(self, rotated_wcs, case):
        pix = case['sky_cls'](CENTER, **case['sky_kw']).to_pixel(rotated_wcs)
        assert isinstance(pix, case['pix_cls'])
        for attr in case['size_attrs']:
            assert getattr(pix, attr) > 0

    @pytest.mark.parametrize('case', APERTURE_CASES)
    def test_sip_wcs(self, sip_wcs, case):
        """
        The Jacobian (distortion) path is used automatically for a SIP
        WCS.
        """
        pix = case['sky_cls'](CENTER, **case['sky_kw']).to_pixel(sip_wcs)
        assert isinstance(pix, case['pix_cls'])
        for attr in case['size_attrs']:
            assert getattr(pix, attr) > 0


class TestPixelToSky:
    """
    Converting a pixel aperture to a sky aperture must return the
    matching sky class with positive angular shape parameters.
    """

    @pytest.mark.parametrize('case', APERTURE_CASES)
    def test_simple_wcs(self, simple_wcs, case):
        sky = case['pix_cls'](PIX_CENTER, **case['pix_kw']).to_sky(
            simple_wcs)
        assert isinstance(sky, case['sky_cls'])
        for attr in case['size_attrs']:
            assert getattr(sky, attr) > 0 * u.arcsec

    @pytest.mark.parametrize('case', APERTURE_CASES)
    def test_sip_wcs(self, sip_wcs, case):
        sky = case['pix_cls'](PIX_CENTER, **case['pix_kw']).to_sky(sip_wcs)
        assert isinstance(sky, case['sky_cls'])
        for attr in case['size_attrs']:
            assert getattr(sky, attr) > 0 * u.arcsec


class TestRoundtripSkyPixelSky:
    """
    A sky -> pixel -> sky roundtrip must recover the original shape
    parameters and rotation angle.
    """

    @pytest.mark.parametrize('case', APERTURE_CASES)
    def test_simple_wcs(self, simple_wcs, case):
        sky = case['sky_cls'](CENTER, **case['sky_kw'])
        sky_rt = sky.to_pixel(simple_wcs).to_sky(simple_wcs)
        assert isinstance(sky_rt, case['sky_cls'])
        for attr in case['size_attrs']:
            assert_quantity_allclose(getattr(sky_rt, attr),
                                     getattr(sky, attr))
        if case['has_angle']:
            assert_quantity_allclose(sky_rt.theta, sky.theta,
                                     atol=1e-9 * u.deg)

    @pytest.mark.parametrize('case', APERTURE_CASES)
    def test_rotated_wcs(self, rotated_wcs, case):
        sky = case['sky_cls'](CENTER, **case['sky_kw'])
        sky_rt = sky.to_pixel(rotated_wcs).to_sky(rotated_wcs)
        for attr in case['size_attrs']:
            assert_quantity_allclose(getattr(sky_rt, attr),
                                     getattr(sky, attr))
        if case['has_angle']:
            assert_quantity_allclose(sky_rt.theta, sky.theta,
                                     atol=1e-9 * u.deg)

    @pytest.mark.parametrize('case', APERTURE_CASES)
    def test_sip_wcs(self, sip_wcs, case):
        """
        Roundtrip through a distorted (SIP) WCS, exercising the Jacobian
        code path.
        """
        sky = case['sky_cls'](CENTER, **case['sky_kw'])
        sky_rt = sky.to_pixel(sip_wcs).to_sky(sip_wcs)
        for attr in case['size_attrs']:
            assert_quantity_allclose(getattr(sky_rt, attr),
                                     getattr(sky, attr), rtol=1e-3)
        if case['has_angle']:
            assert_quantity_allclose(sky_rt.theta, sky.theta,
                                     atol=1e-3 * u.deg)


class TestRoundtripPixelSkyPixel:
    """
    A pixel -> sky -> pixel roundtrip must recover the original center,
    shape parameters, and rotation angle.
    """

    @pytest.mark.parametrize('case', APERTURE_CASES)
    def test_simple_wcs(self, simple_wcs, case):
        pix = case['pix_cls'](PIX_CENTER, **case['pix_kw'])
        pix_rt = pix.to_sky(simple_wcs).to_pixel(simple_wcs)
        assert_allclose(pix_rt.positions, pix.positions)
        for attr in case['size_attrs']:
            assert_allclose(getattr(pix_rt, attr), getattr(pix, attr))
        if case['has_angle']:
            assert_quantity_allclose(pix_rt.theta, pix.theta,
                                     atol=1e-9 * u.deg)

    @pytest.mark.parametrize('case', APERTURE_CASES)
    def test_rotated_wcs(self, rotated_wcs, case):
        pix = case['pix_cls'](PIX_CENTER, **case['pix_kw'])
        pix_rt = pix.to_sky(rotated_wcs).to_pixel(rotated_wcs)
        assert_allclose(pix_rt.positions, pix.positions)
        for attr in case['size_attrs']:
            assert_allclose(getattr(pix_rt, attr), getattr(pix, attr))
        if case['has_angle']:
            assert_quantity_allclose(pix_rt.theta, pix.theta,
                                     atol=1e-9 * u.deg)

    @pytest.mark.parametrize('case', APERTURE_CASES)
    def test_sip_wcs(self, sip_wcs, case):
        pix = case['pix_cls'](PIX_CENTER, **case['pix_kw'])
        pix_rt = pix.to_sky(sip_wcs).to_pixel(sip_wcs)
        assert_allclose(pix_rt.positions, pix.positions, atol=1e-6)
        for attr in case['size_attrs']:
            assert_allclose(getattr(pix_rt, attr), getattr(pix, attr),
                            rtol=1e-3)
        if case['has_angle']:
            assert_quantity_allclose(pix_rt.theta, pix.theta,
                                     atol=1e-3 * u.deg)


@pytest.mark.skipif(not HAS_GWCS, reason='gwcs is required')
class TestGWCSRoundtrip:
    """
    Roundtrip conversions through a generalized WCS (`gwcs.wcs.WCS`).

    A gWCS has no ``has_distortion`` attribute, so the Jacobian path is
    always used.
    """

    @pytest.mark.parametrize('case', APERTURE_CASES)
    def test_sky_pixel_sky(self, case):
        wcs = make_gwcs((100, 100))
        center = wcs.pixel_to_world(50, 50)
        sky = case['sky_cls'](center, **case['sky_kw'])
        sky_rt = sky.to_pixel(wcs).to_sky(wcs)
        assert isinstance(sky_rt, case['sky_cls'])
        for attr in case['size_attrs']:
            assert_quantity_allclose(getattr(sky_rt, attr),
                                     getattr(sky, attr), rtol=1e-5)
        if case['has_angle']:
            assert_quantity_allclose(sky_rt.theta, sky.theta,
                                     atol=1e-4 * u.deg)

    @pytest.mark.parametrize('case', APERTURE_CASES)
    def test_pixel_sky_pixel(self, case):
        wcs = make_gwcs((100, 100))
        pix = case['pix_cls']((50, 50), **case['pix_kw'])
        pix_rt = pix.to_sky(wcs).to_pixel(wcs)
        assert_allclose(pix_rt.positions, pix.positions, rtol=1e-5)
        for attr in case['size_attrs']:
            assert_allclose(getattr(pix_rt, attr), getattr(pix, attr),
                            rtol=1e-5)
        if case['has_angle']:
            assert_quantity_allclose(pix_rt.theta, pix.theta,
                                     atol=1e-4 * u.deg)


class TestSkyPixelAngleConvention:
    """
    Verify the sky/pixel angle conventions on an axis-aligned WCS where
    North = +y.

    For a directed sky aperture, ``theta`` is the position angle (PA) of
    the width axis measured from North, counterclockwise. For a directed
    pixel aperture, ``theta`` is measured counterclockwise from the
    positive ``x`` axis. For a standard axis-aligned WCS (RA increasing
    to the left, equal isotropic pixel scale, North = +y), these two
    conventions differ by exactly 90 deg:

        sky.to_pixel(wcs).theta == sky.theta + 90 deg   (mod 360 deg)
        pixel.to_sky(wcs).theta == pixel.theta - 90 deg (mod 360 deg)

    These tests pin that convention down so that any future refactor of
    the internal ``_wcs_helpers`` cannot silently re-introduce a
    symmetric 90 deg offset bug, which is invisible to roundtrip-only
    tests because the offset cancels in sky -> pixel -> sky (and vice
    versa).
    """

    @pytest.mark.parametrize('case', DIRECTED_CASES)
    @pytest.mark.parametrize('theta_deg', ANGLES_DEG)
    def test_sky_to_pixel_theta(self, simple_wcs, case, theta_deg):
        """
        Verify ``sky.to_pixel`` returns ``theta + 90 deg`` (mod 360).
        """
        sky, _ = _build_pair(case, theta_deg * u.deg)
        pix = sky.to_pixel(simple_wcs)
        diff = _angle_diff_deg(pix.theta, (theta_deg + 90) * u.deg)
        assert abs(diff) < 2e-5

    @pytest.mark.parametrize('case', DIRECTED_CASES)
    @pytest.mark.parametrize('theta_deg', ANGLES_DEG)
    def test_pixel_to_sky_theta(self, simple_wcs, case, theta_deg):
        """
        Verify ``pixel.to_sky`` returns ``theta - 90 deg`` (mod 360).
        """
        _, pix = _build_pair(case, theta_deg * u.deg)
        sky = pix.to_sky(simple_wcs)
        diff = _angle_diff_deg(sky.theta, (theta_deg - 90) * u.deg)
        assert abs(diff) < 2e-5

    def test_north_is_plus_y(self, simple_wcs):
        """
        Test that a sky aperture at PA=0 (North) maps to a pixel
        aperture pointing along +y, i.e., pixel theta = 90 deg.
        """
        sky = SkyEllipticalAperture(CENTER, a=2 * u.arcsec,
                                    b=1 * u.arcsec, theta=0 * u.deg)
        pix = sky.to_pixel(simple_wcs)
        diff = _angle_diff_deg(pix.theta, 90 * u.deg)
        assert abs(diff) < 2e-5

    def test_east_is_minus_x(self, simple_wcs):
        """
        Test that a sky aperture at PA=90 deg (East) maps to a pixel
        aperture pointing along -x, i.e., pixel theta = 180 deg.
        """
        sky = SkyEllipticalAperture(CENTER, a=2 * u.arcsec,
                                    b=1 * u.arcsec, theta=90 * u.deg)
        pix = sky.to_pixel(simple_wcs)
        diff = _angle_diff_deg(pix.theta, 180 * u.deg)
        assert abs(diff) < 2e-5


class TestShearedWCSRoundtrip:
    """
    Verify that directed apertures round-trip exactly through a sheared
    WCS, where the pixel x and y axes are not perpendicular on the sky.
    """

    @pytest.mark.parametrize('case', DIRECTED_CASES)
    @pytest.mark.parametrize('theta_deg', SHEAR_ANGLES_DEG)
    def test_sky_pixel_sky(self, sheared_wcs, case, theta_deg):
        sky, _ = _build_pair(case, theta_deg * u.deg)
        sky_rt = sky.to_pixel(sheared_wcs).to_sky(sheared_wcs)
        diff = _angle_diff_deg(sky_rt.theta, theta_deg * u.deg)
        assert abs(diff) < 1e-6

        # Shape parameters must round-trip too.
        for attr in ('a', 'b', 'w', 'h', 'a_in', 'a_out', 'b_in', 'b_out',
                     'w_in', 'w_out', 'h_in', 'h_out'):
            if hasattr(sky, attr):
                assert u.allclose(getattr(sky_rt, attr),
                                  getattr(sky, attr), rtol=1e-6)

    @pytest.mark.parametrize('case', DIRECTED_CASES)
    @pytest.mark.parametrize('theta_deg', SHEAR_ANGLES_DEG)
    def test_pixel_sky_pixel(self, sheared_wcs, case, theta_deg):
        _, pix = _build_pair(case, theta_deg * u.deg)
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

    @pytest.mark.parametrize('theta_deg', SHEAR_ANGLES_DEG)
    @pytest.mark.parametrize(
        'sky_cls', [SkyRectangularAperture, SkyEllipticalAperture])
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

    @pytest.mark.parametrize('theta_deg', SHEAR_ANGLES_DEG)
    @pytest.mark.parametrize(
        'pix_cls', [RectangularAperture, EllipticalAperture])
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


class TestFlippedParityWCS:
    """
    Regression tests for a flipped-parity WCS (North down, East left;
    positive-determinant pixel scale matrix).

    Such a WCS previously produced apertures that were mirrored about
    the x-axis. For the flipped axis-aligned WCS, North = -y and East =
    -x, so the sky/pixel angle relation is ``pixel.theta == 270 deg -
    sky.theta`` (mod 360).
    """

    def test_flipped_wcs_has_positive_parity(self, flipped_wcs):
        """
        The flipped WCS fixture must have a positive-determinant pixel
        scale matrix (parity = +1).
        """
        assert np.linalg.det(flipped_wcs.pixel_scale_matrix) > 0

    def test_north_is_minus_y(self, flipped_wcs):
        """
        A sky aperture at PA=0 (North) maps to a pixel aperture pointing
        along -y (down), i.e., pixel theta = 270 deg.
        """
        sky = SkyEllipticalAperture(CENTER, a=2 * u.arcsec, b=1 * u.arcsec,
                                    theta=0 * u.deg)
        pix = sky.to_pixel(flipped_wcs)
        diff = _angle_diff_deg(pix.theta, 270 * u.deg)
        assert abs(diff) < 2e-5

    def test_east_is_minus_x(self, flipped_wcs):
        """
        A sky aperture at PA=90 deg (East) maps to a pixel aperture
        pointing along -x (left), i.e., pixel theta = 180 deg.
        """
        sky = SkyEllipticalAperture(CENTER, a=2 * u.arcsec, b=1 * u.arcsec,
                                    theta=90 * u.deg)
        pix = sky.to_pixel(flipped_wcs)
        diff = _angle_diff_deg(pix.theta, 180 * u.deg)
        assert abs(diff) < 2e-5

    @pytest.mark.parametrize('case', DIRECTED_CASES)
    @pytest.mark.parametrize('theta_deg', ANGLES_DEG)
    def test_sky_to_pixel_theta(self, flipped_wcs, case, theta_deg):
        """
        Verify ``sky.to_pixel`` returns ``270 deg - theta`` (mod 360)
        for the flipped-parity WCS (not the mirrored value).
        """
        sky, _ = _build_pair(case, theta_deg * u.deg)
        pix = sky.to_pixel(flipped_wcs)
        diff = _angle_diff_deg(pix.theta, (270 - theta_deg) * u.deg)
        assert abs(diff) < 2e-5

    @pytest.mark.parametrize('case', DIRECTED_CASES)
    @pytest.mark.parametrize('theta_deg', ANGLES_DEG)
    def test_pixel_to_sky_theta(self, flipped_wcs, case, theta_deg):
        """
        Verify ``pixel.to_sky`` returns ``270 deg - theta`` (mod 360)
        for the flipped-parity WCS (the inverse of the above relation).
        """
        _, pix = _build_pair(case, theta_deg * u.deg)
        sky = pix.to_sky(flipped_wcs)
        diff = _angle_diff_deg(sky.theta, (270 - theta_deg) * u.deg)
        assert abs(diff) < 2e-5

    @pytest.mark.parametrize('case', DIRECTED_CASES)
    @pytest.mark.parametrize('theta_deg', ANGLES_DEG)
    def test_sky_pixel_sky_roundtrip_theta(self, flipped_wcs, case, theta_deg):
        """
        Verify ``sky -> pixel -> sky`` preserves the original theta for
        the flipped-parity WCS.
        """
        sky, _ = _build_pair(case, theta_deg * u.deg)
        sky_rt = sky.to_pixel(flipped_wcs).to_sky(flipped_wcs)
        diff = _angle_diff_deg(sky_rt.theta, theta_deg * u.deg)
        assert abs(diff) < 2e-5
