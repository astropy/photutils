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
from photutils.utils.tests.wcs_test_helpers import CountingWCS

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


def _make_sip_wcs(*, ra_deg=100.0, dec_deg=30.0):
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


@pytest.fixture
def swapped_wcs():
    """
    A TAN WCS with the latitude axis first (Dec, RA).

    The CD matrix maps the same sky footprint as ``simple_wcs``, so
    every conversion must give the same aperture.
    """
    cdelt = 0.1 / 3600
    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [10.5, 10.5]
    wcs.wcs.crval = [CENTER.dec.deg, CENTER.ra.deg]
    wcs.wcs.cd = [[0.0, cdelt], [-cdelt, 0.0]]
    wcs.wcs.ctype = ['DEC--TAN', 'RA---TAN']

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


class TestSwappedAxisWCS:
    """
    A WCS with the latitude axis first must give the same apertures as
    the equivalent WCS with the longitude axis first.

    A roundtrip alone would not catch a Jacobian built from swapped
    longitude and latitude, because the forward and inverse conversions
    would share the same error.
    """

    @pytest.mark.parametrize('case', APERTURE_CASES)
    def test_sky_to_pixel(self, simple_wcs, swapped_wcs, case):
        sky = case['sky_cls'](CENTER, **case['sky_kw'])
        expected = sky.to_pixel(simple_wcs)
        pix = sky.to_pixel(swapped_wcs)
        assert_allclose(pix.positions, expected.positions, atol=1e-8)
        for attr in case['size_attrs']:
            assert_allclose(getattr(pix, attr), getattr(expected, attr),
                            rtol=1e-9)
        if case['has_angle']:
            assert_quantity_allclose(pix.theta, expected.theta,
                                     atol=1e-6 * u.deg)

    @pytest.mark.parametrize('case', APERTURE_CASES)
    def test_pixel_to_sky(self, simple_wcs, swapped_wcs, case):
        pix = case['pix_cls'](PIX_CENTER, **case['pix_kw'])
        expected = pix.to_sky(simple_wcs)
        sky = pix.to_sky(swapped_wcs)
        assert sky.positions.separation(expected.positions).arcsec < 1e-8
        for attr in case['size_attrs']:
            assert_quantity_allclose(getattr(sky, attr),
                                     getattr(expected, attr), rtol=1e-9)
        if case['has_angle']:
            assert_quantity_allclose(sky.theta, expected.theta,
                                     atol=1e-6 * u.deg)


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
    height). The converted theta must round-trip exactly.

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
    Regression tests for a flipped-parity WCS (North down, East left,
    with a positive-determinant pixel scale matrix).

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


def _make_nonsquare_wcs():
    """
    Non-distorted TAN WCS with non-square pixels (0.03 x 0.05 deg).
    """
    wcs = WCS(naxis=2)
    wcs.wcs.crpix = list(PIX_CENTER)
    wcs.wcs.crval = [CENTER.ra.deg, CENTER.dec.deg]
    wcs.wcs.cdelt = [-0.03, 0.05]
    wcs.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    return wcs


def _pixel_ellipse_boundary(aperture, n_points=90):
    """
    Return the (x, y) pixel coordinates of points on the boundary of a
    single-position elliptical pixel aperture.
    """
    t = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    theta = aperture.theta
    if not isinstance(theta, float):
        theta = theta.to_value(u.rad)
    x0, y0 = np.atleast_2d(aperture.positions)[0]
    xe = aperture.a * np.cos(t)
    ye = aperture.b * np.sin(t)
    x = x0 + xe * np.cos(theta) - ye * np.sin(theta)
    y = y0 + xe * np.sin(theta) + ye * np.cos(theta)
    return x, y


class TestCircleAsEllipse:
    """
    Tests for the ``as_ellipse`` keyword of the circular aperture
    ``to_pixel`` and ``to_sky`` methods.

    A circle on the sky maps to an ellipse in pixels (and vice versa)
    whenever the pixels are not square or the WCS is sheared. The
    keyword returns that exact ellipse instead of the mean-scale circle.
    """

    NONSQUARE_X_ARCSEC = 0.03 * 3600
    NONSQUARE_Y_ARCSEC = 0.05 * 3600

    def test_default_returns_circle(self, simple_wcs):
        sky = SkyCircularAperture(CENTER, r=1 * u.arcsec)
        assert isinstance(sky.to_pixel(simple_wcs), CircularAperture)
        pix = CircularAperture(PIX_CENTER, r=2)
        assert isinstance(pix.to_sky(simple_wcs), SkyCircularAperture)

    def test_sky_to_pixel_nonsquare(self):
        wcs = _make_nonsquare_wcs()
        sky = SkyCircularAperture(CENTER, r=1 * u.arcsec)
        aper = sky.to_pixel(wcs, as_ellipse=True)
        assert isinstance(aper, EllipticalAperture)
        assert_allclose(aper.a, 1 / self.NONSQUARE_X_ARCSEC, rtol=1e-6)
        assert_allclose(aper.b, 1 / self.NONSQUARE_Y_ARCSEC, rtol=1e-6)
        # The major axis is along x
        assert_allclose(np.sin(aper.theta.to_value(u.rad)), 0, atol=1e-6)

    def test_pixel_to_sky_nonsquare(self):
        wcs = _make_nonsquare_wcs()
        # Place the aperture on the tangent point (CRPIX is 1-based),
        # where the pixel y axis points exactly North. One pixel away
        # the meridians have already converged by 0.02 deg with these
        # 0.05 deg pixels.
        pix = CircularAperture((PIX_CENTER[0] - 1, PIX_CENTER[1] - 1), r=2)
        aper = pix.to_sky(wcs, as_ellipse=True)
        assert isinstance(aper, SkyEllipticalAperture)
        y_arcsec = self.NONSQUARE_Y_ARCSEC * u.arcsec
        x_arcsec = self.NONSQUARE_X_ARCSEC * u.arcsec
        assert_quantity_allclose(aper.a, 2 * y_arcsec, rtol=1e-6)
        assert_quantity_allclose(aper.b, 2 * x_arcsec, rtol=1e-6)
        # The major axis is along y, which is North, so the position
        # angle is 0 mod 180.
        assert_allclose(np.sin(aper.theta.to_value(u.rad)), 0, atol=1e-6)

    def test_sky_annulus_to_pixel_nonsquare(self):
        wcs = _make_nonsquare_wcs()
        sky = SkyCircularAnnulus(CENTER, r_in=1 * u.arcsec, r_out=2 * u.arcsec)
        aper = sky.to_pixel(wcs, as_ellipse=True)
        assert isinstance(aper, EllipticalAnnulus)
        assert_allclose(aper.a_in, 1 / self.NONSQUARE_X_ARCSEC, rtol=1e-6)
        assert_allclose(aper.a_out, 2 / self.NONSQUARE_X_ARCSEC, rtol=1e-6)
        assert_allclose(aper.b_in, 1 / self.NONSQUARE_Y_ARCSEC, rtol=1e-6)
        assert_allclose(aper.b_out, 2 / self.NONSQUARE_Y_ARCSEC, rtol=1e-6)

    def test_pixel_annulus_to_sky_nonsquare(self):
        wcs = _make_nonsquare_wcs()
        pix = CircularAnnulus(PIX_CENTER, r_in=2, r_out=4)
        aper = pix.to_sky(wcs, as_ellipse=True)
        assert isinstance(aper, SkyEllipticalAnnulus)
        y_arcsec = self.NONSQUARE_Y_ARCSEC * u.arcsec
        x_arcsec = self.NONSQUARE_X_ARCSEC * u.arcsec
        assert_quantity_allclose(aper.a_in, 2 * y_arcsec, rtol=1e-6)
        assert_quantity_allclose(aper.a_out, 4 * y_arcsec, rtol=1e-6)
        assert_quantity_allclose(aper.b_in, 2 * x_arcsec, rtol=1e-6)
        assert_quantity_allclose(aper.b_out, 4 * x_arcsec, rtol=1e-6)

    def test_square_pixels_give_circle(self, simple_wcs):
        sky = SkyCircularAperture(CENTER, r=1 * u.arcsec)
        aper = sky.to_pixel(simple_wcs, as_ellipse=True)
        circle = sky.to_pixel(simple_wcs)
        assert_allclose(aper.a, aper.b, rtol=1e-8)
        assert_allclose(aper.a, circle.r, rtol=1e-8)

    @pytest.mark.parametrize('wcs_name', ['sheared_wcs', 'sip_wcs'])
    def test_pixel_ellipse_traces_sky_circle(self, wcs_name, request):
        # Every point on the pixel ellipse boundary must lie at the
        # circle radius from the center on the sky.
        wcs = request.getfixturevalue(wcs_name)
        radius = 2 * u.arcsec
        sky = SkyCircularAperture(CENTER, r=radius)
        aper = sky.to_pixel(wcs, as_ellipse=True)
        x, y = _pixel_ellipse_boundary(aper)
        separations = CENTER.separation(wcs.pixel_to_world(x, y))
        assert_quantity_allclose(separations, radius, rtol=1e-4)

    def test_sky_circle_roundtrip_through_ellipse(self, sheared_wcs):
        sky = SkyCircularAperture(CENTER, r=2 * u.arcsec)
        back = sky.to_pixel(sheared_wcs, as_ellipse=True).to_sky(sheared_wcs)
        assert_quantity_allclose(back.a, 2 * u.arcsec, rtol=1e-6)
        assert_quantity_allclose(back.b, 2 * u.arcsec, rtol=1e-6)

    def test_pixel_circle_roundtrip_through_ellipse(self, sheared_wcs):
        pix = CircularAperture(PIX_CENTER, r=3)
        back = pix.to_sky(sheared_wcs, as_ellipse=True).to_pixel(sheared_wcs)
        assert_allclose(back.a, 3, rtol=1e-6)
        assert_allclose(back.b, 3, rtol=1e-6)

    def test_multiple_positions(self):
        wcs = _make_nonsquare_wcs()
        positions = [(10.5, 10.5), (12.0, 8.0)]
        pix = CircularAperture(positions, r=2)
        aper = pix.to_sky(wcs, as_ellipse=True)
        assert aper.positions.shape == (2,)
        assert np.isscalar(aper.a.value)
        sky = SkyCircularAperture(wcs.pixel_to_world(*np.transpose(positions)),
                                  r=1 * u.arcsec)
        aper = sky.to_pixel(wcs, as_ellipse=True)
        assert aper.positions.shape == (2, 2)
        assert np.isscalar(aper.a)

    def test_as_ellipse_is_keyword_only(self, simple_wcs):
        sky = SkyCircularAperture(CENTER, r=1 * u.arcsec)
        match = 'positional argument'
        with pytest.raises(TypeError, match=match):
            sky.to_pixel(simple_wcs, True)  # noqa: FBT003


# Sky apertures and to_pixel keywords for the single-evaluation tests
_SKY_APERTURE_CASES = [
    pytest.param(SkyCircularAperture(CENTER, r=1 * u.arcsec), {},
                 id='circle'),
    pytest.param(SkyCircularAperture(CENTER, r=1 * u.arcsec),
                 {'as_ellipse': True}, id='circle_ellipse'),
    pytest.param(SkyCircularAnnulus(CENTER, r_in=1 * u.arcsec,
                                    r_out=2 * u.arcsec), {},
                 id='circle_annulus'),
    pytest.param(SkyCircularAnnulus(CENTER, r_in=1 * u.arcsec,
                                    r_out=2 * u.arcsec),
                 {'as_ellipse': True}, id='circle_annulus_ellipse'),
    pytest.param(SkyEllipticalAperture(CENTER, a=2 * u.arcsec,
                                       b=1 * u.arcsec,
                                       theta=30 * u.deg), {},
                 id='ellipse'),
    pytest.param(SkyEllipticalAnnulus(CENTER, a_in=1 * u.arcsec,
                                      a_out=2 * u.arcsec,
                                      b_out=1 * u.arcsec,
                                      theta=30 * u.deg), {},
                 id='ellipse_annulus'),
    pytest.param(SkyRectangularAperture(CENTER, w=2 * u.arcsec,
                                        h=1 * u.arcsec,
                                        theta=30 * u.deg), {},
                 id='rectangle'),
    pytest.param(SkyRectangularAnnulus(CENTER, w_in=1 * u.arcsec,
                                       w_out=2 * u.arcsec,
                                       h_out=1 * u.arcsec,
                                       theta=30 * u.deg), {},
                 id='rectangle_annulus'),
]


# Pixel apertures for the single-evaluation tests
_PIXEL_APERTURE_CASES = [
    pytest.param(CircularAperture(PIX_CENTER, r=3.0), id='circle'),
    pytest.param(CircularAnnulus(PIX_CENTER, r_in=3.0, r_out=5.0),
                 id='circle_annulus'),
    pytest.param(EllipticalAperture(PIX_CENTER, a=5.0, b=3.0, theta=0.5),
                 id='ellipse'),
    pytest.param(EllipticalAnnulus(PIX_CENTER, a_in=3.0, a_out=5.0,
                                   b_out=3.0, theta=0.5),
                 id='ellipse_annulus'),
    pytest.param(RectangularAperture(PIX_CENTER, w=5.0, h=3.0, theta=0.5),
                 id='rectangle'),
    pytest.param(RectangularAnnulus(PIX_CENTER, w_in=3.0, w_out=5.0,
                                    h_out=3.0, theta=0.5),
                 id='rectangle_annulus'),
]


class TestAnnulusIndependentInnerShape:
    """
    Tests that an annulus whose inner shape has a different aspect ratio
    from the outer one converts each shape independently.

    The annuli convert both shapes in one WCS evaluation. Each size must
    match the conversion of the matching simple aperture, and the angle
    is that of the outer shape.
    """

    @pytest.mark.parametrize('wcs_name', ['sheared_wcs', 'sip_wcs'])
    def test_elliptical_pixel_to_sky(self, wcs_name, request):
        wcs = request.getfixturevalue(wcs_name)
        theta = 45 * u.deg
        annulus = EllipticalAnnulus(PIX_CENTER, a_in=2, a_out=4, b_out=2,
                                    b_in=1.5, theta=theta)
        outer = EllipticalAperture(PIX_CENTER, a=4, b=2, theta=theta)
        inner = EllipticalAperture(PIX_CENTER, a=2, b=1.5, theta=theta)
        sky = annulus.to_sky(wcs)
        sky_outer = outer.to_sky(wcs)
        sky_inner = inner.to_sky(wcs)
        assert_quantity_allclose(sky.a_out, sky_outer.a, rtol=1e-12)
        assert_quantity_allclose(sky.b_out, sky_outer.b, rtol=1e-12)
        assert_quantity_allclose(sky.a_in, sky_inner.a, rtol=1e-12)
        assert_quantity_allclose(sky.b_in, sky_inner.b, rtol=1e-12)
        assert_quantity_allclose(sky.theta, sky_outer.theta,
                                 atol=1e-10 * u.deg)

    @pytest.mark.parametrize('wcs_name', ['sheared_wcs', 'sip_wcs'])
    def test_elliptical_sky_to_pixel(self, wcs_name, request):
        wcs = request.getfixturevalue(wcs_name)
        theta = 30 * u.deg
        annulus = SkyEllipticalAnnulus(CENTER, a_in=1 * u.arcsec,
                                       a_out=2 * u.arcsec,
                                       b_out=1 * u.arcsec,
                                       b_in=0.75 * u.arcsec, theta=theta)
        outer = SkyEllipticalAperture(CENTER, a=2 * u.arcsec,
                                      b=1 * u.arcsec, theta=theta)
        inner = SkyEllipticalAperture(CENTER, a=1 * u.arcsec,
                                      b=0.75 * u.arcsec, theta=theta)
        pix = annulus.to_pixel(wcs)
        pix_outer = outer.to_pixel(wcs)
        pix_inner = inner.to_pixel(wcs)
        assert_allclose(pix.a_out, pix_outer.a, rtol=1e-12)
        assert_allclose(pix.b_out, pix_outer.b, rtol=1e-12)
        assert_allclose(pix.a_in, pix_inner.a, rtol=1e-12)
        assert_allclose(pix.b_in, pix_inner.b, rtol=1e-12)
        assert_quantity_allclose(pix.theta, pix_outer.theta,
                                 atol=1e-10 * u.deg)

    @pytest.mark.parametrize('wcs_name', ['sheared_wcs', 'sip_wcs'])
    def test_rectangular_pixel_to_sky(self, wcs_name, request):
        wcs = request.getfixturevalue(wcs_name)
        theta = 45 * u.deg
        annulus = RectangularAnnulus(PIX_CENTER, w_in=2, w_out=4, h_out=2,
                                     h_in=1.5, theta=theta)
        outer = RectangularAperture(PIX_CENTER, w=4, h=2, theta=theta)
        inner = RectangularAperture(PIX_CENTER, w=2, h=1.5, theta=theta)
        sky = annulus.to_sky(wcs)
        sky_outer = outer.to_sky(wcs)
        sky_inner = inner.to_sky(wcs)
        assert_quantity_allclose(sky.w_out, sky_outer.w, rtol=1e-12)
        assert_quantity_allclose(sky.h_out, sky_outer.h, rtol=1e-12)
        assert_quantity_allclose(sky.w_in, sky_inner.w, rtol=1e-12)
        assert_quantity_allclose(sky.h_in, sky_inner.h, rtol=1e-12)
        assert_quantity_allclose(sky.theta, sky_outer.theta,
                                 atol=1e-10 * u.deg)

    @pytest.mark.parametrize('wcs_name', ['sheared_wcs', 'sip_wcs'])
    def test_rectangular_sky_to_pixel(self, wcs_name, request):
        wcs = request.getfixturevalue(wcs_name)
        theta = 30 * u.deg
        annulus = SkyRectangularAnnulus(CENTER, w_in=1 * u.arcsec,
                                        w_out=2 * u.arcsec,
                                        h_out=1 * u.arcsec,
                                        h_in=0.75 * u.arcsec, theta=theta)
        outer = SkyRectangularAperture(CENTER, w=2 * u.arcsec,
                                       h=1 * u.arcsec, theta=theta)
        inner = SkyRectangularAperture(CENTER, w=1 * u.arcsec,
                                       h=0.75 * u.arcsec, theta=theta)
        pix = annulus.to_pixel(wcs)
        pix_outer = outer.to_pixel(wcs)
        pix_inner = inner.to_pixel(wcs)
        assert_allclose(pix.w_out, pix_outer.w, rtol=1e-12)
        assert_allclose(pix.h_out, pix_outer.h, rtol=1e-12)
        assert_allclose(pix.w_in, pix_inner.w, rtol=1e-12)
        assert_allclose(pix.h_in, pix_inner.h, rtol=1e-12)
        assert_quantity_allclose(pix.theta, pix_outer.theta,
                                 atol=1e-10 * u.deg)


class TestSingleWCSEvaluation:
    """
    Tests that each conversion evaluates the WCS as few times as
    possible.

    A sky-to-pixel conversion inverts the WCS once, for the aperture
    positions, reuses that pixel position for the shape conversion,
    and evaluates the low-level forward transform once for the local
    Jacobian. A pixel-to-sky conversion evaluates the high-level forward
    transform once for the positions and the low-level one once for the
    Jacobian. The annuli convert both of their shapes within those same
    evaluations.
    """

    @pytest.mark.parametrize(('aperture', 'kwargs'), _SKY_APERTURE_CASES)
    @pytest.mark.parametrize('wcs_name', ['simple_wcs', 'sip_wcs'])
    def test_sky_to_pixel(self, aperture, kwargs, wcs_name, request):
        real_wcs = request.getfixturevalue(wcs_name)
        expected = aperture.to_pixel(real_wcs, **kwargs)
        wcs = CountingWCS(real_wcs)
        result = aperture.to_pixel(wcs, **kwargs)
        assert wcs.n_world_to_pixel == 1
        assert wcs.n_pixel_to_world_values == 1
        assert wcs.n_pixel_to_world == 0
        assert type(result) is type(expected)
        assert_allclose(result.positions, expected.positions, atol=1e-8)

    @pytest.mark.parametrize('aperture', _PIXEL_APERTURE_CASES)
    @pytest.mark.parametrize('wcs_name', ['simple_wcs', 'sip_wcs'])
    def test_pixel_to_sky(self, aperture, wcs_name, request):
        real_wcs = request.getfixturevalue(wcs_name)
        expected = aperture.to_sky(real_wcs)
        wcs = CountingWCS(real_wcs)
        result = aperture.to_sky(wcs)
        assert wcs.n_world_to_pixel == 0
        assert wcs.n_pixel_to_world == 1
        assert wcs.n_pixel_to_world_values == 1
        assert type(result) is type(expected)
        assert result.positions.separation(expected.positions).arcsec < 1e-9
