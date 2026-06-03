# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the _wcs_helpers module.
"""

import astropy.units as u
import numpy as np
import pytest
from astropy.coordinates import Angle, SkyCoord
from astropy.io.fits import Header
from astropy.wcs import WCS as APWCS
from numpy.testing import assert_allclose

from photutils.utils._wcs_helpers import (compute_local_wcs_jacobian,
                                          jacobian_pixel_to_sky_mean_scale,
                                          jacobian_sky_to_pixel_mean_scale,
                                          pixel_shape_to_sky_svd,
                                          pixel_to_sky_mean_scale,
                                          sky_shape_to_pixel_svd,
                                          sky_to_pixel_mean_scale,
                                          wcs_pixel_scale_angle)
from photutils.utils.tests.conftest import WCS_CDELT_ARCSEC, WCS_CENTER


@pytest.fixture
def center_xy_coord(simple_wcs):
    """
    Return the center (x, y) tuple at CRPIX of the simple WCS.
    """
    x, y = simple_wcs.world_to_pixel(WCS_CENTER)
    return (x, y)


class TestComputeLocalWCSJacobian:
    """
    Tests for `compute_local_wcs_jacobian`.
    """

    def test_shape(self, simple_wcs):
        """
        The Jacobian must be a 2x2 array.
        """
        jac = compute_local_wcs_jacobian(WCS_CENTER, simple_wcs)
        assert jac.shape == (2, 2)

    def test_simple_wcs_diagonal(self, simple_wcs):
        """
        For an axis-aligned TAN WCS the Jacobian should be nearly
        diagonal with magnitudes ~ 1/WCS_CDELT_ARCSEC.
        """
        jac = compute_local_wcs_jacobian(WCS_CENTER, simple_wcs)
        # Off-diagonal elements should be near zero
        assert_allclose(jac[0, 1], 0.0, atol=1e-4)
        assert_allclose(jac[1, 0], 0.0, atol=1e-4)
        # Diagonal: RA axis (pix/arcsec) is negative (RA increases left)
        expected_scale = 1.0 / WCS_CDELT_ARCSEC
        assert_allclose(np.abs(jac[0, 0]), expected_scale)
        assert_allclose(np.abs(jac[1, 1]), expected_scale)

    def test_rotated_wcs(self, rotated_wcs):
        """
        For a rotated WCS the off-diagonal elements should be nonzero,
        but the singular values should still match 1/WCS_CDELT_ARCSEC.
        """
        jac = compute_local_wcs_jacobian(WCS_CENTER, rotated_wcs)
        sv = np.linalg.svd(jac, compute_uv=False)
        expected_scale = 1.0 / WCS_CDELT_ARCSEC
        assert_allclose(sv, expected_scale, rtol=1e-6)

    def test_sip_wcs(self, sip_wcs):
        """
        For a SIP WCS the Jacobian should still be close to the
        undistorted value near the reference pixel.
        """
        jac = compute_local_wcs_jacobian(WCS_CENTER, sip_wcs)
        sv = np.linalg.svd(jac, compute_uv=False)
        expected_scale = 1.0 / WCS_CDELT_ARCSEC
        assert_allclose(sv, expected_scale, rtol=1e-5)

    def test_inverse_of_forward(self, simple_wcs):
        """
        The Jacobian should be the inverse of the forward
        d(sky)/d(pixel) matrix derived from 1-pixel offsets.
        """
        jac = compute_local_wcs_jacobian(WCS_CENTER, simple_wcs)
        # A 1-pixel step should map to ~CDELT arcsec in sky
        forward = np.linalg.inv(jac)
        # Diagonal magnitudes should be ~WCS_CDELT_ARCSEC
        assert_allclose(np.abs(forward[0, 0]), WCS_CDELT_ARCSEC)
        assert_allclose(np.abs(forward[1, 1]), WCS_CDELT_ARCSEC)

    def test_determinant_sign(self, simple_wcs):
        """
        Standard WCS (RA increasing to the left) should have negative
        determinant.
        """
        jac = compute_local_wcs_jacobian(WCS_CENTER, simple_wcs)
        assert np.linalg.det(jac) < 0


class TestWcsPixelScaleAngle:
    """
    Tests for `wcs_pixel_scale_angle`.
    """

    def test_return_types(self, simple_wcs):
        """
        Should return (tuple, float, Angle).
        """
        xy_coord, scale, angle = wcs_pixel_scale_angle(
            WCS_CENTER, simple_wcs)
        assert isinstance(xy_coord, tuple)
        assert isinstance(scale, float)
        assert isinstance(angle, Angle)

    def test_simple_wcs_scale(self, simple_wcs):
        """
        For a simple TAN WCS, scale should equal CDELT in arcsec/pixel.
        """
        _, scale, _ = wcs_pixel_scale_angle(WCS_CENTER, simple_wcs)
        assert_allclose(scale, WCS_CDELT_ARCSEC)

    def test_simple_wcs_angle(self, simple_wcs):
        """
        For an axis-aligned TAN WCS with CDELT=[-c, c], North is along
        +y, so the angle should be ~90 degrees.
        """
        _, _, angle = wcs_pixel_scale_angle(WCS_CENTER, simple_wcs)
        assert_allclose(angle.deg, 90.0)

    def test_angle_wrapped(self, simple_wcs):
        """
        The angle should be in [0, 360) degrees.
        """
        _, _, angle = wcs_pixel_scale_angle(WCS_CENTER, simple_wcs)
        assert 0.0 <= angle.deg < 360.0

    def test_rotated_wcs_angle(self, rotated_wcs):
        """
        For a 25-degree rotated WCS, the North angle should shift by
        ~25 degrees from the axis-aligned value (~90 deg).
        """
        _, _, angle = wcs_pixel_scale_angle(WCS_CENTER, rotated_wcs)
        # The rotation should be about 90 - 25 = 65 degrees
        assert_allclose(angle.deg, 90.0 - 25.0)

    def test_rotated_wcs_scale(self, rotated_wcs):
        """
        Rotation should not change the pixel scale.
        """
        _, scale, _ = wcs_pixel_scale_angle(WCS_CENTER, rotated_wcs)
        assert_allclose(scale, WCS_CDELT_ARCSEC)

    def test_nonsquare_wcs_scale(self, nonsquare_wcs):
        """
        For non-square pixels the scale should be the geometric mean.
        """
        _, scale, _ = wcs_pixel_scale_angle(WCS_CENTER, nonsquare_wcs)
        expected = np.sqrt(0.03 * 0.05) * 3600
        assert_allclose(scale, expected, rtol=1e-5)

    def test_pixel_coordinate(self, simple_wcs):
        """
        The returned xy_coord should match world_to_pixel.
        """
        xy_coord, _, _ = wcs_pixel_scale_angle(WCS_CENTER, simple_wcs)
        x_exp, y_exp = simple_wcs.world_to_pixel(WCS_CENTER)
        assert_allclose(xy_coord[0], x_exp)
        assert_allclose(xy_coord[1], y_exp)

    def test_off_center_position(self, simple_wcs):
        """
        Test a position away from the WCS reference pixel.
        """
        skycoord = SkyCoord(100.5 * u.deg, 30.5 * u.deg)
        _, scale, angle = wcs_pixel_scale_angle(skycoord, simple_wcs)
        assert scale > 0
        assert 0.0 <= angle.deg < 360.0


class TestJacobianMeanScale:
    """
    Tests for `jacobian_sky_to_pixel_mean_scale` and
    `jacobian_pixel_to_sky_mean_scale`.
    """

    def test_sky_to_pixel_return_types(self, simple_wcs):
        """
        Should return (tuple, float).
        """
        pix_position, scale = jacobian_sky_to_pixel_mean_scale(
            WCS_CENTER, simple_wcs)
        assert isinstance(pix_position, tuple)
        assert isinstance(scale, (float, np.floating))

    def test_pixel_to_sky_return_types(self, simple_wcs, center_xy_coord):
        """
        Should return (SkyCoord, float).
        """
        sky_position, scale = jacobian_pixel_to_sky_mean_scale(
            center_xy_coord, simple_wcs)
        assert isinstance(sky_position, SkyCoord)
        assert isinstance(scale, (float, np.floating))

    def test_sky_to_pixel_simple_scale(self, simple_wcs):
        """
        For an isotropic WCS, the mean scale should equal
        1/WCS_CDELT_ARCSEC.
        """
        _, scale = jacobian_sky_to_pixel_mean_scale(
            WCS_CENTER, simple_wcs)
        assert_allclose(scale, 1.0 / WCS_CDELT_ARCSEC)

    def test_pixel_to_sky_simple_scale(self, simple_wcs, center_xy_coord):
        """
        For an isotropic WCS, the mean scale should equal WCS_CDELT_ARCSEC.
        """
        _, scale = jacobian_pixel_to_sky_mean_scale(
            center_xy_coord, simple_wcs)
        assert_allclose(scale, WCS_CDELT_ARCSEC)

    def test_roundtrip_scale(self, simple_wcs):
        """
        Sky -> pixel mean_scale * pixel -> sky mean_scale should ~ 1.
        """
        center_pix, s2p = jacobian_sky_to_pixel_mean_scale(
            WCS_CENTER, simple_wcs)
        _, p2s = jacobian_pixel_to_sky_mean_scale(center_pix, simple_wcs)
        assert_allclose(s2p * p2s, 1.0)

    def test_sip_wcs_positive(self, sip_wcs):
        """
        Mean scale should be positive for distorted WCS.
        """
        _, scale = jacobian_sky_to_pixel_mean_scale(
            WCS_CENTER, sip_wcs)
        assert scale > 0

    def test_center_coordinates(self, simple_wcs):
        """
        The returned pix_position should match world_to_pixel.
        """
        pix_position, _ = jacobian_sky_to_pixel_mean_scale(
            WCS_CENTER, simple_wcs)
        x_exp, y_exp = simple_wcs.world_to_pixel(WCS_CENTER)
        assert_allclose(pix_position[0], x_exp)
        assert_allclose(pix_position[1], y_exp)


class TestDispatchMeanScale:
    """
    Tests for `sky_to_pixel_mean_scale` and `pixel_to_sky_mean_scale`
    dispatch helpers.
    """

    def test_no_distortion_returns(self, simple_wcs):
        """
        Should return (tuple, float) for non-distorted WCS.
        """
        pix_position, scale = sky_to_pixel_mean_scale(WCS_CENTER, simple_wcs)
        assert isinstance(pix_position, tuple)
        assert isinstance(scale, float)

    def test_distortion_returns(self, sip_wcs):
        """
        Should return (tuple, float/np.floating) for distorted WCS.
        """
        pix_position, scale = sky_to_pixel_mean_scale(WCS_CENTER, sip_wcs)
        assert isinstance(pix_position, tuple)
        assert isinstance(scale, (float, np.floating))

    def test_no_distortion_scale(self, simple_wcs):
        """
        For a simple WCS, mean scale should be 1/WCS_CDELT_ARCSEC.
        """
        _, scale = sky_to_pixel_mean_scale(WCS_CENTER, simple_wcs)
        assert_allclose(scale, 1.0 / WCS_CDELT_ARCSEC)

    def test_distortion_scale(self, sip_wcs):
        """
        For a SIP WCS near the reference pixel, the mean scale should be
        close to the undistorted value.
        """
        _, scale = sky_to_pixel_mean_scale(WCS_CENTER, sip_wcs)
        assert_allclose(scale, 1.0 / WCS_CDELT_ARCSEC, rtol=1e-6)

    def test_pixel_to_sky_no_distortion(self, simple_wcs, center_xy_coord):
        """
        For a simple WCS, pixel_to_sky mean scale should be
        WCS_CDELT_ARCSEC.
        """
        sky_position, scale = pixel_to_sky_mean_scale(
            center_xy_coord, simple_wcs)
        assert isinstance(sky_position, SkyCoord)
        assert_allclose(scale, WCS_CDELT_ARCSEC)

    def test_pixel_to_sky_distortion(self, sip_wcs):
        """
        For a SIP WCS, pixel_to_sky mean scale should be close to
        WCS_CDELT_ARCSEC near the reference pixel.
        """
        xy_coord = (9.5, 9.5)
        sky_position, scale = pixel_to_sky_mean_scale(xy_coord, sip_wcs)
        assert isinstance(sky_position, SkyCoord)
        assert_allclose(scale, WCS_CDELT_ARCSEC, rtol=1e-6)

    def test_roundtrip(self, simple_wcs):
        """
        Sky -> pixel mean_scale * pixel -> sky mean_scale should ~ 1.
        """
        center_pix, s2p = sky_to_pixel_mean_scale(
            WCS_CENTER, simple_wcs)
        _, p2s = pixel_to_sky_mean_scale(center_pix, simple_wcs)
        assert_allclose(s2p * p2s, 1.0)

    def test_roundtrip_sip(self, sip_wcs):
        """
        Roundtrip with SIP WCS should give product ~ 1.
        """
        center_pix, s2p = sky_to_pixel_mean_scale(
            WCS_CENTER, sip_wcs)
        _, p2s = pixel_to_sky_mean_scale(center_pix, sip_wcs)
        assert_allclose(s2p * p2s, 1.0)

    def test_consistency_offset_jacobian(self, simple_wcs):
        """
        For a simple WCS, both mean-scale paths should agree.
        """
        # Offset path (via dispatch)
        c1, s1 = sky_to_pixel_mean_scale(WCS_CENTER, simple_wcs)

        # Jacobian path (direct call)
        c2, s2 = jacobian_sky_to_pixel_mean_scale(
            WCS_CENTER, simple_wcs)

        assert_allclose(c1[0], c2[0])
        assert_allclose(c1[1], c2[1])
        assert_allclose(s1, s2)


class TestGWCSDispatch:
    """
    Test that dispatch helpers correctly handle WCS objects without
    the ``has_distortion`` attribute (e.g., GWCS), defaulting to the
    Jacobian path.
    """

    @pytest.fixture
    def mock_gwcs(self, simple_wcs):
        """
        Create a mock WCS that wraps a simple WCS but has no
        has_distortion attribute, simulating GWCS behavior.
        """
        class MockGWCS:
            def __init__(self, real_wcs):
                self._wcs = real_wcs

            def world_to_pixel(self, *args, **kwargs):
                return self._wcs.world_to_pixel(*args, **kwargs)

            def pixel_to_world(self, *args, **kwargs):
                return self._wcs.pixel_to_world(*args, **kwargs)

        return MockGWCS(simple_wcs)

    def test_no_has_distortion_attr(self, mock_gwcs):
        """
        The mock should not have has_distortion.
        """
        assert not hasattr(mock_gwcs, 'has_distortion')

    def test_sky_to_pixel_mean_scale_uses_jacobian(self, mock_gwcs):
        """
        Without has_distortion, should use the Jacobian path.
        """
        pix_position, scale = sky_to_pixel_mean_scale(WCS_CENTER, mock_gwcs)
        assert isinstance(pix_position, tuple)
        assert scale > 0

    def test_pixel_to_sky_mean_scale_uses_jacobian(self, mock_gwcs):
        """
        Without has_distortion, should use the Jacobian path.
        """
        xy_coord = (9.5, 9.5)
        sky_position, scale = pixel_to_sky_mean_scale(xy_coord, mock_gwcs)
        assert isinstance(sky_position, SkyCoord)
        assert scale > 0

    def test_gwcs_scale_matches_simple(self, mock_gwcs, simple_wcs):
        """
        The mock GWCS (Jacobian path) should give scales close to the
        simple WCS (offset path).
        """
        _, scale_offset = sky_to_pixel_mean_scale(
            WCS_CENTER, simple_wcs)
        _, scale_jac = sky_to_pixel_mean_scale(
            WCS_CENTER, mock_gwcs)
        assert_allclose(scale_offset, scale_jac)


class TestNonsquarePixels:
    """
    Tests for WCS with non-square pixels to verify scale separation.
    """

    def test_nonsquare_mean_scale(self, nonsquare_wcs):
        """
        The mean scale should be near the arithmetic mean of the two
        singular values (1/cdelt_x and 1/cdelt_y in pix/arcsec).
        """
        _, scale = jacobian_sky_to_pixel_mean_scale(
            WCS_CENTER, nonsquare_wcs)
        # Arithmetic mean of singular values (SVD):
        # 1/cdelt_x_arcsec and 1/cdelt_y_arcsec
        cdelt_x = 0.03 * 3600
        cdelt_y = 0.05 * 3600
        expected = 0.5 * (1.0 / cdelt_x + 1.0 / cdelt_y)
        assert_allclose(scale, expected, rtol=1e-6)


class TestSVDShapeConversions:
    """
    Tests for `pixel_shape_to_sky_svd` and `sky_shape_to_pixel_svd`.
    """

    def test_pixel_to_sky_return_types(self, simple_wcs, center_xy_coord):
        """
        Should return (SkyCoord, float, float, Angle).
        """
        center, w, h, angle = pixel_shape_to_sky_svd(
            center_xy_coord, simple_wcs, 10.0, 5.0, 0.5)
        assert isinstance(center, SkyCoord)
        assert isinstance(w, (float, np.floating))
        assert isinstance(h, (float, np.floating))
        assert isinstance(angle, Angle)

    def test_sky_to_pixel_return_types(self, simple_wcs):
        """
        Should return (tuple, float, float, Angle).
        """
        center, w, h, angle = sky_shape_to_pixel_svd(
            WCS_CENTER, simple_wcs, 36.0, 18.0, 0.5)
        assert isinstance(center, tuple)
        assert isinstance(w, (float, np.floating))
        assert isinstance(h, (float, np.floating))
        assert isinstance(angle, Angle)

    def test_roundtrip_sky_pixel_sky(self, simple_wcs):
        """
        Sky -> pixel -> sky should recover the original ellipse.
        """
        sky_w, sky_h, sky_a = 36.0, 18.0, 0.5
        center_pix, pw, ph, pa = sky_shape_to_pixel_svd(
            WCS_CENTER, simple_wcs, sky_w, sky_h, sky_a)
        _, rw, rh, ra = pixel_shape_to_sky_svd(
            center_pix, simple_wcs, pw, ph, pa.rad)
        assert_allclose(rw, sky_w, rtol=1e-6)
        assert_allclose(rh, sky_h, rtol=1e-6)
        assert_allclose(ra.rad, sky_a, rtol=1e-4)

    def test_roundtrip_pixel_sky_pixel(self, simple_wcs, center_xy_coord):
        """
        Pixel -> sky -> pixel should recover the original ellipse.
        """
        pix_w, pix_h, pix_a = 10.0, 5.0, 0.3
        _, sw, sh, sa = pixel_shape_to_sky_svd(
            center_xy_coord, simple_wcs, pix_w, pix_h, pix_a)
        _, rw, rh, ra = sky_shape_to_pixel_svd(
            WCS_CENTER, simple_wcs, sw, sh, sa.rad)
        assert_allclose(rw, pix_w, rtol=1e-6)
        assert_allclose(rh, pix_h, rtol=1e-6)
        assert_allclose(ra.rad, pix_a, rtol=1e-4)

    def test_simple_wcs_width_height_scale(self, simple_wcs, center_xy_coord):
        """
        For a simple WCS, pixel dimensions should scale by
        WCS_CDELT_ARCSEC.
        """
        pix_w, pix_h = 10.0, 5.0
        _, sw, sh, _ = pixel_shape_to_sky_svd(
            center_xy_coord, simple_wcs, pix_w, pix_h, 0.0)
        assert_allclose(sw, pix_w * WCS_CDELT_ARCSEC, rtol=1e-5)
        assert_allclose(sh, pix_h * WCS_CDELT_ARCSEC, rtol=1e-5)

    def test_height_larger_than_width(self, simple_wcs, center_xy_coord):
        """
        When height > width, the SVD should still correctly assign
        widths and heights.
        """
        pix_w, pix_h = 5.0, 10.0
        _, sw, sh, _ = pixel_shape_to_sky_svd(
            center_xy_coord, simple_wcs, pix_w, pix_h, 0.0)
        # Width should be smaller than height in sky coords too
        assert sw < sh

    def test_sip_wcs_positive_sizes(self, sip_wcs):
        """
        Sizes should be positive for distorted WCS.
        """
        xy_coord = (9.5, 9.5)
        _, sw, sh, _ = pixel_shape_to_sky_svd(
            xy_coord, sip_wcs, 8.0, 4.0, 0.0)
        assert sw > 0
        assert sh > 0

    def test_sip_wcs_roundtrip(self, sip_wcs):
        """
        Roundtrip with SIP WCS should recover the original ellipse.
        """
        sky_w, sky_h, sky_a = 0.36, 0.18, 0.7
        center_pix, pw, ph, pa = sky_shape_to_pixel_svd(
            WCS_CENTER, sip_wcs, sky_w, sky_h, sky_a)
        _, rw, rh, ra = pixel_shape_to_sky_svd(
            center_pix, sip_wcs, pw, ph, pa.rad)
        assert_allclose(rw, sky_w, rtol=1e-5)
        assert_allclose(rh, sky_h, rtol=1e-5)
        assert_allclose(ra.rad, sky_a, rtol=1e-4)

    def test_angle_wrapped(self, simple_wcs, center_xy_coord):
        """
        The output angle should be in [0, 360) degrees.
        """
        _, _, _, angle = pixel_shape_to_sky_svd(
            center_xy_coord, simple_wcs, 10.0, 5.0, 0.5)
        assert 0.0 <= angle.deg < 360.0


def _make_sip_wcs(ra_deg, dec_deg):
    """
    Build a small TAN-SIP WCS centered at (ra_deg, dec_deg).

    The SIP terms are tiny but nonzero, ensuring that
    ``compute_local_wcs_jacobian`` is exercised (the jacobian is only
    used for distorted WCS).
    """
    header = Header()
    header['NAXIS'] = 2
    header['NAXIS1'] = 20
    header['NAXIS2'] = 20
    header['CRPIX1'] = 10.5
    header['CRPIX2'] = 10.5
    header['CRVAL1'] = ra_deg
    header['CRVAL2'] = dec_deg
    header['CTYPE1'] = 'RA---TAN-SIP'
    header['CTYPE2'] = 'DEC--TAN-SIP'
    cdelt = WCS_CDELT_ARCSEC / 3600.0
    header['CD1_1'] = -cdelt
    header['CD1_2'] = 0.0
    header['CD2_1'] = 0.0
    header['CD2_2'] = cdelt
    header['A_ORDER'] = 2
    header['A_2_0'] = 1e-6
    header['B_ORDER'] = 2
    header['B_0_2'] = 1e-6
    return APWCS(header)


POLE_DEC_LIST = [80.0, 89.0, 89.99, -80.0, -89.99]
RA_WRAP_LIST = [0.0, 0.001, 359.999]


class TestComputeLocalWCSJacobianPoleAndWrap:
    """
    Regression tests for ``compute_local_wcs_jacobian`` at WCS centers
    where the flat-sky finite-difference formula failed.

    The following cases are covered:

    * RA = 0 (any of the three sample points crosses the 0 / 360 cut)
    * High |declination| (small ``cos(dec)`` makes the small-angle
      ``xi = cos(dec) * dRA`` approximation fail, and the (xi, eta)
      sampling becomes wildly nonlinear near the pole)

    These tests pin the near-singular values of the Jacobian to the
    expected ``1 / WCS_CDELT_ARCSEC`` value.
    """

    @pytest.mark.parametrize('center_dec', POLE_DEC_LIST)
    def test_jacobian_well_conditioned_at_pole(self, center_dec):
        wcs = _make_sip_wcs(0.0, center_dec)
        sc = SkyCoord(0 * u.deg, center_dec * u.deg)
        jac = compute_local_wcs_jacobian(sc, wcs)
        sv = np.linalg.svd(jac, compute_uv=False)
        assert_allclose(sv, 1.0 / WCS_CDELT_ARCSEC, rtol=1e-3)

    @pytest.mark.parametrize('center_ra', RA_WRAP_LIST)
    def test_jacobian_well_conditioned_at_ra_wrap(self, center_ra):
        wcs = _make_sip_wcs(center_ra, 30.0)
        sc = SkyCoord(center_ra * u.deg, 30 * u.deg)
        jac = compute_local_wcs_jacobian(sc, wcs)
        sv = np.linalg.svd(jac, compute_uv=False)
        assert_allclose(sv, 1.0 / WCS_CDELT_ARCSEC, rtol=1e-3)

    @pytest.mark.parametrize('center_dec', POLE_DEC_LIST)
    def test_jacobian_parity_correct_at_pole(self, center_dec):
        """
        The determinant of the Jacobian (parity) must be negative for
        a standard RA-increases-to-the-left WCS at all declinations,
        including near the poles.
        """
        wcs = _make_sip_wcs(0.0, center_dec)
        sc = SkyCoord(0 * u.deg, center_dec * u.deg)
        jac = compute_local_wcs_jacobian(sc, wcs)
        assert np.linalg.det(jac) < 0


class TestSkyEllipseSVDPoleAndWrap:
    """
    End-to-end regression tests via ``sky_shape_to_pixel_svd``.

    A directed sky shape converted at the troublesome positions must
    round-trip back to itself.
    """

    @pytest.mark.parametrize('center_dec', POLE_DEC_LIST)
    def test_roundtrip_at_pole(self, center_dec):
        wcs = _make_sip_wcs(0.0, center_dec)
        sc = SkyCoord(0 * u.deg, center_dec * u.deg)
        center, pw, ph, pa = sky_shape_to_pixel_svd(
            sc, wcs, 2.0, 1.0, np.deg2rad(30))
        assert pw > 1.0
        assert ph > 1.0
        _, sw, sh, _ = pixel_shape_to_sky_svd(
            center, wcs, pw, ph, pa.to(u.rad).value)
        assert_allclose(sw, 2.0, rtol=1e-3)
        assert_allclose(sh, 1.0, rtol=1e-3)

    @pytest.mark.parametrize('center_ra', RA_WRAP_LIST)
    def test_roundtrip_at_ra_wrap(self, center_ra):
        wcs = _make_sip_wcs(center_ra, 30.0)
        sc = SkyCoord(center_ra * u.deg, 30 * u.deg)
        center, pw, ph, pa = sky_shape_to_pixel_svd(
            sc, wcs, 2.0, 1.0, np.deg2rad(30))
        assert pw > 1.0
        assert ph > 1.0
        _, sw, sh, _ = pixel_shape_to_sky_svd(
            center, wcs, pw, ph, pa.to(u.rad).value)
        assert_allclose(sw, 2.0, rtol=1e-3)
        assert_allclose(sh, 1.0, rtol=1e-3)
