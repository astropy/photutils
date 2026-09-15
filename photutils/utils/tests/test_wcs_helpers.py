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

from photutils.datasets import make_gwcs
from photutils.utils._optional_deps import HAS_GWCS
from photutils.utils._wcs_helpers import (compute_local_wcs_jacobian,
                                          compute_pixel_scale_angles,
                                          compute_pixel_to_sky_jacobians,
                                          compute_pixel_to_sky_mean_scales,
                                          pixel_shape_to_sky_svd,
                                          pixel_to_sky_mean_scale,
                                          pixel_to_sky_svd_scales,
                                          sky_shape_to_pixel_svd,
                                          sky_to_pixel_mean_scale,
                                          sky_to_pixel_svd_scales,
                                          wcs_pixel_scale_angle)
from photutils.utils.tests.conftest import WCS_CDELT_ARCSEC, WCS_CENTER

# WCS centers that historically broke the flat-sky finite-difference
# Jacobian and SVD shape conversions:
#
# * near the celestial poles, where the small ``cos(dec)`` factor makes
#   the ``xi = cos(dec) * dRA`` small-angle approximation fail and the
#   (xi, eta) sampling becomes wildly nonlinear, and
#
# * across the RA = 0 / 360 wraparound, where naive longitude
#   subtraction crosses the 0 / 360 cut.
TROUBLESOME_CENTERS = [
    pytest.param(0.0, 80.0, id='pole_dec=80'),
    pytest.param(0.0, 89.0, id='pole_dec=89'),
    pytest.param(0.0, 89.99, id='pole_dec=89.99'),
    pytest.param(0.0, -80.0, id='pole_dec=-80'),
    pytest.param(0.0, -89.99, id='pole_dec=-89.99'),
    pytest.param(0.0, 30.0, id='ra_wrap=0'),
    pytest.param(0.001, 30.0, id='ra_wrap=0.001'),
    pytest.param(359.999, 30.0, id='ra_wrap=359.999'),
]


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

    @pytest.mark.parametrize(('center_ra', 'center_dec'), TROUBLESOME_CENTERS)
    def test_well_conditioned_at_pole_and_wrap(self, center_ra, center_dec):
        """
        Test that the near-singular values of the Jacobian match the
        expected ``1 / WCS_CDELT_ARCSEC`` value even near the celestial
        poles and across the RA = 0 / 360 wraparound.
        """
        wcs = _make_sip_wcs(center_ra, center_dec)
        skycoord = SkyCoord(center_ra * u.deg, center_dec * u.deg)
        jac = compute_local_wcs_jacobian(skycoord, wcs)
        sv = np.linalg.svd(jac, compute_uv=False)
        assert_allclose(sv, 1.0 / WCS_CDELT_ARCSEC, rtol=1e-3)

    @pytest.mark.parametrize(('center_ra', 'center_dec'), TROUBLESOME_CENTERS)
    def test_parity_at_pole_and_wrap(self, center_ra, center_dec):
        """
        Test that the determinant (parity) of the Jacobian stays
        negative for a standard RA-increases-to-the-left WCS at all
        declinations and RA values, including near the poles and across
        the wraparound.
        """
        wcs = _make_sip_wcs(center_ra, center_dec)
        skycoord = SkyCoord(center_ra * u.deg, center_dec * u.deg)
        jac = compute_local_wcs_jacobian(skycoord, wcs)
        assert np.linalg.det(jac) < 0


def _make_rotated_wcs(rotation_deg, scale_arcsec=0.25,
                      crval=(150.0, 30.0)):
    wcs = APWCS(naxis=2)
    wcs.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    wcs.wcs.crpix = [50.0, 50.0]
    wcs.wcs.crval = list(crval)
    theta = np.deg2rad(rotation_deg)
    scale = scale_arcsec / 3600.0
    wcs.wcs.cd = scale * np.array(
        [[-np.cos(theta), np.sin(theta)],
         [np.sin(theta), np.cos(theta)]])
    return wcs


def test_compute_pixel_to_sky_jacobians():
    """
    Test the vectorized forward Jacobians against the inverse of the
    scalar compute_local_wcs_jacobian at several positions.
    """
    wcs = _make_rotated_wcs(30.0)
    x = np.array([10.0, 50.0, 80.3])
    y = np.array([20.0, 50.0, 61.7])

    jacs = compute_pixel_to_sky_jacobians(x, y, wcs)
    assert jacs.shape == (3, 2, 2)

    for i in range(x.size):
        skycoord = wcs.pixel_to_world(x[i], y[i])
        jac_inv = compute_local_wcs_jacobian(skycoord, wcs)
        assert_allclose(jacs[i], np.linalg.inv(jac_inv), rtol=1e-4)


def test_compute_pixel_to_sky_jacobians_scale():
    """
    Test that an axis-aligned WCS gives |F| entries equal to the pixel
    scale in arcsec.
    """
    wcs = _make_rotated_wcs(0.0, scale_arcsec=0.5, crval=(150.0, 0.0))
    jacs = compute_pixel_to_sky_jacobians(np.array([50.0]),
                                          np.array([50.0]), wcs)
    assert_allclose(np.abs(jacs[0]),
                    [[0.5, 0.0], [0.0, 0.5]], atol=1e-4)


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


class TestMeanScale:
    """
    Tests for `sky_to_pixel_mean_scale` and `pixel_to_sky_mean_scale`.
    """

    def test_sky_to_pixel_return_types(self, simple_wcs):
        """
        Should return (tuple, float).
        """
        pix_position, scale = sky_to_pixel_mean_scale(WCS_CENTER, simple_wcs)
        assert isinstance(pix_position, tuple)
        assert isinstance(scale, float)

    def test_pixel_to_sky_return_types(self, simple_wcs, center_xy_coord):
        """
        Should return (SkyCoord, float).
        """
        sky_position, scale = pixel_to_sky_mean_scale(center_xy_coord,
                                                      simple_wcs)
        assert isinstance(sky_position, SkyCoord)
        assert isinstance(scale, float)

    def test_sky_to_pixel_simple_scale(self, simple_wcs):
        """
        For an isotropic WCS, the mean scale should equal
        1/WCS_CDELT_ARCSEC.
        """
        _, scale = sky_to_pixel_mean_scale(WCS_CENTER, simple_wcs)
        assert_allclose(scale, 1.0 / WCS_CDELT_ARCSEC)

    def test_pixel_to_sky_simple_scale(self, simple_wcs, center_xy_coord):
        """
        For an isotropic WCS, the mean scale should equal WCS_CDELT_ARCSEC.
        """
        _, scale = pixel_to_sky_mean_scale(center_xy_coord, simple_wcs)
        assert_allclose(scale, WCS_CDELT_ARCSEC)

    def test_sky_to_pixel_sip_scale(self, sip_wcs):
        """
        For a SIP WCS near the reference pixel, the mean scale should be
        close to the undistorted value.
        """
        _, scale = sky_to_pixel_mean_scale(WCS_CENTER, sip_wcs)
        assert_allclose(scale, 1.0 / WCS_CDELT_ARCSEC, rtol=1e-6)

    def test_pixel_to_sky_sip_scale(self, sip_wcs):
        """
        For a SIP WCS near the reference pixel, the mean scale should be
        close to the undistorted value.
        """
        sky_position, scale = pixel_to_sky_mean_scale((9.5, 9.5), sip_wcs)
        assert isinstance(sky_position, SkyCoord)
        assert_allclose(scale, WCS_CDELT_ARCSEC, rtol=1e-6)

    @pytest.mark.parametrize('wcs_name', ['simple_wcs', 'sip_wcs'])
    def test_roundtrip_scale(self, wcs_name, request):
        """
        Sky -> pixel mean_scale * pixel -> sky mean_scale should ~ 1.
        """
        wcs = request.getfixturevalue(wcs_name)
        center_pix, s2p = sky_to_pixel_mean_scale(WCS_CENTER, wcs)
        _, p2s = pixel_to_sky_mean_scale(center_pix, wcs)
        assert_allclose(s2p * p2s, 1.0)

    def test_center_coordinates(self, simple_wcs):
        """
        The returned pix_position should match world_to_pixel.
        """
        pix_position, _ = sky_to_pixel_mean_scale(WCS_CENTER, simple_wcs)
        x_exp, y_exp = simple_wcs.world_to_pixel(WCS_CENTER)
        assert_allclose(pix_position[0], x_exp)
        assert_allclose(pix_position[1], y_exp)

    def test_nonsquare_mean_scale(self, nonsquare_wcs):
        """
        For non-square pixels the mean scale should be the arithmetic
        mean of the two singular values (1/cdelt_x and 1/cdelt_y in
        pix/arcsec).
        """
        _, scale = sky_to_pixel_mean_scale(WCS_CENTER, nonsquare_wcs)
        cdelt_x = 0.03 * 3600
        cdelt_y = 0.05 * 3600
        expected = 0.5 * (1.0 / cdelt_x + 1.0 / cdelt_y)
        assert_allclose(scale, expected, rtol=1e-6)

    def test_nonsquare_same_for_sip(self, nonsquare_wcs):
        """
        The scale formula must not depend on whether the WCS carries
        distortion terms. A SIP WCS with the same non-square pixel
        scales and negligible coefficients gives the same scale.
        """
        header = nonsquare_wcs.to_header()
        header['CTYPE1'] = 'RA---TAN-SIP'
        header['CTYPE2'] = 'DEC--TAN-SIP'
        header['A_ORDER'] = header['B_ORDER'] = 2
        header['A_2_0'] = header['B_0_2'] = 1e-12
        sip_wcs = APWCS(header)
        assert sip_wcs.has_distortion
        _, scale = sky_to_pixel_mean_scale(WCS_CENTER, nonsquare_wcs)
        _, scale_sip = sky_to_pixel_mean_scale(WCS_CENTER, sip_wcs)
        assert_allclose(scale_sip, scale, rtol=1e-8)


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

    @pytest.mark.parametrize('angle_deg', [0.0, 40.0, 130.0])
    def test_circular_input_preserves_angle(self, rotated_wcs, angle_deg):
        """
        For a circular input shape (width == height), the SVD principal
        axis is arbitrary, so the helper falls back to the mapped width
        semi-axis to preserve the input rotation angle.

        The angle must round-trip through pixel -> sky -> pixel.
        """
        pixcoord = (9.5, 9.5)
        angle_rad = np.deg2rad(angle_deg)
        sky_center, sw, sh, sky_angle = pixel_shape_to_sky_svd(
            pixcoord, rotated_wcs, 6.0, 6.0, angle_rad)
        assert_allclose(sw, sh, rtol=1e-8)

        _, pw, ph, pix_angle = sky_shape_to_pixel_svd(
            sky_center, rotated_wcs, sw, sh, sky_angle.rad)
        assert_allclose(pw, ph, rtol=1e-8)
        diff = (pix_angle.deg - angle_deg + 180) % 360 - 180
        assert_allclose(diff, 0.0, atol=1e-6)

    @pytest.mark.parametrize(('center_ra', 'center_dec'), TROUBLESOME_CENTERS)
    def test_roundtrip_at_pole_and_wrap(self, center_ra, center_dec):
        """
        Test that a directed sky shape converted to pixels and back
        round-trips to itself near the celestial poles and
        across the RA = 0 / 360 wraparound.
        """
        wcs = _make_sip_wcs(center_ra, center_dec)
        skycoord = SkyCoord(center_ra * u.deg, center_dec * u.deg)
        center, pw, ph, pa = sky_shape_to_pixel_svd(
            skycoord, wcs, 2.0, 1.0, np.deg2rad(30))
        assert pw > 1.0
        assert ph > 1.0
        _, sw, sh, _ = pixel_shape_to_sky_svd(
            center, wcs, pw, ph, pa.to_value(u.radian))
        assert_allclose(sw, 2.0, rtol=1e-3)
        assert_allclose(sh, 1.0, rtol=1e-3)


def _project_sky_ellipse_boundary(skycoord, wcs, a_arcsec, b_arcsec, pa_rad, *,
                                  n_points=720):
    """
    Project the boundary of a sky ellipse to pixel coordinates.

    The boundary points are computed in the tangent plane (``xi``
    = East, ``eta`` = North) and offset from ``skycoord`` using
    great-circle geometry, then converted to pixel coordinates with
    ``wcs.world_to_pixel``. This is an independent ground truth for the
    pixel image of a sky ellipse.
    """
    theta = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    xi = (a_arcsec * np.cos(theta) * np.sin(pa_rad)
          + b_arcsec * np.sin(theta) * np.cos(pa_rad))
    eta = (a_arcsec * np.cos(theta) * np.cos(pa_rad)
           - b_arcsec * np.sin(theta) * np.sin(pa_rad))
    sep = np.hypot(xi, eta) * u.arcsec
    posang = np.arctan2(xi, eta) * u.rad
    pts = skycoord.directional_offset_by(posang, sep)
    x, y = wcs.world_to_pixel(pts)
    return np.asarray(x, dtype=float), np.asarray(y, dtype=float)


def _ellipse_implicit_residual(x, y, center, width, height, angle_rad):
    """
    Evaluate the implicit ellipse equation for points ``(x, y)``.

    Returns ``(u / a)**2 + (v / b)**2`` where ``(u, v)`` are the point
    offsets from ``center`` rotated into the ellipse frame and ``a``,
    ``b`` are the semi-axes. Points exactly on the ellipse boundary
    yield 1.
    """
    cx, cy = center
    dx = x - cx
    dy = y - cy
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    u_axis = dx * cos_a + dy * sin_a
    v_axis = -dx * sin_a + dy * cos_a
    return (u_axis / (width / 2)) ** 2 + (v_axis / (height / 2)) ** 2


class TestFlippedParityWCS:
    """
    Regression tests for sky <-> pixel shape conversions with a
    flipped-parity WCS (North down, East left, positive determinant).

    Such a WCS previously produced apertures that were mirrored about
    the x-axis relative to the correct orientation.
    """

    PA_DEGS = (0.0, 30.0, 75.0, 130.0, 228.0, 310.0)

    def test_flipped_wcs_has_positive_parity(self, flipped_wcs):
        """
        The flipped WCS fixture must have a positive-determinant pixel
        scale matrix (parity = +1), opposite the standard convention.
        """
        assert np.linalg.det(flipped_wcs.pixel_scale_matrix) > 0

    @pytest.mark.parametrize('pa_deg', PA_DEGS)
    def test_sky_to_pixel_matches_projection(self, flipped_wcs, pa_deg):
        """
        The pixel ellipse from ``sky_shape_to_pixel_svd`` must match
        the true projection of the sky ellipse boundary (not its mirror
        image) for a flipped-parity WCS.
        """
        a_arcsec, b_arcsec = 3.0, 1.5
        pa_rad = np.deg2rad(pa_deg)
        center, pw, ph, pangle = sky_shape_to_pixel_svd(
            WCS_CENTER, flipped_wcs, 2 * a_arcsec, 2 * b_arcsec, pa_rad)

        x, y = _project_sky_ellipse_boundary(
            WCS_CENTER, flipped_wcs, a_arcsec, b_arcsec, pa_rad)
        resid = _ellipse_implicit_residual(
            x, y, center, pw, ph, pangle.rad)
        # All projected boundary points must lie on the pixel ellipse.
        assert_allclose(resid, 1.0, atol=1e-3)

    @pytest.mark.parametrize('pa_deg', PA_DEGS)
    def test_normal_wcs_matches_projection(self, simple_wcs, pa_deg):
        """
        The same check for a standard-parity WCS (a control case that
        already worked before the fix).
        """
        a_arcsec, b_arcsec = 3.0, 1.5
        pa_rad = np.deg2rad(pa_deg)
        center, pw, ph, pangle = sky_shape_to_pixel_svd(
            WCS_CENTER, simple_wcs, 2 * a_arcsec, 2 * b_arcsec, pa_rad)

        x, y = _project_sky_ellipse_boundary(
            WCS_CENTER, simple_wcs, a_arcsec, b_arcsec, pa_rad)
        resid = _ellipse_implicit_residual(
            x, y, center, pw, ph, pangle.rad)
        assert_allclose(resid, 1.0, atol=1e-3)

    def test_roundtrip_sky_pixel_sky(self, flipped_wcs):
        """
        Sky -> pixel -> sky should recover the original ellipse for a
        flipped-parity WCS.
        """
        sky_w, sky_h, sky_a = 6.0, 3.0, np.deg2rad(228.0)
        center_pix, pw, ph, pa = sky_shape_to_pixel_svd(
            WCS_CENTER, flipped_wcs, sky_w, sky_h, sky_a)
        _, rw, rh, ra = pixel_shape_to_sky_svd(
            center_pix, flipped_wcs, pw, ph, pa.rad)
        assert_allclose(rw, sky_w, rtol=1e-6)
        assert_allclose(rh, sky_h, rtol=1e-6)
        assert_allclose(ra.rad, sky_a, rtol=1e-4)


def _make_sheared_wcs():
    """
    Build an anisotropic, sheared (non-orthogonal CD matrix) TAN WCS.

    The unequal, non-orthogonal CD terms make the local Jacobian both
    anisotropic and rotated, so the SVD principal axes have a nontrivial
    orientation. This exercises the angle convention of the scale
    helpers.
    """
    wcs = APWCS(naxis=2)
    wcs.wcs.crpix = [50, 50]
    wcs.wcs.crval = [WCS_CENTER.ra.deg, WCS_CENTER.dec.deg]
    wcs.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    wcs.wcs.cd = np.array([[-0.0008, 0.0003],
                           [0.0002, 0.0005]])
    return wcs


def _project_pixel_circle_to_sky_tangent(pixcoord, wcs, radius_pix, *,
                                         n_points=720):
    """
    Project a pixel circle boundary to tangent-plane sky offsets.

    The boundary points of a pixel circle are converted to sky
    coordinates and expressed as tangent-plane offsets (``xi`` = East,
    ``eta`` = North) in arcsec relative to the circle center, using
    great-circle separation and position angle. This is an independent
    ground truth for the sky image of a pixel circle.
    """
    theta = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    cx, cy = pixcoord
    x = cx + radius_pix * np.cos(theta)
    y = cy + radius_pix * np.sin(theta)
    center = wcs.pixel_to_world(cx, cy)
    pts = wcs.pixel_to_world(x, y)
    arcsec_per_rad = 3600.0 * np.degrees(1)
    sep = center.separation(pts).rad
    posang = center.position_angle(pts).rad
    xi = sep * np.sin(posang) * arcsec_per_rad
    eta = sep * np.cos(posang) * arcsec_per_rad
    return np.asarray(xi, dtype=float), np.asarray(eta, dtype=float)


def _sky_ellipse_implicit_residual(xi, eta, width, height, pa_rad):
    """
    Evaluate the implicit sky-ellipse equation in the tangent plane.

    The major axis (of full length ``width``) is at position angle
    ``pa_rad`` measured from North (``eta``) toward East (``xi``).
    Points exactly on the ellipse boundary yield 1.
    """
    cos_pa = np.cos(pa_rad)
    sin_pa = np.sin(pa_rad)
    p_major = xi * sin_pa + eta * cos_pa
    p_minor = xi * cos_pa - eta * sin_pa
    return (p_major / (width / 2)) ** 2 + (p_minor / (height / 2)) ** 2


class TestSVDScales:
    """
    Tests for `sky_to_pixel_svd_scales` and `pixel_to_sky_svd_scales`.
    """

    def test_sky_to_pixel_return_types(self, simple_wcs):
        """
        Should return (tuple, float, float, Angle).
        """
        center, smaj, smin, angle = sky_to_pixel_svd_scales(
            WCS_CENTER, simple_wcs)
        assert isinstance(center, tuple)
        assert isinstance(smaj, (float, np.floating))
        assert isinstance(smin, (float, np.floating))
        assert isinstance(angle, Angle)

    def test_pixel_to_sky_return_types(self, simple_wcs, center_xy_coord):
        """
        Should return (SkyCoord, float, float, Angle).
        """
        center, smaj, smin, angle = pixel_to_sky_svd_scales(
            center_xy_coord, simple_wcs)
        assert isinstance(center, SkyCoord)
        assert isinstance(smaj, (float, np.floating))
        assert isinstance(smin, (float, np.floating))
        assert isinstance(angle, Angle)

    def test_sky_to_pixel_simple_isotropic(self, simple_wcs):
        """
        For a non-distorted, square-pixel WCS the two scale factors are
        equal and match 1 / WCS_CDELT_ARCSEC (pixels per arcsec).
        """
        _, smaj, smin, angle = sky_to_pixel_svd_scales(
            WCS_CENTER, simple_wcs)
        expected = 1.0 / WCS_CDELT_ARCSEC
        assert_allclose(smaj, expected, rtol=1e-6)
        assert_allclose(smin, expected, rtol=1e-6)
        assert 0.0 <= angle.deg < 360.0

    def test_pixel_to_sky_simple_isotropic(self, simple_wcs, center_xy_coord):
        """
        For a non-distorted, square-pixel WCS the two scale factors are
        equal and match WCS_CDELT_ARCSEC (arcsec per pixel).
        """
        _, smaj, smin, angle = pixel_to_sky_svd_scales(
            center_xy_coord, simple_wcs)
        assert_allclose(smaj, WCS_CDELT_ARCSEC, rtol=1e-6)
        assert_allclose(smin, WCS_CDELT_ARCSEC, rtol=1e-6)
        assert 0.0 <= angle.deg < 360.0

    def test_sky_to_pixel_nonsquare_anisotropic(self, nonsquare_wcs):
        """
        For non-square pixels the major scale exceeds the minor scale,
        and both match the inverse of the corresponding pixel scales.
        """
        _, smaj, smin, _ = sky_to_pixel_svd_scales(WCS_CENTER, nonsquare_wcs)
        assert smaj > smin
        # cdelt = [-0.03, 0.05] deg, so pixels/arcsec = 1 / (cdelt_arcsec)
        assert_allclose(smaj, 1.0 / (0.03 * 3600), rtol=1e-5)
        assert_allclose(smin, 1.0 / (0.05 * 3600), rtol=1e-5)

    def test_pixel_to_sky_nonsquare_anisotropic(self, nonsquare_wcs):
        """
        For non-square pixels the major scale exceeds the minor scale,
        and both match the corresponding pixel scales (arcsec/pixel).
        """
        center, smaj, smin, _ = pixel_to_sky_svd_scales((9.5, 9.5),
                                                        nonsquare_wcs)
        assert isinstance(center, SkyCoord)
        assert smaj > smin
        assert_allclose(smaj, 0.05 * 3600, rtol=1e-5)
        assert_allclose(smin, 0.03 * 3600, rtol=1e-5)

    def test_scales_descending(self, sip_wcs):
        """
        The returned scales are singular values in descending order, so
        the major scale is always >= the minor scale.
        """
        _, smaj_s, smin_s, _ = sky_to_pixel_svd_scales(WCS_CENTER, sip_wcs)
        _, smaj_p, smin_p, _ = pixel_to_sky_svd_scales((9.5, 9.5), sip_wcs)
        assert smaj_s >= smin_s
        assert smaj_p >= smin_p

    @pytest.mark.parametrize('wcs_name', ['rotated_wcs', 'flipped_wcs'])
    def test_sky_to_pixel_matches_projection(self, wcs_name, request):
        """
        A circular sky region converted with the SVD scales must match
        the true projection of the sky circle boundary, including the
        correct major-axis orientation and parity.
        """
        wcs = request.getfixturevalue(wcs_name)
        radius = 5.0  # arcsec
        center, smaj, smin, pixel_angle = sky_to_pixel_svd_scales(
            WCS_CENTER, wcs)
        x, y = _project_sky_ellipse_boundary(
            WCS_CENTER, wcs, radius, radius, 0.0)
        resid = _ellipse_implicit_residual(
            x, y, center, 2 * radius * smaj, 2 * radius * smin,
            pixel_angle.rad)
        assert_allclose(resid, 1.0, atol=1e-3)

    def test_sky_to_pixel_matches_projection_sheared(self):
        """
        For an anisotropic, sheared WCS the projected sky circle is an
        ellipse whose principal axes and orientation must match the SVD
        scales and pixel angle.
        """
        wcs = _make_sheared_wcs()
        radius = 20.0  # arcsec
        center, smaj, smin, pixel_angle = sky_to_pixel_svd_scales(
            WCS_CENTER, wcs)
        assert smaj > smin
        x, y = _project_sky_ellipse_boundary(
            WCS_CENTER, wcs, radius, radius, 0.0)
        resid = _ellipse_implicit_residual(
            x, y, center, 2 * radius * smaj, 2 * radius * smin,
            pixel_angle.rad)
        assert_allclose(resid, 1.0, atol=2e-3)

    def test_pixel_to_sky_matches_projection_sheared(self):
        """
        For an anisotropic, sheared WCS the sky image of a pixel circle
        is an ellipse whose principal axes and position angle must match
        the SVD scales and sky angle.
        """
        wcs = _make_sheared_wcs()
        pixcoord = (50.0, 50.0)
        radius_pix = 10.0
        _, smaj, smin, sky_angle = pixel_to_sky_svd_scales(pixcoord, wcs)
        assert smaj > smin
        xi, eta = _project_pixel_circle_to_sky_tangent(
            pixcoord, wcs, radius_pix)
        resid = _sky_ellipse_implicit_residual(
            xi, eta, 2 * radius_pix * smaj, 2 * radius_pix * smin,
            sky_angle.rad)
        assert_allclose(resid, 1.0, atol=2e-3)

    def test_inverse_scale_consistency(self):
        """
        The singular values of the Jacobian and its inverse are
        reciprocals, so the pixel-to-sky major (minor) scale equals the
        reciprocal of the sky-to-pixel minor (major) scale.
        """
        wcs = _make_sheared_wcs()
        center_pix, smaj_p, smin_p, _ = sky_to_pixel_svd_scales(
            WCS_CENTER, wcs)
        _, smaj_s, smin_s, _ = pixel_to_sky_svd_scales(center_pix, wcs)
        assert_allclose(smaj_s, 1.0 / smin_p, rtol=1e-6)
        assert_allclose(smin_s, 1.0 / smaj_p, rtol=1e-6)

    def test_angles_wrapped(self):
        """
        Both helpers return angles wrapped to [0, 360) degrees.
        """
        wcs = _make_sheared_wcs()
        _, _, _, pixel_angle = sky_to_pixel_svd_scales(WCS_CENTER, wcs)
        _, _, _, sky_angle = pixel_to_sky_svd_scales((50.0, 50.0), wcs)
        assert 0.0 <= pixel_angle.deg < 360.0
        assert 0.0 <= sky_angle.deg < 360.0


def _make_quadratic_sip_wcs(coeff=1e-3, cross=0.0):
    """
    Build a TAN-SIP WCS whose distortion is a pure quadratic in the
    pixel offset from CRPIX.

    At CRPIX the quadratic term has zero slope, so the true local scale
    is exactly CDELT. A one-sided finite difference is biased there by
    a fraction ``coeff``, while a central difference is exact for a
    quadratic.

    ``cross`` adds an ``A_0_2`` term that bends the image of a step
    North into x. It leaves the true direction of North at CRPIX along
    +y, but a one-sided offset North is deflected by ``cross`` times the
    step, which biases the North angle.
    """
    header = Header()
    header['NAXIS'] = 2
    header['NAXIS1'] = 100
    header['NAXIS2'] = 100
    header['CRPIX1'] = 50.0
    header['CRPIX2'] = 50.0
    header['CRVAL1'] = 150.0
    header['CRVAL2'] = 0.0
    header['CTYPE1'] = 'RA---TAN-SIP'
    header['CTYPE2'] = 'DEC--TAN-SIP'
    cdelt = WCS_CDELT_ARCSEC / 3600.0
    header['CD1_1'] = -cdelt
    header['CD1_2'] = 0.0
    header['CD2_1'] = 0.0
    header['CD2_2'] = cdelt
    header['A_ORDER'] = 2
    header['A_2_0'] = coeff
    header['A_0_2'] = cross
    header['B_ORDER'] = 2
    header['B_0_2'] = coeff
    return APWCS(header)


class TestCentralDifferences:
    """
    Tests that the finite-difference Jacobians are unbiased where the
    distortion has curvature.
    """

    def test_pixel_to_sky_jacobians_unbiased_at_crpix(self):
        wcs = _make_quadratic_sip_wcs()
        x = np.array([wcs.wcs.crpix[0] - 1.0])
        y = np.array([wcs.wcs.crpix[1] - 1.0])
        jac = compute_pixel_to_sky_jacobians(x, y, wcs)[0]
        assert_allclose(np.abs(np.diag(jac)), WCS_CDELT_ARCSEC, rtol=1e-6)

    def test_local_wcs_jacobian_unbiased_at_crval(self):
        wcs = _make_quadratic_sip_wcs()
        skycoord = SkyCoord(150.0 * u.deg, 0.0 * u.deg)
        jac = compute_local_wcs_jacobian(skycoord, wcs)
        assert_allclose(np.abs(np.diag(jac)), 1.0 / WCS_CDELT_ARCSEC,
                        rtol=1e-6)

    def test_pixel_scale_unbiased_at_crval(self):
        wcs = _make_quadratic_sip_wcs()
        skycoord = SkyCoord(150.0 * u.deg, 0.0 * u.deg)
        _, scale, _ = wcs_pixel_scale_angle(skycoord, wcs)
        assert_allclose(scale, WCS_CDELT_ARCSEC, rtol=1e-6)

    def test_north_angle_unbiased_at_crval(self):
        # CD1_1 < 0 and CD2_2 > 0, so North is along +y at CRPIX
        wcs = _make_quadratic_sip_wcs(cross=1e-3)
        skycoord = SkyCoord(150.0 * u.deg, 0.0 * u.deg)
        _, _, angle = wcs_pixel_scale_angle(skycoord, wcs)
        assert_allclose(angle.deg, 90.0, atol=1e-4)


@pytest.mark.skipif(not HAS_GWCS, reason='gwcs is required')
class TestGWCSBoundingBox:
    """
    Tests that the finite differences ignore a gwcs bounding box.

    The half-pixel offsets used to build the Jacobian at a source in the
    last pixel of the array fall outside the bounding box. gwcs returns
    NaN there when the box is honored, which would make the Jacobian of
    every edge source NaN.
    """

    @pytest.fixture
    def bounded_gwcs(self):
        shape = (50, 60)
        gwcs = make_gwcs(shape)
        gwcs.bounding_box = ((-0.5, shape[1] - 0.5), (-0.5, shape[0] - 0.5))
        return gwcs

    def test_pixel_to_sky_jacobians_finite_at_edge(self, bounded_gwcs):
        x = np.array([59.4, 0.0, 30.0])
        y = np.array([10.0, 49.4, -0.4])
        jacs = compute_pixel_to_sky_jacobians(x, y, bounded_gwcs)
        assert np.all(np.isfinite(jacs))

    def test_edge_jacobian_matches_interior(self, bounded_gwcs):
        # The gwcs has no distortion, so the Jacobian is the same
        # everywhere
        jacs = compute_pixel_to_sky_jacobians(np.array([59.4, 30.0]),
                                              np.array([10.0, 25.0]),
                                              bounded_gwcs)
        assert_allclose(jacs[0], jacs[1], rtol=1e-6)

    def test_local_wcs_jacobian_finite_at_edge(self, bounded_gwcs):
        skycoord = bounded_gwcs.pixel_to_world(59.4, 10.0)
        jac = compute_local_wcs_jacobian(skycoord, bounded_gwcs)
        assert np.all(np.isfinite(jac))

    def test_pixel_scale_angle_finite_at_edge(self, bounded_gwcs):
        skycoord = bounded_gwcs.pixel_to_world(59.4, 10.0)
        _, scale, angle = wcs_pixel_scale_angle(skycoord, bounded_gwcs)
        assert np.isfinite(scale)
        assert np.isfinite(angle.deg)


class TestVectorizedScalesAndAngles:
    """
    Tests for the vectorized pixel scale and North angle helpers.

    Each must reproduce the per-source function it replaces at every
    position, without calling the WCS inverse once per source.
    """

    positions = (np.array([3.0, 10.0, 16.5]), np.array([4.0, 10.0, 2.2]))

    @pytest.mark.parametrize('wcs_name', ['simple_wcs', 'rotated_wcs',
                                          'nonsquare_wcs', 'sip_wcs'])
    def test_mean_scales_match_per_source(self, wcs_name, request):
        wcs = request.getfixturevalue(wcs_name)
        x, y = self.positions
        scales = compute_pixel_to_sky_mean_scales(x, y, wcs)
        assert scales.shape == (3,)
        for i in range(x.size):
            _, expected = pixel_to_sky_mean_scale((x[i], y[i]), wcs)
            assert_allclose(scales[i], expected, rtol=1e-8)

    @pytest.mark.parametrize('wcs_name', ['simple_wcs', 'rotated_wcs',
                                          'nonsquare_wcs', 'sip_wcs'])
    def test_scale_angles_match_per_source(self, wcs_name, request):
        wcs = request.getfixturevalue(wcs_name)
        x, y = self.positions
        scales, angles = compute_pixel_scale_angles(x, y, wcs)
        assert scales.shape == (3,)
        assert isinstance(angles, Angle)
        assert angles.shape == (3,)
        for i in range(x.size):
            skycoord = wcs.pixel_to_world(x[i], y[i])
            _, scale, angle = wcs_pixel_scale_angle(skycoord, wcs)
            assert_allclose(scales[i], scale, rtol=1e-6)
            assert_allclose(angles[i].deg, angle.deg, atol=1e-4)

    def test_angles_wrapped(self, flipped_wcs):
        _, angles = compute_pixel_scale_angles(*self.positions, flipped_wcs)
        assert np.all((angles.deg >= 0) & (angles.deg < 360))

    def test_scalar_inputs(self, simple_wcs):
        scales = compute_pixel_to_sky_mean_scales(10.0, 10.0, simple_wcs)
        assert scales.shape == (1,)
        scales, angles = compute_pixel_scale_angles(10.0, 10.0, simple_wcs)
        assert scales.shape == (1,)
        assert angles.shape == (1,)

    @pytest.mark.parametrize('wcs_name', ['rotated_wcs', 'flipped_wcs',
                                          'nonsquare_wcs', 'sip_wcs'])
    def test_north_angle_direction(self, wcs_name, request):
        # Solve for the pixel step that moves exactly North on the sky
        # and check that it points along the returned angle. The flipped
        # WCS has a negative Jacobian determinant.
        wcs = request.getfixturevalue(wcs_name)
        x, y = self.positions
        _, angles = compute_pixel_scale_angles(x, y, wcs)
        jacs = compute_pixel_to_sky_jacobians(x, y, wcs)
        north = np.tile([0.0, 1.0], (x.size, 1))[..., np.newaxis]
        north_pix = np.linalg.solve(jacs, north)[..., 0]
        expected = np.degrees(np.arctan2(north_pix[:, 1], north_pix[:, 0]))
        assert_allclose(angles.wrap_at(180 * u.deg).deg, expected, atol=1e-8)


class _CountingWCS:
    """
    Wrapper that counts the calls to the WCS transform methods.
    """

    def __init__(self, real_wcs):
        self._wcs = real_wcs
        self.has_distortion = getattr(real_wcs, 'has_distortion', True)
        self.n_pixel_to_world = 0
        self.n_world_to_pixel = 0

    def pixel_to_world(self, *args, **kwargs):
        self.n_pixel_to_world += 1
        return self._wcs.pixel_to_world(*args, **kwargs)

    def world_to_pixel(self, *args, **kwargs):
        self.n_world_to_pixel += 1
        return self._wcs.world_to_pixel(*args, **kwargs)


def _reference_jacobians(x, y, wcs):
    """
    Forward Jacobians from great-circle separations and position angles
    of the half-pixel offset points, independent of the implementation
    under test.
    """
    x = np.atleast_1d(x).astype(float)
    y = np.atleast_1d(y).astype(float)
    sky0 = wcs.pixel_to_world(x, y)
    arcsec_per_rad = 3600.0 * np.degrees(1)
    jac = np.empty((x.size, 2, 2))
    for col, (dx, dy) in enumerate(((0.5, 0.0), (0.0, 0.5))):
        lo = wcs.pixel_to_world(x - dx, y - dy)
        hi = wcs.pixel_to_world(x + dx, y + dy)
        s_lo, p_lo = sky0.separation(lo).rad, sky0.position_angle(lo).rad
        s_hi, p_hi = sky0.separation(hi).rad, sky0.position_angle(hi).rad
        jac[:, 0, col] = (s_hi * np.sin(p_hi) - s_lo * np.sin(p_lo))
        jac[:, 1, col] = (s_hi * np.cos(p_hi) - s_lo * np.cos(p_lo))
    return jac * arcsec_per_rad


class TestJacobianEvaluation:
    """
    Tests for how the vectorized Jacobian evaluates the WCS.
    """

    def test_flattens_inputs(self, sip_wcs):
        yy, xx = np.mgrid[2:5, 3:7].astype(float)
        jacs = compute_pixel_to_sky_jacobians(xx, yy, sip_wcs)
        expected = compute_pixel_to_sky_jacobians(xx.ravel(), yy.ravel(),
                                                  sip_wcs)
        assert jacs.shape == (12, 2, 2)
        assert_allclose(jacs, expected, rtol=1e-12)

    def test_size_mismatch(self, sip_wcs):
        match = 'x and y must have the same size'
        with pytest.raises(ValueError, match=match):
            compute_pixel_to_sky_jacobians([1.0, 2.0], [1.0], sip_wcs)

    def test_single_wcs_call(self, sip_wcs):
        wcs = _CountingWCS(sip_wcs)
        compute_pixel_to_sky_jacobians(np.array([5.0, 12.0]),
                                       np.array([7.0, 3.0]), wcs)
        assert wcs.n_pixel_to_world == 1
        assert wcs.n_world_to_pixel == 0

    @pytest.mark.parametrize('wcs_name', ['simple_wcs', 'rotated_wcs',
                                          'nonsquare_wcs', 'flipped_wcs',
                                          'sip_wcs'])
    def test_agrees_with_separation_position_angle(self, wcs_name, request):
        wcs = request.getfixturevalue(wcs_name)
        x = np.array([3.0, 9.5, 16.2])
        y = np.array([4.0, 9.5, 2.7])
        jacs = compute_pixel_to_sky_jacobians(x, y, wcs)
        expected = _reference_jacobians(x, y, wcs)
        assert_allclose(jacs, expected, rtol=1e-7, atol=1e-7)

    @pytest.mark.parametrize(('center_ra', 'center_dec'), TROUBLESOME_CENTERS)
    def test_agrees_near_pole_and_wrap(self, center_ra, center_dec):
        wcs = _make_sip_wcs(center_ra, center_dec)
        x = np.array([9.5, 3.0])
        y = np.array([9.5, 15.0])
        jacs = compute_pixel_to_sky_jacobians(x, y, wcs)
        expected = _reference_jacobians(x, y, wcs)
        assert_allclose(jacs, expected, rtol=1e-7, atol=1e-7)

    @pytest.mark.parametrize('dec', [90.0, -90.0])
    def test_exact_pole(self, dec):
        # Longitude is degenerate at the pole, but the pixel area and
        # the Jacobian must stay finite and correct.
        wcs = _make_sip_wcs(0.0, dec)
        jac = compute_pixel_to_sky_jacobians(9.5, 9.5, wcs)[0]
        assert np.all(np.isfinite(jac))
        assert_allclose(np.abs(np.linalg.det(jac)), WCS_CDELT_ARCSEC**2,
                        rtol=1e-6)


class TestMeanScaleClosedForm:
    """
    Tests that the mean pixel scale equals the mean of the singular
    values of the Jacobian.
    """

    @pytest.mark.parametrize('wcs_name', ['simple_wcs', 'rotated_wcs',
                                          'nonsquare_wcs', 'flipped_wcs',
                                          'sip_wcs'])
    def test_vectorized_matches_svd(self, wcs_name, request):
        wcs = request.getfixturevalue(wcs_name)
        x = np.array([3.0, 9.5, 16.2])
        y = np.array([4.0, 9.5, 2.7])
        scales = compute_pixel_to_sky_mean_scales(x, y, wcs)
        jacs = compute_pixel_to_sky_jacobians(x, y, wcs)
        expected = np.linalg.svd(jacs, compute_uv=False).mean(axis=1)
        assert_allclose(scales, expected, rtol=1e-12)

    def test_scalar_matches_svd(self, nonsquare_wcs):
        _, scale = sky_to_pixel_mean_scale(WCS_CENTER, nonsquare_wcs)
        jac = compute_local_wcs_jacobian(WCS_CENTER, nonsquare_wcs)
        expected = np.linalg.svd(jac, compute_uv=False).mean()
        assert_allclose(scale, expected, rtol=1e-12)


class TestKnownPixelPosition:
    """
    Tests for the ``pixcoord`` keyword of the sky-input helpers.

    When the pixel position of the sky coordinate is already known,
    the helpers must use it instead of inverting the WCS, and must
    return the same results as when they invert it themselves. The
    known position is taken from the WCS inverse so that both paths
    evaluate the Jacobian at the same place.
    """

    @staticmethod
    def _known(wcs, x, y):
        skycoord = wcs.pixel_to_world(x, y)
        pixcoord = tuple(float(v) for v in wcs.world_to_pixel(skycoord))
        return skycoord, pixcoord

    @pytest.fixture
    def known(self, sip_wcs):
        return self._known(sip_wcs, 12.0, 7.0)

    @pytest.mark.parametrize('wcs_name', ['simple_wcs', 'sip_wcs'])
    def test_mean_scale(self, wcs_name, request):
        real_wcs = request.getfixturevalue(wcs_name)
        skycoord, pixcoord = self._known(real_wcs, 12.0, 7.0)
        center, scale = sky_to_pixel_mean_scale(skycoord, real_wcs)
        wcs = _CountingWCS(real_wcs)
        center2, scale2 = sky_to_pixel_mean_scale(skycoord, wcs,
                                                  pixcoord=pixcoord)
        assert wcs.n_world_to_pixel == 0
        assert_allclose(center2, center, atol=1e-10)
        assert_allclose(scale2, scale, rtol=1e-12)

    def test_scale_angle(self, sip_wcs, known):
        skycoord, pixcoord = known
        center, scale, angle = wcs_pixel_scale_angle(skycoord, sip_wcs)
        wcs = _CountingWCS(sip_wcs)
        center2, scale2, angle2 = wcs_pixel_scale_angle(skycoord, wcs,
                                                        pixcoord=pixcoord)
        assert wcs.n_world_to_pixel == 0
        assert_allclose(center2, center, atol=1e-10)
        assert_allclose(scale2, scale, rtol=1e-12)
        assert_allclose(angle2.deg, angle.deg, atol=1e-10)

    def test_svd_scales(self, sip_wcs, known):
        skycoord, pixcoord = known
        expected = sky_to_pixel_svd_scales(skycoord, sip_wcs)
        wcs = _CountingWCS(sip_wcs)
        result = sky_to_pixel_svd_scales(skycoord, wcs, pixcoord=pixcoord)
        assert wcs.n_world_to_pixel == 0
        assert_allclose(result[0], expected[0], atol=1e-10)
        assert_allclose(result[1:3], expected[1:3], rtol=1e-12)
        assert_allclose(result[3].deg, expected[3].deg, atol=1e-10)

    def test_shape_svd(self, sip_wcs, known):
        skycoord, pixcoord = known
        args = (2.0, 1.0, 0.3)
        expected = sky_shape_to_pixel_svd(skycoord, sip_wcs, *args)
        wcs = _CountingWCS(sip_wcs)
        result = sky_shape_to_pixel_svd(skycoord, wcs, *args,
                                        pixcoord=pixcoord)
        assert wcs.n_world_to_pixel == 0
        assert_allclose(result[0], expected[0], atol=1e-10)
        assert_allclose(result[1:3], expected[1:3], rtol=1e-12)
        assert_allclose(result[3].deg, expected[3].deg, atol=1e-10)
