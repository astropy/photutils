# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the _wcs_helpers module.
"""

import astropy.units as u
import numpy as np
import pytest
from astropy.coordinates import Angle, SkyCoord
from numpy.testing import assert_allclose

from photutils.utils._wcs_helpers import (compute_local_wcs_jacobian,
                                          jacobian_pixel_to_sky_mean_scale,
                                          jacobian_pixel_to_sky_scales,
                                          jacobian_sky_to_pixel_mean_scale,
                                          jacobian_sky_to_pixel_scales,
                                          pixel_ellipse_to_sky_svd,
                                          pixel_to_sky_mean_scale,
                                          pixel_to_sky_scales,
                                          pixel_to_sky_svd_scales,
                                          sky_ellipse_to_pixel_svd,
                                          sky_to_pixel_mean_scale,
                                          sky_to_pixel_scales,
                                          sky_to_pixel_svd_scales,
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


class TestJacobianDirectionalScales:
    """
    Tests for `jacobian_sky_to_pixel_scales` and
    `jacobian_pixel_to_sky_scales`.
    """

    @pytest.mark.parametrize('sky_angle_deg', [0, 30, 90, 150, 270])
    def test_sky_to_pixel_return_types(self, simple_wcs,
                                       sky_angle_deg):
        """
        Should return (tuple, float, float, Angle).
        """
        sky_angle_rad = np.radians(sky_angle_deg)
        pix_position, sw, sh, pangle = jacobian_sky_to_pixel_scales(
            WCS_CENTER, simple_wcs, sky_angle_rad)
        assert isinstance(pix_position, tuple)
        assert isinstance(sw, (float, np.floating))
        assert isinstance(sh, (float, np.floating))
        assert isinstance(pangle, Angle)

    @pytest.mark.parametrize('pixel_angle_deg', [0, 45, 90, 180, 315])
    def test_pixel_to_sky_return_types(self, simple_wcs, center_xy_coord,
                                       pixel_angle_deg):
        """
        Should return (SkyCoord, float, float, Angle).
        """
        pixel_angle_rad = np.radians(pixel_angle_deg)
        sky_position, sw, sh, sangle = jacobian_pixel_to_sky_scales(
            center_xy_coord, simple_wcs, pixel_angle_rad)
        assert isinstance(sky_position, SkyCoord)
        assert isinstance(sw, (float, np.floating))
        assert isinstance(sh, (float, np.floating))
        assert isinstance(sangle, Angle)

    def test_sky_to_pixel_simple_scales(self, simple_wcs):
        """
        For a simple WCS, the directional scale factors should both
        equal 1/WCS_CDELT_ARCSEC (pixels per arcsec).
        """
        _, sw, sh, _ = jacobian_sky_to_pixel_scales(
            WCS_CENTER, simple_wcs, 0.0)
        expected = 1.0 / WCS_CDELT_ARCSEC
        assert_allclose(sw, expected)
        assert_allclose(sh, expected)

    def test_pixel_to_sky_simple_scales(self, simple_wcs, center_xy_coord):
        """
        For a simple WCS, the directional scale factors should both
        equal WCS_CDELT_ARCSEC (arcsec per pixel).
        """
        _, sw, sh, _ = jacobian_pixel_to_sky_scales(
            center_xy_coord, simple_wcs, 0.0)
        assert_allclose(sw, WCS_CDELT_ARCSEC)
        assert_allclose(sh, WCS_CDELT_ARCSEC)

    def test_pixel_angle_wrapped(self, simple_wcs):
        """
        The pixel angle must be in [0, 360) degrees.
        """
        _, _, _, pangle = jacobian_sky_to_pixel_scales(
            WCS_CENTER, simple_wcs, 0.0)
        assert 0.0 <= pangle.deg < 360.0

    def test_sky_angle_wrapped(self, simple_wcs, center_xy_coord):
        """
        The sky angle must be in [0, 360) degrees.
        """
        _, _, _, sangle = jacobian_pixel_to_sky_scales(
            center_xy_coord, simple_wcs, 0.0)
        assert 0.0 <= sangle.deg < 360.0

    @pytest.mark.parametrize('angle_deg', [0, 30, 90, 270])
    def test_roundtrip_angle(self, simple_wcs, angle_deg):
        """
        Converting sky -> pixel -> sky should recover the original
        angle.
        """
        sky_angle_rad = np.radians(angle_deg)
        center_pix, _, _, pangle = jacobian_sky_to_pixel_scales(
            WCS_CENTER, simple_wcs, sky_angle_rad)
        _, _, _, sangle = jacobian_pixel_to_sky_scales(
            center_pix, simple_wcs, pangle.rad)
        assert_allclose(sangle.deg % 360, angle_deg % 360, atol=1e-6)

    @pytest.mark.parametrize('angle_deg', [0, 45, 135])
    def test_roundtrip_scales(self, simple_wcs, angle_deg):
        """
        Converting sky -> pixel -> sky should recover the original
        scale factors.
        """
        sky_angle_rad = np.radians(angle_deg)
        center_pix, sw, sh, pangle = jacobian_sky_to_pixel_scales(
            WCS_CENTER, simple_wcs, sky_angle_rad)
        _, sw_rt, sh_rt, _ = jacobian_pixel_to_sky_scales(
            center_pix, simple_wcs, pangle.rad)
        # sw (pix/arcsec) * sw_rt (arcsec/pix) ~ 1
        assert_allclose(sw * sw_rt, 1.0)
        assert_allclose(sh * sh_rt, 1.0)

    def test_sip_wcs_positive_scales(self, sip_wcs):
        """
        Scale factors should be positive even for distorted WCS.
        """
        _, sw, sh, _ = jacobian_sky_to_pixel_scales(
            WCS_CENTER, sip_wcs, np.radians(30))
        assert sw > 0
        assert sh > 0

    def test_rotated_wcs(self, rotated_wcs):
        """
        For a rotated (but isotropic) WCS, scales should still equal
        1/WCS_CDELT_ARCSEC.
        """
        _, sw, sh, _ = jacobian_sky_to_pixel_scales(
            WCS_CENTER, rotated_wcs, 0.0)
        expected = 1.0 / WCS_CDELT_ARCSEC
        assert_allclose(sw, expected, rtol=1e-6)
        assert_allclose(sh, expected, rtol=1e-6)


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


class TestDispatchScales:
    """
    Tests for `sky_to_pixel_scales` and `pixel_to_sky_scales` dispatch
    helpers that route between offset and Jacobian methods based on
    ``wcs.has_distortion``.
    """

    def test_no_distortion_dispatches_offset(self, simple_wcs):
        """
        For a non-distorted WCS, the dispatch helper should produce
        equal w/h scales (isotropic offset method).
        """
        assert not simple_wcs.has_distortion
        _, sw, sh, _ = sky_to_pixel_scales(WCS_CENTER, simple_wcs, 0.0)
        assert_allclose(sw, sh)

    def test_distortion_dispatches_jacobian(self, sip_wcs):
        """
        For a SIP WCS, the dispatch helper should use the Jacobian path.
        """
        assert sip_wcs.has_distortion
        _, sw, sh, _ = sky_to_pixel_scales(
            WCS_CENTER, sip_wcs, np.radians(30))
        # Scales should be close but not necessarily identical
        assert sw > 0
        assert sh > 0

    @pytest.mark.parametrize('angle_deg', [0, 45, 90, 180])
    def test_sky_to_pixel_return_types(self, simple_wcs,
                                       angle_deg):
        """
        Should return (tuple, float, float, Angle).
        """
        pix_position, _sw, _sh, pangle = sky_to_pixel_scales(
            WCS_CENTER, simple_wcs, np.radians(angle_deg))
        assert isinstance(pix_position, tuple)
        assert isinstance(pangle, Angle)
        assert 0.0 <= pangle.deg < 360.0

    @pytest.mark.parametrize('angle_deg', [0, 45, 90, 180])
    def test_pixel_to_sky_return_types(self, simple_wcs, center_xy_coord,
                                       angle_deg):
        """
        Should return (SkyCoord, float, float, Angle).
        """
        sky_position, _sw, _sh, sangle = pixel_to_sky_scales(
            center_xy_coord, simple_wcs, np.radians(angle_deg))
        assert isinstance(sky_position, SkyCoord)
        assert isinstance(sangle, Angle)
        assert 0.0 <= sangle.deg < 360.0

    @pytest.mark.parametrize('angle_deg', [0, 30, 90, 270])
    def test_roundtrip_simple_wcs(self, simple_wcs,
                                  angle_deg):
        """
        Converting sky -> pixel -> sky with a simple WCS should recover
        the original angle.
        """
        sky_angle_rad = np.radians(angle_deg)
        center_pix, _, _, pangle = sky_to_pixel_scales(
            WCS_CENTER, simple_wcs, sky_angle_rad)
        _, _, _, sangle = pixel_to_sky_scales(
            center_pix, simple_wcs, pangle.rad)
        assert_allclose(sangle.deg % 360, angle_deg % 360, atol=1e-6)

    @pytest.mark.parametrize('angle_deg', [0, 45, 90])
    def test_roundtrip_sip_wcs(self, sip_wcs, angle_deg):
        """
        Roundtrip through the Jacobian path should recover the original
        angle within tolerance.
        """
        sky_angle_rad = np.radians(angle_deg)
        center_pix, _, _, pangle = sky_to_pixel_scales(
            WCS_CENTER, sip_wcs, sky_angle_rad)
        _, _, _, sangle = pixel_to_sky_scales(
            center_pix, sip_wcs, pangle.rad)
        assert_allclose(sangle.deg % 360, angle_deg % 360, atol=1e-6)

    @pytest.mark.parametrize('angle_deg', [0, 45, 90])
    def test_roundtrip_rotated_wcs(self, rotated_wcs,
                                   angle_deg):
        """
        Roundtrip through a rotated WCS should recover the original
        angle.
        """
        sky_angle_rad = np.radians(angle_deg)
        center_pix, _, _, pangle = sky_to_pixel_scales(
            WCS_CENTER, rotated_wcs, sky_angle_rad)
        _, _, _, sangle = pixel_to_sky_scales(
            center_pix, rotated_wcs, pangle.rad)
        diff = abs(sangle.deg - angle_deg) % 360
        assert min(diff, 360 - diff) < 0.5

    def test_simple_wcs_scale_value(self, simple_wcs):
        """
        For a simple WCS, scales should be 1/WCS_CDELT_ARCSEC.
        """
        _, sw, sh, _ = sky_to_pixel_scales(
            WCS_CENTER, simple_wcs, 0.0)
        expected = 1.0 / WCS_CDELT_ARCSEC
        assert_allclose(sw, expected)
        assert_allclose(sh, expected)

    def test_pixel_to_sky_scale_value(self, simple_wcs, center_xy_coord):
        """
        For a simple WCS, scales should be WCS_CDELT_ARCSEC.
        """
        _, sw, sh, _ = pixel_to_sky_scales(
            center_xy_coord, simple_wcs, 0.0)
        assert_allclose(sw, WCS_CDELT_ARCSEC)
        assert_allclose(sh, WCS_CDELT_ARCSEC)

    def test_zero_angle(self, simple_wcs):
        """
        A zero sky angle should work without error.
        """
        _center, sw, sh, _pangle = sky_to_pixel_scales(
            WCS_CENTER, simple_wcs, 0.0)
        assert sw > 0
        assert sh > 0

    def test_two_pi_angle(self, simple_wcs):
        """
        An angle of 2*pi (360 degrees) should be equivalent to 0.
        """
        _, sw_0, sh_0, pa_0 = sky_to_pixel_scales(
            WCS_CENTER, simple_wcs, 0.0)
        _, sw_2pi, sh_2pi, pa_2pi = sky_to_pixel_scales(
            WCS_CENTER, simple_wcs, 2 * np.pi)
        assert_allclose(sw_0, sw_2pi, rtol=1e-10)
        assert_allclose(sh_0, sh_2pi, rtol=1e-10)
        assert_allclose(pa_0.deg % 360, pa_2pi.deg % 360, atol=1e-6)

    def test_negative_angle(self, simple_wcs):
        """
        A negative sky angle should be handled correctly.
        """
        _, sw, sh, _pangle = sky_to_pixel_scales(
            WCS_CENTER, simple_wcs, -np.pi / 4)
        assert sw > 0
        assert sh > 0

    def test_consistency_offset_jacobian(self, simple_wcs):
        """
        For a simple WCS (no distortion), the offset method and the
        Jacobian method should give consistent results.
        """
        sky_angle_rad = np.radians(30.0)

        # Offset path (via dispatch)
        c1, sw1, sh1, pa1 = sky_to_pixel_scales(
            WCS_CENTER, simple_wcs, sky_angle_rad)

        # Jacobian path (direct call)
        c2, sw2, sh2, pa2 = jacobian_sky_to_pixel_scales(
            WCS_CENTER, simple_wcs, sky_angle_rad)

        assert_allclose(c1[0], c2[0])
        assert_allclose(c1[1], c2[1])
        assert_allclose(sw1, sw2)
        assert_allclose(sh1, sh2)
        assert_allclose(pa1.deg % 360, pa2.deg % 360, atol=1e-5)

    def test_consistency_pixel_offset_jacobian(self, simple_wcs,
                                               center_xy_coord):
        """
        For a simple WCS, both paths should give consistent pixel ->
        sky results.
        """
        pixel_angle_rad = np.radians(45.0)

        # Offset path (via dispatch)
        c1, sw1, sh1, sa1 = pixel_to_sky_scales(
            center_xy_coord, simple_wcs, pixel_angle_rad)

        # Jacobian path (direct call)
        c2, sw2, sh2, sa2 = jacobian_pixel_to_sky_scales(
            center_xy_coord, simple_wcs, pixel_angle_rad)

        assert_allclose(c1.ra.deg, c2.ra.deg)
        assert_allclose(c1.dec.deg, c2.dec.deg)
        assert_allclose(sw1, sw2)
        assert_allclose(sh1, sh2)
        assert_allclose(sa1.deg % 360, sa2.deg % 360, atol=1e-5)


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

    def test_sky_to_pixel_scales_uses_jacobian(self, mock_gwcs):
        """
        Without has_distortion, should use the Jacobian path and still
        produce valid results.
        """
        pix_position, sw, sh, pangle = sky_to_pixel_scales(
            WCS_CENTER, mock_gwcs, 0.0)
        assert isinstance(pix_position, tuple)
        assert sw > 0
        assert sh > 0
        assert isinstance(pangle, Angle)

    def test_pixel_to_sky_scales_uses_jacobian(self, mock_gwcs):
        """
        Without has_distortion, pixel_to_sky_scales should use the
        Jacobian path.
        """
        xy_coord = (9.5, 9.5)
        sky_position, sw, sh, _sangle = pixel_to_sky_scales(
            xy_coord, mock_gwcs, 0.0)
        assert isinstance(sky_position, SkyCoord)
        assert sw > 0
        assert sh > 0

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

    def test_nonsquare_directional_scales(self, nonsquare_wcs):
        """
        For non-square pixels, the directional scales should differ.
        """
        _, sw, sh, _ = jacobian_sky_to_pixel_scales(
            WCS_CENTER, nonsquare_wcs, 0.0)
        # Scale factors should differ since x and y pixel scales differ
        assert not np.isclose(sw, sh)


class TestSVDEllipseConversions:
    """
    Tests for `pixel_ellipse_to_sky_svd` and `sky_ellipse_to_pixel_svd`.
    """

    def test_pixel_to_sky_return_types(self, simple_wcs, center_xy_coord):
        """
        Should return (SkyCoord, float, float, Angle).
        """
        center, w, h, angle = pixel_ellipse_to_sky_svd(
            center_xy_coord, simple_wcs, 10.0, 5.0, 0.5)
        assert isinstance(center, SkyCoord)
        assert isinstance(w, (float, np.floating))
        assert isinstance(h, (float, np.floating))
        assert isinstance(angle, Angle)

    def test_sky_to_pixel_return_types(self, simple_wcs):
        """
        Should return (tuple, float, float, Angle).
        """
        center, w, h, angle = sky_ellipse_to_pixel_svd(
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
        center_pix, pw, ph, pa = sky_ellipse_to_pixel_svd(
            WCS_CENTER, simple_wcs, sky_w, sky_h, sky_a)
        _, rw, rh, ra = pixel_ellipse_to_sky_svd(
            center_pix, simple_wcs, pw, ph, pa.rad)
        assert_allclose(rw, sky_w, rtol=1e-6)
        assert_allclose(rh, sky_h, rtol=1e-6)
        assert_allclose(ra.rad, sky_a, rtol=1e-4)

    def test_roundtrip_pixel_sky_pixel(self, simple_wcs, center_xy_coord):
        """
        Pixel -> sky -> pixel should recover the original ellipse.
        """
        pix_w, pix_h, pix_a = 10.0, 5.0, 0.3
        _, sw, sh, sa = pixel_ellipse_to_sky_svd(
            center_xy_coord, simple_wcs, pix_w, pix_h, pix_a)
        _, rw, rh, ra = sky_ellipse_to_pixel_svd(
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
        _, sw, sh, _ = pixel_ellipse_to_sky_svd(
            center_xy_coord, simple_wcs, pix_w, pix_h, 0.0)
        assert_allclose(sw, pix_w * WCS_CDELT_ARCSEC, rtol=1e-5)
        assert_allclose(sh, pix_h * WCS_CDELT_ARCSEC, rtol=1e-5)

    def test_height_larger_than_width(self, simple_wcs, center_xy_coord):
        """
        When height > width, the SVD should still correctly assign
        widths and heights.
        """
        pix_w, pix_h = 5.0, 10.0
        _, sw, sh, _ = pixel_ellipse_to_sky_svd(
            center_xy_coord, simple_wcs, pix_w, pix_h, 0.0)
        # Width should be smaller than height in sky coords too
        assert sw < sh

    def test_sip_wcs_positive_sizes(self, sip_wcs):
        """
        Sizes should be positive for distorted WCS.
        """
        xy_coord = (9.5, 9.5)
        _, sw, sh, _ = pixel_ellipse_to_sky_svd(
            xy_coord, sip_wcs, 8.0, 4.0, 0.0)
        assert sw > 0
        assert sh > 0

    def test_sip_wcs_roundtrip(self, sip_wcs):
        """
        Roundtrip with SIP WCS should recover the original ellipse.
        """
        sky_w, sky_h, sky_a = 0.36, 0.18, 0.7
        center_pix, pw, ph, pa = sky_ellipse_to_pixel_svd(
            WCS_CENTER, sip_wcs, sky_w, sky_h, sky_a)
        _, rw, rh, ra = pixel_ellipse_to_sky_svd(
            center_pix, sip_wcs, pw, ph, pa.rad)
        assert_allclose(rw, sky_w, rtol=1e-5)
        assert_allclose(rh, sky_h, rtol=1e-5)
        assert_allclose(ra.rad, sky_a, rtol=1e-4)

    def test_angle_wrapped(self, simple_wcs, center_xy_coord):
        """
        The output angle should be in [0, 360) degrees.
        """
        _, _, _, angle = pixel_ellipse_to_sky_svd(
            center_xy_coord, simple_wcs, 10.0, 5.0, 0.5)
        assert 0.0 <= angle.deg < 360.0


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

    def test_simple_wcs_isotropic(self, simple_wcs):
        """
        For an isotropic WCS, major and minor scales should be equal.
        """
        _, smaj, smin, _ = sky_to_pixel_svd_scales(
            WCS_CENTER, simple_wcs)
        expected = 1.0 / WCS_CDELT_ARCSEC
        assert_allclose(smaj, expected, rtol=1e-5)
        assert_allclose(smin, expected, rtol=1e-5)

    def test_pixel_to_sky_simple_scales(self, simple_wcs, center_xy_coord):
        """
        For an isotropic WCS, pixel-to-sky scales should equal
        WCS_CDELT_ARCSEC.
        """
        _, smaj, smin, _ = pixel_to_sky_svd_scales(
            center_xy_coord, simple_wcs)
        assert_allclose(smaj, WCS_CDELT_ARCSEC, rtol=1e-5)
        assert_allclose(smin, WCS_CDELT_ARCSEC, rtol=1e-5)

    def test_major_geq_minor(self, simple_wcs):
        """
        The major scale should always be >= minor scale (SVD ordering).
        """
        _, smaj, smin, _ = sky_to_pixel_svd_scales(
            WCS_CENTER, simple_wcs)
        assert smaj >= smin

    def test_nonsquare_different_scales(self, nonsquare_wcs):
        """
        For non-square pixels, major and minor scales should differ.
        """
        _, smaj, smin, _ = sky_to_pixel_svd_scales(
            WCS_CENTER, nonsquare_wcs)
        assert not np.isclose(smaj, smin)

    def test_roundtrip_inverse(self, simple_wcs, center_xy_coord):
        """
        The product of sky->pixel major scale and pixel->sky major scale
        should be ~1.
        """
        _, smaj_s2p, smin_s2p, _ = sky_to_pixel_svd_scales(
            WCS_CENTER, simple_wcs)
        _, smaj_p2s, smin_p2s, _ = pixel_to_sky_svd_scales(
            center_xy_coord, simple_wcs)
        assert_allclose(smaj_s2p * smaj_p2s, 1.0, rtol=1e-6)
        assert_allclose(smin_s2p * smin_p2s, 1.0, rtol=1e-6)

    def test_angle_wrapped(self, simple_wcs):
        """
        The output angle should be in [0, 360) degrees.
        """
        _, _, _, angle = sky_to_pixel_svd_scales(
            WCS_CENTER, simple_wcs)
        assert 0.0 <= angle.deg < 360.0

    def test_sip_wcs_positive_scales(self, sip_wcs):
        """
        Scales should be positive for distorted WCS.
        """
        _, smaj, smin, _ = sky_to_pixel_svd_scales(
            WCS_CENTER, sip_wcs)
        assert smaj > 0
        assert smin > 0
