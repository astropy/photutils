# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the _wcs_helpers module.
"""

import astropy.units as u
import numpy as np
import pytest
from astropy.coordinates import Angle, SkyCoord
from astropy.wcs import WCS as APWCS
from astropy.wcs.wcsapi import HighLevelWCSWrapper
from numpy.testing import assert_allclose

from photutils.datasets import make_gwcs
from photutils.utils._optional_deps import HAS_GWCS
from photutils.utils._wcs_helpers import (compute_local_wcs_jacobian,
                                          compute_pixel_to_sky_jacobians,
                                          compute_pixel_to_sky_mean_scales,
                                          pixel_shape_to_sky_svd,
                                          pixel_to_sky_mean_scale,
                                          pixel_to_sky_svd_scales,
                                          sky_shape_to_pixel_svd,
                                          sky_to_pixel_mean_scale,
                                          sky_to_pixel_svd_scales)
from photutils.utils.tests.wcs_test_helpers import (WCS_CDELT_ARCSEC,
                                                    WCS_CENTER, CountingWCS,
                                                    make_sip_wcs)

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

    The SIP terms are tiny but nonzero, so the distortion code paths
    are exercised.
    """
    center = SkyCoord(ra_deg * u.deg, dec_deg * u.deg)
    return make_sip_wcs(center=center, coeffs={'A_2_0': 1e-6, 'B_0_2': 1e-6})


class TestComputeLocalWCSJacobian:
    """
    Tests for `compute_local_wcs_jacobian`.
    """

    def test_shape(self, simple_wcs):
        """
        The Jacobian must be a 2x2 array.
        """
        jac = compute_local_wcs_jacobian(simple_wcs, WCS_CENTER)
        assert jac.shape == (2, 2)

    def test_simple_wcs_diagonal(self, simple_wcs):
        """
        For an axis-aligned TAN WCS the Jacobian should be nearly
        diagonal with magnitudes ~ 1/WCS_CDELT_ARCSEC.
        """
        jac = compute_local_wcs_jacobian(simple_wcs, WCS_CENTER)
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
        jac = compute_local_wcs_jacobian(rotated_wcs, WCS_CENTER)
        sv = np.linalg.svd(jac, compute_uv=False)
        expected_scale = 1.0 / WCS_CDELT_ARCSEC
        assert_allclose(sv, expected_scale, rtol=1e-6)

    def test_sip_wcs(self, sip_wcs):
        """
        For a SIP WCS the Jacobian should still be close to the
        undistorted value near the reference pixel.
        """
        jac = compute_local_wcs_jacobian(sip_wcs, WCS_CENTER)
        sv = np.linalg.svd(jac, compute_uv=False)
        expected_scale = 1.0 / WCS_CDELT_ARCSEC
        assert_allclose(sv, expected_scale, rtol=1e-5)

    def test_inverse_of_forward(self, simple_wcs):
        """
        The Jacobian should be the inverse of the forward
        d(sky)/d(pixel) matrix derived from central differences over one
        pixel.
        """
        jac = compute_local_wcs_jacobian(simple_wcs, WCS_CENTER)
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
        jac = compute_local_wcs_jacobian(simple_wcs, WCS_CENTER)
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
        jac = compute_local_wcs_jacobian(wcs, skycoord)
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
        jac = compute_local_wcs_jacobian(wcs, skycoord)
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

    jacs = compute_pixel_to_sky_jacobians(wcs, x, y)
    assert jacs.shape == (3, 2, 2)

    for i in range(x.size):
        skycoord = wcs.pixel_to_world(x[i], y[i])
        jac_inv = compute_local_wcs_jacobian(wcs, skycoord)
        assert_allclose(jacs[i], np.linalg.inv(jac_inv), rtol=1e-4)


def test_compute_pixel_to_sky_jacobians_scale():
    """
    Test that an axis-aligned WCS gives |F| entries equal to the pixel
    scale in arcsec.
    """
    wcs = _make_rotated_wcs(0.0, scale_arcsec=0.5, crval=(150.0, 0.0))
    jacs = compute_pixel_to_sky_jacobians(wcs,
                                          np.array([50.0]), np.array([50.0]))
    assert_allclose(np.abs(jacs[0]),
                    [[0.5, 0.0], [0.0, 0.5]], atol=1e-4)


class TestMeanScale:
    """
    Tests for `sky_to_pixel_mean_scale` and `pixel_to_sky_mean_scale`.
    """

    def test_sky_to_pixel_return_types(self, simple_wcs):
        """
        Should return (tuple, float).
        """
        pix_position, scale = sky_to_pixel_mean_scale(simple_wcs, WCS_CENTER)
        assert isinstance(pix_position, tuple)
        assert isinstance(scale, float)

    def test_pixel_to_sky_return_type(self, simple_wcs, center_xy_coord):
        """
        Should return a float.
        """
        scale = pixel_to_sky_mean_scale(simple_wcs, center_xy_coord)
        assert isinstance(scale, float)

    def test_sky_to_pixel_simple_scale(self, simple_wcs):
        """
        For an isotropic WCS, the mean scale should equal
        1/WCS_CDELT_ARCSEC.
        """
        _, scale = sky_to_pixel_mean_scale(simple_wcs, WCS_CENTER)
        assert_allclose(scale, 1.0 / WCS_CDELT_ARCSEC)

    def test_pixel_to_sky_simple_scale(self, simple_wcs, center_xy_coord):
        """
        For an isotropic WCS, the mean scale should equal WCS_CDELT_ARCSEC.
        """
        scale = pixel_to_sky_mean_scale(simple_wcs, center_xy_coord)
        assert_allclose(scale, WCS_CDELT_ARCSEC)

    def test_sky_to_pixel_sip_scale(self, sip_wcs):
        """
        For a SIP WCS near the reference pixel, the mean scale should be
        close to the undistorted value.
        """
        _, scale = sky_to_pixel_mean_scale(sip_wcs, WCS_CENTER)
        assert_allclose(scale, 1.0 / WCS_CDELT_ARCSEC, rtol=1e-6)

    def test_pixel_to_sky_sip_scale(self, sip_wcs):
        """
        For a SIP WCS near the reference pixel, the mean scale should be
        close to the undistorted value.
        """
        scale = pixel_to_sky_mean_scale(sip_wcs, (9.5, 9.5))
        assert_allclose(scale, WCS_CDELT_ARCSEC, rtol=1e-6)

    @pytest.mark.parametrize('wcs_name', ['simple_wcs', 'rotated_wcs',
                                          'nonsquare_wcs', 'flipped_wcs',
                                          'swapped_wcs', 'sip_wcs'])
    def test_roundtrip_scale(self, wcs_name, request):
        """
        Sky -> pixel mean_scale * pixel -> sky mean_scale must be
        exactly 1, even where the Jacobian is anisotropic.
        """
        wcs = request.getfixturevalue(wcs_name)
        center_pix, s2p = sky_to_pixel_mean_scale(wcs, WCS_CENTER)
        p2s = pixel_to_sky_mean_scale(wcs, center_pix)
        assert_allclose(s2p * p2s, 1.0, rtol=1e-12)

    def test_roundtrip_scale_off_axis(self):
        """
        Far from the tangent point of a wide-field TAN projection the
        Jacobian is anisotropic even for square pixels, and the scale
        must still invert exactly.
        """
        wcs = APWCS(naxis=2)
        wcs.wcs.ctype = ['RA---TAN', 'DEC--TAN']
        wcs.wcs.crpix = [50.5, 50.5]
        wcs.wcs.crval = [WCS_CENTER.ra.deg, WCS_CENTER.dec.deg]
        wcs.wcs.cdelt = [-0.02, 0.02]
        pixcoord = (-50.5, 299.5)
        jac = compute_pixel_to_sky_jacobians(wcs, *pixcoord)[0]
        s_max, s_min = np.linalg.svd(jac, compute_uv=False)
        assert (s_max - s_min) / (s_max + s_min) > 1e-3
        p2s = pixel_to_sky_mean_scale(wcs, pixcoord)
        skycoord = wcs.pixel_to_world(*pixcoord)
        _, s2p = sky_to_pixel_mean_scale(wcs, skycoord)
        assert_allclose(s2p * p2s, 1.0, rtol=1e-12)

    def test_center_coordinates(self, simple_wcs):
        """
        The returned pix_position should match world_to_pixel.
        """
        pix_position, _ = sky_to_pixel_mean_scale(simple_wcs, WCS_CENTER)
        x_exp, y_exp = simple_wcs.world_to_pixel(WCS_CENTER)
        assert_allclose(pix_position[0], x_exp)
        assert_allclose(pix_position[1], y_exp)

    def test_nonsquare_mean_scale(self, nonsquare_wcs):
        """
        For non-square pixels the mean scale should be the geometric
        mean of the two singular values (1/cdelt_x and 1/cdelt_y in
        pix/arcsec).
        """
        _, scale = sky_to_pixel_mean_scale(nonsquare_wcs, WCS_CENTER)
        cdelt_x = 0.03 * 3600
        cdelt_y = 0.05 * 3600
        expected = 1.0 / np.sqrt(cdelt_x * cdelt_y)
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
        _, scale = sky_to_pixel_mean_scale(nonsquare_wcs, WCS_CENTER)
        _, scale_sip = sky_to_pixel_mean_scale(sip_wcs, WCS_CENTER)
        assert_allclose(scale_sip, scale, rtol=1e-8)


class TestSVDShapeConversions:
    """
    Tests for `pixel_shape_to_sky_svd` and `sky_shape_to_pixel_svd`.
    """

    def test_pixel_to_sky_return_types(self, simple_wcs, center_xy_coord):
        """
        Should return (float, float, Angle).
        """
        w, h, angle = pixel_shape_to_sky_svd(
            simple_wcs, center_xy_coord, 10.0, 5.0, 0.5)
        assert isinstance(w, (float, np.floating))
        assert isinstance(h, (float, np.floating))
        assert isinstance(angle, Angle)

    def test_sky_to_pixel_return_types(self, simple_wcs):
        """
        Should return (tuple, float, float, Angle).
        """
        center, w, h, angle = sky_shape_to_pixel_svd(
            simple_wcs, WCS_CENTER, 36.0, 18.0, 0.5)
        assert isinstance(center, tuple)
        assert isinstance(w, (float, np.floating))
        assert isinstance(h, (float, np.floating))
        assert isinstance(angle, Angle)

    @pytest.mark.parametrize('wcs_name', ['sip_wcs', 'nonsquare_wcs',
                                          'flipped_wcs'])
    def test_pixel_to_sky_array_shapes(self, wcs_name, request):
        """
        Array widths and heights give the per-shape scalar results in
        one low-level WCS evaluation. The middle shape is circular, so
        the circular-input angle path is exercised as well.
        """
        real_wcs = request.getfixturevalue(wcs_name)
        widths = np.array([10.0, 6.0, 4.0])
        heights = np.array([5.0, 6.0, 7.0])
        wcs = CountingWCS(real_wcs)
        w, h, angle = pixel_shape_to_sky_svd(wcs, (12.0, 7.0), widths,
                                             heights, 0.3)
        assert wcs.n_pixel_to_world_values == 1
        assert wcs.n_pixel_to_world == 0
        assert w.shape == h.shape == angle.shape == (3,)
        for i in range(3):
            w1, h1, a1 = pixel_shape_to_sky_svd(
                real_wcs, (12.0, 7.0), widths[i], heights[i], 0.3)
            assert_allclose(w[i], w1, rtol=1e-12)
            assert_allclose(h[i], h1, rtol=1e-12)
            assert_allclose(angle[i].deg, a1.deg, atol=1e-10)

    @pytest.mark.parametrize('wcs_name', ['sip_wcs', 'nonsquare_wcs',
                                          'flipped_wcs'])
    def test_sky_to_pixel_array_shapes(self, wcs_name, request):
        """
        Array widths and heights give the per-shape scalar results with
        one low-level WCS evaluation and, given the pixel position, no
        inversion.
        """
        real_wcs = request.getfixturevalue(wcs_name)
        widths = np.array([1.0, 0.6, 0.4])
        heights = np.array([0.5, 0.6, 0.7])
        pixcoord = tuple(float(v) for v in real_wcs.world_to_pixel(WCS_CENTER))
        wcs = CountingWCS(real_wcs)
        center, w, h, angle = sky_shape_to_pixel_svd(
            wcs, WCS_CENTER, widths, heights, 0.3, pixcoord=pixcoord)
        assert wcs.n_pixel_to_world_values == 1
        assert wcs.n_pixel_to_world == 0
        assert wcs.n_world_to_pixel == 0
        assert w.shape == h.shape == angle.shape == (3,)
        for i in range(3):
            c1, w1, h1, a1 = sky_shape_to_pixel_svd(
                real_wcs, WCS_CENTER, widths[i], heights[i], 0.3)
            assert_allclose(w[i], w1, rtol=1e-12)
            assert_allclose(h[i], h1, rtol=1e-12)
            assert_allclose(angle[i].deg, a1.deg, atol=1e-10)
            assert_allclose(center, c1, atol=1e-10)

    def test_shape_broadcasting(self, sip_wcs):
        """
        A scalar width broadcasts against an array of heights, and
        scalar inputs give scalar outputs.
        """
        w, h, angle = pixel_shape_to_sky_svd(sip_wcs, (12.0, 7.0), 8.0,
                                             [4.0, 8.0], 0.3)
        assert w.shape == h.shape == angle.shape == (2,)
        w1, h1, a1 = pixel_shape_to_sky_svd(sip_wcs, (12.0, 7.0), 8.0,
                                            8.0, 0.3)
        assert isinstance(w1, float)
        assert isinstance(h1, float)
        assert a1.isscalar
        assert_allclose(w[1], w1, rtol=1e-12)
        assert_allclose(angle[1].deg, a1.deg, atol=1e-10)

    def test_roundtrip_sky_pixel_sky(self, simple_wcs):
        """
        Sky -> pixel -> sky should recover the original ellipse.
        """
        sky_w, sky_h, sky_a = 36.0, 18.0, 0.5
        center_pix, pw, ph, pa = sky_shape_to_pixel_svd(
            simple_wcs, WCS_CENTER, sky_w, sky_h, sky_a)
        rw, rh, ra = pixel_shape_to_sky_svd(
            simple_wcs, center_pix, pw, ph, pa.rad)
        assert_allclose(rw, sky_w, rtol=1e-6)
        assert_allclose(rh, sky_h, rtol=1e-6)
        assert_allclose(ra.rad, sky_a, rtol=1e-4)

    def test_roundtrip_pixel_sky_pixel(self, simple_wcs, center_xy_coord):
        """
        Pixel -> sky -> pixel should recover the original ellipse.
        """
        pix_w, pix_h, pix_a = 10.0, 5.0, 0.3
        sw, sh, sa = pixel_shape_to_sky_svd(
            simple_wcs, center_xy_coord, pix_w, pix_h, pix_a)
        _, rw, rh, ra = sky_shape_to_pixel_svd(
            simple_wcs, WCS_CENTER, sw, sh, sa.rad)
        assert_allclose(rw, pix_w, rtol=1e-6)
        assert_allclose(rh, pix_h, rtol=1e-6)
        assert_allclose(ra.rad, pix_a, rtol=1e-4)

    def test_simple_wcs_width_height_scale(self, simple_wcs, center_xy_coord):
        """
        For a simple WCS, pixel dimensions should scale by
        WCS_CDELT_ARCSEC.
        """
        pix_w, pix_h = 10.0, 5.0
        sw, sh, _ = pixel_shape_to_sky_svd(
            simple_wcs, center_xy_coord, pix_w, pix_h, 0.0)
        assert_allclose(sw, pix_w * WCS_CDELT_ARCSEC, rtol=1e-5)
        assert_allclose(sh, pix_h * WCS_CDELT_ARCSEC, rtol=1e-5)

    def test_height_larger_than_width(self, simple_wcs, center_xy_coord):
        """
        When height > width, the SVD should still correctly assign
        widths and heights.
        """
        pix_w, pix_h = 5.0, 10.0
        sw, sh, _ = pixel_shape_to_sky_svd(
            simple_wcs, center_xy_coord, pix_w, pix_h, 0.0)
        # Width should be smaller than height in sky coords too
        assert sw < sh

    def test_sip_wcs_positive_sizes(self, sip_wcs):
        """
        Sizes should be positive for distorted WCS.
        """
        xy_coord = (9.5, 9.5)
        sw, sh, _ = pixel_shape_to_sky_svd(
            sip_wcs, xy_coord, 8.0, 4.0, 0.0)
        assert sw > 0
        assert sh > 0

    def test_sip_wcs_roundtrip(self, sip_wcs):
        """
        Roundtrip with SIP WCS should recover the original ellipse.
        """
        sky_w, sky_h, sky_a = 0.36, 0.18, 0.7
        center_pix, pw, ph, pa = sky_shape_to_pixel_svd(
            sip_wcs, WCS_CENTER, sky_w, sky_h, sky_a)
        rw, rh, ra = pixel_shape_to_sky_svd(
            sip_wcs, center_pix, pw, ph, pa.rad)
        assert_allclose(rw, sky_w, rtol=1e-5)
        assert_allclose(rh, sky_h, rtol=1e-5)
        assert_allclose(ra.rad, sky_a, rtol=1e-4)

    def test_angle_wrapped(self, simple_wcs, center_xy_coord):
        """
        The output angle should be in [0, 360) degrees.
        """
        _, _, angle = pixel_shape_to_sky_svd(
            simple_wcs, center_xy_coord, 10.0, 5.0, 0.5)
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
        sw, sh, sky_angle = pixel_shape_to_sky_svd(
            rotated_wcs, pixcoord, 6.0, 6.0, angle_rad)
        assert_allclose(sw, sh, rtol=1e-8)

        sky_center = rotated_wcs.pixel_to_world(*pixcoord)
        _, pw, ph, pix_angle = sky_shape_to_pixel_svd(
            rotated_wcs, sky_center, sw, sh, sky_angle.rad)
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
            wcs, skycoord, 2.0, 1.0, np.deg2rad(30))
        assert pw > 1.0
        assert ph > 1.0
        sw, sh, _ = pixel_shape_to_sky_svd(
            wcs, center, pw, ph, pa.to_value(u.radian))
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
            flipped_wcs, WCS_CENTER, 2 * a_arcsec, 2 * b_arcsec, pa_rad)

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
            simple_wcs, WCS_CENTER, 2 * a_arcsec, 2 * b_arcsec, pa_rad)

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
            flipped_wcs, WCS_CENTER, sky_w, sky_h, sky_a)
        rw, rh, ra = pixel_shape_to_sky_svd(
            flipped_wcs, center_pix, pw, ph, pa.rad)
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
            simple_wcs, WCS_CENTER)
        assert isinstance(center, tuple)
        assert isinstance(smaj, (float, np.floating))
        assert isinstance(smin, (float, np.floating))
        assert isinstance(angle, Angle)

    def test_pixel_to_sky_return_types(self, simple_wcs, center_xy_coord):
        """
        Should return (float, float, Angle).
        """
        smaj, smin, angle = pixel_to_sky_svd_scales(
            simple_wcs, center_xy_coord)
        assert isinstance(smaj, (float, np.floating))
        assert isinstance(smin, (float, np.floating))
        assert isinstance(angle, Angle)

    def test_sky_to_pixel_simple_isotropic(self, simple_wcs):
        """
        For a non-distorted, square-pixel WCS the two scale factors are
        equal and match 1 / WCS_CDELT_ARCSEC (pixels per arcsec).
        """
        _, smaj, smin, angle = sky_to_pixel_svd_scales(
            simple_wcs, WCS_CENTER)
        expected = 1.0 / WCS_CDELT_ARCSEC
        assert_allclose(smaj, expected, rtol=1e-6)
        assert_allclose(smin, expected, rtol=1e-6)
        assert 0.0 <= angle.deg < 360.0

    def test_pixel_to_sky_simple_isotropic(self, simple_wcs, center_xy_coord):
        """
        For a non-distorted, square-pixel WCS the two scale factors are
        equal and match WCS_CDELT_ARCSEC (arcsec per pixel).
        """
        smaj, smin, angle = pixel_to_sky_svd_scales(
            simple_wcs, center_xy_coord)
        assert_allclose(smaj, WCS_CDELT_ARCSEC, rtol=1e-6)
        assert_allclose(smin, WCS_CDELT_ARCSEC, rtol=1e-6)
        assert 0.0 <= angle.deg < 360.0

    def test_sky_to_pixel_nonsquare_anisotropic(self, nonsquare_wcs):
        """
        For non-square pixels the major scale exceeds the minor scale,
        and both match the inverse of the corresponding pixel scales.
        """
        _, smaj, smin, _ = sky_to_pixel_svd_scales(nonsquare_wcs, WCS_CENTER)
        assert smaj > smin
        # cdelt = [-0.03, 0.05] deg, so pixels/arcsec = 1 / (cdelt_arcsec)
        assert_allclose(smaj, 1.0 / (0.03 * 3600), rtol=1e-5)
        assert_allclose(smin, 1.0 / (0.05 * 3600), rtol=1e-5)

    def test_pixel_to_sky_nonsquare_anisotropic(self, nonsquare_wcs):
        """
        For non-square pixels the major scale exceeds the minor scale,
        and both match the corresponding pixel scales (arcsec/pixel).
        """
        smaj, smin, _ = pixel_to_sky_svd_scales(nonsquare_wcs, (9.5, 9.5))
        assert smaj > smin
        assert_allclose(smaj, 0.05 * 3600, rtol=1e-5)
        assert_allclose(smin, 0.03 * 3600, rtol=1e-5)

    def test_scales_descending(self, sip_wcs):
        """
        The returned scales are singular values in descending order, so
        the major scale is always >= the minor scale.
        """
        _, smaj_s, smin_s, _ = sky_to_pixel_svd_scales(sip_wcs, WCS_CENTER)
        smaj_p, smin_p, _ = pixel_to_sky_svd_scales(sip_wcs, (9.5, 9.5))
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
            wcs, WCS_CENTER)
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
            wcs, WCS_CENTER)
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
        smaj, smin, sky_angle = pixel_to_sky_svd_scales(wcs, pixcoord)
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
            wcs, WCS_CENTER)
        smaj_s, smin_s, _ = pixel_to_sky_svd_scales(wcs, center_pix)
        assert_allclose(smaj_s, 1.0 / smin_p, rtol=1e-6)
        assert_allclose(smin_s, 1.0 / smaj_p, rtol=1e-6)

    def test_angles_wrapped(self):
        """
        Both helpers return angles wrapped to [0, 360) degrees.
        """
        wcs = _make_sheared_wcs()
        _, _, _, pixel_angle = sky_to_pixel_svd_scales(wcs, WCS_CENTER)
        _, _, sky_angle = pixel_to_sky_svd_scales(wcs, (50.0, 50.0))
        assert 0.0 <= pixel_angle.deg < 360.0
        assert 0.0 <= sky_angle.deg < 360.0


def _make_quadratic_sip_wcs(coeff=1e-3):
    """
    Build a TAN-SIP WCS whose distortion is a pure quadratic in the
    pixel offset from CRPIX.

    At CRPIX the quadratic term has zero slope, so the true local scale
    is exactly CDELT. A one-sided finite difference is biased there by
    a fraction ``coeff``, while a central difference is exact for a
    quadratic.

    """
    center = SkyCoord(150.0 * u.deg, 0.0 * u.deg)
    coeffs = {'A_2_0': coeff, 'B_0_2': coeff}
    return make_sip_wcs((100, 100), center=center, coeffs=coeffs)


class TestCentralDifferences:
    """
    Tests that the finite-difference Jacobians are unbiased where the
    distortion has curvature.
    """

    def test_pixel_to_sky_jacobians_unbiased_at_crpix(self):
        wcs = _make_quadratic_sip_wcs()
        x = np.array([wcs.wcs.crpix[0] - 1.0])
        y = np.array([wcs.wcs.crpix[1] - 1.0])
        jac = compute_pixel_to_sky_jacobians(wcs, x, y)[0]
        assert_allclose(np.abs(np.diag(jac)), WCS_CDELT_ARCSEC, rtol=1e-6)

    def test_local_wcs_jacobian_unbiased_at_crval(self):
        wcs = _make_quadratic_sip_wcs()
        skycoord = SkyCoord(150.0 * u.deg, 0.0 * u.deg)
        jac = compute_local_wcs_jacobian(wcs, skycoord)
        assert_allclose(np.abs(np.diag(jac)), 1.0 / WCS_CDELT_ARCSEC,
                        rtol=1e-6)


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
        jacs = compute_pixel_to_sky_jacobians(bounded_gwcs, x, y)
        assert np.all(np.isfinite(jacs))

    def test_edge_jacobian_matches_interior(self, bounded_gwcs):
        # The gwcs has no distortion, so the Jacobian is the same
        # everywhere
        jacs = compute_pixel_to_sky_jacobians(bounded_gwcs,
                                              np.array([59.4, 30.0]),
                                              np.array([10.0, 25.0]))
        assert_allclose(jacs[0], jacs[1], rtol=1e-6)

    def test_local_wcs_jacobian_finite_at_edge(self, bounded_gwcs):
        skycoord = bounded_gwcs.pixel_to_world(59.4, 10.0)
        jac = compute_local_wcs_jacobian(bounded_gwcs, skycoord)
        assert np.all(np.isfinite(jac))

    def test_wrapped_gwcs_finite_at_edge(self, bounded_gwcs):
        """
        Test that a high-level wrapper around a gwcs object does not
        break the finite-difference Jacobian evaluation at the edge of
        the bounding box.
        """
        wrapped = HighLevelWCSWrapper(bounded_gwcs)
        jacs = compute_pixel_to_sky_jacobians(wrapped, 59.4, 10.0)
        expected = compute_pixel_to_sky_jacobians(bounded_gwcs, 59.4, 10.0)
        assert np.all(np.isfinite(jacs))
        assert_allclose(jacs, expected, rtol=1e-12)
        skycoord = bounded_gwcs.pixel_to_world(59.4, 10.0)
        jac = compute_local_wcs_jacobian(wrapped, skycoord)
        assert np.all(np.isfinite(jac))


class TestVectorizedMeanScales:
    """
    Tests for the vectorized mean pixel scale helper.

    It must reproduce the per-source function at every position,
    without calling the WCS inverse once per source.
    """

    positions = (np.array([3.0, 10.0, 16.5]), np.array([4.0, 10.0, 2.2]))

    @pytest.mark.parametrize('wcs_name', ['simple_wcs', 'rotated_wcs',
                                          'nonsquare_wcs', 'swapped_wcs',
                                          'sip_wcs'])
    def test_mean_scales_match_per_source(self, wcs_name, request):
        wcs = request.getfixturevalue(wcs_name)
        x, y = self.positions
        scales = compute_pixel_to_sky_mean_scales(wcs, x, y)
        assert scales.shape == (3,)
        for i in range(x.size):
            expected = pixel_to_sky_mean_scale(wcs, (x[i], y[i]))
            assert_allclose(scales[i], expected, rtol=1e-8)

    def test_scalar_inputs(self, simple_wcs):
        scales = compute_pixel_to_sky_mean_scales(simple_wcs, 10.0, 10.0)
        assert scales.shape == (1,)


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
        jacs = compute_pixel_to_sky_jacobians(sip_wcs, xx, yy)
        expected = compute_pixel_to_sky_jacobians(sip_wcs, xx.ravel(),
                                                  yy.ravel())
        assert jacs.shape == (12, 2, 2)
        assert_allclose(jacs, expected, rtol=1e-12)

    def test_size_mismatch(self, sip_wcs):
        match = 'x and y must have the same size'
        with pytest.raises(ValueError, match=match):
            compute_pixel_to_sky_jacobians(sip_wcs, [1.0, 2.0], [1.0])

    def test_single_low_level_wcs_call(self, sip_wcs):
        """
        Test that the vectorized Jacobian evaluation calls the WCS only
        once for all positions, rather than once per position.
        """
        # The Jacobian needs only world coordinate values, so the
        # high-level API (which builds SkyCoord objects) is not used.
        wcs = CountingWCS(sip_wcs)
        compute_pixel_to_sky_jacobians(wcs, np.array([5.0, 12.0]),
                                       np.array([7.0, 3.0]))
        assert wcs.n_pixel_to_world_values == 1
        assert wcs.n_pixel_to_world == 0
        assert wcs.n_world_to_pixel == 0

    @pytest.mark.parametrize('func', [
        pixel_to_sky_mean_scale,
        pixel_to_sky_svd_scales,
        lambda wcs, pixcoord: pixel_shape_to_sky_svd(wcs, pixcoord, 2.0, 1.0,
                                                     0.3),
    ], ids=['mean_scale', 'svd_scales', 'shape_svd'])
    def test_pixel_to_sky_helpers_single_call(self, sip_wcs, func):
        pixcoord = (5.0, 12.0)
        expected = func(sip_wcs, pixcoord)
        wcs = CountingWCS(sip_wcs)
        result = func(wcs, pixcoord)
        assert wcs.n_pixel_to_world_values == 1
        assert wcs.n_pixel_to_world == 0
        assert wcs.n_world_to_pixel == 0
        if not isinstance(result, tuple):
            result = (result,)
            expected = (expected,)
        for value, expected_value in zip(result, expected, strict=True):
            if isinstance(value, Angle):
                assert_allclose(value.deg, expected_value.deg, atol=1e-10)
            else:
                assert_allclose(value, expected_value, rtol=1e-12)

    def test_world_axis_units(self, sip_wcs):
        """
        Test that the vectorized Jacobian evaluation respects the
        ``world_axis_units`` of the WCS, rather than assuming degrees.
        """
        # The low-level API returns values in the WCS world axis units,
        # which must be converted rather than assumed to be degrees.
        class RadianWCS(CountingWCS):
            world_axis_units = ('rad', 'rad')

            def pixel_to_world_values(self, *args, **kwargs):
                lon, lat = super().pixel_to_world_values(*args, **kwargs)
                return np.deg2rad(lon), np.deg2rad(lat)

        x = np.array([5.0, 12.0])
        y = np.array([7.0, 3.0])
        expected = compute_pixel_to_sky_jacobians(sip_wcs, x, y)
        jacs = compute_pixel_to_sky_jacobians(RadianWCS(sip_wcs), x, y)
        assert_allclose(jacs, expected, rtol=1e-12)

    @pytest.mark.parametrize('wcs_name', ['simple_wcs', 'rotated_wcs',
                                          'nonsquare_wcs', 'flipped_wcs',
                                          'swapped_wcs', 'sip_wcs'])
    def test_agrees_with_separation_position_angle(self, wcs_name, request):
        wcs = request.getfixturevalue(wcs_name)
        x = np.array([3.0, 9.5, 16.2])
        y = np.array([4.0, 9.5, 2.7])
        jacs = compute_pixel_to_sky_jacobians(wcs, x, y)
        expected = _reference_jacobians(x, y, wcs)
        assert_allclose(jacs, expected, rtol=1e-7, atol=1e-7)

    def test_world_axis_order(self, simple_wcs, swapped_wcs):
        """
        Test that a WCS with the latitude axis first gives the same
        Jacobians as the equivalent WCS with the longitude axis first.
        """
        x = np.array([3.0, 9.5, 16.2])
        y = np.array([4.0, 9.5, 2.7])
        expected = compute_pixel_to_sky_jacobians(simple_wcs, x, y)
        jacs = compute_pixel_to_sky_jacobians(swapped_wcs, x, y)
        assert_allclose(jacs, expected, rtol=1e-9)
        skycoord = simple_wcs.pixel_to_world(x[0], y[0])
        expected = compute_local_wcs_jacobian(simple_wcs, skycoord)
        jac = compute_local_wcs_jacobian(swapped_wcs, skycoord)
        assert_allclose(jac, expected, rtol=1e-9)

    def test_non_celestial_wcs(self):
        """
        Test that a WCS without a celestial longitude and latitude pair
        is rejected with a clear message.
        """
        wcs = APWCS(naxis=2)
        wcs.wcs.ctype = ['LINEAR', 'LINEAR']
        wcs.wcs.cdelt = [0.1, 0.1]
        match = 'exactly one celestial longitude axis'
        with pytest.raises(ValueError, match=match):
            compute_pixel_to_sky_jacobians(wcs, 5.0, 5.0)

    @pytest.mark.parametrize(('center_ra', 'center_dec'), TROUBLESOME_CENTERS)
    def test_agrees_near_pole_and_wrap(self, center_ra, center_dec):
        wcs = _make_sip_wcs(center_ra, center_dec)
        x = np.array([9.5, 3.0])
        y = np.array([9.5, 15.0])
        jacs = compute_pixel_to_sky_jacobians(wcs, x, y)
        expected = _reference_jacobians(x, y, wcs)
        assert_allclose(jacs, expected, rtol=1e-7, atol=1e-7)

    @pytest.mark.parametrize('dec', [90.0, -90.0])
    def test_exact_pole(self, dec):
        # Longitude is degenerate at the pole, but the pixel area and
        # the Jacobian must stay finite and correct.
        wcs = _make_sip_wcs(0.0, dec)
        jac = compute_pixel_to_sky_jacobians(wcs, 9.5, 9.5)[0]
        assert np.all(np.isfinite(jac))
        assert_allclose(np.abs(np.linalg.det(jac)), WCS_CDELT_ARCSEC**2,
                        rtol=1e-6)


class TestCovarianceTransport:
    """
    Monte Carlo tests of the pixel-to-sky error covariance transport.

    A pixel error covariance is mapped to the local tangent plane as ``F
    @ cov @ F.T`` with the forward Jacobian ``F``. Pixel positions drawn
    from that covariance and converted with the high-level WCS interface
    must scatter on the sky by the transported amount, along East and
    North and in their correlation.
    """

    PIX_COV = np.array([[0.04, 0.01], [0.01, 0.09]])
    N_DRAW = 50_000

    @pytest.mark.parametrize('wcs_name', ['sheared', 'swapped_wcs',
                                          'sip_wcs'])
    def test_scatter_matches_transport(self, wcs_name, request):
        if wcs_name == 'sheared':
            wcs = _make_sheared_wcs()
            xy0 = np.array([50.0, 50.0])
        else:
            wcs = request.getfixturevalue(wcs_name)
            xy0 = np.array([12.3, 7.6])
        jac = compute_pixel_to_sky_jacobians(wcs, *xy0)[0]
        sky_cov = jac @ self.PIX_COV @ jac.T

        rng = np.random.default_rng(0)
        draws = rng.multivariate_normal(xy0, self.PIX_COV, size=self.N_DRAW)
        center = wcs.pixel_to_world(*xy0)
        coords = wcs.pixel_to_world(draws[:, 0], draws[:, 1])
        sep = center.separation(coords).rad
        pa = center.position_angle(coords).rad
        arcsec_per_rad = 3600.0 * np.degrees(1)
        east = sep * np.sin(pa) * arcsec_per_rad
        north = sep * np.cos(pa) * arcsec_per_rad
        mc_cov = np.cov(east, north)

        # The Monte Carlo precision of a standard deviation is 1 /
        # sqrt(2 N) = 0.3%, so the tolerances are several sigma.
        assert_allclose(np.sqrt(np.diag(mc_cov)), np.sqrt(np.diag(sky_cov)),
                        rtol=0.02)
        mc_corr = mc_cov[0, 1] / np.sqrt(mc_cov[0, 0] * mc_cov[1, 1])
        corr = sky_cov[0, 1] / np.sqrt(sky_cov[0, 0] * sky_cov[1, 1])
        assert_allclose(mc_corr, corr, atol=0.02)


class TestMeanScaleClosedForm:
    """
    Tests that the mean pixel scale equals the geometric mean of the
    singular values of the Jacobian.
    """

    @pytest.mark.parametrize('wcs_name', ['simple_wcs', 'rotated_wcs',
                                          'nonsquare_wcs', 'flipped_wcs',
                                          'swapped_wcs', 'sip_wcs'])
    def test_vectorized_matches_svd(self, wcs_name, request):
        wcs = request.getfixturevalue(wcs_name)
        x = np.array([3.0, 9.5, 16.2])
        y = np.array([4.0, 9.5, 2.7])
        scales = compute_pixel_to_sky_mean_scales(wcs, x, y)
        jacs = compute_pixel_to_sky_jacobians(wcs, x, y)
        expected = np.sqrt(np.prod(np.linalg.svd(jacs, compute_uv=False),
                                   axis=1))
        assert_allclose(scales, expected, rtol=1e-12)

    def test_scalar_matches_svd(self, nonsquare_wcs):
        _, scale = sky_to_pixel_mean_scale(nonsquare_wcs, WCS_CENTER)
        jac = compute_local_wcs_jacobian(nonsquare_wcs, WCS_CENTER)
        expected = np.sqrt(np.prod(np.linalg.svd(jac, compute_uv=False)))
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
        center, scale = sky_to_pixel_mean_scale(real_wcs, skycoord)
        wcs = CountingWCS(real_wcs)
        center2, scale2 = sky_to_pixel_mean_scale(wcs, skycoord,
                                                  pixcoord=pixcoord)
        assert wcs.n_world_to_pixel == 0
        assert_allclose(center2, center, atol=1e-10)
        assert_allclose(scale2, scale, rtol=1e-12)

    def test_svd_scales(self, sip_wcs, known):
        skycoord, pixcoord = known
        expected = sky_to_pixel_svd_scales(sip_wcs, skycoord)
        wcs = CountingWCS(sip_wcs)
        result = sky_to_pixel_svd_scales(wcs, skycoord, pixcoord=pixcoord)
        assert wcs.n_world_to_pixel == 0
        assert_allclose(result[0], expected[0], atol=1e-10)
        assert_allclose(result[1:3], expected[1:3], rtol=1e-12)
        assert_allclose(result[3].deg, expected[3].deg, atol=1e-10)

    def test_shape_svd(self, sip_wcs, known):
        skycoord, pixcoord = known
        args = (2.0, 1.0, 0.3)
        expected = sky_shape_to_pixel_svd(sip_wcs, skycoord, *args)
        wcs = CountingWCS(sip_wcs)
        result = sky_shape_to_pixel_svd(wcs, skycoord, *args,
                                        pixcoord=pixcoord)
        assert wcs.n_world_to_pixel == 0
        assert_allclose(result[0], expected[0], atol=1e-10)
        assert_allclose(result[1:3], expected[1:3], rtol=1e-12)
        assert_allclose(result[3].deg, expected[3].deg, atol=1e-10)
