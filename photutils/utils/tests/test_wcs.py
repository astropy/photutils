# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the wcs module.
"""

import astropy.units as u
import numpy as np
import pytest
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_area
from numpy.testing import assert_allclose

from photutils.datasets import make_gwcs
from photutils.utils._optional_deps import HAS_GWCS
from photutils.utils.tests.wcs_test_helpers import (WCS_CDELT_ARCSEC,
                                                    WCS_CENTER, make_sip_wcs)
from photutils.utils.wcs import compute_pixel_areas, pixel_area_map

UNIFORM_AREA = WCS_CDELT_ARCSEC**2


def _make_sip_wcs(shape, coeff=2e-5):
    """
    Build a TAN-SIP WCS whose pixel area varies smoothly across an array
    of the given shape.
    """
    return make_sip_wcs(shape, coeffs={'A_2_0': coeff, 'B_0_2': coeff})


def _make_wide_tan_wcs(shape, deg_per_pix=0.01):
    """
    Build a TAN WCS whose pixels are large enough that the pixel area
    varies non-polynomially across the field.

    A quadratic SIP distortion is reproduced exactly by the bicubic
    spline in ``pixel_area_map``, so it cannot detect an interpolation
    error. The gnomonic area factor of a wide field can.
    """
    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [shape[1] / 2 + 0.5, shape[0] / 2 + 0.5]
    wcs.wcs.crval = [WCS_CENTER.ra.deg, WCS_CENTER.dec.deg]
    wcs.wcs.cdelt = [-deg_per_pix, deg_per_pix]
    wcs.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    return wcs


def _make_cd_wcs(cd_deg):
    """
    Build a TAN WCS with the given CD matrix in degrees per pixel.
    """
    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [10.5, 10.5]
    wcs.wcs.crval = [WCS_CENTER.ra.deg, WCS_CENTER.dec.deg]
    wcs.wcs.cd = cd_deg
    wcs.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    return wcs


class TestComputePixelAreas:
    def test_uniform_wcs(self, simple_wcs):
        x = np.array([2.0, 10.0, 17.3])
        y = np.array([1.5, 10.0, 4.0])
        areas = compute_pixel_areas(simple_wcs, x, y)
        assert areas.shape == (3,)
        assert_allclose(areas, UNIFORM_AREA, rtol=1e-6)

    def test_scalar_input(self, simple_wcs):
        area = compute_pixel_areas(simple_wcs, 10.0, 10.0)
        assert isinstance(area, float)
        assert_allclose(area, UNIFORM_AREA, rtol=1e-6)

    @pytest.mark.parametrize('shape', [(3, 7), (4, 3), (2, 3, 4)])
    def test_input_shape_preserved(self, shape):
        wcs = _make_sip_wcs((200, 200))
        rng = np.random.default_rng(0)
        x = rng.uniform(0, 199, shape)
        y = rng.uniform(0, 199, shape)
        areas = compute_pixel_areas(wcs, x, y)
        assert areas.shape == shape
        expected = compute_pixel_areas(wcs, x.ravel(), y.ravel())
        assert_allclose(areas.ravel(), expected, rtol=1e-12)

    def test_shape_mismatch(self, simple_wcs):
        match = 'x and y must have the same shape'
        with pytest.raises(ValueError, match=match):
            compute_pixel_areas(simple_wcs, [1.0, 2.0], [1.0, 2.0, 3.0])

    def test_nonsquare_pixels(self, nonsquare_wcs):
        expected = (0.03 * 3600) * (0.05 * 3600)
        area = compute_pixel_areas(nonsquare_wcs, 10.0, 10.0)
        assert_allclose(area, expected, rtol=1e-6)

    def test_sheared_wcs(self):
        cd = np.array([[-2.0e-5, 0.5e-5], [0.3e-5, 3.0e-5]])
        wcs = _make_cd_wcs(cd)
        expected = np.abs(np.linalg.det(cd)) * 3600**2
        area = compute_pixel_areas(wcs, 10.0, 10.0)
        assert_allclose(area, expected, rtol=1e-6)

    def test_flipped_parity_is_positive(self, flipped_wcs):
        area = compute_pixel_areas(flipped_wcs, 10.0, 10.0)
        assert area > 0
        assert_allclose(area, UNIFORM_AREA, rtol=1e-6)

    def test_varies_with_distortion(self):
        wcs = _make_sip_wcs((200, 200))
        areas = compute_pixel_areas(wcs, np.array([10.0, 190.0]),
                                    np.array([10.0, 190.0]))
        assert not np.isclose(areas[0], areas[1], rtol=1e-4)


class TestPixelAreaMap:
    def test_uniform_wcs(self, simple_wcs):
        shape = (20, 30)
        area = pixel_area_map(simple_wcs, shape)
        assert isinstance(area, np.ndarray)
        assert area.shape == shape
        assert_allclose(area, UNIFORM_AREA, rtol=1e-6)

    def test_matches_point_values_with_distortion(self):
        shape = (300, 200)
        wcs = _make_sip_wcs(shape)
        area = pixel_area_map(wcs, shape)

        # The distortion must actually produce a gradient
        assert np.ptp(area) / area.mean() > 1e-3

        rng = np.random.default_rng(0)
        y = rng.integers(0, shape[0], 50)
        x = rng.integers(0, shape[1], 50)
        expected = compute_pixel_areas(wcs, x, y)
        assert_allclose(area[y, x], expected, rtol=1e-6)

    @pytest.mark.parametrize('shape', [(1, 1), (2, 5), (10, 10), (65, 3),
                                       (3, 130)])
    def test_small_shapes(self, shape):
        wcs = _make_sip_wcs(shape)
        area = pixel_area_map(wcs, shape)
        assert area.shape == shape
        yy, xx = np.mgrid[:shape[0], :shape[1]]
        expected = compute_pixel_areas(wcs, xx.ravel(), yy.ravel())
        assert_allclose(area.ravel(), expected, rtol=1e-6)

    def test_matches_point_values_wide_field(self):
        shape = (300, 200)
        wcs = _make_wide_tan_wcs(shape)
        area = pixel_area_map(wcs, shape)

        # The gnomonic projection must actually produce a gradient
        assert np.ptp(area) / area.mean() > 1e-3

        rng = np.random.default_rng(0)
        y = rng.integers(0, shape[0], 50)
        x = rng.integers(0, shape[1], 50)
        expected = compute_pixel_areas(wcs, x, y)
        assert_allclose(area[y, x], expected, rtol=1e-6)

    def test_step_independent(self):
        shape = (300, 200)
        wcs = _make_wide_tan_wcs(shape)
        fine = pixel_area_map(wcs, shape, step=16)
        coarse = pixel_area_map(wcs, shape, step=128)
        assert_allclose(fine, coarse, rtol=1e-6)

    def test_step_capped(self):
        """
        Test that a step larger than min(shape) // 8 is reduced to
        it, so the result is identical to that of the capped step and
        differs from what the uncapped coarse grid would give.
        """
        shape = (64, 48)
        wcs = _make_wide_tan_wcs(shape, deg_per_pix=0.2)
        capped = pixel_area_map(wcs, shape, step=6)
        assert np.array_equal(pixel_area_map(wcs, shape, step=1000), capped)
        assert np.array_equal(pixel_area_map(wcs, shape, step=7), capped)
        assert not np.array_equal(pixel_area_map(wcs, shape, step=5), capped)

    @pytest.mark.parametrize('step', [0, -4, 2.5, True])
    def test_invalid_step(self, simple_wcs, step):
        match = 'step must be a positive integer'
        with pytest.raises(ValueError, match=match):
            pixel_area_map(simple_wcs, (20, 20), step=step)

    @pytest.mark.parametrize('shape', [(20,), (20, 20, 20), (0, 20),
                                       (20, -1), (20.0, 20), (True, 20)])
    def test_invalid_shape(self, simple_wcs, shape):
        match = 'shape must be two positive integers'
        with pytest.raises(ValueError, match=match):
            pixel_area_map(simple_wcs, shape)

    @pytest.mark.skipif(not HAS_GWCS, reason='gwcs is required')
    def test_gwcs_bounding_box(self):
        shape = (40, 50)
        gwcs = make_gwcs(shape)
        gwcs.bounding_box = ((-0.5, shape[1] - 0.5), (-0.5, shape[0] - 0.5))
        area = pixel_area_map(gwcs, shape)
        assert np.all(np.isfinite(area))
        # make_gwcs has a uniform 0.1 arcsec pixel scale
        assert_allclose(area, 0.01, rtol=1e-6)

    def test_matches_astropy_proj_plane_area(self, simple_wcs):
        expected = (proj_plane_pixel_area(simple_wcs) * u.deg**2).to_value(
            u.arcsec**2)
        area = pixel_area_map(simple_wcs, (20, 20))
        assert_allclose(area, expected, rtol=1e-6)
