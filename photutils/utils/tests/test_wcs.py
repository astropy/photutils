# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the wcs module.
"""

import astropy.units as u
import numpy as np
import pytest
from astropy.io.fits import Header
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_area
from numpy.testing import assert_allclose

from photutils.datasets import make_gwcs
from photutils.utils._optional_deps import HAS_GWCS
from photutils.utils.tests.conftest import WCS_CDELT_ARCSEC, WCS_CENTER
from photutils.utils.wcs import compute_pixel_areas, pixel_area_map

UNIFORM_AREA = WCS_CDELT_ARCSEC**2


def _make_sip_wcs(shape, coeff=2e-5):
    """
    Build a TAN-SIP WCS whose pixel area varies smoothly across an array
    of the given shape.
    """
    header = Header()
    header['NAXIS'] = 2
    header['NAXIS1'] = shape[1]
    header['NAXIS2'] = shape[0]
    header['CRPIX1'] = shape[1] / 2
    header['CRPIX2'] = shape[0] / 2
    header['CRVAL1'] = WCS_CENTER.ra.deg
    header['CRVAL2'] = WCS_CENTER.dec.deg
    header['CTYPE1'] = 'RA---TAN-SIP'
    header['CTYPE2'] = 'DEC--TAN-SIP'
    cdelt = WCS_CDELT_ARCSEC / 3600.0
    header['CD1_1'] = -cdelt
    header['CD1_2'] = 0.0
    header['CD2_1'] = 0.0
    header['CD2_2'] = cdelt
    header['A_ORDER'] = 2
    header['A_2_0'] = coeff
    header['B_ORDER'] = 2
    header['B_0_2'] = coeff
    return WCS(header)


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

    def test_step_independent(self):
        shape = (300, 200)
        wcs = _make_sip_wcs(shape)
        fine = pixel_area_map(wcs, shape, step=16)
        coarse = pixel_area_map(wcs, shape, step=128)
        assert_allclose(fine, coarse, rtol=1e-6)

    @pytest.mark.parametrize('step', [0, -4, 2.5])
    def test_invalid_step(self, simple_wcs, step):
        match = 'step must be a positive integer'
        with pytest.raises(ValueError, match=match):
            pixel_area_map(simple_wcs, (20, 20), step=step)

    @pytest.mark.parametrize('shape', [(20,), (20, 20, 20), (0, 20),
                                       (20, -1)])
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
