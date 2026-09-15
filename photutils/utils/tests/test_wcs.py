# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the wcs module.
"""

import warnings

import astropy.units as u
import numpy as np
import pytest
from astropy.utils.exceptions import AstropyUserWarning
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_area
from numpy.testing import assert_allclose

from photutils.datasets import make_gwcs
from photutils.utils._optional_deps import HAS_GWCS
from photutils.utils.tests.wcs_test_helpers import (WCS_CDELT_ARCSEC,
                                                    WCS_CENTER, make_sip_wcs)
from photutils.utils.wcs import compute_pixel_area_map, compute_pixel_areas

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
    spline in ``compute_pixel_area_map``, so it cannot detect an interpolation
    error. The gnomonic area factor of a wide field can.
    """
    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [shape[1] / 2 + 0.5, shape[0] / 2 + 0.5]
    wcs.wcs.crval = [WCS_CENTER.ra.deg, WCS_CENTER.dec.deg]
    wcs.wcs.cdelt = [-deg_per_pix, deg_per_pix]
    wcs.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    return wcs


ALLSKY_SHAPE = (180, 360)


def _make_allsky_wcs(projection):
    """
    Build an all-sky WCS with 1 degree pixels in the given
    projection (e.g., ``'CAR'`` or ``'AIT'``) for an array of shape
    ``ALLSKY_SHAPE``.
    """
    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [ALLSKY_SHAPE[1] / 2 + 0.5, ALLSKY_SHAPE[0] / 2 + 0.5]
    wcs.wcs.crval = [0.0, 0.0]
    wcs.wcs.cdelt = [-1.0, 1.0]
    wcs.wcs.ctype = [f'RA---{projection}', f'DEC--{projection}']
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

    @pytest.mark.parametrize('bad', [np.nan, np.inf, -np.inf])
    def test_nonfinite_position(self, simple_wcs, bad):
        x = np.array([10.0, bad, 10.0])
        y = np.array([10.0, 10.0, bad])
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            areas = compute_pixel_areas(simple_wcs, x, y)
        assert_allclose(areas[0], UNIFORM_AREA, rtol=1e-6)
        assert np.all(np.isnan(areas[1:]))

    def test_outside_projection(self):
        # An all-sky Aitoff map has corners outside the projection
        wcs = _make_allsky_wcs('AIT')
        areas = compute_pixel_areas(wcs, [0.0, 180.0], [0.0, 90.0])
        assert np.isnan(areas[0])
        assert np.isfinite(areas[1])
        assert areas[1] > 0

    def test_varies_with_distortion(self):
        wcs = _make_sip_wcs((200, 200))
        areas = compute_pixel_areas(wcs, np.array([10.0, 190.0]),
                                    np.array([10.0, 190.0]))
        assert not np.isclose(areas[0], areas[1], rtol=1e-4)


class TestPixelAreaMap:
    def test_uniform_wcs(self, simple_wcs):
        shape = (20, 30)
        area = compute_pixel_area_map(simple_wcs, shape)
        assert isinstance(area, np.ndarray)
        assert area.shape == shape
        assert_allclose(area, UNIFORM_AREA, rtol=1e-6)

    def test_matches_point_values_with_distortion(self):
        shape = (300, 200)
        wcs = _make_sip_wcs(shape)
        area = compute_pixel_area_map(wcs, shape)

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
        area = compute_pixel_area_map(wcs, shape)
        assert area.shape == shape
        yy, xx = np.mgrid[:shape[0], :shape[1]]
        expected = compute_pixel_areas(wcs, xx.ravel(), yy.ravel())
        assert_allclose(area.ravel(), expected, rtol=1e-6)

    def test_matches_point_values_wide_field(self):
        shape = (300, 200)
        wcs = _make_wide_tan_wcs(shape)
        area = compute_pixel_area_map(wcs, shape)

        # The gnomonic projection must actually produce a gradient
        assert np.ptp(area) / area.mean() > 1e-3

        rng = np.random.default_rng(0)
        y = rng.integers(0, shape[0], 50)
        x = rng.integers(0, shape[1], 50)
        expected = compute_pixel_areas(wcs, x, y)
        assert_allclose(area[y, x], expected, rtol=1e-6)

    def test_step_independent(self):
        shape = (1040, 1040)
        wcs = _make_wide_tan_wcs(shape)
        fine = compute_pixel_area_map(wcs, shape, step=16)
        coarse = compute_pixel_area_map(wcs, shape, step=128)
        assert_allclose(fine, coarse, rtol=1e-6)

    @pytest.mark.parametrize('step', [7, 1000])
    def test_step_capped(self, step):
        """
        Test that a step larger than min(shape) // 8 is reduced to
        it with a warning, so the result is identical to that of the
        largest allowed step and differs from a smaller step.
        """
        shape = (64, 48)
        wcs = _make_wide_tan_wcs(shape, deg_per_pix=0.2)
        capped = compute_pixel_area_map(wcs, shape, step=6)
        match = f'step={step} was reduced to 6 '
        with pytest.warns(AstropyUserWarning, match=match):
            area = compute_pixel_area_map(wcs, shape, step=step)
        assert np.array_equal(area, capped)
        smaller = compute_pixel_area_map(wcs, shape, step=5)
        assert not np.array_equal(smaller, capped)

    def test_default_step_small_image(self):
        """
        Test that the default step is silently reduced on a small image
        and equals the largest allowed step.
        """
        shape = (64, 48)
        wcs = _make_wide_tan_wcs(shape, deg_per_pix=0.2)
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            area = compute_pixel_area_map(wcs, shape)
            expected = compute_pixel_area_map(wcs, shape, step=6)
        assert np.array_equal(area, expected)

    def test_default_step_large_image(self):
        shape = (600, 520)
        wcs = _make_wide_tan_wcs(shape)
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            area = compute_pixel_area_map(wcs, shape)
            expected = compute_pixel_area_map(wcs, shape, step=64)
        assert np.array_equal(area, expected)

    @pytest.mark.parametrize('projection', ['CAR', 'AIT'])
    def test_coarse_grid_outside_projection(self, projection):
        """
        Test all-sky maps whose padded coarse grid leaves the valid
        region of the projection.

        Every pixel of the CAR map is valid, while the AIT map has
        corners outside the projection. In both cases the map must
        match the directly evaluated areas pixel for pixel, including
        the NaN pattern.
        """
        wcs = _make_allsky_wcs(projection)
        yy, xx = np.mgrid[:ALLSKY_SHAPE[0], :ALLSKY_SHAPE[1]]
        expected = compute_pixel_areas(wcs, xx, yy)
        area = compute_pixel_area_map(wcs, ALLSKY_SHAPE)
        assert area.shape == ALLSKY_SHAPE
        assert_allclose(area, expected, rtol=1e-12, equal_nan=True)

        finite = np.isfinite(expected)
        if projection == 'CAR':
            assert np.all(finite)
        else:
            assert 0 < finite.mean() < 1

    @pytest.mark.parametrize('step', [0, -4, 2.5, True])
    def test_invalid_step(self, simple_wcs, step):
        match = 'step must be a positive integer or None'
        with pytest.raises(ValueError, match=match):
            compute_pixel_area_map(simple_wcs, (20, 20), step=step)

    @pytest.mark.parametrize('shape', [(20,), (20, 20, 20), (0, 20),
                                       (20, -1), (20.0, 20), (True, 20)])
    def test_invalid_shape(self, simple_wcs, shape):
        match = 'shape must be two positive integers'
        with pytest.raises(ValueError, match=match):
            compute_pixel_area_map(simple_wcs, shape)

    @pytest.mark.skipif(not HAS_GWCS, reason='gwcs is required')
    def test_gwcs_bounding_box(self):
        shape = (40, 50)
        gwcs = make_gwcs(shape)
        gwcs.bounding_box = ((-0.5, shape[1] - 0.5), (-0.5, shape[0] - 0.5))
        area = compute_pixel_area_map(gwcs, shape)
        assert np.all(np.isfinite(area))
        # make_gwcs has a uniform 0.1 arcsec pixel scale
        assert_allclose(area, 0.01, rtol=1e-6)

    def test_matches_astropy_proj_plane_area(self, simple_wcs):
        expected = (proj_plane_pixel_area(simple_wcs) * u.deg**2).to_value(
            u.arcsec**2)
        area = compute_pixel_area_map(simple_wcs, (20, 20))
        assert_allclose(area, expected, rtol=1e-6)
