# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the curve_of_growth module.
"""

import astropy.units as u
import numpy as np
import pytest
from astropy.modeling.models import Gaussian2D
from astropy.utils.exceptions import AstropyUserWarning
from numpy.testing import assert_allclose, assert_equal

from photutils.aperture import (CircularAperture, EllipticalAperture,
                                RectangularAperture)
from photutils.profiles import (CurveOfGrowth, EllipticalCurveOfGrowth,
                                EnsquaredCurveOfGrowth)
from photutils.utils._optional_deps import HAS_MATPLOTLIB


class TestCurveOfGrowth:
    def test_basic(self, profile_data):
        """
        Test basic CurveOfGrowth functionality.
        """
        xycen, data, _, _ = profile_data

        radii = np.arange(1, 37)
        cg1 = CurveOfGrowth(data, xycen, radii, error=None, mask=None)

        assert_equal(cg1.radius, radii)
        assert cg1.area.shape == (36,)
        assert cg1.profile.shape == (36,)
        assert cg1.profile_error.shape == (0,)
        assert_allclose(cg1.area[0], np.pi)

        assert len(cg1.apertures) == 36
        assert isinstance(cg1.apertures[0], CircularAperture)

        radii = np.arange(1, 36)
        cg2 = CurveOfGrowth(data, xycen, radii, error=None, mask=None)
        assert cg2.area[0] > 0.0
        assert isinstance(cg2.apertures[0], CircularAperture)

    def test_units(self, profile_data):
        """
        Test CurveOfGrowth with units.
        """
        xycen, data, error, _ = profile_data

        radii = np.arange(1, 36)
        unit = u.Jy
        cg1 = CurveOfGrowth(data << unit, xycen, radii,
                            error=error << unit, mask=None)

        assert cg1.profile.unit == unit
        assert cg1.profile_error.unit == unit

        match = 'must all have the same units'
        with pytest.raises(ValueError, match=match):
            CurveOfGrowth(data << unit, xycen, radii, error=error, mask=None)

    def test_error(self, profile_data):
        """
        Test CurveOfGrowth with error array.
        """
        xycen, data, error, _ = profile_data

        radii = np.arange(1, 36)
        cg1 = CurveOfGrowth(data, xycen, radii, error=error, mask=None)

        assert cg1.profile.shape == (35,)
        assert cg1.profile_error.shape == (35,)

    def test_mask(self, profile_data):
        """
        Test CurveOfGrowth with a mask.
        """
        xycen, data, error, mask = profile_data

        radii = np.arange(1, 36)
        cg1 = CurveOfGrowth(data, xycen, radii, error=error, mask=None)
        cg2 = CurveOfGrowth(data, xycen, radii, error=error, mask=mask)

        assert cg1.profile.sum() > cg2.profile.sum()
        assert np.sum(cg1.profile_error**2) > np.sum(cg2.profile_error**2)

    def test_normalize(self, profile_data):
        """
        Test CurveOfGrowth normalize and unnormalize methods.
        """
        xycen, data, _, _ = profile_data

        radii = np.arange(1, 36)
        cg1 = CurveOfGrowth(data, xycen, radii, error=None, mask=None)
        cg2 = CurveOfGrowth(data, xycen, radii, error=None, mask=None)

        profile1 = cg1.profile
        cg1.normalize()
        profile2 = cg1.profile
        assert np.mean(profile2) < np.mean(profile1)

        cg1.unnormalize()
        assert_allclose(cg1.profile, cg2.profile)

        cg1.normalize(method='sum')
        cg1.normalize(method='max')
        cg1.unnormalize()
        assert_allclose(cg1.profile, cg2.profile)

        cg1.normalize(method='max')
        cg1.normalize(method='sum')
        cg1.normalize(method='max')
        cg1.normalize(method='max')
        cg1.unnormalize()
        assert_allclose(cg1.profile, cg2.profile)

        cg1.normalize(method='sum')
        profile3 = cg1.profile
        assert np.mean(profile3) < np.mean(profile1)

        cg1.unnormalize()
        assert_allclose(cg1.profile, cg2.profile)

        match = "invalid method, must be 'max' or 'sum'"
        with pytest.raises(ValueError, match=match):
            cg1.normalize(method='invalid')

        cg1.__dict__['_raw_profile'] = cg1.profile - np.nanmax(cg1.profile)
        match = 'The profile cannot be normalized'
        with pytest.warns(AstropyUserWarning, match=match):
            cg1.normalize(method='max')

    def test_interp(self, profile_data):
        """
        Test CurveOfGrowth encircled energy interpolation methods.
        """
        xycen, data, _, _ = profile_data
        radii = np.arange(1, 36)
        cg1 = CurveOfGrowth(data, xycen, radii, error=None, mask=None)
        cg1.normalize()
        ee_radii = np.array([0, 5, 10, 20, 25, 50], dtype=float)
        ee_vals = cg1.calc_ee_at_radius(ee_radii)
        ee_expected = np.array([np.nan, 0.1176754, 0.39409357, 0.86635049,
                                0.95805792, np.nan])
        assert_allclose(ee_vals, ee_expected, rtol=1e-6)

        rr = cg1.calc_radius_at_ee(ee_vals)
        ee_radii[[0, -1]] = np.nan
        assert_allclose(rr, ee_radii, rtol=1e-6)

        radii = np.linspace(0.1, 36, 200)
        cg1 = CurveOfGrowth(data, xycen, radii, error=None, mask=None,
                            method='center')
        ee_vals = cg1.calc_ee_at_radius(ee_radii)
        rr = cg1.calc_radius_at_ee(ee_vals)
        assert_allclose(cg1.calc_ee_at_radius(rr), ee_vals)

    def test_inputs(self, profile_data):
        """
        Test CurveOfGrowth input validation.
        """
        xycen, data, error, _ = profile_data

        match = 'radii must be > 0'
        radii = np.arange(10)
        with pytest.raises(ValueError, match=match):
            CurveOfGrowth(data, xycen, radii, error=None, mask=None)

        match = 'radii must be a 1D array and have at least two values'
        with pytest.raises(ValueError, match=match):
            CurveOfGrowth(data, xycen, [1], error=None, mask=None)
        with pytest.raises(ValueError, match=match):
            CurveOfGrowth(data, xycen, np.arange(1, 7).reshape(2, 3),
                          error=None, mask=None)

        match = 'radii must be strictly increasing'
        radii = np.arange(1, 10)[::-1]
        with pytest.raises(ValueError, match=match):
            CurveOfGrowth(data, xycen, radii, error=None, mask=None)

        unit1 = u.Jy
        unit2 = u.km
        radii = np.arange(1, 36)
        match = 'must all have the same units'
        with pytest.raises(ValueError, match=match):
            CurveOfGrowth(data << unit1, xycen, radii,
                          error=error << unit2)

    def test_no_mutation(self, profile_data):
        """
        Test that input data, error, mask, and radii arrays are not
        mutated by CurveOfGrowth.
        """
        xycen, data, error, mask = profile_data

        # Introduce a NaN to trigger the badmask / mask-merge code path.
        # Use a pixel outside the fixture's masked region (mask[:39, :50]).
        data2 = data.copy()
        data2[50, 70] = np.nan
        mask2 = mask.copy()
        data2_orig = data2.copy()
        error_orig = error.copy()
        mask2_orig = mask2.copy()
        radii = np.arange(1, 36)
        radii_orig = radii.copy()

        match = 'Input data contains nonfinite values'
        with pytest.warns(AstropyUserWarning, match=match):
            CurveOfGrowth(data2, xycen, radii, error=error, mask=mask2)

        assert_equal(data2, data2_orig)
        assert_equal(error, error_orig)
        assert_equal(mask2, mask2_orig)
        assert_equal(radii, radii_orig)

    @pytest.mark.skipif(not HAS_MATPLOTLIB, reason='matplotlib is required')
    def test_plot(self, profile_data):
        """
        Test CurveOfGrowth plot methods.
        """
        xycen, data, error, _ = profile_data

        radii = np.arange(1, 36)
        cg1 = CurveOfGrowth(data, xycen, radii, error=None, mask=None)
        cg1.plot()
        match = 'Errors were not input'
        with pytest.warns(AstropyUserWarning, match=match):
            cg1.plot_error()

        cg2 = CurveOfGrowth(data, xycen, radii, error=error, mask=None)
        cg2.plot()
        pc1 = cg2.plot_error()
        assert_allclose(pc1.get_facecolor(), [[0.5, 0.5, 0.5, 0.3]])
        pc2 = cg2.plot_error(facecolor='blue')
        assert_allclose(pc2.get_facecolor(), [[0, 0, 1, 1]])

        unit = u.Jy
        cg3 = CurveOfGrowth(data << unit, xycen, radii,
                            error=error << unit, mask=None)
        cg3.plot()
        cg3.plot_error()

    def test_all_masked(self, profile_data):
        """
        Test CurveOfGrowth with all data masked.

        When every pixel is masked the profile should be all zero
        (zero flux in every aperture).
        """
        xycen, data, _, _ = profile_data

        all_mask = np.ones(data.shape, dtype=bool)
        radii = np.arange(1, 36)
        cg = CurveOfGrowth(data, xycen, radii, error=None, mask=all_mask)
        assert_allclose(cg.profile, 0.0)

    @pytest.mark.skipif(not HAS_MATPLOTLIB, reason='matplotlib is required')
    def test_plot_error_none(self, profile_data):
        """
        Test that ``plot_error()`` returns ``None`` when no errors were
        input.
        """
        xycen, data, _, _ = profile_data

        radii = np.arange(1, 36)
        cg = CurveOfGrowth(data, xycen, radii, error=None)
        match = 'Errors were not input'
        with pytest.warns(AstropyUserWarning, match=match):
            result = cg.plot_error()
        assert result is None

    def test_repr(self, profile_data):
        """
        Test __repr__ output format.
        """
        xycen, data, _, _ = profile_data

        radii = np.arange(1, 36)
        cg = CurveOfGrowth(data, xycen, radii)
        r = repr(cg)
        assert 'CurveOfGrowth' in r
        assert f'xycen={xycen}' in r
        assert f'n_radii={len(radii)}' in r
        assert 'normalized=False' in r


class TestEnsquaredCurveOfGrowth:
    def test_basic(self, profile_data):
        """
        Test basic EnsquaredCurveOfGrowth functionality.
        """
        xycen, data, _, _ = profile_data

        half_sizes = np.arange(1, 37)
        ecg1 = EnsquaredCurveOfGrowth(data, xycen, half_sizes, error=None,
                                      mask=None)

        assert_equal(ecg1.half_size, half_sizes)
        assert_equal(ecg1.radius, half_sizes)
        assert ecg1.area.shape == (36,)
        assert ecg1.profile.shape == (36,)
        assert ecg1.profile_error.shape == (0,)
        assert_allclose(ecg1.area[0], 4.0)  # 2x2 square

        assert len(ecg1.apertures) == 36
        assert isinstance(ecg1.apertures[0], RectangularAperture)

        half_sizes = np.arange(1, 36)
        ecg2 = EnsquaredCurveOfGrowth(data, xycen, half_sizes, error=None,
                                      mask=None)
        assert ecg2.area[0] > 0.0
        assert isinstance(ecg2.apertures[0], RectangularAperture)

    def test_units(self, profile_data):
        """
        Test EnsquaredCurveOfGrowth with units.
        """
        xycen, data, error, _ = profile_data

        half_sizes = np.arange(1, 36)
        unit = u.Jy
        ecg1 = EnsquaredCurveOfGrowth(data << unit, xycen, half_sizes,
                                      error=error << unit, mask=None)

        assert ecg1.profile.unit == unit
        assert ecg1.profile_error.unit == unit

        match = 'must all have the same units'
        with pytest.raises(ValueError, match=match):
            EnsquaredCurveOfGrowth(data << unit, xycen, half_sizes,
                                   error=error, mask=None)

    def test_error(self, profile_data):
        """
        Test EnsquaredCurveOfGrowth with error array.
        """
        xycen, data, error, _ = profile_data

        half_sizes = np.arange(1, 36)
        ecg1 = EnsquaredCurveOfGrowth(data, xycen, half_sizes, error=error,
                                      mask=None)

        assert ecg1.profile.shape == (35,)
        assert ecg1.profile_error.shape == (35,)

    def test_mask(self, profile_data):
        """
        Test EnsquaredCurveOfGrowth with a mask.
        """
        xycen, data, error, mask = profile_data

        half_sizes = np.arange(1, 36)
        ecg1 = EnsquaredCurveOfGrowth(data, xycen, half_sizes, error=error,
                                      mask=None)
        ecg2 = EnsquaredCurveOfGrowth(data, xycen, half_sizes, error=error,
                                      mask=mask)

        assert ecg1.profile.sum() > ecg2.profile.sum()
        assert np.sum(ecg1.profile_error**2) > np.sum(ecg2.profile_error**2)

    def test_normalize(self, profile_data):
        """
        Test EnsquaredCurveOfGrowth normalize and unnormalize methods.
        """
        xycen, data, _, _ = profile_data

        half_sizes = np.arange(1, 36)
        ecg1 = EnsquaredCurveOfGrowth(data, xycen, half_sizes, error=None,
                                      mask=None)
        ecg2 = EnsquaredCurveOfGrowth(data, xycen, half_sizes, error=None,
                                      mask=None)

        profile1 = ecg1.profile
        ecg1.normalize()
        profile2 = ecg1.profile
        assert np.mean(profile2) < np.mean(profile1)

        ecg1.unnormalize()
        assert_allclose(ecg1.profile, ecg2.profile)

        ecg1.normalize(method='sum')
        ecg1.normalize(method='max')
        ecg1.unnormalize()
        assert_allclose(ecg1.profile, ecg2.profile)

        ecg1.normalize(method='max')
        ecg1.normalize(method='sum')
        ecg1.normalize(method='max')
        ecg1.normalize(method='max')
        ecg1.unnormalize()
        assert_allclose(ecg1.profile, ecg2.profile)

        ecg1.normalize(method='sum')
        profile3 = ecg1.profile
        assert np.mean(profile3) < np.mean(profile1)

        ecg1.unnormalize()
        assert_allclose(ecg1.profile, ecg2.profile)

        match = "invalid method, must be 'max' or 'sum'"
        with pytest.raises(ValueError, match=match):
            ecg1.normalize(method='invalid')

        ecg1.__dict__['_raw_profile'] = ecg1.profile - np.nanmax(ecg1.profile)
        match = 'The profile cannot be normalized'
        with pytest.warns(AstropyUserWarning, match=match):
            ecg1.normalize(method='max')

    def test_interp(self, profile_data):
        """
        Test EnsquaredCurveOfGrowth encircled energy interpolation
        methods.
        """
        xycen, data, _, _ = profile_data
        half_sizes = np.arange(1, 36)
        ecg1 = EnsquaredCurveOfGrowth(data, xycen, half_sizes, error=None,
                                      mask=None)
        ecg1.normalize()
        ee_half_sizes = np.array([0, 5, 10, 20, 25, 50], dtype=float)
        ee_vals = ecg1.calc_ee_at_half_size(ee_half_sizes)
        half_sizes_back = ecg1.calc_half_size_at_ee(ee_vals)
        ee_half_sizes[[0, -1]] = np.nan
        assert_allclose(half_sizes_back, ee_half_sizes, rtol=1e-6)

        half_sizes = np.linspace(0.1, 36, 200)
        ecg1 = EnsquaredCurveOfGrowth(data, xycen, half_sizes, error=None,
                                      mask=None, method='center')
        ee_vals = ecg1.calc_ee_at_half_size(ee_half_sizes)
        half_sizes_back = ecg1.calc_half_size_at_ee(ee_vals)
        assert_allclose(ecg1.calc_ee_at_half_size(half_sizes_back), ee_vals)

    def test_inputs(self, profile_data):
        """
        Test EnsquaredCurveOfGrowth input validation.
        """
        xycen, data, error, _ = profile_data

        match = 'half_sizes must be > 0'
        half_sizes = np.arange(10)
        with pytest.raises(ValueError, match=match):
            EnsquaredCurveOfGrowth(data, xycen, half_sizes, error=None,
                                   mask=None)

        match = 'half_sizes must be a 1D array and have at least two values'
        with pytest.raises(ValueError, match=match):
            EnsquaredCurveOfGrowth(data, xycen, [1], error=None, mask=None)
        with pytest.raises(ValueError, match=match):
            EnsquaredCurveOfGrowth(data, xycen,
                                   np.arange(1, 7).reshape(2, 3),
                                   error=None, mask=None)

        match = 'half_sizes must be strictly increasing'
        half_sizes = np.arange(1, 10)[::-1]
        with pytest.raises(ValueError, match=match):
            EnsquaredCurveOfGrowth(data, xycen, half_sizes, error=None,
                                   mask=None)

        match = 'half_sizes must be a plain array of pixel values'
        half_sizes = np.arange(1, 10) << u.pix
        with pytest.raises(TypeError, match=match):
            EnsquaredCurveOfGrowth(data, xycen, half_sizes, error=None,
                                   mask=None)

        unit1 = u.Jy
        unit2 = u.km
        half_sizes = np.arange(1, 36)
        match = 'must all have the same units'
        with pytest.raises(ValueError, match=match):
            EnsquaredCurveOfGrowth(data << unit1, xycen, half_sizes,
                                   error=error << unit2)

    def test_no_mutation(self, profile_data):
        """
        Test that input data, error, mask, and half_sizes arrays are not
        mutated by EnsquaredCurveOfGrowth.
        """
        xycen, data, error, mask = profile_data

        data2 = data.copy()
        data2[50, 70] = np.nan
        mask2 = mask.copy()
        data2_orig = data2.copy()
        error_orig = error.copy()
        mask2_orig = mask2.copy()
        half_sizes = np.arange(1, 36)
        half_sizes_orig = half_sizes.copy()

        match = 'Input data contains nonfinite values'
        with pytest.warns(AstropyUserWarning, match=match):
            EnsquaredCurveOfGrowth(data2, xycen, half_sizes, error=error,
                                   mask=mask2)

        assert_equal(data2, data2_orig)
        assert_equal(error, error_orig)
        assert_equal(mask2, mask2_orig)
        assert_equal(half_sizes, half_sizes_orig)

    @pytest.mark.skipif(not HAS_MATPLOTLIB, reason='matplotlib is required')
    def test_plot(self, profile_data):
        """
        Test EnsquaredCurveOfGrowth plot methods.
        """
        xycen, data, error, _ = profile_data

        half_sizes = np.arange(1, 36)
        ecg1 = EnsquaredCurveOfGrowth(data, xycen, half_sizes, error=None,
                                      mask=None)
        ecg1.plot()
        match = 'Errors were not input'
        with pytest.warns(AstropyUserWarning, match=match):
            ecg1.plot_error()

        ecg2 = EnsquaredCurveOfGrowth(data, xycen, half_sizes, error=error,
                                      mask=None)
        ecg2.plot()
        pc1 = ecg2.plot_error()
        assert_allclose(pc1.get_facecolor(), [[0.5, 0.5, 0.5, 0.3]])
        pc2 = ecg2.plot_error(facecolor='blue')
        assert_allclose(pc2.get_facecolor(), [[0, 0, 1, 1]])

        unit = u.Jy
        ecg3 = EnsquaredCurveOfGrowth(data << unit, xycen, half_sizes,
                                      error=error << unit, mask=None)
        ecg3.plot()
        ecg3.plot_error()

    def test_repr(self, profile_data):
        """
        Test __repr__ output format.
        """
        xycen, data, _, _ = profile_data

        half_sizes = np.arange(1, 36)
        ecg = EnsquaredCurveOfGrowth(data, xycen, half_sizes)
        r = repr(ecg)
        assert 'EnsquaredCurveOfGrowth' in r
        assert f'xycen={xycen}' in r
        assert f'n_half_sizes={len(half_sizes)}' in r
        assert 'normalized=False' in r


class TestEllipticalCurveOfGrowth:
    def test_basic(self, profile_data):
        """
        Test basic EllipticalCurveOfGrowth functionality.
        """
        xycen, data, _, _ = profile_data

        radii = np.arange(1, 37)
        ecg1 = EllipticalCurveOfGrowth(data, xycen, radii, axis_ratio=0.5,
                                       error=None, mask=None)

        assert_equal(ecg1.radius, radii)
        assert ecg1.area.shape == (36,)
        assert ecg1.profile.shape == (36,)
        assert ecg1.profile_error.shape == (0,)
        assert_allclose(ecg1.area[0], np.pi * 1.0 * 0.5)

        assert len(ecg1.apertures) == 36
        assert isinstance(ecg1.apertures[0], EllipticalAperture)

        radii = np.arange(1, 36)
        ecg2 = EllipticalCurveOfGrowth(data, xycen, radii, axis_ratio=0.5,
                                       error=None, mask=None)
        assert ecg2.area[0] > 0.0
        assert isinstance(ecg2.apertures[0], EllipticalAperture)

    def test_axis_ratio_1(self, profile_data):
        """
        Test that axis_ratio=1 gives the same result as CurveOfGrowth.
        """
        xycen, data, error, _ = profile_data
        radii = np.arange(1, 36)
        cog = CurveOfGrowth(data, xycen, radii, error=error, mask=None)
        ecg = EllipticalCurveOfGrowth(data, xycen, radii, axis_ratio=1.0,
                                      error=error, mask=None)
        assert_allclose(ecg.profile, cog.profile)
        assert_allclose(ecg.profile_error, cog.profile_error)

    def test_units(self, profile_data):
        """
        Test EllipticalCurveOfGrowth with units.
        """
        xycen, data, error, _ = profile_data

        radii = np.arange(1, 36)
        unit = u.Jy
        ecg1 = EllipticalCurveOfGrowth(data << unit, xycen, radii,
                                       axis_ratio=0.5,
                                       error=error << unit, mask=None)

        assert ecg1.profile.unit == unit
        assert ecg1.profile_error.unit == unit

        match = 'must all have the same units'
        with pytest.raises(ValueError, match=match):
            EllipticalCurveOfGrowth(data << unit, xycen, radii,
                                    axis_ratio=0.5, error=error, mask=None)

    def test_error(self, profile_data):
        """
        Test EllipticalCurveOfGrowth with error array.
        """
        xycen, data, error, _ = profile_data

        radii = np.arange(1, 36)
        ecg1 = EllipticalCurveOfGrowth(data, xycen, radii, axis_ratio=0.5,
                                       error=error, mask=None)

        assert ecg1.profile.shape == (35,)
        assert ecg1.profile_error.shape == (35,)

    def test_mask(self, profile_data):
        """
        Test EllipticalCurveOfGrowth with a mask.
        """
        xycen, data, error, mask = profile_data

        radii = np.arange(1, 36)
        ecg1 = EllipticalCurveOfGrowth(data, xycen, radii, axis_ratio=0.5,
                                       error=error, mask=None)
        ecg2 = EllipticalCurveOfGrowth(data, xycen, radii, axis_ratio=0.5,
                                       error=error, mask=mask)

        assert ecg1.profile.sum() > ecg2.profile.sum()
        assert np.sum(ecg1.profile_error**2) > np.sum(ecg2.profile_error**2)

    def test_normalize(self, profile_data):
        """
        Test EllipticalCurveOfGrowth normalize and unnormalize methods.
        """
        xycen, data, _, _ = profile_data

        radii = np.arange(1, 36)
        ecg1 = EllipticalCurveOfGrowth(data, xycen, radii, axis_ratio=0.5,
                                       error=None, mask=None)
        ecg2 = EllipticalCurveOfGrowth(data, xycen, radii, axis_ratio=0.5,
                                       error=None, mask=None)

        profile1 = ecg1.profile
        ecg1.normalize()
        profile2 = ecg1.profile
        assert np.mean(profile2) < np.mean(profile1)

        ecg1.unnormalize()
        assert_allclose(ecg1.profile, ecg2.profile)

        ecg1.normalize(method='sum')
        ecg1.normalize(method='max')
        ecg1.unnormalize()
        assert_allclose(ecg1.profile, ecg2.profile)

        ecg1.normalize(method='max')
        ecg1.normalize(method='sum')
        ecg1.normalize(method='max')
        ecg1.normalize(method='max')
        ecg1.unnormalize()
        assert_allclose(ecg1.profile, ecg2.profile)

        ecg1.normalize(method='sum')
        profile3 = ecg1.profile
        assert np.mean(profile3) < np.mean(profile1)

        ecg1.unnormalize()
        assert_allclose(ecg1.profile, ecg2.profile)

        match = "invalid method, must be 'max' or 'sum'"
        with pytest.raises(ValueError, match=match):
            ecg1.normalize(method='invalid')

        ecg1.__dict__['_raw_profile'] = ecg1.profile - np.nanmax(ecg1.profile)
        match = 'The profile cannot be normalized'
        with pytest.warns(AstropyUserWarning, match=match):
            ecg1.normalize(method='max')

    def test_interp(self, profile_data):
        """
        Test EllipticalCurveOfGrowth encircled energy interpolation
        methods.
        """
        xycen, data, _, _ = profile_data
        radii = np.arange(1, 36)
        ecg1 = EllipticalCurveOfGrowth(data, xycen, radii, axis_ratio=0.5,
                                       error=None, mask=None)
        ecg1.normalize()
        ee_radii = np.array([0, 5, 10, 20, 25, 50], dtype=float)
        ee_vals = ecg1.calc_ee_at_radius(ee_radii)
        radii_back = ecg1.calc_radius_at_ee(ee_vals)
        ee_radii[[0, -1]] = np.nan
        assert_allclose(radii_back, ee_radii, rtol=1e-6)

        radii = np.linspace(0.1, 36, 200)
        ecg1 = EllipticalCurveOfGrowth(data, xycen, radii, axis_ratio=0.5,
                                       error=None, mask=None,
                                       method='center')
        ee_vals = ecg1.calc_ee_at_radius(ee_radii)
        radii_back = ecg1.calc_radius_at_ee(ee_vals)
        assert_allclose(ecg1.calc_ee_at_radius(radii_back), ee_vals)

    def test_inputs(self, profile_data):
        """
        Test EllipticalCurveOfGrowth input validation.
        """
        xycen, data, error, _ = profile_data

        match = 'radii must be > 0'
        radii = np.arange(10)
        with pytest.raises(ValueError, match=match):
            EllipticalCurveOfGrowth(data, xycen, radii, axis_ratio=0.5,
                                    error=None, mask=None)

        match = 'radii must be a 1D array and have at least two values'
        with pytest.raises(ValueError, match=match):
            EllipticalCurveOfGrowth(data, xycen, [1], axis_ratio=0.5,
                                    error=None, mask=None)
        with pytest.raises(ValueError, match=match):
            EllipticalCurveOfGrowth(data, xycen,
                                    np.arange(1, 7).reshape(2, 3),
                                    axis_ratio=0.5, error=None, mask=None)

        match = 'radii must be strictly increasing'
        radii = np.arange(1, 10)[::-1]
        with pytest.raises(ValueError, match=match):
            EllipticalCurveOfGrowth(data, xycen, radii, axis_ratio=0.5,
                                    error=None, mask=None)

        match = 'axis_ratio must be in the range 0 < axis_ratio <= 1'
        radii = np.arange(1, 36)
        with pytest.raises(ValueError, match=match):
            EllipticalCurveOfGrowth(data, xycen, radii, axis_ratio=0.0)
        with pytest.raises(ValueError, match=match):
            EllipticalCurveOfGrowth(data, xycen, radii, axis_ratio=-0.5)
        with pytest.raises(ValueError, match=match):
            EllipticalCurveOfGrowth(data, xycen, radii, axis_ratio=1.5)

        unit1 = u.Jy
        unit2 = u.km
        radii = np.arange(1, 36)
        match = 'must all have the same units'
        with pytest.raises(ValueError, match=match):
            EllipticalCurveOfGrowth(data << unit1, xycen, radii,
                                    axis_ratio=0.5,
                                    error=error << unit2)

    def test_no_mutation(self, profile_data):
        """
        Test that input data, error, mask, and radii arrays are not
        mutated by EllipticalCurveOfGrowth.
        """
        xycen, data, error, mask = profile_data

        data2 = data.copy()
        data2[50, 70] = np.nan
        mask2 = mask.copy()
        data2_orig = data2.copy()
        error_orig = error.copy()
        mask2_orig = mask2.copy()
        radii = np.arange(1, 36)
        radii_orig = radii.copy()

        match = 'Input data contains nonfinite values'
        with pytest.warns(AstropyUserWarning, match=match):
            EllipticalCurveOfGrowth(data2, xycen, radii, axis_ratio=0.5,
                                    error=error, mask=mask2)

        assert_equal(data2, data2_orig)
        assert_equal(error, error_orig)
        assert_equal(mask2, mask2_orig)
        assert_equal(radii, radii_orig)

    @pytest.mark.skipif(not HAS_MATPLOTLIB, reason='matplotlib is required')
    def test_plot(self, profile_data):
        """
        Test EllipticalCurveOfGrowth plot methods.
        """
        xycen, data, error, _ = profile_data

        radii = np.arange(1, 36)
        ecg1 = EllipticalCurveOfGrowth(data, xycen, radii, axis_ratio=0.5,
                                       error=None, mask=None)
        ecg1.plot()
        match = 'Errors were not input'
        with pytest.warns(AstropyUserWarning, match=match):
            ecg1.plot_error()

        ecg2 = EllipticalCurveOfGrowth(data, xycen, radii, axis_ratio=0.5,
                                       error=error, mask=None)
        ecg2.plot()
        pc1 = ecg2.plot_error()
        assert_allclose(pc1.get_facecolor(), [[0.5, 0.5, 0.5, 0.3]])
        pc2 = ecg2.plot_error(facecolor='blue')
        assert_allclose(pc2.get_facecolor(), [[0, 0, 1, 1]])

        unit = u.Jy
        ecg3 = EllipticalCurveOfGrowth(data << unit, xycen, radii,
                                       axis_ratio=0.5,
                                       error=error << unit, mask=None)
        ecg3.plot()
        ecg3.plot_error()

    def test_repr(self, profile_data):
        """
        Test __repr__ output format.
        """
        xycen, data, _, _ = profile_data

        radii = np.arange(1, 36)
        ecg = EllipticalCurveOfGrowth(data, xycen, radii, axis_ratio=0.5,
                                      theta=0.7)
        r = repr(ecg)
        assert 'EllipticalCurveOfGrowth' in r
        assert f'xycen={xycen}' in r
        assert f'n_radii={len(radii)}' in r
        assert 'axis_ratio=0.5' in r
        assert 'theta=0.7' in r
        assert 'normalized=False' in r


INTERP_CASES = [
    (CurveOfGrowth, {}, 'calc_ee_at_radius', 'calc_radius_at_ee'),
    (EnsquaredCurveOfGrowth, {}, 'calc_ee_at_half_size',
     'calc_half_size_at_ee'),
    (EllipticalCurveOfGrowth, {'axis_ratio': 0.5}, 'calc_ee_at_radius',
     'calc_radius_at_ee'),
]


@pytest.mark.parametrize(('cls', 'kwargs', 'ee_name', 'size_name'),
                         INTERP_CASES)
class TestNonMonotonicInterpolation:
    """
    Tests of the encircled energy interpolation methods for profiles
    that are not monotonically increasing.
    """

    @staticmethod
    def make_profile(profile_cls, kwargs, values, *, unit=None):
        """
        Make a curve of growth whose raw profile is set to ``values``.
        """
        sizes = np.arange(1, len(values) + 1, dtype=float)
        data = np.ones((51, 51))
        if unit is not None:
            data = data << unit
            values = values << unit
        cog = profile_cls(data, (25, 25), sizes, **kwargs)
        cog.__dict__['_raw_profile'] = values
        return cog

    def test_noisy_profile(self, cls, kwargs, ee_name, size_name):
        """
        Test that the two methods are consistent for a noisy profile.
        """
        # A compact source, so that the noise dominates the outer
        # part of the profile
        xycen = (50.0, 40.0)
        yy, xx = np.mgrid[0:81, 0:101]
        data = Gaussian2D(21.0, *xycen, 4.0, 4.0)(xx, yy)
        rng = np.random.default_rng(0)
        data += rng.normal(0.0, 1.0, data.shape)
        sizes = np.arange(1, 36)
        cog = cls(data, xycen, sizes, **kwargs)
        cog.normalize()
        assert np.any(np.diff(cog.profile) < 0)
        calc_ee = getattr(cog, ee_name)
        calc_size = getattr(cog, size_name)

        # The interpolated curve passes through the measured profile,
        # so it also decreases where the profile does
        assert_allclose(calc_ee(sizes), cog.profile)
        fine_sizes = np.linspace(1, 35, 1000)
        assert np.any(np.diff(calc_ee(fine_sizes)) < 0)

        # Every value in the profile range has a size, the size never
        # decreases with the encircled energy, and the round-trip is
        # exact
        ee = np.linspace(cog.profile[0], 1.0, 200)
        result = calc_size(ee)
        assert np.all(np.isfinite(result))
        assert np.all(np.diff(result) >= 0)
        assert_allclose(calc_ee(result), ee, atol=1e-12)

        # The maximum is reached at the size of the profile maximum
        assert_allclose(calc_size(1.0), sizes[np.argmax(cog.profile)])

        # Values outside the profile range have no size
        outside = calc_size([cog.profile[0] - 0.01, 1.01])
        assert np.all(np.isnan(outside))

    def test_dip(self, cls, kwargs, ee_name, size_name):
        """
        Test the values across a dip in the profile.
        """
        values = np.array([1.0, 2.0, 3.0, 2.5, 2.8, 4.0, 5.0])
        cog = self.make_profile(cls, kwargs, values)
        calc_ee = getattr(cog, ee_name)
        calc_size = getattr(cog, size_name)

        # The curve follows the profile through the dip
        assert_allclose(calc_ee(np.arange(1, 8)), values)
        assert 2.5 < calc_ee(3.5) < 3.0

        # The smallest size that reaches the value is returned. The
        # values 2.6 and 2.8 are also reached again after the dip.
        assert_allclose(calc_size([1.0, 3.0, 4.0, 5.0]), [1, 3, 6, 7])
        assert 2 < calc_size(2.6) < 3
        assert 2 < calc_size(2.8) < 3
        result = calc_size(3.5)
        assert 5 < result < 6
        assert_allclose(calc_ee(result), 3.5)

    def test_decreasing_start(self, cls, kwargs, ee_name, size_name):
        """
        Test a profile that does not increase at its smallest sizes.
        """
        values = np.array([2.0, 2.0, 1.0, 3.0, 4.0])
        cog = self.make_profile(cls, kwargs, values)
        calc_ee = getattr(cog, ee_name)
        calc_size = getattr(cog, size_name)
        assert_allclose(calc_ee([1.0, 2.0, 3.0]), [2.0, 2.0, 1.0])
        assert_allclose(calc_size([2.0, 1.0, 4.0]), [1, 3, 5])
        # A value below the first profile value is first reached where
        # the profile decreases
        result = calc_size(1.5)
        assert 2 < result < 3
        assert_allclose(calc_ee(result), 1.5)
        result = calc_size(2.5)
        assert 3 < result < 4
        assert_allclose(calc_ee(result), 2.5)
        assert np.all(np.isnan(calc_size([0.5, 4.5])))

    def test_nonfinite(self, cls, kwargs, ee_name, size_name):
        """
        Test that nonfinite profile values are skipped.
        """
        values = np.array([1.0, 3.0, np.nan, 5.0, 7.0])
        cog = self.make_profile(cls, kwargs, values)
        calc_ee = getattr(cog, ee_name)
        calc_size = getattr(cog, size_name)
        assert_allclose(calc_ee([2.0, 4.0]), [3.0, 5.0])
        assert 2 < calc_ee(3.0) < 5
        assert_allclose(calc_size([3.0, 5.0, 7.0]), [2, 4, 5])
        assert np.isnan(calc_size(np.nan))

        values = np.array([np.nan, np.nan, 1.0, np.nan])
        cog = self.make_profile(cls, kwargs, values)
        match = 'must have at least two finite values'
        with pytest.raises(ValueError, match=match):
            getattr(cog, ee_name)(2.0)
        with pytest.raises(ValueError, match=match):
            getattr(cog, size_name)(1.0)

    def test_shapes(self, cls, kwargs, ee_name, size_name):
        """
        Test that the output has the shape of the input.
        """
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        cog = self.make_profile(cls, kwargs, values)
        calc_ee = getattr(cog, ee_name)
        calc_size = getattr(cog, size_name)
        assert np.shape(calc_ee(2.5)) == ()
        assert np.shape(calc_size(2.5)) == ()
        ee = np.array([[1.5, 2.5], [3.5, 4.5]])
        assert calc_size(ee).shape == (2, 2)
        assert_allclose(calc_size(ee), ee)
        assert_allclose(calc_ee(ee), ee)

        # A dimensionless quantity is accepted for a profile without
        # units
        assert_allclose(calc_size(2.5 * u.dimensionless_unscaled), 2.5)

    def test_units(self, cls, kwargs, ee_name, size_name):
        """
        Test the interpolation methods for a profile with units.
        """
        values = np.array([1.0, 2.0, 3.0, 2.5, 2.8, 4.0, 5.0])
        unit = u.Jy
        cog = self.make_profile(cls, kwargs, values, unit=unit)
        calc_ee = getattr(cog, ee_name)
        calc_size = getattr(cog, size_name)

        ee = calc_ee([2.0, 4.0])
        assert ee.unit == unit
        assert_allclose(ee.value, [2.0, 2.5])

        # The sizes are in pixels and have no units
        result = calc_size([2.0, 4.0] * unit)
        assert not isinstance(result, u.Quantity)
        assert_allclose(result, [2, 6])
        assert_allclose(calc_size([2000.0, 4000.0] * u.mJy), [2, 6])

        # Values without units are in the profile units
        assert_allclose(calc_size([2.0, 4.0]), [2, 6])

        match = 'not convertible'
        with pytest.raises(u.UnitConversionError, match=match):
            calc_size(2.0 * u.m)

        # The sizes are in pixels and cannot have units
        match = 'must be in pixels and cannot be a Quantity'
        with pytest.raises(TypeError, match=match):
            calc_ee(2.0 * u.pix)

        # A normalized profile is dimensionless
        cog.normalize()
        ee = calc_ee([2.0, 7.0])
        assert ee.unit == u.dimensionless_unscaled
        assert_allclose(ee.value, [0.4, 1.0])
        assert_allclose(calc_size([0.4, 1.0]), [2, 7])
        assert_allclose(calc_size([0.4, 1.0] * u.dimensionless_unscaled),
                        [2, 7])
        assert_allclose(calc_size([40.0, 100.0] * u.percent), [2, 7])
