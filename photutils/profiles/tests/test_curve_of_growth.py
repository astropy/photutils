# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the curve_of_growth module.
"""

import astropy.units as u
import numpy as np
import pytest
from astropy.utils.exceptions import AstropyUserWarning
from numpy.testing import assert_allclose, assert_equal

from photutils.aperture import (CircularAperture, EllipticalAperture,
                                RectangularAperture)
from photutils.profiles import (CurveOfGrowth, EllipticalCurveOfGrowth,
                                EnsquaredCurveOfGrowth, ProfileBase)
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

        match = 'invalid method, must be "max" or "sum"'
        with pytest.raises(ValueError, match=match):
            cg1.normalize(method='invalid')

        cg1.__dict__['profile'] -= np.nanmax(cg1.__dict__['profile'])
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
        match = 'The curve-of-growth profile is not monotonically increasing'
        with pytest.raises(ValueError, match=match):
            cg1.calc_radius_at_ee(ee_vals)

    def test_interp_nonmonotonic_start(self, profile_data):
        """
        Test that `calc_radius_at_ee` raises ValueError when the profile
        is non-monotonic at the very first point (covers the len(radius)
        < 2 branch).
        """
        xycen, data, _, _ = profile_data
        radii = np.arange(1, 36)
        cg1 = CurveOfGrowth(data, xycen, radii, error=None, mask=None)

        # Force non-monotonicity at the first point: diff[0] = 0
        # (non-positive) so idx=0 and radius[0:0] has fewer than 2
        # elements
        profile = cg1.profile.copy()
        profile[0] = profile[1]
        cg1.__dict__['profile'] = profile

        match = 'The curve-of-growth profile is not monotonically increasing'
        with pytest.raises(ValueError, match=match):
            cg1.calc_radius_at_ee(np.array([0.5]))

    def test_trim_to_monotonic_nan(self):
        """
        Test that `_trim_to_monotonic` keeps only the leading finite
        segment when NaN values are present (covers the NaN-trimming
        branch in core.py).
        """
        xarr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        profile = np.array([1.0, 3.0, np.nan, 5.0, 7.0])
        xarr_out, profile_out = ProfileBase._trim_to_monotonic(
            xarr, profile, 'test')
        assert_equal(xarr_out, np.array([1.0, 2.0]))
        assert_equal(profile_out, np.array([1.0, 3.0]))

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

        match = 'Input data contains non-finite values'
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

        match = 'invalid method, must be "max" or "sum"'
        with pytest.raises(ValueError, match=match):
            ecg1.normalize(method='invalid')

        ecg1.__dict__['profile'] -= np.nanmax(ecg1.__dict__['profile'])
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
        match = 'The ensquared curve-of-growth profile is not monotonically'
        with pytest.raises(ValueError, match=match):
            ecg1.calc_half_size_at_ee(ee_vals)

    def test_interp_nonmonotonic_start(self, profile_data):
        """
        Test that `calc_half_size_at_ee` raises ValueError when the
        profile is non-monotonic at the very first point (covers the
        len(half_size) < 2 branch).
        """
        xycen, data, _, _ = profile_data
        half_sizes = np.arange(1, 36)
        ecg1 = EnsquaredCurveOfGrowth(data, xycen, half_sizes, error=None,
                                      mask=None)
        # Force non-monotonicity at the first point
        profile = ecg1.profile.copy()
        profile[0] = profile[1]
        ecg1.__dict__['profile'] = profile

        match = 'The ensquared curve-of-growth profile is not monotonically'
        with pytest.raises(ValueError, match=match):
            ecg1.calc_half_size_at_ee(np.array([0.5]))

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

        match = 'radii must be a 1D array and have at least two values'
        with pytest.raises(ValueError, match=match):
            EnsquaredCurveOfGrowth(data, xycen, [1], error=None, mask=None)
        with pytest.raises(ValueError, match=match):
            EnsquaredCurveOfGrowth(data, xycen,
                                   np.arange(1, 7).reshape(2, 3),
                                   error=None, mask=None)

        match = 'radii must be strictly increasing'
        half_sizes = np.arange(1, 10)[::-1]
        with pytest.raises(ValueError, match=match):
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

        match = 'Input data contains non-finite values'
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

        match = 'invalid method, must be "max" or "sum"'
        with pytest.raises(ValueError, match=match):
            ecg1.normalize(method='invalid')

        ecg1.__dict__['profile'] -= np.nanmax(ecg1.__dict__['profile'])
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
        match = 'The elliptical curve-of-growth profile is not monotonically'
        with pytest.raises(ValueError, match=match):
            ecg1.calc_radius_at_ee(ee_vals)

    def test_interp_nonmonotonic_start(self, profile_data):
        """
        Test that `calc_radius_at_ee` raises ValueError when the
        profile is non-monotonic at the very first point (covers the
        len(radius) < 2 branch).
        """
        xycen, data, _, _ = profile_data
        radii = np.arange(1, 36)
        ecg1 = EllipticalCurveOfGrowth(data, xycen, radii, axis_ratio=0.5,
                                       error=None, mask=None)
        # Force non-monotonicity at the first point
        profile = ecg1.profile.copy()
        profile[0] = profile[1]
        ecg1.__dict__['profile'] = profile

        match = 'The elliptical curve-of-growth profile is not monotonically'
        with pytest.raises(ValueError, match=match):
            ecg1.calc_radius_at_ee(np.array([0.5]))

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

        match = 'Input data contains non-finite values'
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
        ecg = EllipticalCurveOfGrowth(data, xycen, radii, axis_ratio=0.5)
        r = repr(ecg)
        assert 'EllipticalCurveOfGrowth' in r
        assert f'xycen={xycen}' in r
        assert f'n_radii={len(radii)}' in r
        assert 'normalized=False' in r
