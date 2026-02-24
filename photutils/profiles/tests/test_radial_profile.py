# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the radial_profile module.
"""

import warnings

import astropy.units as u
import numpy as np
import pytest
from astropy.modeling.models import Gaussian1D, Moffat1D
from astropy.utils.exceptions import AstropyUserWarning
from numpy.testing import assert_allclose, assert_equal

from photutils.aperture import CircularAnnulus, CircularAperture
from photutils.profiles import RadialProfile
from photutils.utils._optional_deps import HAS_MATPLOTLIB


class TestRadialProfile:
    def test_basic(self, profile_data):
        """
        Test basic RadialProfile functionality.
        """
        xycen, data, _, _ = profile_data

        edge_radii = np.arange(36)
        rp1 = RadialProfile(data, xycen, edge_radii, error=None, mask=None)

        assert_equal(rp1.radius, np.arange(35) + 0.5)
        assert rp1.area.shape == (35,)
        assert rp1.profile.shape == (35,)
        assert rp1.profile_error.shape == (0,)
        assert rp1.area[0] > 0.0

        assert len(rp1.apertures) == 35
        assert isinstance(rp1.apertures[0], CircularAperture)
        assert isinstance(rp1.apertures[1], CircularAnnulus)

        edge_radii = np.arange(36) + 0.1
        rp2 = RadialProfile(data, xycen, edge_radii, error=None, mask=None)
        assert isinstance(rp2.apertures[0], CircularAnnulus)

    def test_normalization(self, profile_data):
        """
        Test RadialProfile normalize and unnormalize methods.
        """
        xycen, data, error, _ = profile_data

        edge_radii = np.arange(36)
        rp1 = RadialProfile(data, xycen, edge_radii, error=error, mask=None)

        profile = rp1.profile
        profile_error = rp1.profile_error
        data_profile = rp1.data_profile
        rp1.normalize()
        assert np.max(rp1.profile) == 1.0
        assert np.max(rp1.profile_error) <= np.max(profile_error)
        assert np.max(rp1.data_profile) <= np.max(data_profile)

        rp1.unnormalize()
        assert_allclose(rp1.profile, profile)
        assert_allclose(rp1.profile_error, profile_error)
        assert_allclose(rp1.data_profile, data_profile)

    def test_data(self, profile_data):
        """
        Test RadialProfile data_radius and data_profile attributes.
        """
        xycen, data, _, _ = profile_data

        edge_radii = np.arange(36)
        rp1 = RadialProfile(data, xycen, edge_radii, error=None, mask=None)

        data_radius = rp1.data_radius
        data_profile = rp1.data_profile
        assert np.max(data_radius) <= np.max(edge_radii)
        assert data_radius.shape == data_profile.shape
        assert np.min(data_profile) >= np.min(data)
        assert np.max(data_profile) <= np.max(data)

    def test_data_mask(self, profile_data):
        """
        Test that masked pixels are excluded from data_radius and
        data_profile.
        """
        xycen, data, _, mask = profile_data

        edge_radii = np.arange(36)
        rp1 = RadialProfile(data, xycen, edge_radii, mask=None)
        rp2 = RadialProfile(data, xycen, edge_radii, mask=mask)

        # Applying a mask should reduce the number of data points returned
        assert len(rp2.data_radius) < len(rp1.data_radius)
        assert rp2.data_radius.shape == rp2.data_profile.shape

        # Auto-masked non-finite pixels should also be excluded
        xcen, ycen = xycen
        data2 = data.copy()
        data2[int(ycen), int(xcen)] = np.nan
        match = 'Input data contains non-finite values'
        with pytest.warns(AstropyUserWarning, match=match):
            rp3 = RadialProfile(data2, xycen, edge_radii, mask=None)
        assert np.all(np.isfinite(rp3.data_profile))
        assert len(rp3.data_profile) < len(rp1.data_profile)

    def test_inputs(self, profile_data):
        """
        Test RadialProfile input validation.
        """
        xycen, data, _, _ = profile_data

        match = 'minimum radii must be >= 0'
        edge_radii = np.arange(-1, 10)
        with pytest.raises(ValueError, match=match):
            RadialProfile(data, xycen, edge_radii, error=None, mask=None)

        match = 'radii must be a 1D array and have at least two values'
        edge_radii = [1]
        with pytest.raises(ValueError, match=match):
            RadialProfile(data, xycen, edge_radii, error=None, mask=None)

        edge_radii = np.arange(6).reshape(2, 3)
        with pytest.raises(ValueError, match=match):
            RadialProfile(data, xycen, edge_radii, error=None, mask=None)

        match = 'radii must be strictly increasing'
        edge_radii = np.arange(10)[::-1]
        with pytest.raises(ValueError, match=match):
            RadialProfile(data, xycen, edge_radii, error=None, mask=None)

        match = 'error must have the same shape as data'
        edge_radii = np.arange(10)
        with pytest.raises(ValueError, match=match):
            RadialProfile(data, xycen, edge_radii, error=np.ones(3),
                          mask=None)

        match = 'mask must have the same shape as data'
        edge_radii = np.arange(10)
        mask = np.ones(3, dtype=bool)
        with pytest.raises(ValueError, match=match):
            RadialProfile(data, xycen, edge_radii, error=None, mask=mask)

    def test_gaussian(self, profile_data):
        """
        Test RadialProfile Gaussian fit attributes.
        """
        xycen, data, _, _ = profile_data

        edge_radii = np.arange(36)
        rp1 = RadialProfile(data, xycen, edge_radii, error=None, mask=None)

        assert isinstance(rp1.gaussian_fit, Gaussian1D)
        assert rp1.gaussian_profile.shape == (35,)
        assert rp1.gaussian_fwhm < 23.6

        edge_radii = np.arange(201)
        rp2 = RadialProfile(data, xycen, edge_radii, error=None, mask=None)
        assert isinstance(rp2.gaussian_fit, Gaussian1D)
        assert rp2.gaussian_profile.shape == (200,)
        assert rp2.gaussian_fwhm < 23.6

    def test_unit(self, profile_data):
        """
        Test RadialProfile with units.
        """
        xycen, data, error, _ = profile_data

        edge_radii = np.arange(36)
        unit = u.Jy
        rp1 = RadialProfile(data << unit, xycen, edge_radii,
                            error=error << unit, mask=None)
        assert rp1.profile.unit == unit
        assert rp1.profile_error.unit == unit

        match = 'must all have the same units'
        with pytest.raises(ValueError, match=match):
            RadialProfile(data << unit, xycen, edge_radii, error=error,
                          mask=None)

    def test_no_mutation(self, profile_data):
        """
        Test that input data, error, mask, and radii arrays are not
        mutated by RadialProfile.
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
        edge_radii = np.arange(36)
        radii_orig = edge_radii.copy()

        match = 'Input data contains non-finite values'
        with pytest.warns(AstropyUserWarning, match=match):
            RadialProfile(data2, xycen, edge_radii, error=error, mask=mask2)

        assert_equal(data2, data2_orig)
        assert_equal(error, error_orig)
        assert_equal(mask2, mask2_orig)
        assert_equal(edge_radii, radii_orig)

    def test_error(self, profile_data):
        """
        Test RadialProfile with error array.
        """
        xycen, data, error, _ = profile_data

        edge_radii = np.arange(36)
        rp1 = RadialProfile(data, xycen, edge_radii, error=error, mask=None)

        assert_equal(rp1.radius, np.arange(35) + 0.5)
        assert rp1.area.shape == (35,)
        assert rp1.profile.shape == (35,)
        assert rp1.profile_error.shape == (35,)

        assert len(rp1.apertures) == 35
        assert isinstance(rp1.apertures[0], CircularAperture)
        assert isinstance(rp1.apertures[1], CircularAnnulus)

    def test_gaussian_zero_sum(self, profile_data):
        """
        Test that ``gaussian_fit`` issues a warning and falls back to
        ``std=1.0`` when the profile sum is zero (covers the
        ``sum_profile == 0`` branch in ``gaussian_fit``).
        """
        xycen, data, _, _ = profile_data

        # All-zero data produces a zero-sum profile
        zero_data = np.zeros_like(data)
        edge_radii = np.arange(36)
        rp = RadialProfile(zero_data, xycen, edge_radii)

        with pytest.warns(AstropyUserWarning) as warning_list:
            gfit = rp.gaussian_fit

        messages = [str(w.message) for w in warning_list]
        assert any('The profile sum is zero' in m for m in messages)

        # The fit should still return a Gaussian1D model
        assert isinstance(gfit, Gaussian1D)

    def test_normalize_nan(self, profile_data):
        """
        If the profile has NaNs (e.g., aperture outside the image), make
        sure the normalization ignores them.
        """
        xycen, data, _, _ = profile_data

        edge_radii = np.arange(101)
        rp1 = RadialProfile(data, xycen, edge_radii)
        rp1.normalize()
        assert not np.isnan(rp1.profile[0])

    def test_nonfinite(self, profile_data):
        """
        Test RadialProfile handling of non-finite data values.
        """
        xycen, data, error, _ = profile_data
        data2 = data.copy()
        data2[40, 40] = np.nan
        mask = ~np.isfinite(data2)

        edge_radii = np.arange(36)
        rp1 = RadialProfile(data, xycen, edge_radii, error=None, mask=mask)

        rp2 = RadialProfile(data2, xycen, edge_radii, error=error, mask=mask)
        assert_allclose(rp1.profile, rp2.profile)

        match = 'Input data contains non-finite values'
        with pytest.warns(AstropyUserWarning, match=match):
            rp3 = RadialProfile(data2, xycen, edge_radii, error=error,
                                mask=None)
        assert_allclose(rp1.profile, rp3.profile)

        error2 = error.copy()
        error2[40, 40] = np.inf
        with pytest.warns(AstropyUserWarning, match=match):
            rp4 = RadialProfile(data, xycen, edge_radii, error=error2,
                                mask=None)
        assert_allclose(rp1.profile, rp4.profile)

    def test_all_masked(self, profile_data):
        """
        Test RadialProfile with all data masked.

        When every pixel is masked the profile should be all NaN
        (division by zero area).
        """
        xycen, data, _, _ = profile_data

        all_mask = np.ones(data.shape, dtype=bool)
        edge_radii = np.arange(36)
        rp = RadialProfile(data, xycen, edge_radii, error=None, mask=all_mask)
        assert np.all(np.isnan(rp.profile))

    @pytest.mark.skipif(not HAS_MATPLOTLIB, reason='matplotlib is required')
    def test_plot(self, profile_data):
        """
        Test RadialProfile plot methods.
        """
        xycen, data, error, _ = profile_data

        edge_radii = np.arange(36)
        rp1 = RadialProfile(data, xycen, edge_radii, error=None)
        rp1.plot()
        match = 'Errors were not input'
        with pytest.warns(AstropyUserWarning, match=match):
            rp1.plot_error()

        rp2 = RadialProfile(data, xycen, edge_radii, error=error)
        rp2.plot()
        pc1 = rp2.plot_error()
        assert_allclose(pc1.get_facecolor(), [[0.5, 0.5, 0.5, 0.3]])
        pc2 = rp2.plot_error(facecolor='blue')
        assert_allclose(pc2.get_facecolor(), [[0, 0, 1, 1]])

        unit = u.Jy
        rp3 = RadialProfile(data << unit, xycen, edge_radii,
                            error=error << unit)
        rp3.plot()
        rp3.plot_error()

    @pytest.mark.skipif(not HAS_MATPLOTLIB, reason='matplotlib is required')
    def test_plot_error_none(self, profile_data):
        """
        Test that ``plot_error()`` returns ``None`` when no errors were
        input.
        """
        xycen, data, _, _ = profile_data

        edge_radii = np.arange(36)
        rp = RadialProfile(data, xycen, edge_radii, error=None)
        match = 'Errors were not input'
        with pytest.warns(AstropyUserWarning, match=match):
            result = rp.plot_error()
        assert result is None

    def test_gaussian_std_guard(self, profile_data):
        """
        Test that the ``std = max(std, 1.0)`` guard is exercised when
        the computed std from a near-delta-function profile is close to
        zero.
        """
        xycen, data, _, _ = profile_data

        # Create a near-delta-function: all zero except one pixel at the
        # center so the weighted mean radius is nearly zero.
        delta = np.zeros_like(data)
        cy, cx = round(xycen[1]), round(xycen[0])
        delta[cy, cx] = 1.0

        edge_radii = np.arange(36)
        rp = RadialProfile(delta, xycen, edge_radii)
        # The radii warning may also be emitted because the fitted
        # stddev can be small enough that radius.min() > 0.3 * stddev.
        with pytest.warns(AstropyUserWarning):
            gfit = rp.gaussian_fit
        # The initial std guess gets clamped to 1.0, and the fit should
        # still produce a valid Gaussian
        assert isinstance(gfit, Gaussian1D)
        assert gfit.stddev.value > 0

    def test_gaussian_radii_warning(self, profile_data):
        """
        Test that a warning is issued when the input radii do not extend
        close to the source center.
        """
        xycen, data, _, _ = profile_data

        # Use edge_radii that start far from the center so that
        # radius.min() > 0.3 * stddev.
        edge_radii = np.arange(20, 36)
        rp = RadialProfile(data, xycen, edge_radii)
        match = 'Gaussian fit may be unreliable'
        with pytest.warns(AstropyUserWarning, match=match):
            _ = rp.gaussian_fit

    def test_repr(self, profile_data):
        """
        Test __repr__ output format.
        """
        xycen, data, _, _ = profile_data

        edge_radii = np.arange(36)
        rp = RadialProfile(data, xycen, edge_radii)
        r = repr(rp)
        assert 'RadialProfile' in r
        assert f'xycen={xycen}' in r
        assert f'n_radii={len(edge_radii)}' in r
        assert 'normalized=False' in r

        rp.normalize()
        r = repr(rp)
        assert 'normalized=True' in r

    def test_gaussian_fit_all_masked(self, profile_data):
        """
        Test that gaussian_fit returns None when the profile is entirely
        masked.
        """
        xycen, data, _, _ = profile_data

        mask = np.ones(data.shape, dtype=bool)
        edge_radii = np.arange(36)
        rp = RadialProfile(data, xycen, edge_radii, mask=mask)
        match = 'The radial profile is entirely non-finite or masked'
        with pytest.warns(AstropyUserWarning, match=match):
            result = rp.gaussian_fit
        assert result is None
        assert rp.gaussian_profile is None
        assert rp.gaussian_fwhm is None

    def test_normalize_all_nan(self, profile_data):
        """
        Test that normalize warns when the profile is all NaN.
        """
        xycen, data, _, _ = profile_data

        mask = np.ones(data.shape, dtype=bool)
        edge_radii = np.arange(36)
        rp = RadialProfile(data, xycen, edge_radii, mask=mask)
        match = 'The profile cannot be normalized'
        with pytest.warns(AstropyUserWarning, match=match):
            rp.normalize()

    def test_moffat(self, profile_data):
        """
        Test RadialProfile Moffat fit attributes.
        """
        xycen, data, _, _ = profile_data

        edge_radii = np.arange(36)
        rp = RadialProfile(data, xycen, edge_radii, error=None, mask=None)

        assert isinstance(rp.moffat_fit, Moffat1D)
        assert rp.moffat_profile.shape == (35,)
        assert rp.moffat_fwhm > 0
        assert rp.moffat_fwhm < 30.0

        # Check that x_0 is fixed at 0
        assert rp.moffat_fit.x_0.value == 0.0

        edge_radii = np.arange(201)
        rp2 = RadialProfile(data, xycen, edge_radii, error=None, mask=None)
        assert isinstance(rp2.moffat_fit, Moffat1D)
        assert rp2.moffat_profile.shape == (200,)
        assert rp2.moffat_fwhm > 0

    def test_moffat_fwhm_consistency(self, profile_data):
        """
        Test that ``moffat_fwhm`` is consistent with
        ``moffat_fit.fwhm``.
        """
        xycen, data, _, _ = profile_data

        edge_radii = np.arange(36)
        rp = RadialProfile(data, xycen, edge_radii)
        assert_allclose(rp.moffat_fwhm, rp.moffat_fit.fwhm)

    def test_moffat_profile_values(self, profile_data):
        """
        Test that ``moffat_profile`` equals the fit model evaluated at
        the profile radii.
        """
        xycen, data, _, _ = profile_data

        edge_radii = np.arange(36)
        rp = RadialProfile(data, xycen, edge_radii)
        expected = rp.moffat_fit(rp.radius)
        assert_allclose(rp.moffat_profile, expected)

    def test_moffat_zero_sum(self, profile_data):
        """
        Test that ``moffat_fit`` issues a warning and falls back to
        ``gamma=1.0`` when the profile sum is zero.
        """
        xycen, data, _, _ = profile_data

        zero_data = np.zeros_like(data)
        edge_radii = np.arange(36)
        rp = RadialProfile(zero_data, xycen, edge_radii)

        with pytest.warns(AstropyUserWarning) as warning_list:
            mfit = rp.moffat_fit

        messages = [str(w.message) for w in warning_list]
        assert any('The profile sum is zero' in m for m in messages)
        assert isinstance(mfit, Moffat1D)

    def test_moffat_fit_all_masked(self, profile_data):
        """
        Test that ``moffat_fit`` returns ``None`` when the profile is
        entirely masked.
        """
        xycen, data, _, _ = profile_data

        mask = np.ones(data.shape, dtype=bool)
        edge_radii = np.arange(36)
        rp = RadialProfile(data, xycen, edge_radii, mask=mask)
        match = 'The radial profile is entirely non-finite or masked'
        with pytest.warns(AstropyUserWarning, match=match):
            result = rp.moffat_fit
        assert result is None
        assert rp.moffat_profile is None
        assert rp.moffat_fwhm is None

    def test_moffat_no_above_half_max(self, profile_data):
        """
        Test that ``moffat_fit`` handles the case where no profile
        values are above half the maximum (gamma fallback to 1.0).
        """
        xycen, data, _, _ = profile_data

        # Create a near-delta-function profile so that all annular
        # averages may be below half-max (only center pixel has flux).
        delta = np.zeros_like(data)
        cy, cx = round(xycen[1]), round(xycen[0])
        delta[cy, cx] = 1.0

        edge_radii = np.arange(36)
        rp = RadialProfile(delta, xycen, edge_radii)
        # The fit may warn about convergence for such a narrow profile.
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', AstropyUserWarning)
            mfit = rp.moffat_fit
        assert isinstance(mfit, Moffat1D)
        assert mfit.gamma.value > 0
        assert mfit.alpha.value >= 1

    def test_moffat_lazyproperty(self, profile_data):
        """
        Test that ``moffat_fit``, ``moffat_profile``, and
        ``moffat_fwhm`` are lazily computed and cached.
        """
        xycen, data, _, _ = profile_data

        edge_radii = np.arange(36)
        rp = RadialProfile(data, xycen, edge_radii)

        fit1 = rp.moffat_fit
        fit2 = rp.moffat_fit
        assert fit1 is fit2

        prof1 = rp.moffat_profile
        prof2 = rp.moffat_profile
        assert prof1 is prof2

        fwhm1 = rp.moffat_fwhm
        fwhm2 = rp.moffat_fwhm
        assert fwhm1 == fwhm2

    def test_moffat_normalized(self, profile_data):
        """
        Test that normalizing the profile *after* fitting does not
        change the Moffat fit (lazyproperty caching).
        """
        xycen, data, _, _ = profile_data

        edge_radii = np.arange(36)
        rp = RadialProfile(data, xycen, edge_radii)

        fwhm_before = rp.moffat_fwhm
        rp.normalize()
        # Because moffat_fit is a lazyproperty computed before
        # normalization, the FWHM should be the same.
        assert rp.moffat_fwhm == fwhm_before

    def test_moffat_with_error(self, profile_data):
        """
        Test Moffat fit with error input.
        """
        xycen, data, error, _ = profile_data

        edge_radii = np.arange(36)
        rp = RadialProfile(data, xycen, edge_radii, error=error)
        assert isinstance(rp.moffat_fit, Moffat1D)
        assert rp.moffat_profile.shape == rp.profile.shape
        assert rp.moffat_fwhm > 0

    def test_moffat_with_mask(self, profile_data):
        """
        Test Moffat fit with a partial mask.
        """
        xycen, data, _, mask = profile_data

        edge_radii = np.arange(36)
        rp = RadialProfile(data, xycen, edge_radii, mask=mask)
        assert isinstance(rp.moffat_fit, Moffat1D)
        assert rp.moffat_fwhm > 0

    def test_moffat_nonfinite_data(self, profile_data):
        """
        Test Moffat fit with non-finite data values.
        """
        xycen, data, error, _ = profile_data

        data2 = data.copy()
        data2[40, 40] = np.nan
        edge_radii = np.arange(36)
        match = 'Input data contains non-finite values'
        with pytest.warns(AstropyUserWarning, match=match):
            rp = RadialProfile(data2, xycen, edge_radii, error=error)

        assert isinstance(rp.moffat_fit, Moffat1D)
        assert rp.moffat_fwhm > 0
