# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the _utils module.
"""

import numpy as np
import pytest
from astropy.modeling.models import Gaussian2D
from astropy.utils.exceptions import AstropyUserWarning
from numpy.testing import assert_allclose, assert_array_equal

from photutils.centroids._utils import (_gaussian2d_moments,
                                        _process_data_mask, _validate_data,
                                        _validate_gaussian_inputs,
                                        _validate_mask_shape)


class TestValidateData:
    """
    Tests for _validate_data.
    """

    def test_converts_to_float(self):
        data = np.array([[1, 2], [3, 4]], dtype=int)
        result = _validate_data(data)
        assert result.dtype == float

    def test_2d_default(self):
        data = np.ones((4, 4))
        result = _validate_data(data)
        assert result.shape == (4, 4)

    def test_wrong_ndim_raises(self):
        data = np.ones((4, 4, 4))
        match = 'data must be a 2D array'
        with pytest.raises(ValueError, match=match):
            _validate_data(data, ndim=2)

    def test_1d_valid(self):
        data = np.ones(10)
        result = _validate_data(data, ndim=1)
        assert result.ndim == 1

    def test_ndim_none_accepts_any_shape(self):
        for shape in [(5,), (5, 5), (5, 5, 5)]:
            result = _validate_data(np.ones(shape), ndim=None)
            assert result.shape == shape


class TestValidateMaskShape:
    """
    Tests for _validate_mask_shape.
    """

    def test_none_mask_passes(self):
        data = np.ones((4, 4))
        _validate_mask_shape(data, None)

    def test_matching_shape_passes(self):
        data = np.ones((4, 4))
        mask = np.zeros((4, 4), dtype=bool)
        _validate_mask_shape(data, mask)

    def test_mismatched_shape_raises(self):
        data = np.ones((4, 4))
        mask = np.zeros((2, 2), dtype=bool)
        match = 'data and mask must have the same shape'
        with pytest.raises(ValueError, match=match):
            _validate_mask_shape(data, mask)


class TestProcessDataMask:
    """
    Tests for _process_data_mask.
    """

    def test_finite_data_unchanged(self):
        data = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = _process_data_mask(data, None)
        assert_array_equal(result, data)

    def test_nan_fills_and_warns(self):
        data = np.array([[1.0, np.nan], [3.0, 4.0]])
        match = 'Input data contains non-finite values'
        with pytest.warns(AstropyUserWarning, match=match):
            result = _process_data_mask(data, None)
        assert np.isnan(result[0, 1])

    def test_nan_fills_with_custom_fill_value(self):
        data = np.array([[1.0, np.nan], [3.0, 4.0]])
        with pytest.warns(AstropyUserWarning):
            result = _process_data_mask(data, None, fill_value=0.0)
        assert result[0, 1] == 0.0

    def test_mask_fills_fill_value(self):
        data = np.array([[1.0, 2.0], [3.0, 4.0]])
        mask = np.array([[False, True], [False, False]])
        result = _process_data_mask(data, mask, fill_value=0.0)
        assert result[0, 1] == 0.0

    def test_masked_nan_no_warning(self):
        data = np.array([[1.0, np.nan], [3.0, 4.0]])
        mask = np.array([[False, True], [False, False]])
        # Masked NaN should not trigger a warning
        result = _process_data_mask(data, mask, fill_value=0.0)
        assert result[0, 1] == 0.0

    def test_ndim_validation(self):
        data = np.ones((4, 4, 4))
        match = 'data must be a 2D array'
        with pytest.raises(ValueError, match=match):
            _process_data_mask(data, None, ndim=2)

    def test_mask_shape_validation(self):
        data = np.ones((4, 4))
        mask = np.zeros((2, 2), dtype=bool)
        match = 'data and mask must have the same shape'
        with pytest.raises(ValueError, match=match):
            _process_data_mask(data, mask)

    def test_masked_array_no_mutation(self):
        """
        Input mask must not be mutated when data is a MaskedArray.
        """
        masked_data = np.ma.array([[1.0, 2.0], [3.0, 4.0]],
                                  mask=[[False, True], [False, False]])
        input_mask = np.zeros((2, 2), dtype=bool)
        input_mask_orig = input_mask.copy()

        _process_data_mask(masked_data, input_mask)
        assert_array_equal(input_mask, input_mask_orig)

    def test_masked_array_returns_ndarray(self):
        """
        MaskedArray input must return a plain ndarray, not MaskedArray.
        """
        masked_data = np.ma.array([[1.0, 2.0], [3.0, 4.0]],
                                  mask=[[False, True], [False, False]])
        result = _process_data_mask(masked_data, None)
        assert not isinstance(result, np.ma.MaskedArray)
        assert isinstance(result, np.ndarray)

    def test_masked_array_mask_combined(self):
        """
        MaskedArray mask and keyword mask are combined correctly.
        """
        masked_data = np.ma.array([[1.0, 2.0], [3.0, 4.0]],
                                  mask=[[False, True], [False, False]])
        extra_mask = np.array([[True, False], [False, False]])
        result = _process_data_mask(masked_data, extra_mask, fill_value=0.0)
        # Both (0,0) from extra_mask and (0,1) from MaskedArray should
        # be 0
        assert result[0, 0] == 0.0
        assert result[0, 1] == 0.0


class TestValidateGaussianInputs:
    """
    Tests for _validate_gaussian_inputs.
    """

    @pytest.fixture
    def gauss_data(self):
        rng = np.random.default_rng(0)
        return np.abs(rng.standard_normal((20, 20))) + 1.0

    def test_no_error_no_mask(self, gauss_data):
        data, mask, error = _validate_gaussian_inputs(gauss_data, None, None)
        assert error is None
        assert data.shape == gauss_data.shape
        assert mask.shape == gauss_data.shape
        assert mask.dtype == bool

    def test_error_shape_mismatch_raises(self, gauss_data):
        error = np.ones((5, 5))
        match = 'data and error must have the same shape'
        with pytest.raises(ValueError, match=match):
            _validate_gaussian_inputs(gauss_data, None, error)

    def test_nan_in_error_sets_combined_mask(self, gauss_data):
        error = np.ones_like(gauss_data)
        error[5, 5] = np.nan
        data, combined_mask, _ = _validate_gaussian_inputs(
            gauss_data, None, error)
        assert combined_mask[5, 5]
        assert data[5, 5] == 0.0

    def test_nan_in_data_sets_combined_mask(self, gauss_data):
        gauss_data = gauss_data.copy()
        gauss_data[3, 3] = np.nan
        match = 'Input data contains non-finite values'
        with pytest.warns(AstropyUserWarning, match=match):
            data, combined_mask, _ = _validate_gaussian_inputs(
                gauss_data, None, None)
        assert combined_mask[3, 3]
        assert data[3, 3] == 0.0

    def test_input_mask_not_mutated(self, gauss_data):
        error = np.ones_like(gauss_data)
        error[5, 5] = np.nan
        mask = np.zeros(gauss_data.shape, dtype=bool)
        mask_orig = mask.copy()
        _validate_gaussian_inputs(gauss_data, mask, error)
        assert_array_equal(mask, mask_orig)

    def test_input_error_not_mutated(self, gauss_data):
        error = np.ones_like(gauss_data)
        error[5, 5] = np.nan
        error_orig = error.copy()
        _validate_gaussian_inputs(gauss_data, None, error)
        assert_array_equal(error, error_orig)

    def test_data_not_mutated_when_error_nan(self, gauss_data):
        """
        data must not be mutated when NaN in error extends combined_mask
        beyond the positions already handled by _process_data_mask.

        When mask=None and data is clean, _process_data_mask returns
        the original object without copying. If error then contributes
        new NaN positions, the code must not write 0.0 directly into
        the caller's array.
        """
        error = np.ones_like(gauss_data)
        error[5, 5] = np.nan
        data_orig = gauss_data.copy()
        _validate_gaussian_inputs(gauss_data, None, error)
        assert_array_equal(gauss_data, data_orig)

    def test_error_zeroed_at_combined_mask(self, gauss_data):
        mask = np.zeros(gauss_data.shape, dtype=bool)
        mask[2, 2] = True
        error = np.ones_like(gauss_data)
        error[5, 5] = np.nan
        _, _, out_error = _validate_gaussian_inputs(gauss_data, mask, error)
        assert out_error[2, 2] == 0.0   # masked by input mask
        assert out_error[5, 5] == 0.0   # masked by NaN in error

    def test_no_copy_when_no_invalids(self, gauss_data):
        """
        Test that error values are unchanged when there are no NaNs in
        data or error, and mask is None.
        """
        error = np.ones_like(gauss_data)
        _, _, out_error = _validate_gaussian_inputs(gauss_data, None, error)
        assert_array_equal(out_error, error)


class TestGaussian2DMoments:
    """
    Tests for _gaussian2d_moments.
    """

    def test_symmetric_gaussian(self):
        """
        Circular Gaussian: centroid and equal stddevs are recovered.
        """
        xcen, ycen, std = 25.0, 25.0, 5.0
        model = Gaussian2D(1.0, xcen, ycen, x_stddev=std, y_stddev=std)
        y, x = np.mgrid[0:50, 0:50]
        data = model(x, y)
        (amplitude,
         x_mean,
         y_mean,
         x_stddev,
         y_stddev,
         theta,
         ) = _gaussian2d_moments(data)

        assert_allclose(amplitude, 1.0, atol=0.01)
        assert_allclose(x_mean, xcen, atol=0.01)
        assert_allclose(y_mean, ycen, atol=0.01)
        assert_allclose(x_stddev, std, atol=0.1)
        assert_allclose(y_stddev, std, atol=0.1)
        assert_allclose(theta, 0, atol=0.05)

    @pytest.mark.parametrize('theta_in',
                             np.deg2rad((0, 22, 37, 45, 60, 88, 90)))
    def test_asymmetric_gaussian_theta(self, theta_in):
        """
        Axis-aligned elliptical Gaussian (theta=0): centroid and axis
        stddevs are recovered. The larger sigma maps to x_stddev.
        """
        ampl = 3.5
        xcen, ycen = 30.0, 20.0
        x_std, y_std = 6.0, 3.0
        model = Gaussian2D(ampl, xcen, ycen, x_stddev=x_std, y_stddev=y_std,
                           theta=theta_in)
        y, x = np.mgrid[0:50, 0:50]
        data = model(x, y)
        (amplitude,
         x_mean,
         y_mean,
         x_stddev,
         y_stddev,
         theta,
         ) = _gaussian2d_moments(data)

        assert_allclose(amplitude, ampl, atol=0.01)
        assert_allclose(x_mean, xcen, atol=0.02)
        assert_allclose(y_mean, ycen, atol=0.02)
        assert_allclose(x_stddev, x_std, atol=0.1)
        assert_allclose(y_stddev, y_std, atol=0.1)
        assert_allclose(theta, theta_in, atol=0.05)
