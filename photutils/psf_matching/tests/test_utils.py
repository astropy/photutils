# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the utils module.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from photutils.psf_matching.tests.conftest import _make_gaussian_psf
from photutils.psf_matching.utils import (_apply_window_to_fourier,
                                          _convert_psf_to_otf, _validate_psf,
                                          _validate_window_array, resize_psf)


class TestValidatePSF:
    def test_valid_psf(self):
        """
        Test that a valid PSF passes validation without error.
        """
        psf = _make_gaussian_psf(5, 1.5)
        _validate_psf(psf, 'psf')  # should not raise

    def test_non_2d(self):
        """
        Test that non-2D array raises ValueError.
        """
        match = 'psf must be a 2D array'
        with pytest.raises(ValueError, match=match):
            _validate_psf(np.ones(5), 'psf')

    def test_even_shape(self):
        """
        Test that even-shaped array raises ValueError.
        """
        psf = np.zeros((4, 4))
        psf[2, 2] = 1.0
        match = 'must have odd dimensions'
        with pytest.raises(ValueError, match=match):
            _validate_psf(psf, 'psf')

    def test_nan_inf_values(self):
        """
        Test that NaN or Inf values raise ValueError.
        """
        psf = np.zeros((25, 25))
        psf[12, 12] = np.nan
        match = 'contains NaN or Inf values'
        with pytest.raises(ValueError, match=match):
            _validate_psf(psf, 'psf')

        psf[12, 12] = np.inf
        with pytest.raises(ValueError, match=match):
            _validate_psf(psf, 'psf')

    def test_zero_psf_raises(self):
        """
        Test that an all-zero PSF raises ValueError (cannot be
        normalized).
        """
        psf = np.zeros((5, 5))
        match = 'must have a non-zero sum'
        with pytest.raises(ValueError, match=match):
            _validate_psf(psf, 'psf')


class TestValidateWindowArray:
    def test_valid_window(self):
        """
        Test that a valid window array passes validation.
        """
        shape = (25, 25)
        window = np.ones(shape)
        _validate_window_array(window, shape)  # should not raise

    def test_not_2d(self):
        """
        Test that non-2D window raises ValueError.
        """
        match = 'window function must return a 2D array'
        with pytest.raises(ValueError, match=match):
            _validate_window_array(np.ones(10), (10,))

    def test_wrong_shape(self):
        """
        Test that wrong-shaped window raises ValueError.
        """
        match = 'window function must return an array with shape'
        with pytest.raises(ValueError, match=match):
            _validate_window_array(np.ones((10, 10)), (25, 25))

    def test_values_below_zero(self):
        """
        Test that window values < 0 raise ValueError.
        """
        arr = np.ones((10, 10))
        arr[0, 0] = -0.1
        match = 'window function values must be in the range'
        with pytest.raises(ValueError, match=match):
            _validate_window_array(arr, (10, 10))

    def test_values_above_one(self):
        """
        Test that window values > 1 raise ValueError.
        """
        arr = np.ones((10, 10))
        arr[0, 0] = 1.5
        match = 'window function values must be in the range'
        with pytest.raises(ValueError, match=match):
            _validate_window_array(arr, (10, 10))


class TestConvertPsfToOtf:
    def test_zero_psf(self):
        """
        Test that an all-zero PSF returns an all-zero OTF.
        """
        psf = np.zeros((3, 3))
        otf = _convert_psf_to_otf(psf, (5, 5))
        assert_allclose(otf, 0.0)

    def test_output_shape(self):
        """
        Test that the output OTF has the requested shape.
        """
        psf = np.array([[0, -1, 0],
                        [-1, 4, -1],
                        [0, -1, 0]])
        otf = _convert_psf_to_otf(psf, (25, 25))
        assert otf.shape == (25, 25)

    def test_delta_function(self):
        """
        Test that a delta function produces a flat OTF.
        """
        # A single-pixel PSF at the center of a 1x1 array
        psf = np.array([[1.0]])
        otf = _convert_psf_to_otf(psf, (5, 5))
        assert_allclose(np.abs(otf), 1.0)

    def test_power_spectrum_shift_invariant(self):
        """
        Test that |OTF|^2 is the same regardless of PSF centering.

        The power spectrum should be independent of the input kernel
        position because the circular shift only affects the phase.
        """
        laplacian = np.array([[0, -1, 0],
                              [-1, 4, -1],
                              [0, -1, 0]])
        shape = (25, 25)

        otf = _convert_psf_to_otf(laplacian, shape)
        power = np.abs(otf) ** 2

        # Compare to a naive approach (no circular shift)
        padded = np.zeros(shape)
        padded[:3, :3] = laplacian
        otf_naive = np.fft.fft2(padded)
        power_naive = np.abs(otf_naive) ** 2

        assert_allclose(power, power_naive)

    def test_psf_larger_than_shape(self):
        """
        Test that a PSF larger than the target shape raises ValueError.
        """
        psf = np.ones((7, 7))
        match = 'PSF shape.*is larger than the target shape'
        with pytest.raises(ValueError, match=match):
            _convert_psf_to_otf(psf, (5, 5))

    def test_laplacian_dc_is_zero(self):
        """
        Test that the Laplacian OTF has zero power at DC.

        The Laplacian sums to zero, so its DC component should be zero.
        """
        laplacian = np.array([[0, -1, 0],
                              [-1, 4, -1],
                              [0, -1, 0]])
        otf = _convert_psf_to_otf(laplacian, (25, 25))
        assert_allclose(otf[0, 0], 0.0, atol=1e-15)

    def test_normalized_psf_dc_component(self):
        """
        Test that a normalized PSF has DC component equal to 1.0.

        For a normalized PSF (sum = 1), the DC component of the OTF
        should be 1.0.
        """
        psf = _make_gaussian_psf(5, 1.5)
        otf = _convert_psf_to_otf(psf, (25, 25))
        assert_allclose(otf[0, 0], 1.0, rtol=1e-10)

    def test_non_square_shapes(self):
        """
        Test with non-square PSF and output shapes.
        """
        # 3x5 PSF to 11x21 output
        psf = np.zeros((3, 5))
        psf[1, 2] = 1.0  # center at (1, 2)
        otf = _convert_psf_to_otf(psf, (11, 21))
        assert otf.shape == (11, 21)
        assert_allclose(np.abs(otf), 1.0, rtol=1e-10)

    def test_no_padding_needed(self):
        """
        Test when PSF size equals output size (no padding).
        """
        psf = _make_gaussian_psf(7, 1.5)
        otf = _convert_psf_to_otf(psf, (7, 7))
        assert otf.shape == (7, 7)
        assert_allclose(otf[0, 0], 1.0, rtol=1e-10)

    def test_symmetric_psf_real_otf(self):
        """
        Test that a symmetric PSF produces a nearly real OTF.

        A centered, symmetric PSF should produce an OTF with negligible
        imaginary components.
        """
        psf = _make_gaussian_psf(7, 2.0)
        otf = _convert_psf_to_otf(psf, (25, 25))
        # Imaginary components should be negligible
        assert np.max(np.abs(otf.imag)) < 1e-10

    def test_larger_psf_sizes(self):
        """
        Test with various PSF sizes to ensure centering works correctly.
        """
        for psf_size in [5, 7, 9]:
            psf = _make_gaussian_psf(psf_size, psf_size / 4.0)
            otf = _convert_psf_to_otf(psf, (25, 25))
            # Normalized PSF should have DC component of 1.0
            assert_allclose(otf[0, 0], 1.0, rtol=1e-10)
            # OTF should be nearly real for symmetric PSF
            assert np.max(np.abs(otf.imag)) < 1e-10


class TestResizePSF:
    def test_resize(self):
        """
        Test that resizing returns an odd-shaped output.

        For a (5,5) input with ratio=2.0, ceil gives 10 (even), so one
        pixel is added to give (11, 11).
        """
        psf = _make_gaussian_psf(5, 1.5)
        result = resize_psf(psf, 0.1, 0.05)
        assert result.shape == (11, 11)
        assert result.shape[0] % 2 == 1
        assert result.shape[1] % 2 == 1

    def test_resize_odd_output(self):
        """
        Test that resizing to a naturally odd output shape is unchanged.
        """
        psf = _make_gaussian_psf(5, 1.5)
        result = resize_psf(psf, 0.1, 0.1)  # ratio=1.0 -> 5x5 (odd)
        assert result.shape == (5, 5)

    def test_resize_always_odd(self):
        """
        Test that the output is always odd across a range of ratios.
        """
        psf = _make_gaussian_psf(5, 1.5)
        for scale_out in [0.04, 0.05, 0.06, 0.07, 0.08]:
            result = resize_psf(psf, 0.1, scale_out)
            assert result.shape[0] % 2 == 1, (
                f'Even output shape {result.shape} for scale={scale_out}')
            assert result.shape[1] % 2 == 1, (
                f'Even output shape {result.shape} for scale={scale_out}')

    def test_non_2d(self):
        """
        Test that non-2D PSF raises ValueError.
        """
        match = 'psf must be a 2D array'
        with pytest.raises(ValueError, match=match):
            resize_psf(np.ones(5), 0.1, 0.05)

    def test_even_shape(self):
        """
        Test that even-shaped PSF raises ValueError.
        """
        psf = np.zeros((4, 4))
        psf[2, 2] = 1.0
        match = 'must have odd dimensions'
        with pytest.raises(ValueError, match=match):
            resize_psf(psf, 0.1, 0.05)

    def test_non_positive_input_scale(self):
        """
        Test that negative input_pixel_scale raises ValueError.
        """
        psf = _make_gaussian_psf(5, 1.5)
        match = 'must be positive'
        with pytest.raises(ValueError, match=match):
            resize_psf(psf, -0.1, 0.05)

    def test_non_positive_output_scale(self):
        """
        Test that negative output_pixel_scale raises ValueError.
        """
        psf = _make_gaussian_psf(5, 1.5)
        match = 'must be positive'
        with pytest.raises(ValueError, match=match):
            resize_psf(psf, 0.1, -0.05)

    def test_zero_scale(self):
        """
        Test that zero input_pixel_scale raises ValueError.
        """
        psf = _make_gaussian_psf(5, 1.5)
        match = 'must be positive'
        with pytest.raises(ValueError, match=match):
            resize_psf(psf, 0.0, 0.05)


class TestApplyWindowToFourier:
    def test_basic(self):
        """
        Test that _apply_window_to_fourier applies a window to a
        Fourier array.
        """
        shape = (11, 11)
        fourier_array = np.ones(shape, dtype=complex)

        def uniform_window(shape):
            return np.ones(shape)

        result = _apply_window_to_fourier(fourier_array, uniform_window,
                                          shape)
        assert result.shape == shape
        assert np.allclose(result, fourier_array)

    def test_zero_window(self):
        """
        Test that a zero window zeros out the Fourier array.
        """
        shape = (11, 11)
        fourier_array = np.ones(shape, dtype=complex)

        def zero_window(shape):
            return np.zeros(shape)

        result = _apply_window_to_fourier(fourier_array, zero_window, shape)
        assert np.allclose(result, 0.0)

    def test_invalid_window_raises(self):
        """
        Test that an invalid window function raises ValueError.
        """
        shape = (11, 11)
        fourier_array = np.ones(shape, dtype=complex)

        def bad_window(shape):  # noqa: ARG001
            return np.ones((5, 5))  # wrong shape

        match = 'window function must return an array with shape'
        with pytest.raises(ValueError, match=match):
            _apply_window_to_fourier(fourier_array, bad_window, shape)
