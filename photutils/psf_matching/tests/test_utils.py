# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the utils module.
"""

import numpy as np
import pytest
from astropy.modeling.models import Gaussian2D
from astropy.utils.exceptions import AstropyUserWarning
from numpy.testing import assert_allclose

from photutils.psf_matching.utils import (_convert_psf_to_otf, _validate_psf,
                                          _validate_window_array, resize_psf)


def _make_gaussian_psf(size, std):
    """
    Make a centered, normalized 2D Gaussian PSF.
    """
    cen = (size - 1) / 2.0
    yy, xx = np.mgrid[0:size, 0:size]
    model = Gaussian2D(1.0, cen, cen, std, std)
    psf = model(xx, yy)
    return psf / psf.sum()


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

    def test_not_centered(self):
        """
        Test that non-centered PSF produces a warning.
        """
        psf = np.zeros((5, 5))
        psf[0, 0] = 1.0
        match = r'The peak .* is not centered'
        with pytest.warns(AstropyUserWarning, match=match):
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

    def test_negative_values(self):
        """
        Test that negative values produce a warning.
        """
        psf = _make_gaussian_psf(25, 3.0)
        psf[0, 0] = -0.1
        match = 'contains negative values'
        with pytest.warns(AstropyUserWarning, match=match):
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


class TestResizePSF:
    def test_resize(self):
        """
        Test basic PSF resizing from one pixel scale to another.
        """
        psf = _make_gaussian_psf(5, 1.5)
        result = resize_psf(psf, 0.1, 0.05)
        assert result.shape == (10, 10)

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

    def test_not_centered(self):
        """
        Test that non-centered PSF produces a warning.
        """
        psf = np.zeros((5, 5))
        psf[0, 0] = 1.0
        match = r'The peak .* is not centered'
        with pytest.warns(AstropyUserWarning, match=match):
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
