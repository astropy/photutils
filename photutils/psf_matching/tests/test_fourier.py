# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the fourier module.
"""

import numpy as np
import pytest
from astropy.modeling.fitting import TRFLSQFitter
from astropy.modeling.models import Gaussian2D
from astropy.utils.exceptions import AstropyDeprecationWarning
from numpy.testing import assert_allclose

from photutils.psf_matching.fourier import (create_matching_kernel,
                                            make_kernel, make_wiener_kernel)
from photutils.psf_matching.tests.conftest import _make_gaussian_psf
from photutils.psf_matching.windows import SplitCosineBellWindow


class TestMakeKernel:
    def test_with_window(self, psf1, psf2):
        """
        Test with noiseless 2D Gaussians and a window.
        """
        size = psf1.shape[0]
        cen = (size - 1) / 2.0
        yy, xx = np.mgrid[0:size, 0:size]
        window = SplitCosineBellWindow(0.0, 0.2)
        kernel = make_kernel(psf1, psf2, window=window)

        fitter = TRFLSQFitter()
        gm1 = Gaussian2D(1.0, cen, cen, 3.0, 3.0)
        gfit = fitter(gm1, xx, yy, kernel)
        assert_allclose(gfit.x_stddev, gfit.y_stddev)
        assert_allclose(gfit.x_stddev, np.sqrt(25 - 9), atol=0.06)

    def test_without_window(self, psf1, psf2):
        """
        Test without a window function.
        """
        kernel = make_kernel(psf1, psf2)
        assert kernel.shape == psf1.shape
        assert_allclose(kernel.sum(), 1.0)

    def test_shape_mismatch(self, psf1):
        """
        Test that mismatched PSF shapes raise ValueError.
        """
        psf_small = _make_gaussian_psf(5, 1.5)
        match = 'must have the same shape'
        with pytest.raises(ValueError, match=match):
            make_kernel(psf1, psf_small)

    def test_non_2d_source(self, psf2):
        """
        Test that non-2D source PSF raises ValueError.
        """
        match = 'source_psf must be a 2D array'
        with pytest.raises(ValueError, match=match):
            make_kernel(np.ones(25), psf2)

    def test_non_2d_target(self, psf1):
        """
        Test that non-2D target PSF raises ValueError.
        """
        match = 'target_psf must be a 2D array'
        with pytest.raises(ValueError, match=match):
            make_kernel(psf1, np.ones(25))

    def test_even_shape(self):
        """
        Test that even-shaped PSFs raise ValueError.
        """
        psf = np.zeros((4, 4))
        psf[2, 2] = 1.0
        match = 'must have odd dimensions'
        with pytest.raises(ValueError, match=match):
            make_kernel(psf, psf)

    def test_non_callable_window(self, psf1, psf2):
        """
        Test that non-callable window raises TypeError.
        """
        match = 'window must be a callable'
        with pytest.raises(TypeError, match=match):
            make_kernel(psf1, psf2, window='bad')

    def test_regularization(self, psf1, psf2):
        """
        Test with an aggressive regularization threshold.
        """
        kernel = make_kernel(psf1, psf2, regularization=0.5)
        assert kernel.shape == psf1.shape
        assert_allclose(kernel.sum(), 1.0)

    def test_regularization_zero(self, psf1, psf2):
        """
        Test with regularization=0 (minimum thresholding).
        """
        kernel = make_kernel(psf1, psf2, regularization=0)
        assert kernel.shape == psf1.shape
        assert_allclose(kernel.sum(), 1.0)

    def test_regularization_negative(self, psf1, psf2):
        """
        Test that negative regularization raises an error.
        """
        match = 'regularization must be in the range'
        with pytest.raises(ValueError, match=match):
            make_kernel(psf1, psf2, regularization=-0.1)

    def test_regularization_greater_than_one(self, psf1, psf2):
        """
        Test that regularization > 1 raises an error.
        """
        match = 'regularization must be in the range'
        with pytest.raises(ValueError, match=match):
            make_kernel(psf1, psf2, regularization=1.5)

    def test_regularization_invalid(self, psf1, psf2):
        """
        Test that regularization=1 raises an error (range is [0, 1)).
        """
        match = 'regularization must be in the range'
        with pytest.raises(ValueError, match=match):
            make_kernel(psf1, psf2, regularization=1.0)

    def test_window_not_2d(self, psf1, psf2):
        """
        Test that window function returning non-2D array raises error.
        """
        def bad_window(shape):
            return np.ones(shape[0])  # 1D array

        match = 'window function must return a 2D array'
        with pytest.raises(ValueError, match=match):
            make_kernel(psf1, psf2, window=bad_window)

    def test_window_wrong_shape(self, psf1, psf2):
        """
        Test that window function returning wrong shape raises error.
        """
        def bad_window(shape):  # noqa: ARG001
            return np.ones((10, 10))  # wrong shape

        match = 'window function must return an array with shape'
        with pytest.raises(ValueError, match=match):
            make_kernel(psf1, psf2, window=bad_window)

    def test_window_values_below_zero(self, psf1, psf2):
        """
        Test that window function with values < 0 raises error.
        """
        def bad_window(shape):
            arr = np.ones(shape)
            arr[0, 0] = -0.1
            return arr

        match = 'window function values must be in the range'
        with pytest.raises(ValueError, match=match):
            make_kernel(psf1, psf2, window=bad_window)

    def test_window_values_above_one(self, psf1, psf2):
        """
        Test that window function with values > 1 raises error.
        """
        def bad_window(shape):
            arr = np.ones(shape)
            arr[0, 0] = 1.5
            return arr

        match = 'window function values must be in the range'
        with pytest.raises(ValueError, match=match):
            make_kernel(psf1, psf2, window=bad_window)

    def test_asymmetric_shape(self):
        """
        Test with asymmetric PSF shapes.
        """
        # Create 51x25 PSFs centered at (y=25, x=12)
        y, x = np.mgrid[0:51, 0:25]
        psf1 = Gaussian2D(100, 12, 25, 3, 3)(x, y)
        psf2 = Gaussian2D(100, 12, 25, 5, 5)(x, y)
        psf1 /= psf1.sum()
        psf2 /= psf2.sum()

        kernel = make_kernel(psf1, psf2)
        assert kernel.shape == (51, 25)
        assert_allclose(kernel.sum(), 1.0)


class TestMakeKernelWiener:
    def test_basic(self, psf1, psf2):
        """
        Test basic Wiener kernel creation with noiseless Gaussians.
        """
        kernel = make_wiener_kernel(psf1, psf2)
        assert kernel.shape == psf1.shape
        assert_allclose(kernel.sum(), 1.0)

    def test_kernel_shape(self, psf1, psf2):
        """
        Test that the kernel has the expected Gaussian shape.

        For two Gaussians with sigma=3 and sigma=5, the matching
        kernel should be a Gaussian with sigma=sqrt(25-9)=4.
        """
        size = psf1.shape[0]
        cen = (size - 1) / 2.0
        yy, xx = np.mgrid[0:size, 0:size]
        kernel = make_wiener_kernel(psf1, psf2)

        fitter = TRFLSQFitter()
        gm1 = Gaussian2D(1.0, cen, cen, 3.0, 3.0)
        gfit = fitter(gm1, xx, yy, kernel)
        assert_allclose(gfit.x_stddev, gfit.y_stddev)
        assert_allclose(gfit.x_stddev, np.sqrt(25 - 9), atol=0.06)

    def test_with_window(self, psf1, psf2):
        """
        Test with a window function applied.
        """
        window = SplitCosineBellWindow(0.0, 0.2)
        kernel = make_wiener_kernel(psf1, psf2, window=window)
        assert kernel.shape == psf1.shape
        assert_allclose(kernel.sum(), 1.0)

    def test_regularization(self, psf1, psf2):
        """
        Test with different regularization strengths.
        """
        kernel_weak = make_wiener_kernel(psf1, psf2, regularization=1e-8)
        kernel_strong = make_wiener_kernel(psf1, psf2, regularization=1e-1)
        # Both should be normalized
        assert_allclose(kernel_weak.sum(), 1.0)
        assert_allclose(kernel_strong.sum(), 1.0)
        # Stronger regularization should produce a smoother kernel
        # (lower max value)
        assert kernel_strong.max() < kernel_weak.max()

    def test_shape_mismatch(self, psf1):
        """
        Test that mismatched PSF shapes raise ValueError.
        """
        g_small = _make_gaussian_psf(5, 1.5)
        match = 'must have the same shape'
        with pytest.raises(ValueError, match=match):
            make_wiener_kernel(psf1, g_small)

    def test_non_2d_source(self, psf2):
        """
        Test that non-2D source PSF raises ValueError.
        """
        match = 'source_psf must be a 2D array'
        with pytest.raises(ValueError, match=match):
            make_wiener_kernel(np.ones(25), psf2)

    def test_non_2d_target(self, psf1):
        """
        Test that non-2D target PSF raises ValueError.
        """
        match = 'target_psf must be a 2D array'
        with pytest.raises(ValueError, match=match):
            make_wiener_kernel(psf1, np.ones(25))

    def test_even_shape(self):
        """
        Test that even-shaped PSFs raise ValueError.
        """
        psf = np.zeros((4, 4))
        psf[2, 2] = 1.0
        match = 'must have odd dimensions'
        with pytest.raises(ValueError, match=match):
            make_wiener_kernel(psf, psf)

    def test_non_callable_window(self, psf1, psf2):
        """
        Test that non-callable window raises TypeError.
        """
        match = 'window must be a callable'
        with pytest.raises(TypeError, match=match):
            make_wiener_kernel(psf1, psf2, window='bad')

    def test_negative_regularization(self, psf1, psf2):
        """
        Test that negative regularization raises ValueError.
        """
        match = 'regularization must be a positive number'
        with pytest.raises(ValueError, match=match):
            make_wiener_kernel(psf1, psf2, regularization=-1.0)

    def test_zero_regularization(self, psf1, psf2):
        """
        Test that zero regularization raises ValueError.
        """
        match = 'regularization must be a positive number'
        with pytest.raises(ValueError, match=match):
            make_wiener_kernel(psf1, psf2, regularization=0.0)

    def test_penalty_string_equals_array(self, psf1, psf2):
        """
        Test that 'laplacian' string gives the same result as the
        explicit Laplacian array.
        """
        laplacian = np.array([[0, -1, 0],
                              [-1, 4, -1],
                              [0, -1, 0]])
        kernel_str = make_wiener_kernel(psf1, psf2, penalty='laplacian')
        kernel_arr = make_wiener_kernel(psf1, psf2, penalty=laplacian)
        assert_allclose(kernel_str, kernel_arr)

    def test_penalty_laplacian_kernel_shape(self, psf1, psf2):
        """
        Test that Laplacian penalty kernel has the expected Gaussian
        shape.
        """
        size = psf1.shape[0]
        cen = (size - 1) / 2.0
        yy, xx = np.mgrid[0:size, 0:size]
        kernel = make_wiener_kernel(psf1, psf2, penalty='laplacian')

        fitter = TRFLSQFitter()
        gm1 = Gaussian2D(1.0, cen, cen, 3.0, 3.0)
        gfit = fitter(gm1, xx, yy, kernel)
        assert_allclose(gfit.x_stddev, gfit.y_stddev)
        assert_allclose(gfit.x_stddev, np.sqrt(25 - 9), atol=0.06)

    def test_penalty_invalid_string(self, psf1, psf2):
        """
        Test that an invalid penalty string raises ValueError.
        """
        match = 'Invalid penalty string'
        with pytest.raises(ValueError, match=match):
            make_wiener_kernel(psf1, psf2, penalty='invalid')

    def test_penalty_invalid_type(self, psf1, psf2):
        """
        Test that an invalid penalty type raises ValueError.
        """
        match = 'penalty must be None'
        with pytest.raises(ValueError, match=match):
            make_wiener_kernel(psf1, psf2, penalty=42)

    def test_penalty_non_2d_array(self, psf1, psf2):
        """
        Test that a non-2D penalty array raises ValueError.
        """
        match = 'penalty array must be 2D'
        with pytest.raises(ValueError, match=match):
            make_wiener_kernel(psf1, psf2, penalty=np.ones(5))

    def test_penalty_psf_too_small_for_laplacian(self):
        """
        Test that a PSF smaller than 3x3 raises ValueError when using
        laplacian penalty.
        """
        # Create 1x1 PSFs (too small for 3x3 laplacian)
        psf1 = np.array([[1.0]])
        psf2 = np.array([[1.0]])
        match = 'PSFs must be at least as large as the penalty operator'
        with pytest.raises(ValueError, match=match):
            make_wiener_kernel(psf1, psf2, penalty='laplacian')

    def test_penalty_psf_too_small_for_biharmonic(self):
        """
        Test that a PSF smaller than 5x5 raises ValueError when using
        biharmonic penalty.
        """
        # Create 3x3 PSFs (too small for 5x5 biharmonic)
        psf1 = _make_gaussian_psf(3, 1.0)
        psf2 = _make_gaussian_psf(3, 1.5)
        match = 'PSFs must be at least as large as the penalty operator'
        with pytest.raises(ValueError, match=match):
            make_wiener_kernel(psf1, psf2, penalty='biharmonic')

    def test_penalty_psf_minimum_size_laplacian(self):
        """
        Test that 3x3 PSF (minimum size) works with laplacian penalty.
        """
        psf1 = _make_gaussian_psf(3, 0.8)
        psf2 = _make_gaussian_psf(3, 1.0)
        kernel = make_wiener_kernel(psf1, psf2, penalty='laplacian')
        assert kernel.shape == (3, 3)
        assert_allclose(kernel.sum(), 1.0)

    def test_penalty_psf_minimum_size_biharmonic(self):
        """
        Test that 5x5 PSF (minimum size) works with biharmonic penalty.
        """
        psf1 = _make_gaussian_psf(5, 1.2)
        psf2 = _make_gaussian_psf(5, 1.5)
        kernel = make_wiener_kernel(psf1, psf2, penalty='biharmonic')
        assert kernel.shape == (5, 5)
        assert_allclose(kernel.sum(), 1.0)

    def test_penalty_custom_array_too_large(self):
        """
        Test that a custom penalty array larger than the PSF raises
        ValueError.
        """
        # Create 3x3 PSFs but 5x5 penalty
        psf1 = _make_gaussian_psf(3, 0.8)
        psf2 = _make_gaussian_psf(3, 1.0)
        penalty = np.ones((5, 5))
        match = 'PSFs must be at least as large as the penalty operator'
        with pytest.raises(ValueError, match=match):
            make_wiener_kernel(psf1, psf2, penalty=penalty)

    def test_penalty_custom_array(self, psf1, psf2):
        """
        Test with a custom 2D penalty array.
        """
        # Use a simple high-pass operator
        penalty = np.array([[0, 0, 0],
                            [0, 1, 0],
                            [0, 0, 0]])
        kernel = make_wiener_kernel(psf1, psf2, penalty=penalty)
        assert kernel.shape == psf1.shape
        assert_allclose(kernel.sum(), 1.0)

    def test_penalty_with_window(self, psf1, psf2):
        """
        Test that penalty and window can be used together.
        """
        window = SplitCosineBellWindow(0.0, 0.2)
        kernel = make_wiener_kernel(psf1, psf2, penalty='laplacian',
                                    window=window)
        assert kernel.shape == psf1.shape
        assert_allclose(kernel.sum(), 1.0)

    def test_penalty_differs_from_scalar(self, psf1, psf2):
        """
        Test that Laplacian penalty gives a different result than scalar
        Tikhonov with the same regularization parameter.
        """
        reg = 1e-4
        kernel_scalar = make_wiener_kernel(psf1, psf2, regularization=reg)
        kernel_laplacian = make_wiener_kernel(psf1, psf2, regularization=reg,
                                              penalty='laplacian')
        assert not np.allclose(kernel_scalar, kernel_laplacian)

    def test_penalty_biharmonic_basic(self, psf1, psf2):
        """
        Test basic biharmonic penalty functionality.
        """
        kernel = make_wiener_kernel(psf1, psf2, penalty='biharmonic')
        assert kernel.shape == psf1.shape
        assert_allclose(kernel.sum(), 1.0)

    def test_penalty_biharmonic_kernel_shape(self, psf1, psf2):
        """
        Test that biharmonic penalty kernel has the expected Gaussian
        shape.
        """
        size = psf1.shape[0]
        cen = (size - 1) / 2.0
        yy, xx = np.mgrid[0:size, 0:size]
        kernel = make_wiener_kernel(psf1, psf2, penalty='biharmonic')

        fitter = TRFLSQFitter()
        gm1 = Gaussian2D(1.0, cen, cen, 3.0, 3.0)
        gfit = fitter(gm1, xx, yy, kernel)
        assert_allclose(gfit.x_stddev, gfit.y_stddev)
        assert_allclose(gfit.x_stddev, np.sqrt(25 - 9), atol=0.08)

    def test_penalty_biharmonic_results(self, psf1, psf2):
        """
        Test that biharmonic penalty gives different results than scalar
        and Laplacian penalties.
        """
        reg = 1e-4
        kernel_scalar = make_wiener_kernel(psf1, psf2, regularization=reg)
        kernel_laplacian = make_wiener_kernel(psf1, psf2, regularization=reg,
                                              penalty='laplacian')
        kernel_biharmonic = make_wiener_kernel(psf1, psf2, regularization=reg,
                                               penalty='biharmonic')
        assert not np.allclose(kernel_scalar, kernel_biharmonic)
        assert not np.allclose(kernel_laplacian, kernel_biharmonic)

    def test_penalty_biharmonic_smoothness(self, psf1, psf2):
        """
        Test that biharmonic penalty produces smoother kernels than
        Laplacian (lower peak value indicates more smoothing).
        """
        reg = 1e-4
        kernel_laplacian = make_wiener_kernel(psf1, psf2, regularization=reg,
                                              penalty='laplacian')
        kernel_biharmonic = make_wiener_kernel(psf1, psf2, regularization=reg,
                                               penalty='biharmonic')
        # Biharmonic should produce a smoother kernel with lower peak
        assert kernel_biharmonic.max() < kernel_laplacian.max()

    def test_penalty_biharmonic_with_window(self, psf1, psf2):
        """
        Test that biharmonic penalty works with window functions.
        """
        window = SplitCosineBellWindow(0.0, 0.2)
        kernel = make_wiener_kernel(psf1, psf2, penalty='biharmonic',
                                    window=window)
        assert kernel.shape == psf1.shape
        assert_allclose(kernel.sum(), 1.0)

    def test_asymmetric_shape(self):
        """
        Test with asymmetric PSF shapes.
        """
        # Create 51x25 PSFs centered at (y=25, x=12)
        y, x = np.mgrid[0:51, 0:25]
        psf1 = Gaussian2D(100, 12, 25, 3, 3)(x, y)
        psf2 = Gaussian2D(100, 12, 25, 5, 5)(x, y)
        psf1 /= psf1.sum()
        psf2 /= psf2.sum()

        kernel = make_wiener_kernel(psf1, psf2)
        assert kernel.shape == (51, 25)
        assert_allclose(kernel.sum(), 1.0)

    def test_window_not_2d(self, psf1, psf2):
        """
        Test that window function returning non-2D array raises error.
        """
        def bad_window(shape):
            return np.ones(shape[0])  # 1D array

        match = 'window function must return a 2D array'
        with pytest.raises(ValueError, match=match):
            make_wiener_kernel(psf1, psf2, window=bad_window)

    def test_window_wrong_shape(self, psf1, psf2):
        """
        Test that window function returning wrong shape raises error.
        """
        def bad_window(shape):  # noqa: ARG001
            return np.ones((10, 10))  # wrong shape

        match = 'window function must return an array with shape'
        with pytest.raises(ValueError, match=match):
            make_wiener_kernel(psf1, psf2, window=bad_window)

    def test_window_values_below_zero(self, psf1, psf2):
        """
        Test that window function with values < 0 raises error.
        """
        def bad_window(shape):
            arr = np.ones(shape)
            arr[0, 0] = -0.1
            return arr

        match = 'window function values must be in the range'
        with pytest.raises(ValueError, match=match):
            make_wiener_kernel(psf1, psf2, window=bad_window)

    def test_window_values_above_one(self, psf1, psf2):
        """
        Test that window function with values > 1 raises error.
        """
        def bad_window(shape):
            arr = np.ones(shape)
            arr[0, 0] = 1.5
            return arr

        match = 'window function values must be in the range'
        with pytest.raises(ValueError, match=match):
            make_wiener_kernel(psf1, psf2, window=bad_window)


class TestCreateMatchingKernelDeprecated:
    def test_deprecation_warning(self, psf1, psf2):
        """
        Test that create_matching_kernel raises a deprecation warning.
        """
        with pytest.warns(AstropyDeprecationWarning):
            kernel = create_matching_kernel(psf1, psf2)
        assert_allclose(kernel.sum(), 1.0)

    def test_deprecation_result_matches_make_kernel(self, psf1, psf2):
        """
        Test that create_matching_kernel returns the same result as
        make_kernel.
        """
        with pytest.warns(AstropyDeprecationWarning):
            kernel_old = create_matching_kernel(psf1, psf2)
        kernel_new = make_kernel(psf1, psf2)
        assert_allclose(kernel_old, kernel_new)
