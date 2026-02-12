# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the fourier module.
"""

import numpy as np
import pytest
from astropy.modeling.fitting import TRFLSQFitter
from astropy.modeling.models import Gaussian2D
from astropy.utils.exceptions import AstropyUserWarning
from numpy.testing import assert_allclose

from photutils.psf_matching.fourier import create_matching_kernel, resize_psf
from photutils.psf_matching.windows import SplitCosineBellWindow


def _make_gaussian_psf(size, std):
    """
    Make a centered, normalized 2D Gaussian PSF.
    """
    cen = (size - 1) / 2.0
    y, x = np.mgrid[0:size, 0:size]
    model = Gaussian2D(1.0, cen, cen, std, std)
    psf = model(x, y)
    return psf / psf.sum()


@pytest.fixture
def psf1():
    """
    Narrow Gaussian PSF (source).
    """
    return _make_gaussian_psf(25, 3.0)


@pytest.fixture
def psf2():
    """
    Broad Gaussian PSF (target).
    """
    return _make_gaussian_psf(25, 5.0)


class TestResizePSF:
    def test_resize(self):
        psf = _make_gaussian_psf(5, 1.5)
        result = resize_psf(psf, 0.1, 0.05)
        assert result.shape == (10, 10)

    def test_non_2d(self):
        match = 'psf must be a 2D array'
        with pytest.raises(ValueError, match=match):
            resize_psf(np.ones(5), 0.1, 0.05)

    def test_even_shape(self):
        psf = np.zeros((4, 4))
        psf[2, 2] = 1.0
        match = 'must have odd dimensions'
        with pytest.raises(ValueError, match=match):
            resize_psf(psf, 0.1, 0.05)

    def test_not_centered(self):
        psf = np.zeros((5, 5))
        psf[0, 0] = 1.0
        match = r'The peak .* is not centered'
        with pytest.warns(AstropyUserWarning, match=match):
            resize_psf(psf, 0.1, 0.05)

    def test_non_positive_input_scale(self):
        psf = _make_gaussian_psf(5, 1.5)
        match = 'must be positive'
        with pytest.raises(ValueError, match=match):
            resize_psf(psf, -0.1, 0.05)

    def test_non_positive_output_scale(self):
        psf = _make_gaussian_psf(5, 1.5)
        match = 'must be positive'
        with pytest.raises(ValueError, match=match):
            resize_psf(psf, 0.1, -0.05)

    def test_zero_scale(self):
        psf = _make_gaussian_psf(5, 1.5)
        match = 'must be positive'
        with pytest.raises(ValueError, match=match):
            resize_psf(psf, 0.0, 0.05)


class TestCreateMatchingKernel:
    def test_with_window(self, psf1, psf2):
        """
        Test with noiseless 2D Gaussians and a window.
        """
        size = psf1.shape[0]
        cen = (size - 1) / 2.0
        y, x = np.mgrid[0:size, 0:size]
        window = SplitCosineBellWindow(0.0, 0.2)
        k = create_matching_kernel(psf1, psf2, window=window)

        fitter = TRFLSQFitter()
        gm1 = Gaussian2D(1.0, cen, cen, 3.0, 3.0)
        gfit = fitter(gm1, x, y, k)
        assert_allclose(gfit.x_stddev, gfit.y_stddev)
        assert_allclose(gfit.x_stddev, np.sqrt(25 - 9), atol=0.06)

    def test_without_window(self, psf1, psf2):
        """
        Test without a window function.
        """
        k = create_matching_kernel(psf1, psf2)
        assert k.shape == psf1.shape
        assert_allclose(k.sum(), 1.0)

    def test_shape_mismatch(self, psf1):
        psf_small = _make_gaussian_psf(5, 1.5)
        match = 'must have the same shape'
        with pytest.raises(ValueError, match=match):
            create_matching_kernel(psf1, psf_small)

    def test_non_2d_source(self, psf2):
        match = 'source_psf must be a 2D array'
        with pytest.raises(ValueError, match=match):
            create_matching_kernel(np.ones(25), psf2)

    def test_non_2d_target(self, psf1):
        match = 'target_psf must be a 2D array'
        with pytest.raises(ValueError, match=match):
            create_matching_kernel(psf1, np.ones(25))

    def test_even_shape(self):
        psf = np.zeros((4, 4))
        psf[2, 2] = 1.0
        match = 'must have odd dimensions'
        with pytest.raises(ValueError, match=match):
            create_matching_kernel(psf, psf)

    def test_source_not_centered(self, psf2):
        psf = np.zeros((25, 25))
        psf[0, 0] = 1.0
        match = r'The peak .* is not centered'
        with pytest.warns(AstropyUserWarning, match=match):
            create_matching_kernel(psf, psf2)

    def test_target_not_centered(self, psf1):
        psf = np.zeros((25, 25))
        psf[0, 0] = 1.0
        match = r'The peak .* is not centered'
        with pytest.warns(AstropyUserWarning, match=match):
            create_matching_kernel(psf1, psf)

    def test_non_callable_window(self, psf1, psf2):
        match = 'window must be a callable'
        with pytest.raises(TypeError, match=match):
            create_matching_kernel(psf1, psf2, window='bad')

    def test_fourier_cutoff(self, psf1, psf2):
        """
        Test with an aggressive fourier cutoff.
        """
        k = create_matching_kernel(psf1, psf2,
                                   fourier_cutoff=0.5)
        assert k.shape == psf1.shape
        assert_allclose(k.sum(), 1.0)

    def test_fourier_cutoff_zero(self, psf1, psf2):
        """
        Test with fourier_cutoff=0 (minimum filtering).
        """
        k = create_matching_kernel(psf1, psf2,
                                   fourier_cutoff=0)
        assert k.shape == psf1.shape
        assert_allclose(k.sum(), 1.0)

    def test_fourier_cutoff_negative(self, psf1, psf2):
        """
        Test that negative fourier_cutoff raises an error.
        """
        match = 'fourier_cutoff must be in the range'
        with pytest.raises(ValueError, match=match):
            create_matching_kernel(psf1, psf2,
                                   fourier_cutoff=-0.1)

    def test_fourier_cutoff_greater_than_one(self, psf1, psf2):
        """
        Test that fourier_cutoff > 1 raises an error.
        """
        match = 'fourier_cutoff must be in the range'
        with pytest.raises(ValueError, match=match):
            create_matching_kernel(psf1, psf2,
                                   fourier_cutoff=1.5)

    def test_window_not_2d(self, psf1, psf2):
        """
        Test that window function returning non-2D array raises error.
        """
        def bad_window(shape):
            return np.ones(shape[0])  # 1D array

        match = 'window function must return a 2D array'
        with pytest.raises(ValueError, match=match):
            create_matching_kernel(psf1, psf2, window=bad_window)

    def test_window_wrong_shape(self, psf1, psf2):
        """
        Test that window function returning wrong shape raises error.
        """
        def bad_window(shape):  # noqa: ARG001
            return np.ones((10, 10))  # wrong shape

        match = 'window function must return an array with shape'
        with pytest.raises(ValueError, match=match):
            create_matching_kernel(psf1, psf2, window=bad_window)

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
            create_matching_kernel(psf1, psf2, window=bad_window)

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
            create_matching_kernel(psf1, psf2, window=bad_window)
