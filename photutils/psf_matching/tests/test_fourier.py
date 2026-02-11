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
    def setup_method(self):
        self.size = 25
        self.g1 = _make_gaussian_psf(self.size, 3.0)
        self.g2 = _make_gaussian_psf(self.size, 5.0)

    def test_with_window(self):
        """
        Test with noiseless 2D Gaussians and a window.
        """
        cen = (self.size - 1) / 2.0
        y, x = np.mgrid[0:self.size, 0:self.size]
        window = SplitCosineBellWindow(0.0, 0.2)
        k = create_matching_kernel(self.g1, self.g2, window=window)

        fitter = TRFLSQFitter()
        gm1 = Gaussian2D(1.0, cen, cen, 3.0, 3.0)
        gfit = fitter(gm1, x, y, k)
        assert_allclose(gfit.x_stddev, gfit.y_stddev)
        assert_allclose(gfit.x_stddev, np.sqrt(25 - 9), atol=0.06)

    def test_without_window(self):
        """
        Test without a window function.
        """
        k = create_matching_kernel(self.g1, self.g2)
        assert k.shape == (self.size, self.size)
        assert_allclose(k.sum(), 1.0)

    def test_shape_mismatch(self):
        g_small = _make_gaussian_psf(5, 1.5)
        match = 'must have the same shape'
        with pytest.raises(ValueError, match=match):
            create_matching_kernel(self.g1, g_small)

    def test_non_2d_source(self):
        match = 'source_psf must be a 2D array'
        with pytest.raises(ValueError, match=match):
            create_matching_kernel(np.ones(25), self.g2)

    def test_non_2d_target(self):
        match = 'target_psf must be a 2D array'
        with pytest.raises(ValueError, match=match):
            create_matching_kernel(self.g1, np.ones(25))

    def test_even_shape(self):
        psf = np.zeros((4, 4))
        psf[2, 2] = 1.0
        match = 'must have odd dimensions'
        with pytest.raises(ValueError, match=match):
            create_matching_kernel(psf, psf)

    def test_source_not_centered(self):
        psf = np.zeros((25, 25))
        psf[0, 0] = 1.0
        match = r'The peak .* is not centered'
        with pytest.warns(AstropyUserWarning, match=match):
            create_matching_kernel(psf, self.g2)

    def test_target_not_centered(self):
        psf = np.zeros((25, 25))
        psf[0, 0] = 1.0
        match = r'The peak .* is not centered'
        with pytest.warns(AstropyUserWarning, match=match):
            create_matching_kernel(self.g1, psf)

    def test_non_callable_window(self):
        match = 'window must be a callable'
        with pytest.raises(TypeError, match=match):
            create_matching_kernel(self.g1, self.g2, window='bad')

    def test_fourier_cutoff(self):
        """
        Test with an aggressive fourier cutoff.
        """
        k = create_matching_kernel(self.g1, self.g2,
                                   fourier_cutoff=0.5)
        assert k.shape == (self.size, self.size)
        assert_allclose(k.sum(), 1.0)
