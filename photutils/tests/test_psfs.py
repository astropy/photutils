# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import division
import numpy as np
from numpy.testing import assert_allclose
from astropy.tests.helper import pytest
from ..psf import GaussianPSF
try:
    from scipy import optimize
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


widths = [0.001, 0.01, 0.1, 1]
sigmas = [0.5, 1., 2.]


@pytest.mark.skipif('not HAS_SCIPY')
@pytest.mark.parametrize(('width'), widths)
def test_subpixel_gauss_psf(width):
    """
    Test subpixel accuracy of Gaussian PSF by checking the sum of pixels.
    """
    gauss_psf = GaussianPSF(width)
    y, x = np.mgrid[-10:11, -10:11]
    assert_allclose(gauss_psf(x, y).sum(), 1)


@pytest.mark.skipif('not HAS_SCIPY')
@pytest.mark.parametrize(('sigma'), sigmas)
def test_gaussian_PSF_integral():
    """
    Test if Gaussian PSF integrates to unity on larger scales.
    """
    psf = GaussianPSF(10, sigma=sigma)
    y, x = np.mgrid[-100:101, -100:101]
    assert_allclose(psf(y, x).sum(), 1)
