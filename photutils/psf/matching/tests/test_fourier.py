# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the fourier module.
"""

import numpy as np
import pytest
from astropy.modeling.fitting import TRFLSQFitter
from astropy.modeling.models import Gaussian2D
from numpy.testing import assert_allclose

from photutils.psf.matching.fourier import create_matching_kernel, resize_psf
from photutils.psf.matching.windows import SplitCosineBellWindow


def test_resize_psf():
    psf1 = np.ones((5, 5))
    psf2 = resize_psf(psf1, 0.1, 0.05)
    assert psf2.shape == (10, 10)


def test_create_matching_kernel():
    """
    Test with noiseless 2D Gaussians.
    """
    size = 25
    cen = (size - 1) / 2.0
    y, x = np.mgrid[0:size, 0:size]
    std1 = 3.0
    std2 = 5.0
    gm1 = Gaussian2D(1.0, cen, cen, std1, std1)
    gm2 = Gaussian2D(1.0, cen, cen, std2, std2)
    g1 = gm1(x, y)
    g2 = gm2(x, y)
    g1 /= g1.sum()
    g2 /= g2.sum()

    window = SplitCosineBellWindow(0.0, 0.2)
    k = create_matching_kernel(g1, g2, window=window)

    fitter = TRFLSQFitter()
    gfit = fitter(gm1, x, y, k)
    assert_allclose(gfit.x_stddev, gfit.y_stddev)
    assert_allclose(gfit.x_stddev, np.sqrt(std2**2 - std1**2), 0.06)


def test_create_matching_kernel_shapes():
    """
    Test with wrong PSF shapes.
    """
    psf1 = np.ones((5, 5))
    psf2 = np.ones((3, 3))
    match = 'source_psf and target_psf must have the same shape'
    with pytest.raises(ValueError, match=match):
        create_matching_kernel(psf1, psf2)
