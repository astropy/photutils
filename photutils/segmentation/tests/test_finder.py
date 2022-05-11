# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the finder module.
"""

from astropy.convolution import Gaussian2DKernel, convolve
from astropy.stats import gaussian_fwhm_to_sigma
import pytest

from ..finder import SourceFinder
from ...datasets import make_100gaussians_image
from ...utils.exceptions import NoDetectionsWarning
from ...utils._optional_deps import HAS_SCIPY, HAS_SKIMAGE  # noqa


class TestSourceFinder:
    data = make_100gaussians_image() - 5.0  # subtract background
    sigma = 3. * gaussian_fwhm_to_sigma  # FWHM = 3.
    kernel = Gaussian2DKernel(sigma, x_size=5, y_size=5)
    convolved_data = convolve(data, kernel, normalize_kernel=True)
    threshold = 1.5 * 2.0
    npixels = 10

    def test_deblend(self):
        finder = SourceFinder(npixels=self.npixels)
        segm = finder(self.convolved_data, self.threshold)
        assert segm.nlabels == 94

    def test_no_deblend(self):
        finder = SourceFinder(npixels=self.npixels, deblend=False)
        segm = finder(self.convolved_data, self.threshold)
        assert segm.nlabels == 86

    def test_no_sources(self):
        finder = SourceFinder(npixels=self.npixels, deblend=True)
        with pytest.warns(NoDetectionsWarning):
            segm = finder(self.convolved_data, 1000)
            assert segm is None
