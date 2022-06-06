# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the finder module.
"""

from astropy.convolution import convolve
import pytest

from ..finder import SourceFinder
from ..utils import make_2dgaussian_kernel
from ...datasets import make_100gaussians_image
from ...utils.exceptions import NoDetectionsWarning
from ...utils._optional_deps import HAS_SCIPY, HAS_SKIMAGE  # noqa


@pytest.mark.skipif('not HAS_SCIPY')
class TestSourceFinder:
    data = make_100gaussians_image() - 5.0  # subtract background
    kernel = make_2dgaussian_kernel(3., size=5)
    convolved_data = convolve(data, kernel, normalize_kernel=True)
    threshold = 1.5 * 2.0
    npixels = 10

    @pytest.mark.skipif('not HAS_SKIMAGE')
    def test_deblend(self):
        finder = SourceFinder(npixels=self.npixels)
        segm = finder(self.convolved_data, self.threshold)
        assert segm.nlabels == 94

    def test_no_deblend(self):
        finder = SourceFinder(npixels=self.npixels, deblend=False)
        segm = finder(self.convolved_data, self.threshold)
        assert segm.nlabels == 85

    def test_no_sources(self):
        finder = SourceFinder(npixels=self.npixels, deblend=True)
        with pytest.warns(NoDetectionsWarning):
            segm = finder(self.convolved_data, 1000)
            assert segm is None
