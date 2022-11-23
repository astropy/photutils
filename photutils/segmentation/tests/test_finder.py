# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the finder module.
"""

import astropy.units as u
import numpy as np
import pytest
from astropy.convolution import convolve

from ...datasets import make_100gaussians_image
from ...utils._optional_deps import HAS_SCIPY, HAS_SKIMAGE  # noqa
from ...utils.exceptions import NoDetectionsWarning
from ..finder import SourceFinder
from ..utils import make_2dgaussian_kernel


@pytest.mark.skipif('not HAS_SCIPY')
class TestSourceFinder:
    data = make_100gaussians_image() - 5.0  # subtract background
    kernel = make_2dgaussian_kernel(3., size=5)
    convolved_data = convolve(data, kernel, normalize_kernel=True)
    threshold = 1.5 * 2.0
    npixels = 10

    @pytest.mark.skipif('not HAS_SKIMAGE')
    def test_deblend(self):
        finder = SourceFinder(npixels=self.npixels, progress_bar=False)
        segm1 = finder(self.convolved_data, self.threshold)
        assert segm1.nlabels == 94

        segm2 = finder(self.convolved_data << u.uJy, self.threshold * u.uJy)
        assert segm2.nlabels == 94
        assert np.all(segm1.data == segm2.data)

    def test_invalid_units(self):
        finder = SourceFinder(npixels=self.npixels, progress_bar=False)
        with pytest.raises(ValueError):
            finder(self.convolved_data << u.uJy, self.threshold)
        with pytest.raises(ValueError):
            finder(self.convolved_data, self.threshold * u.uJy)
        with pytest.raises(ValueError):
            finder(self.convolved_data << u.uJy, self.threshold * u.m)

    def test_no_deblend(self):
        finder = SourceFinder(npixels=self.npixels, deblend=False,
                              progress_bar=False)
        segm = finder(self.convolved_data, self.threshold)
        assert segm.nlabels == 85

    def test_no_sources(self):
        finder = SourceFinder(npixels=self.npixels, deblend=True,
                              progress_bar=False)
        with pytest.warns(NoDetectionsWarning,
                          match='No sources were found'):
            segm = finder(self.convolved_data, 1000)
            assert segm is None
