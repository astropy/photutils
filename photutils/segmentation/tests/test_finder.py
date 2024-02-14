# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the finder module.
"""

import astropy.units as u
import numpy as np
import pytest
from astropy.convolution import convolve
from astropy.modeling.models import Gaussian2D

from photutils.datasets import make_100gaussians_image
from photutils.segmentation.finder import SourceFinder
from photutils.segmentation.utils import make_2dgaussian_kernel
from photutils.utils._optional_deps import HAS_SCIPY, HAS_SKIMAGE
from photutils.utils.exceptions import NoDetectionsWarning


@pytest.mark.skipif(not HAS_SCIPY, reason='scipy is required')
class TestSourceFinder:
    data = make_100gaussians_image() - 5.0  # subtract background
    kernel = make_2dgaussian_kernel(3.0, size=5)
    convolved_data = convolve(data, kernel, normalize_kernel=True)
    threshold = 1.5 * 2.0
    npixels = 10

    @pytest.mark.skipif(not HAS_SKIMAGE, reason='skimage is required')
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

    def test_npixels_tuple(self):
        g1 = Gaussian2D(10, 35, 45, 5, 5)
        g2 = Gaussian2D(10, 50, 50, 5, 5)
        g3 = Gaussian2D(10, 66, 55, 5, 5)
        yy, xx = np.mgrid[0:101, 0:101]
        data = g1(xx, yy) + g2(xx, yy) + g3(xx, yy)

        sf1 = SourceFinder(npixels=200)
        segm1 = sf1(data, threshold=0.1)
        assert segm1.nlabels == 1

        sf2 = SourceFinder(npixels=(200, 5))
        segm2 = sf2(data, threshold=0.1)
        assert segm2.nlabels == 3

    def test_repr(self):
        finder = SourceFinder(npixels=self.npixels, deblend=False,
                              progress_bar=False)
        cls_repr = repr(finder)
        assert cls_repr.startswith(finder.__class__.__name__)
