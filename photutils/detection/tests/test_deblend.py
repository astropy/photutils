# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import numpy as np
from numpy.testing import assert_allclose
from astropy.tests.helper import pytest
from astropy.modeling import models
from ..core import detect_sources
from ..deblend import deblend_sources

try:
    import scipy
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    import skimage
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False


@pytest.mark.skipif('not HAS_SCIPY')
@pytest.mark.skipif('not HAS_SKIMAGE')
class TestDeblendSources(object):
    def setup_class(self):
        g1 = models.Gaussian2D(100, 50, 50, 5, 5)
        g2 = models.Gaussian2D(100, 35, 50, 5, 5)
        g3 = models.Gaussian2D(30, 70, 50, 5, 5)
        y, x = np.mgrid[0:100, 0:100]
        self.data = g1(x, y) + g2(x, y)
        self.data3 = self.data + g3(x, y)
        self.threshold = 10
        self.npixels = 5
        self.segm = detect_sources(self.data, self.threshold, self.npixels)
        self.segm3 = detect_sources(self.data3, self.threshold, self.npixels)

    @pytest.mark.parametrize('mode', ['exponential', 'linear'])
    def test_deblend_sources(self, mode):
        result = deblend_sources(self.data, self.segm, self.npixels,
                                 mode=mode)
        assert result.nlabels == 2
        mask1 = (result.data == 1)
        mask2 = (result.data == 2)
        assert_allclose(len(result.data[mask1]), len(result.data[mask2]))
        assert_allclose(np.sum(self.data[mask1]), np.sum(self.data[mask2]))
        assert_allclose(np.nonzero(self.segm), np.nonzero(result))

    @pytest.mark.parametrize('mode', ['exponential', 'linear'])
    def test_deblend_three_sources(self, mode):
        result = deblend_sources(self.data3, self.segm3, self.npixels,
                                 mode=mode)
        assert result.nlabels == 3
        assert_allclose(np.nonzero(self.segm3), np.nonzero(result))

    def test_deblend_sources_segm_array(self):
        result = deblend_sources(self.data, self.segm.data, self.npixels)
        assert result.nlabels == 2

    def test_segment_img_badshape(self):
        segm_wrong = np.zeros((2, 2))
        with pytest.raises(ValueError):
            deblend_sources(self.data, segm_wrong, self.npixels)

    def test_invalid_nlevels(self):
        with pytest.raises(ValueError):
            deblend_sources(self.data, self.segm, self.npixels, nlevels=0)

    def test_invalid_contrast(self):
        with pytest.raises(ValueError):
            deblend_sources(self.data, self.segm, self.npixels, contrast=-1)

    def test_invalid_mode(self):
        with pytest.raises(ValueError):
            deblend_sources(self.data, self.segm, self.npixels,
                            mode='invalid')
