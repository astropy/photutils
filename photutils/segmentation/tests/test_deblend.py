# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the deblend module.
"""

from astropy.modeling.models import Gaussian2D
from astropy.tests.helper import catch_warnings
from astropy.utils.exceptions import AstropyUserWarning
import numpy as np
from numpy.testing import assert_allclose
import pytest

from ..core import SegmentationImage
from ..deblend import deblend_sources
from ..detect import detect_sources
from ...utils.exceptions import NoDetectionsWarning
from ...utils._optional_deps import HAS_SCIPY, HAS_SKIMAGE  # noqa


@pytest.mark.skipif('not HAS_SCIPY')
@pytest.mark.skipif('not HAS_SKIMAGE')
class TestDeblendSources:
    def setup_class(self):
        g1 = Gaussian2D(100, 50, 50, 5, 5)
        g2 = Gaussian2D(100, 35, 50, 5, 5)
        g3 = Gaussian2D(30, 70, 50, 5, 5)
        y, x = np.mgrid[0:100, 0:100]
        self.x = x
        self.y = y
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
        assert result.nlabels == len(result.slices)
        mask1 = (result.data == 1)
        mask2 = (result.data == 2)
        assert_allclose(len(result.data[mask1]), len(result.data[mask2]))
        assert_allclose(np.sum(self.data[mask1]), np.sum(self.data[mask2]))
        assert_allclose(np.nonzero(self.segm), np.nonzero(result))

    def test_deblend_multiple_sources(self):
        g4 = Gaussian2D(100, 50, 15, 5, 5)
        g5 = Gaussian2D(100, 35, 15, 5, 5)
        g6 = Gaussian2D(100, 50, 85, 5, 5)
        g7 = Gaussian2D(100, 35, 85, 5, 5)
        x = self.x
        y = self.y
        data = self.data + g4(x, y) + g5(x, y) + g6(x, y) + g7(x, y)
        segm = detect_sources(data, self.threshold, self.npixels)
        result = deblend_sources(data, segm, self.npixels)
        assert result.nlabels == 6
        assert result.nlabels == len(result.slices)
        assert result.areas[0] == result.areas[1]
        assert result.areas[0] == result.areas[2]
        assert result.areas[0] == result.areas[3]
        assert result.areas[0] == result.areas[4]
        assert result.areas[0] == result.areas[5]

    def test_deblend_multiple_sources_with_neighbor(self):
        g1 = Gaussian2D(100, 50, 50, 20, 5, theta=45)
        g2 = Gaussian2D(100, 35, 50, 5, 5)
        g3 = Gaussian2D(100, 60, 20, 5, 5)

        x = self.x
        y = self.y
        data = (g1 + g2 + g3)(x, y)
        segm = detect_sources(data, self.threshold, self.npixels)
        result = deblend_sources(data, segm, self.npixels)
        assert result.nlabels == 3

    @pytest.mark.parametrize('contrast, nlabels',
                             ((0.001, 6), (0.017, 5), (0.06, 4), (0.1, 3),
                              (0.15, 2), (0.45, 1)))
    def test_deblend_contrast(self, contrast, nlabels):
        y, x = np.mgrid[0:51, 0:151]
        y0 = 25
        data = (Gaussian2D(9.5, 16, y0, 5, 5)(x, y) +
                Gaussian2D(51, 30, y0, 3, 3)(x, y) +
                Gaussian2D(30, 42, y0, 5, 5)(x, y) +
                Gaussian2D(80, 66, y0, 8, 8)(x, y) +
                Gaussian2D(71, 88, y0, 8, 8)(x, y) +
                Gaussian2D(18, 119, y0, 7, 7)(x, y))

        npixels = 5
        segm = detect_sources(data, 1.0, npixels)
        segm2 = deblend_sources(data, segm, npixels, mode='linear',
                                nlevels=32, contrast=contrast)
        assert segm2.nlabels == nlabels

    def test_deblend_connectivity(self):
        data = np.zeros((51, 51))
        data[15:36, 15:36] = 10.
        data[14, 36] = 1.
        data[13, 37] = 10
        data[14, 14] = 5.
        data[13, 13] = 10.
        data[36, 14] = 10.
        data[37, 13] = 10.
        data[36, 36] = 10.
        data[37, 37] = 10.

        segm = detect_sources(data, 0.1, 1, connectivity=4)
        assert segm.nlabels == 9
        segm2 = deblend_sources(data, segm, 1, mode='linear', connectivity=4)
        assert segm2.nlabels == 9

        segm = detect_sources(data, 0.1, 1, connectivity=8)
        assert segm.nlabels == 1
        segm2 = deblend_sources(data, segm, 1, mode='linear', connectivity=8)
        assert segm2.nlabels == 3

        with pytest.raises(ValueError):
            deblend_sources(data, segm, 1, mode='linear', connectivity=4)

    def test_deblend_label_assignment(self):
        """
        Regression test to ensure newly-deblended labels are unique.
        """

        y, x = np.mgrid[0:201, 0:101]
        y0a = 35
        y1a = 60
        yshift = 100
        y0b = y0a + yshift
        y1b = y1a + yshift
        data = (Gaussian2D(80, 36, y0a, 8, 8)(x, y) +
                Gaussian2D(71, 58, y1a, 8, 8)(x, y) +
                Gaussian2D(30, 36, y1a, 7, 7)(x, y) +
                Gaussian2D(30, 58, y0a, 7, 7)(x, y) +
                Gaussian2D(80, 36, y0b, 8, 8)(x, y) +
                Gaussian2D(71, 58, y1b, 8, 8)(x, y) +
                Gaussian2D(30, 36, y1b, 7, 7)(x, y) +
                Gaussian2D(30, 58, y0b, 7, 7)(x, y))

        npixels = 5
        segm1 = detect_sources(data, 5.0, npixels)
        segm2 = deblend_sources(data, segm1, npixels, mode='linear',
                                nlevels=32, contrast=0.3)
        assert segm2.nlabels == 4

    @pytest.mark.parametrize('mode', ['exponential', 'linear'])
    def test_deblend_sources_norelabel(self, mode):
        result = deblend_sources(self.data, self.segm, self.npixels,
                                 mode=mode, relabel=False)
        assert result.nlabels == 2
        assert len(result.slices) <= result.max_label
        assert len(result.slices) == result.nlabels
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
        segm_wrong = np.ones((2, 2))
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

    def test_invalid_connectivity(self):
        with pytest.raises(ValueError):
            deblend_sources(self.data, self.segm, self.npixels,
                            connectivity='invalid')

    def test_constant_source(self):
        data = self.data.copy()
        data[data.nonzero()] = 1.
        result = deblend_sources(data, self.segm, self.npixels)
        assert_allclose(result, self.segm)

    def test_source_with_negval(self):
        data = self.data.copy()
        data -= 20
        with catch_warnings(AstropyUserWarning) as warning_lines:
            deblend_sources(data, self.segm, self.npixels)
            assert ('contains negative values' in
                    str(warning_lines[0].message))

    def test_source_zero_min(self):
        data = self.data.copy()
        data -= data[self.segm.data > 0].min()
        result1 = deblend_sources(self.data, self.segm, self.npixels)
        result2 = deblend_sources(data, self.segm, self.npixels)
        assert_allclose(result1, result2)

    def test_connectivity(self):
        """Regression test for #341."""
        data = np.zeros((3, 3))
        data[0, 0] = 2
        data[1, 1] = 2
        data[2, 2] = 1
        segm = np.zeros(data.shape, dtype=int)
        segm[data.nonzero()] = 1
        segm = SegmentationImage(segm)
        data = data * 100.
        segm_deblend = deblend_sources(data, segm, npixels=1, connectivity=8)
        assert segm_deblend.nlabels == 1
        with pytest.raises(ValueError):
            deblend_sources(data, segm, npixels=1, connectivity=4)

    def test_data_nan(self):
        """
        Test that deblending occurs even if the data within a segment
        contains one or more NaNs.  Regression test for #658.
        """

        data = self.data.copy()
        data[50, 50] = np.nan
        segm2 = deblend_sources(data, self.segm, 5)
        assert segm2.nlabels == 2

    def test_watershed(self):
        """
        Regression test to ensure watershed input mask is bool array.

        With scikit-image >= 0.13, the mask must be a bool array.  In
        particular, if the mask array contains label 512, the watershed
        algorithm fails.
        """

        segm = self.segm.copy()
        segm.reassign_label(1, 512)
        result = deblend_sources(self.data, segm, self.npixels)
        assert result.nlabels == 2

    def test_nondetection(self):
        """
        Test for case where no sources are detected at one of the
        threshold levels.

        For this case, a `NoDetectionsWarning` should not be raised when
        deblending sources.
        """

        data = np.copy(self.data3)
        data[50, 50] = 1000.
        data[50, 70] = 500.
        self.segm = detect_sources(data, self.threshold, self.npixels)

        with catch_warnings(NoDetectionsWarning) as warning_lines:
            deblend_sources(data, self.segm, self.npixels)
            assert len(warning_lines) == 0

    def test_nonconsecutive_labels(self):
        segm = self.segm.copy()
        segm.reassign_label(1, 1000)
        result = deblend_sources(self.data, segm, self.npixels)
        assert result.nlabels == 2
