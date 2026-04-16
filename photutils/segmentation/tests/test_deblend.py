# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the deblend module.
"""

from unittest.mock import patch

import numpy as np
import pytest
from astropy.modeling.models import Gaussian2D
from astropy.utils.exceptions import AstropyUserWarning
from numpy.testing import assert_allclose, assert_equal

from photutils.segmentation import (SegmentationImage, deblend_sources,
                                    detect_sources)
from photutils.segmentation.deblend import (_DeblendParams,
                                            _SingleSourceDeblender)
from photutils.utils._optional_deps import HAS_SKIMAGE


@pytest.mark.skipif(not HAS_SKIMAGE, reason='skimage is required')
class TestDeblendSources:
    @pytest.fixture(autouse=True)
    def setup(self):
        g1 = Gaussian2D(100, 50, 50, 5, 5)
        g2 = Gaussian2D(100, 35, 50, 5, 5)
        g3 = Gaussian2D(30, 70, 50, 5, 5)
        y, x = np.mgrid[0:100, 0:100]
        self.x = x
        self.y = y
        self.data = g1(x, y) + g2(x, y)
        self.data3 = self.data + g3(x, y)
        self.threshold = 10
        self.n_pixels = 5
        self.segm = detect_sources(self.data, self.threshold, self.n_pixels)
        self.segm3 = detect_sources(self.data3, self.threshold, self.n_pixels)

    @pytest.mark.parametrize('mode', ['exponential', 'linear', 'sinh'])
    def test_deblend_sources(self, mode):
        """
        Test deblend sources.
        """
        result = deblend_sources(self.data, self.segm, self.n_pixels,
                                 mode=mode, progress_bar=False)
        assert result.data.dtype == self.segm.data.dtype

        if mode == 'linear':
            # Test multiprocessing
            result2 = deblend_sources(self.data, self.segm,
                                      self.n_pixels,
                                      mode=mode,
                                      progress_bar=False,
                                      n_processes=2)
            assert_equal(result.data, result2.data)
            assert result2.data.dtype == self.segm.data.dtype

        assert result.n_labels == 2
        assert result.n_labels == len(result.slices)
        mask1 = (result.data == 1)
        mask2 = (result.data == 2)
        assert_allclose(len(result.data[mask1]), len(result.data[mask2]))
        assert_allclose(np.sum(self.data[mask1]), np.sum(self.data[mask2]))
        assert_allclose(np.nonzero(self.segm), np.nonzero(result))
        assert_equal(result.parent_to_deblended_labels, {1: [1, 2]})

    def test_deblend_multiple_sources(self):
        """
        Test deblend multiple sources.
        """
        g4 = Gaussian2D(100, 50, 15, 5, 5)
        g5 = Gaussian2D(100, 35, 15, 5, 5)
        g6 = Gaussian2D(100, 50, 85, 5, 5)
        g7 = Gaussian2D(100, 35, 85, 5, 5)
        x = self.x
        y = self.y
        data = self.data + g4(x, y) + g5(x, y) + g6(x, y) + g7(x, y)
        segm = detect_sources(data, self.threshold, self.n_pixels)
        result = deblend_sources(data, segm, self.n_pixels, progress_bar=False)
        assert result.n_labels == 6
        assert result.n_labels == len(result.slices)
        assert result.areas[0] == result.areas[1]
        assert result.areas[0] == result.areas[2]
        assert result.areas[0] == result.areas[3]
        assert result.areas[0] == result.areas[4]
        assert result.areas[0] == result.areas[5]

    def test_deblend_multiple_sources_with_neighbor(self):
        """
        Test deblend multiple sources with neighbor.
        """
        g1 = Gaussian2D(100, 50, 50, 20, 5, theta=45)
        g2 = Gaussian2D(100, 35, 50, 5, 5)
        g3 = Gaussian2D(100, 60, 20, 5, 5)

        x = self.x
        y = self.y
        data = (g1 + g2 + g3)(x, y)
        segm = detect_sources(data, self.threshold, self.n_pixels)
        result = deblend_sources(data, segm, self.n_pixels, progress_bar=False)
        assert result.n_labels == 3

    def test_deblend_labels(self):
        """
        Test deblend labels.
        """
        g1 = Gaussian2D(100, 50, 50, 20, 5, theta=45)
        g2 = Gaussian2D(100, 35, 50, 5, 5)
        g3 = Gaussian2D(100, 60, 20, 5, 5)
        x = self.x
        y = self.y
        data = (g1 + g2 + g3)(x, y)
        segm = detect_sources(data, self.threshold, self.n_pixels)
        result = deblend_sources(data, segm, self.n_pixels, labels=1,
                                 progress_bar=False)
        assert result.n_labels == 2

    @pytest.mark.parametrize(('contrast', 'nlabels'),
                             [(0.001, 6), (0.017, 5), (0.06, 4), (0.1, 3),
                              (0.15, 2), (0.45, 1)])
    def test_deblend_contrast(self, contrast, nlabels):
        """
        Test deblend contrast.
        """
        y, x = np.mgrid[0:51, 0:151]
        y0 = 25
        data = (Gaussian2D(9.5, 16, y0, 5, 5)(x, y)
                + Gaussian2D(51, 30, y0, 3, 3)(x, y)
                + Gaussian2D(30, 42, y0, 5, 5)(x, y)
                + Gaussian2D(80, 66, y0, 8, 8)(x, y)
                + Gaussian2D(71, 88, y0, 8, 8)(x, y)
                + Gaussian2D(18, 119, y0, 7, 7)(x, y))

        n_pixels = 5
        segm = detect_sources(data, 1.0, n_pixels)
        segm2 = deblend_sources(data, segm, n_pixels, mode='linear',
                                n_levels=32, contrast=contrast,
                                progress_bar=False)
        assert segm2.n_labels == nlabels

    def test_deblend_contrast_levels(self):
        """
        Test deblend contrast levels.

        Regression test for case where contrast=1.0.
        """
        y, x = np.mgrid[0:51, 0:151]
        y0 = 25
        data = (Gaussian2D(9.5, 16, y0, 5, 5)(x, y)
                + Gaussian2D(51, 30, y0, 3, 3)(x, y)
                + Gaussian2D(30, 42, y0, 5, 5)(x, y)
                + Gaussian2D(80, 66, y0, 8, 8)(x, y)
                + Gaussian2D(71, 88, y0, 8, 8)(x, y)
                + Gaussian2D(18, 119, y0, 7, 7)(x, y))

        n_pixels = 5
        segm = detect_sources(data, 1.0, n_pixels)
        for contrast in np.arange(1, 11) / 10.0:
            segm3 = deblend_sources(data, segm, n_pixels, mode='linear',
                                    n_levels=32, contrast=contrast,
                                    progress_bar=False)
            assert segm3.n_labels >= 1

    def test_deblend_connectivity(self):
        """
        Test deblend connectivity.
        """
        data = np.zeros((51, 51))
        data[15:36, 15:36] = 10.0
        data[14, 36] = 1.0
        data[13, 37] = 10
        data[14, 14] = 5.0
        data[13, 13] = 10.0
        data[36, 14] = 10.0
        data[37, 13] = 10.0
        data[36, 36] = 10.0
        data[37, 37] = 10.0

        segm = detect_sources(data, 0.1, 1, connectivity=4)
        assert segm.n_labels == 9
        segm2 = deblend_sources(data, segm, 1, mode='linear', connectivity=4,
                                progress_bar=False)
        assert segm2.n_labels == 9

        segm = detect_sources(data, 0.1, 1, connectivity=8)
        assert segm.n_labels == 1
        segm2 = deblend_sources(data, segm, 1, mode='linear', connectivity=8,
                                progress_bar=False)
        assert segm2.n_labels == 3

        match = 'Deblending failed for source'
        with pytest.raises(ValueError, match=match):
            deblend_sources(data, segm, 1, mode='linear', connectivity=4,
                            progress_bar=False)

    def test_deblend_label_assignment(self):
        """
        Test to ensure newly-deblended labels are unique.
        """
        y, x = np.mgrid[0:201, 0:101]
        y0a = 35
        y1a = 60
        yshift = 100
        y0b = y0a + yshift
        y1b = y1a + yshift
        data = (Gaussian2D(80, 36, y0a, 8, 8)(x, y)
                + Gaussian2D(71, 58, y1a, 8, 8)(x, y)
                + Gaussian2D(30, 36, y1a, 7, 7)(x, y)
                + Gaussian2D(30, 58, y0a, 7, 7)(x, y)
                + Gaussian2D(80, 36, y0b, 8, 8)(x, y)
                + Gaussian2D(71, 58, y1b, 8, 8)(x, y)
                + Gaussian2D(30, 36, y1b, 7, 7)(x, y)
                + Gaussian2D(30, 58, y0b, 7, 7)(x, y))

        n_pixels = 5
        segm1 = detect_sources(data, 5.0, n_pixels)
        segm2 = deblend_sources(data, segm1, n_pixels, mode='linear',
                                n_levels=32, contrast=0.3, progress_bar=False)
        assert segm2.n_labels == 4

    @pytest.mark.parametrize('mode', ['exponential', 'linear'])
    def test_deblend_sources_norelabel(self, mode):
        """
        Test deblend sources norelabel.
        """
        result = deblend_sources(self.data, self.segm, self.n_pixels,
                                 mode=mode, relabel=False, progress_bar=False)
        assert result.n_labels == 2
        assert_equal(result.labels, [2, 3])
        assert_equal(result.parent_to_deblended_labels, {1: [2, 3]})
        assert len(result.slices) <= result.max_label
        assert len(result.slices) == result.n_labels
        assert_allclose(np.nonzero(self.segm), np.nonzero(result))

    @pytest.mark.parametrize('mode', ['exponential', 'linear'])
    def test_deblend_three_sources(self, mode):
        """
        Test deblend three sources.
        """
        result = deblend_sources(self.data3, self.segm3, self.n_pixels,
                                 mode=mode, progress_bar=False)
        assert result.n_labels == 3
        assert_allclose(np.nonzero(self.segm3), np.nonzero(result))

    def test_segmentation_image(self):
        """
        Test segmentation image.
        """
        segm_wrong = np.ones((2, 2), dtype=int)  # ndarray
        match = 'segmentation_image must be a SegmentationImage'
        with pytest.raises(TypeError, match=match):
            deblend_sources(self.data, segm_wrong, self.n_pixels,
                            progress_bar=False)

        segm_wrong = SegmentationImage(segm_wrong)  # wrong shape
        match = 'segmentation_image must have the same shape as data'
        with pytest.raises(ValueError, match=match):
            deblend_sources(self.data, segm_wrong, self.n_pixels,
                            progress_bar=False)

    def test_invalid_n_levels(self):
        """
        Test invalid n_levels.
        """
        match = 'n_levels must be >= 1'
        with pytest.raises(ValueError, match=match):
            deblend_sources(self.data, self.segm, self.n_pixels, n_levels=0,
                            progress_bar=False)

    def test_invalid_contrast(self):
        """
        Test invalid contrast.
        """
        match = 'contrast must be >= 0 and <= 1'
        with pytest.raises(ValueError, match=match):
            deblend_sources(self.data, self.segm, self.n_pixels, contrast=-1,
                            progress_bar=False)

    def test_invalid_mode(self):
        """
        Test invalid mode.
        """
        match = "mode must be 'exponential', 'linear', or 'sinh'"
        with pytest.raises(ValueError, match=match):
            deblend_sources(self.data, self.segm, self.n_pixels,
                            mode='invalid', progress_bar=False)

    def test_invalid_connectivity(self):
        """
        Test invalid connectivity.
        """
        match = 'Invalid connectivity'
        with pytest.raises(ValueError, match=match):
            deblend_sources(self.data, self.segm, self.n_pixels,
                            connectivity='invalid', progress_bar=False)

    def test_constant_source(self):
        """
        Test constant source.
        """
        data = self.data.copy()
        data[data.nonzero()] = 1.0
        result = deblend_sources(data, self.segm, self.n_pixels,
                                 progress_bar=False)
        assert_allclose(result, self.segm)

    def test_source_with_negval(self):
        """
        Test source with negval.
        """
        data = self.data.copy()
        data -= 20
        match = 'The deblending mode of one or more source labels from the'
        with pytest.warns(AstropyUserWarning, match=match):
            segm = deblend_sources(data, self.segm, self.n_pixels,
                                   progress_bar=False)
        assert segm.info['warnings']['nonposmin']['input_labels'] == 1

    def test_source_zero_min(self):
        """
        Test source zero min.
        """
        data = self.data.copy()
        data -= data[self.segm.data > 0].min()
        match = 'The deblending mode of one or more source labels from the'
        with pytest.warns(AstropyUserWarning, match=match):
            segm = deblend_sources(data, self.segm, self.n_pixels,
                                   progress_bar=False)
        assert segm.info['warnings']['nonposmin']['input_labels'] == 1

    def test_connectivity(self):
        """
        Test connectivity.

        Regression test for #341.
        """
        data = np.zeros((3, 3))
        data[0, 0] = 2
        data[1, 1] = 2
        data[2, 2] = 1
        segm = np.zeros(data.shape, dtype=int)
        segm[data.nonzero()] = 1
        segm = SegmentationImage(segm)
        data = data * 100.0
        segm_deblend = deblend_sources(data, segm, n_pixels=1, connectivity=8,
                                       progress_bar=False)
        assert segm_deblend.n_labels == 1
        match = 'Deblending failed for source'
        with pytest.raises(ValueError, match=match):
            deblend_sources(data, segm, n_pixels=1, connectivity=4,
                            progress_bar=False)

    def test_data_nan(self):
        """
        Test that deblending occurs even if the data within a segment
        contains one or more NaNs.

        Regression test for #658.
        """
        data = self.data.copy()
        data[50, 50] = np.nan
        segm2 = deblend_sources(data, self.segm, 5, progress_bar=False)
        assert segm2.n_labels == 2

    def test_watershed(self):
        """
        Test that the watershed input mask is a bool array.

        With scikit-image >= 0.13, the mask must be a bool array. In
        particular, if the mask array contains label 512, the watershed
        algorithm fails.
        """
        segm = self.segm.copy()
        segm.reassign_label(1, 512)
        result = deblend_sources(self.data, segm, self.n_pixels,
                                 progress_bar=False)
        assert result.n_labels == 2

    def test_nondetection(self):
        """
        Test for case where no sources are detected at one of the
        threshold levels.

        For this case, a `NoDetectionsWarning` should not be raised when
        deblending sources.
        """
        data = np.copy(self.data3)
        data[50, 50] = 1000.0
        data[50, 70] = 500.0
        self.segm = detect_sources(data, self.threshold, self.n_pixels)
        deblend_sources(data, self.segm, self.n_pixels, progress_bar=False)

    def test_nonconsecutive_labels(self):
        """
        Test nonconsecutive labels.
        """
        segm = self.segm.copy()
        segm.reassign_label(1, 1000)
        result = deblend_sources(self.data, segm, self.n_pixels,
                                 progress_bar=False)
        assert result.n_labels == 2

    def test_single_source_methods(self):
        """
        Test the multithreshold and make_markers methods of the
        _SingleSourceDeblender class.

        These methods are useful for debugging but are not currently
        used by the deblend_sources function.
        """
        data = self.data3
        segm = self.segm3
        n_pixels = 5
        footprint = np.ones((3, 3))
        deblend_params = _DeblendParams(n_pixels, footprint, 32, 0.001,
                                        'linear')
        single_debl = _SingleSourceDeblender(data, segm.data, 1,
                                             deblend_params)
        segms = single_debl.multithreshold()
        assert len(segms) == 32

        markers = single_debl.make_markers(return_all=True)
        assert len(markers) == 19

    def test_deblend_progress_bar(self):
        """
        Test deblend_sources with progress_bar=True (serial).
        """
        result = deblend_sources(self.data, self.segm, self.n_pixels,
                                 mode='linear', progress_bar=True)
        assert result.n_labels == 2

    def test_deblend_nproc_none(self):
        """
        Test deblend_sources with n_processes=None (auto-detect CPU count).
        """
        result = deblend_sources(self.data, self.segm, self.n_pixels,
                                 mode='linear', progress_bar=False,
                                 n_processes=None)
        assert result.n_labels == 2


@pytest.mark.skipif(not HAS_SKIMAGE, reason='skimage is required')
def test_nmarkers_fallback():
    """
    Test that if there are too many markers, a warning is raised.
    """
    size = 51
    data1 = np.resize([0, 0, 1, 1], size)
    data1 = np.abs(data1 - np.atleast_2d(data1).T) + 2

    for i in range(size):
        if i % 2 == 0:
            data1[i, :] = 1
            data1[:, i] = 1

    data = np.zeros((101, 101))
    data[25:25 + size, 25:25 + size] = data1
    data[50:60, 50:60] = 10.0

    segm = detect_sources(data, 0.01, 10)
    match = 'The deblending mode of one or more source labels from the'
    with pytest.warns(AstropyUserWarning, match=match):
        segm2 = deblend_sources(data, segm, 1, mode='exponential')
    assert segm2.info['warnings']['nmarkers']['input_labels'][0] == 1
    mesg = segm2.info['warnings']['nmarkers']['message']
    assert mesg.startswith('Deblending mode changed')


@pytest.mark.skipif(not HAS_SKIMAGE, reason='skimage is required')
def test_nmarkers_fallback_multiproc():
    """
    Test the nmarkers fallback warning via multiprocessing (n_processes=2).
    This covers the multiprocessing result-processing block for nmarkers.
    """
    size = 51
    data1 = np.resize([0, 0, 1, 1], size)
    data1 = np.abs(data1 - np.atleast_2d(data1).T) + 2

    for i in range(size):
        if i % 2 == 0:
            data1[i, :] = 1
            data1[:, i] = 1

    data = np.zeros((101, 101))
    data[25:25 + size, 25:25 + size] = data1
    data[50:60, 50:60] = 10.0

    segm = detect_sources(data, 0.01, 10)
    match = 'The deblending mode of one or more source labels from the'
    with pytest.warns(AstropyUserWarning, match=match):
        segm2 = deblend_sources(data, segm, 1, mode='exponential',
                                n_processes=2)
    assert segm2.info['warnings']['nmarkers']['input_labels'][0] == 1


@pytest.mark.skipif(not HAS_SKIMAGE, reason='skimage is required')
def test_nonposmin_multiproc():
    """
    Test nonposmin warning via multiprocessing (n_processes=2).
    This covers the multiprocessing result-processing block for
    nonposmin.
    """
    g1 = Gaussian2D(100, 50, 50, 8, 8)
    g2 = Gaussian2D(100, 35, 50, 8, 8)
    yy, xx = np.mgrid[0:101, 0:101]
    data = g1(xx, yy) + g2(xx, yy) - 20  # negative values

    segm = detect_sources(data + 20, 10, 5)  # detect sources on positive data
    match = 'The deblending mode of one or more source labels from the'
    with pytest.warns(AstropyUserWarning, match=match):
        segm2 = deblend_sources(data, segm, 5, progress_bar=False,
                                n_processes=2)
    assert 'nonposmin' in segm2.info['warnings']


@pytest.mark.skipif(not HAS_SKIMAGE, reason='skimage is required')
def test_nmarkers_fallback_returns_none():
    """
    Test that deblend_source returns None when make_markers returns
    None on the linear-mode fallback (second attempt after >200
    markers).
    """
    # Create a source with varying data values so source_min != source_max
    data = np.ones((20, 20)) * 10.0
    data[5:15, 5:15] = 50.0
    data[8:12, 8:12] = 100.0  # peak in center
    segment = np.zeros((20, 20), dtype=int)
    segment[5:15, 5:15] = 1

    deblend_params = _DeblendParams(5, np.ones((3, 3)), 32, 0.001,
                                    'exponential')

    deblender = _SingleSourceDeblender(data, segment, 1, deblend_params)

    call_count = [0]

    def mock_make_markers(*, _return_all=False):
        call_count[0] += 1
        if call_count[0] == 1:
            # First call: return markers with > 200 labels
            markers = np.zeros((20, 20), dtype=int)
            for i in range(201):
                r, c = divmod(i, 20)
                if r < 20 and c < 20:
                    markers[r, c] = i + 1
            return markers
        # Second call (linear fallback): return None
        return None

    with patch.object(deblender, 'make_markers', mock_make_markers):
        result = deblender.deblend_source()

    assert result is None
