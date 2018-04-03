# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
from numpy.testing import assert_allclose
import pytest

from ..core import SegmentationImage

try:
    import scipy    # noqa
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    import skimage    # noqa
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False


@pytest.mark.skipif('not HAS_SKIMAGE')
@pytest.mark.skipif('not HAS_SCIPY')
class TestSegmentationImage(object):
    def setup_class(self):
        self.data = [[1, 1, 0, 0, 4, 4],
                     [0, 0, 0, 0, 0, 4],
                     [0, 0, 3, 3, 0, 0],
                     [7, 0, 0, 0, 0, 5],
                     [7, 7, 0, 5, 5, 5],
                     [7, 7, 0, 0, 5, 5]]
        self.segm = SegmentationImage(self.data)

    def test_array(self):
        assert_allclose(self.segm.data, self.segm.array)
        assert_allclose(self.segm.data, self.segm.__array__())

    def test_copy(self):
        segm = SegmentationImage(self.data)
        segm2 = segm.copy()
        assert segm.data is not segm2.data
        assert segm.labels is not segm2.labels
        segm.data[0, 0] = 100.
        assert segm.data[0, 0] != segm2.data[0, 0]

    def test_negative_data(self):
        data = np.arange(-1, 8).reshape(3, 3)
        with pytest.raises(ValueError):
            SegmentationImage(data)

    def test_zero_label(self):
        with pytest.raises(ValueError):
            self.segm.check_label(0)

    def test_negative_label(self):
        with pytest.raises(ValueError):
            self.segm.check_label(-1)

    def test_invalid_label(self):
        with pytest.raises(ValueError):
            self.segm.check_label(2)

    def test_data_masked(self):
        assert isinstance(self.segm.data_masked, np.ma.MaskedArray)
        assert np.ma.count(self.segm.data_masked) == 18
        assert np.ma.count_masked(self.segm.data_masked) == 18

    def test_labels(self):
        assert_allclose(self.segm.labels, [1, 3, 4, 5, 7])

    def test_nlabels(self):
        assert self.segm.nlabels == 5

    def test_max_label(self):
        assert self.segm.max_label == 7

    def test_missing_labels(self):
        assert_allclose(self.segm.missing_labels, [2, 6])

    def test_areas(self):
        expected = np.array([2, 0, 2, 3, 6, 0, 5])
        assert_allclose(self.segm.areas, expected)

    def test_area(self):
        expected = np.array([2, 0, 2, 3, 6, 0, 5])
        labels = [3, 1, 4]
        idx = np.array(labels) - 1
        assert_allclose(self.segm.areas[idx], expected[idx])

    def test_cmap(self):
        cmap = self.segm.cmap()
        assert len(cmap.colors) == (self.segm.max_label + 1)
        assert_allclose(cmap.colors[0], [0, 0, 0])

    def test_outline_segments(self):
        segm_array = np.zeros((5, 5)).astype(int)
        segm_array[1:4, 1:4] = 2
        segm = SegmentationImage(segm_array)
        segm_array_ref = np.copy(segm_array)
        segm_array_ref[2, 2] = 0
        assert_allclose(segm.outline_segments(), segm_array_ref)

    def test_outline_segments_masked_background(self):
        segm_array = np.zeros((5, 5)).astype(int)
        segm_array[1:4, 1:4] = 2
        segm = SegmentationImage(segm_array)
        segm_array_ref = np.copy(segm_array)
        segm_array_ref[2, 2] = 0
        segm_outlines = segm.outline_segments(mask_background=True)
        assert isinstance(segm_outlines, np.ma.MaskedArray)
        assert np.ma.count(segm_outlines) == 8
        assert np.ma.count_masked(segm_outlines) == 17

    def test_relabel(self):
        segm = SegmentationImage(self.data)
        segm.relabel(labels=[1, 7], new_label=2)
        ref_data = np.array([[2, 2, 0, 0, 4, 4],
                             [0, 0, 0, 0, 0, 4],
                             [0, 0, 3, 3, 0, 0],
                             [2, 0, 0, 0, 0, 5],
                             [2, 2, 0, 5, 5, 5],
                             [2, 2, 0, 0, 5, 5]])
        assert_allclose(segm.data, ref_data)
        assert segm.nlabels == len(segm.slices) - segm.slices.count(None)

    @pytest.mark.parametrize('start_label', [1, 5])
    def test_relabel_sequential(self, start_label):
        segm = SegmentationImage(self.data)
        ref_data = np.array([[1, 1, 0, 0, 3, 3],
                             [0, 0, 0, 0, 0, 3],
                             [0, 0, 2, 2, 0, 0],
                             [5, 0, 0, 0, 0, 4],
                             [5, 5, 0, 4, 4, 4],
                             [5, 5, 0, 0, 4, 4]])
        ref_data[ref_data != 0] += (start_label - 1)
        segm.relabel_sequential(start_label=start_label)
        assert_allclose(segm.data, ref_data)

        # relabel_sequential should do nothing if already sequential
        segm.relabel_sequential(start_label=start_label)
        assert_allclose(segm.data, ref_data)
        assert segm.nlabels == len(segm.slices) - segm.slices.count(None)

    @pytest.mark.parametrize('start_label', [0, -1])
    def test_relabel_sequential_start_invalid(self, start_label):
        with pytest.raises(ValueError):
            segm = SegmentationImage(self.data)
            segm.relabel_sequential(start_label=start_label)

    def test_keep_labels(self):
        ref_data = np.array([[0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0],
                             [0, 0, 3, 3, 0, 0],
                             [0, 0, 0, 0, 0, 5],
                             [0, 0, 0, 5, 5, 5],
                             [0, 0, 0, 0, 5, 5]])
        segm = SegmentationImage(self.data)
        segm.keep_labels([5, 3])
        assert_allclose(segm.data, ref_data)

    def test_keep_labels_relabel(self):
        ref_data = np.array([[0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0],
                             [0, 0, 1, 1, 0, 0],
                             [0, 0, 0, 0, 0, 2],
                             [0, 0, 0, 2, 2, 2],
                             [0, 0, 0, 0, 2, 2]])
        segm = SegmentationImage(self.data)
        segm.keep_labels([5, 3], relabel=True)
        assert_allclose(segm.data, ref_data)

    def test_remove_labels(self):
        ref_data = np.array([[1, 1, 0, 0, 4, 4],
                             [0, 0, 0, 0, 0, 4],
                             [0, 0, 0, 0, 0, 0],
                             [7, 0, 0, 0, 0, 0],
                             [7, 7, 0, 0, 0, 0],
                             [7, 7, 0, 0, 0, 0]])
        segm = SegmentationImage(self.data)
        segm.remove_labels(labels=[5, 3])
        assert_allclose(segm.data, ref_data)

    def test_remove_labels_relabel(self):
        ref_data = np.array([[1, 1, 0, 0, 2, 2],
                             [0, 0, 0, 0, 0, 2],
                             [0, 0, 0, 0, 0, 0],
                             [3, 0, 0, 0, 0, 0],
                             [3, 3, 0, 0, 0, 0],
                             [3, 3, 0, 0, 0, 0]])
        segm = SegmentationImage(self.data)
        segm.remove_labels(labels=[5, 3], relabel=True)
        assert_allclose(segm.data, ref_data)

    def test_remove_border_labels(self):
        ref_data = np.array([[0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0],
                             [0, 0, 3, 3, 0, 0],
                             [0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0]])
        segm = SegmentationImage(self.data)
        segm.remove_border_labels(border_width=1)
        assert_allclose(segm.data, ref_data)

    def test_remove_border_labels_border_width(self):
        with pytest.raises(ValueError):
            segm = SegmentationImage(self.data)
            segm.remove_border_labels(border_width=3)

    def test_remove_masked_labels(self):
        ref_data = np.array([[0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0],
                             [0, 0, 3, 3, 0, 0],
                             [7, 0, 0, 0, 0, 5],
                             [7, 7, 0, 5, 5, 5],
                             [7, 7, 0, 0, 5, 5]])
        segm = SegmentationImage(self.data)
        mask = np.zeros_like(segm.data, dtype=np.bool)
        mask[0, :] = True
        segm.remove_masked_labels(mask)
        assert_allclose(segm.data, ref_data)

    def test_remove_masked_labels_without_partial_overlap(self):
        ref_data = np.array([[0, 0, 0, 0, 4, 4],
                             [0, 0, 0, 0, 0, 4],
                             [0, 0, 3, 3, 0, 0],
                             [7, 0, 0, 0, 0, 5],
                             [7, 7, 0, 5, 5, 5],
                             [7, 7, 0, 0, 5, 5]])
        segm = SegmentationImage(self.data)
        mask = np.zeros_like(segm.data, dtype=np.bool)
        mask[0, :] = True
        segm.remove_masked_labels(mask, partial_overlap=False)
        assert_allclose(segm.data, ref_data)

    def test_remove_masked_segments_mask_shape(self):
        segm = SegmentationImage(np.ones((5, 5)))
        mask = np.zeros((3, 3), dtype=np.bool)
        with pytest.raises(ValueError):
            segm.remove_masked_labels(mask)
