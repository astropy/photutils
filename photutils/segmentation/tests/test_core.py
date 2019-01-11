# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
from numpy.testing import assert_allclose
import pytest

from ..core import Segment, SegmentationImage

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
class TestSegmentationImage:
    def setup_class(self):
        self.data = [[1, 1, 0, 0, 4, 4],
                     [0, 0, 0, 0, 0, 4],
                     [0, 0, 3, 3, 0, 0],
                     [7, 0, 0, 0, 0, 5],
                     [7, 7, 0, 5, 5, 5],
                     [7, 7, 0, 0, 5, 5]]
        self.segm = SegmentationImage(self.data)

    def test_array(self):
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

    @pytest.mark.parametrize('label', [0, -1, 2])
    def test_invalid_label(self, label):
        # test with scalar labels
        with pytest.raises(ValueError):
            self.segm.check_labels(label)

    def test_invalid_label_array(self):
        # test with array of labels
        with pytest.raises(ValueError):
            self.segm.check_labels([0, -1, 2])

    def test_data_ma(self):
        assert isinstance(self.segm.data_ma, np.ma.MaskedArray)
        assert np.ma.count(self.segm.data_ma) == 18
        assert np.ma.count_masked(self.segm.data_ma) == 18

    def test_segments(self):
        assert isinstance(self.segm[0], Segment)
        assert_allclose(self.segm[0].data, self.segm[0].__array__())
        assert self.segm[4].area == self.segm.areas[4]
        assert self.segm[4].slices == self.segm.slices[4]
        assert self.segm[3].bbox.slices == self.segm[3].slices
        assert self.segm[1] is None
        assert self.segm[5] is None

        for i, segment in enumerate(self.segm):
            if segment is not None:
                assert segment.label == (i + 1)

    def test_segment_repr_str(self):
        assert repr(self.segm[0]) == str(self.segm[0])

        props = ['label', 'slices', 'bbox', 'area']
        for prop in props:
            assert '{}:'.format(prop) in repr(self.segm[0])

    def test_segment_data(self):
        assert_allclose(self.segm[4].data.shape, (3, 3))
        assert_allclose(np.unique(self.segm[4].data), [0, 5])

    def test_segment_make_cutout(self):
        cutout = self.segm[4].make_cutout(self.data, masked_array=False)
        assert not np.ma.is_masked(cutout)
        assert_allclose(cutout.shape, (3, 3))

        cutout = self.segm[4].make_cutout(self.data, masked_array=True)
        assert np.ma.is_masked(cutout)
        assert_allclose(cutout.shape, (3, 3))

    def test_segment_make_cutout_input(self):
        with pytest.raises(ValueError):
            self.segm[0].make_cutout(np.arange(10))

    def test_labels(self):
        assert_allclose(self.segm.labels, [1, 3, 4, 5, 7])

    def test_nlabels(self):
        assert self.segm.nlabels == 5

    def test_max_label(self):
        assert self.segm.max_label == 7

    def test_areas(self):
        expected = np.array([2, 0, 2, 3, 6, 0, 5])
        assert_allclose(self.segm.areas, expected)

    def test_is_consecutive(self):
        assert self.segm.is_consecutive is False

    def test_missing_labels(self):
        assert_allclose(self.segm.missing_labels, [2, 6])

    def test_check_labels(self):
        with pytest.raises(ValueError):
            self.segm.check_labels([2])

        with pytest.raises(ValueError):
            self.segm.check_labels([2, 6])

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
    def test_relabel_consecutive(self, start_label):
        segm = SegmentationImage(self.data)
        ref_data = np.array([[1, 1, 0, 0, 3, 3],
                             [0, 0, 0, 0, 0, 3],
                             [0, 0, 2, 2, 0, 0],
                             [5, 0, 0, 0, 0, 4],
                             [5, 5, 0, 4, 4, 4],
                             [5, 5, 0, 0, 4, 4]])
        ref_data[ref_data != 0] += (start_label - 1)
        segm.relabel_consecutive(start_label=start_label)
        assert_allclose(segm.data, ref_data)

        # relabel_consecutive should do nothing if already consecutive
        segm.relabel_consecutive(start_label=start_label)
        assert_allclose(segm.data, ref_data)
        assert segm.nlabels == len(segm.slices) - segm.slices.count(None)

    @pytest.mark.parametrize('start_label', [0, -1])
    def test_relabel_consecutive_start_invalid(self, start_label):
        with pytest.raises(ValueError):
            segm = SegmentationImage(self.data)
            segm.relabel_consecutive(start_label=start_label)

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
