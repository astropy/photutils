# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the core module.
"""

import sys
from collections import defaultdict
from unittest.mock import PropertyMock, patch

import numpy as np
import pytest
from astropy.utils import lazyproperty
from astropy.utils.exceptions import (AstropyDeprecationWarning,
                                      AstropyUserWarning)
from numpy.testing import assert_allclose, assert_equal

from photutils.segmentation.core import Segment, SegmentationImage
from photutils.utils import circular_footprint
from photutils.utils._optional_deps import (HAS_MATPLOTLIB, HAS_RASTERIO,
                                            HAS_REGIONS, HAS_SHAPELY)


@pytest.fixture
def segm_data():
    """
    Reusable 6x6 segmentation data array.
    """
    return np.array([[1, 1, 0, 0, 4, 4],
                     [0, 0, 0, 0, 0, 4],
                     [0, 0, 3, 3, 0, 0],
                     [7, 0, 0, 0, 0, 5],
                     [7, 7, 0, 5, 5, 5],
                     [7, 7, 0, 0, 5, 5]])


class TestSegmentationImage:
    @pytest.fixture(autouse=True)
    def setup(self, segm_data):
        self.data = segm_data
        self.segm = SegmentationImage(self.data)

    def test_array(self):
        """
        Test array.
        """
        assert_allclose(self.segm.data, self.segm.__array__())

    def test_copy(self):
        """
        Test copy.
        """
        segm = SegmentationImage(self.data.copy())
        segm2 = segm.copy()
        assert segm.data is not segm2.data
        assert segm.labels is not segm2.labels
        segm.data[0, 0] = 100.0
        assert segm.data[0, 0] != segm2.data[0, 0]

    def test_slicing(self):
        """
        Test slicing.
        """
        segm2 = self.segm[1:5, 2:5]
        assert segm2.shape == (4, 3)
        assert_equal(segm2.labels, [3, 5])
        assert segm2.data.sum() == 16

        match = 'is not a valid 2D slice object'
        with pytest.raises(TypeError, match=match):
            self.segm[1]
        with pytest.raises(TypeError, match=match):
            self.segm[1:10]

        match = 'The sliced result is empty'
        with pytest.raises(ValueError, match=match):
            self.segm[1:1, 2:4]
        with pytest.raises(ValueError, match=match):
            self.segm[5:2, 0:3]

    def test_labels_via_raw_slices(self):
        """
        Test that labels can be derived from _raw_slices when that
        lazyproperty is already cached.
        """
        segm = SegmentationImage(self.data.copy())
        # Force _raw_slices to be cached
        _ = segm._raw_slices
        # Remove labels from instance dict to force the _raw_slices path
        del segm.__dict__['labels']
        labels = segm.labels
        assert_equal(labels, [1, 3, 4, 5, 7])

    def test_data_all_zeros(self):
        """
        Test data all zeros.
        """
        data = np.zeros((5, 5), dtype=int)
        segm = SegmentationImage(data)
        assert segm.max_label == 0
        assert not segm.is_consecutive
        assert segm.cmap is None
        match = 'Cannot relabel a segmentation image with no non-zero labels'
        with pytest.warns(AstropyUserWarning, match=match):
            segm.relabel_consecutive()

    def test_data_reassignment(self):
        """
        Test data reassignment.
        """
        segm = SegmentationImage(self.data.copy())
        segm.data = self.data[0:3, :].copy()
        assert_equal(segm.labels, [1, 3, 4])

    def test_invalid_data(self):
        """
        Test invalid data.
        """
        # Is float dtype
        data = np.zeros((3, 3), dtype=float)
        match = 'data must have integer type'
        with pytest.raises(TypeError, match=match):
            SegmentationImage(data)

        # Contains a negative value
        data = np.arange(-1, 8).reshape(3, 3).astype(int)
        match = 'The segmentation image cannot contain negative integers'
        with pytest.raises(ValueError, match=match):
            SegmentationImage(data)

        # Is not ndarray
        data = [[1, 1], [0, 1]]
        match = 'Input data must be a numpy array'
        with pytest.raises(TypeError, match=match):
            SegmentationImage(data)

    @pytest.mark.parametrize('label', [0, -1, 2])
    def test_invalid_label(self, label):
        """
        Test invalid label.
        """
        match = 'is invalid'
        with pytest.raises(ValueError, match=match):
            self.segm.check_label(label)
        with pytest.raises(ValueError, match=match):
            self.segm.check_labels(label)

    def test_invalid_label_array(self):
        """
        Test invalid label array.
        """
        match = 'are invalid'
        with pytest.raises(ValueError, match=match):
            self.segm.check_labels([0, -1, 2])

    def test_data_masked(self):
        """
        Test data_masked.
        """
        assert isinstance(self.segm.data_masked, np.ma.MaskedArray)
        assert np.ma.count(self.segm.data_masked) == 18
        assert np.ma.count_masked(self.segm.data_masked) == 18

    def test_segments(self):
        """
        Test segments.
        """
        assert isinstance(self.segm.segments[0], Segment)
        assert_allclose(self.segm.segments[0].data,
                        self.segm.segments[0].__array__())

        assert (self.segm.segments[0].data_masked.shape
                == self.segm.segments[0].data.shape)
        assert (self.segm.segments[0].data_masked.filled(0.0).sum()
                == self.segm.segments[0].data.sum())

        label = 4
        idx = self.segm.get_index(label)
        assert self.segm.segments[idx].label == label
        assert self.segm.segments[idx].area == self.segm.areas[idx]
        assert self.segm.segments[idx].slices == self.segm.slices[idx]
        assert self.segm.segments[idx].bbox == self.segm.bbox[idx]

    def test_repr_str(self):
        """
        Test repr str.
        """
        assert repr(self.segm) == str(self.segm)

        props = ['shape', 'n_labels']
        for prop in props:
            assert f'{prop}:' in repr(self.segm)

    def test_segment_repr_str(self):
        """
        Test segment repr str.
        """
        props = ['label', 'slices', 'area']
        for prop in props:
            assert f'{prop}:' in repr(self.segm.segments[0])

    @pytest.mark.skipif(not HAS_RASTERIO, reason='rasterio is required')
    @pytest.mark.skipif(not HAS_SHAPELY, reason='shapely is required')
    def test_segment_repr_svg_with_polygon(self):
        """
        Test _repr_svg_ returns SVG string when polygon is present.
        """
        segment = self.segm.segments[0]
        assert segment.polygon is not None
        svg = segment._repr_svg_()
        assert svg is not None
        assert isinstance(svg, str)

    def test_segment_repr_svg_without_polygon(self):
        """
        Test _repr_svg_ returns None when polygon is None.
        """
        segment = self.segm.segments[0]
        # Create a Segment without a polygon
        seg_no_poly = Segment(self.segm.data, segment.label,
                              segment.slices, segment.bbox,
                              segment.area, polygon=None)
        assert seg_no_poly._repr_svg_() is None

    def test_segment_array(self):
        """
        Test that Segment.__array__ returns the correct labeled cutout.
        """
        segment = self.segm.segments[0]  # label=1
        arr = segment.__array__()
        assert arr.shape == segment.data.shape
        assert_allclose(arr, segment.data)
        # Only the label and 0 should appear
        assert set(np.unique(arr)) <= {0, segment.label}

    def test_segment_data(self):
        """
        Test segment data.
        """
        assert_allclose(self.segm.segments[3].data.shape, (3, 3))
        assert_allclose(np.unique(self.segm.segments[3].data), [0, 5])

    def test_segment_make_cutout(self):
        """
        Test segment make cutout.
        """
        cutout = self.segm.segments[3].make_cutout(self.data,
                                                   masked_array=False)
        assert not np.ma.is_masked(cutout)
        assert_allclose(cutout.shape, (3, 3))

        cutout = self.segm.segments[3].make_cutout(self.data,
                                                   masked_array=True)
        assert np.ma.is_masked(cutout)
        assert_allclose(cutout.shape, (3, 3))

    def test_segment_make_cutout_input(self):
        """
        Test segment make cutout input.
        """
        match = 'data must have the same shape as the segmentation array'
        with pytest.raises(ValueError, match=match):
            self.segm.segments[0].make_cutout(np.arange(10))

    def test_segment_no_full_array_reference(self):
        """
        Test that Segment stores only a cutout copy, not a reference
        to the full segmentation array.
        """
        large = np.zeros((1000, 1000), dtype=np.int32)
        large[10:20, 10:20] = 1
        segm = SegmentationImage(large)
        segment = segm.segments[0]

        # The cutout should be much smaller than the full array
        assert segment._segment_data_cutout.shape == (10, 10)
        assert segment._segment_data_shape == (1000, 1000)

        # Delete the SegmentationImage; the segment should not keep
        # the full array alive
        full_refcount = sys.getrefcount(large)
        del segm
        assert sys.getrefcount(large) < full_refcount

    def test_shape(self):
        """
        Test that the shape lazyproperty returns the correct shape.
        """
        assert self.segm.shape == (6, 6)

    def test_lazyproperties_class_cache(self):
        """
        Test that _lazyproperties is cached on the class and shared
        across instances.
        """
        segm2 = SegmentationImage(self.data.copy())
        result1 = self.segm._lazyproperties
        result2 = segm2._lazyproperties
        assert result1 is result2

    def test_labels(self):
        """
        Test labels.
        """
        assert_allclose(self.segm.labels, [1, 3, 4, 5, 7])

    def test_n_labels(self):
        """
        Test n_labels.
        """
        assert self.segm.n_labels == 5

    def test_max_label(self):
        """
        Test max label.
        """
        assert self.segm.max_label == 7

    def test_get_index_invalid(self):
        """
        Test get_index with an invalid label.
        """
        match = 'is invalid'
        with pytest.raises(ValueError, match=match):
            self.segm.get_index(999)
        with pytest.raises(ValueError, match=match):
            self.segm.get_index(0)

    def test_get_indices_invalid(self):
        """
        Test get_indices with invalid labels.
        """
        match = 'is invalid'
        with pytest.raises(ValueError, match=match):
            self.segm.get_indices([1, 999])

        match = 'are invalid'
        with pytest.raises(ValueError, match=match):
            self.segm.get_indices([999, 888])

    def test_areas(self):
        """
        Test areas.
        """
        expected = np.array([2, 2, 3, 6, 5])
        assert_allclose(self.segm.areas, expected)

        assert (self.segm.get_area(1)
                == self.segm.areas[self.segm.get_index(1)])
        assert_allclose(self.segm.get_areas(self.segm.labels),
                        self.segm.areas)

    def test_background_area(self):
        """
        Test background area.
        """
        assert self.segm.background_area == 18

    def test_is_consecutive(self):
        """
        Test is consecutive.
        """
        assert not self.segm.is_consecutive

        data = np.array([[2, 2, 0], [0, 3, 3], [0, 0, 4]], dtype=np.int32)
        segm = SegmentationImage(data)
        dtype = segm.data.dtype
        assert not segm.is_consecutive  # does not start with label=1

        segm.relabel_consecutive(start_label=1)
        assert segm.is_consecutive
        assert segm.data.dtype == dtype

    def test_missing_labels(self):
        """
        Test missing labels.
        """
        assert_allclose(self.segm.missing_labels, [2, 6])

    def test_missing_labels_dtype(self):
        """
        Test that missing_labels dtype matches the segmentation image
        label dtype.
        """
        assert self.segm.missing_labels.dtype == self.segm.labels.dtype

    def test_missing_labels_empty(self):
        """
        Test that missing_labels dtype matches the segmentation image
        label dtype when there are no labels.
        """
        segm = SegmentationImage(np.zeros((5, 5), dtype=np.int32))
        assert_equal(segm.missing_labels, [])
        assert segm.missing_labels.dtype == segm.data.dtype

        segm = SegmentationImage(np.zeros((5, 5), dtype=int))
        assert_equal(segm.missing_labels, [])
        assert segm.missing_labels.dtype == segm.data.dtype

    def test_check_labels(self):
        """
        Test check labels.
        """
        match = 'is invalid'
        with pytest.raises(ValueError, match=match):
            self.segm.check_label(2)
        with pytest.raises(ValueError, match=match):
            self.segm.check_labels([2])

        match = 'are invalid'
        with pytest.raises(ValueError, match=match):
            self.segm.check_labels([2, 6])

    @pytest.mark.parametrize(('label', 'expected'), [
        (1, (0, 1, 0, 2)),
        (3, (2, 3, 2, 4)),
        (4, (0, 2, 4, 6)),
        (5, (3, 6, 3, 6)),
        (7, (3, 6, 0, 2)),
    ])
    def test_bbox_values(self, label, expected):
        """
        Test that bbox returns correct bounding box coordinates for
        each label.
        """
        from photutils.aperture import BoundingBox

        idx = self.segm.get_index(label)
        bbox = self.segm.bbox[idx]
        assert isinstance(bbox, BoundingBox)
        assert (bbox.iymin, bbox.iymax, bbox.ixmin, bbox.ixmax) == expected

    def test_bbox_1d(self):
        """
        Test bbox 1d.
        """
        segm = SegmentationImage(np.array([0, 0, 1, 1, 0, 2, 2, 0]))
        match = "The 'bbox' attribute requires a 2D segmentation image"
        with pytest.raises(ValueError, match=match):
            _ = segm.bbox

    @pytest.mark.skipif(not HAS_MATPLOTLIB, reason='matplotlib is required')
    def test_reset_cmap(self):
        """
        Test reset cmap.
        """
        segm = self.segm.copy()
        cmap = segm.cmap.copy()
        segm.reset_cmap(seed=123)
        assert not np.array_equal(cmap.colors, segm.cmap.colors)

    @pytest.mark.skipif(not HAS_MATPLOTLIB, reason='matplotlib is required')
    def test_make_cmap(self):
        """
        Test make cmap.
        """
        cmap = self.segm.make_cmap()
        assert len(cmap.colors) == (self.segm.max_label + 1)
        assert_allclose(cmap.colors[0], [0, 0, 0, 1])

        assert_allclose(self.segm.cmap.colors,
                        self.segm.make_cmap(background_color='#000000ff',
                                            seed=0).colors)

    @pytest.mark.skipif(not HAS_MATPLOTLIB, reason='matplotlib is required')
    @pytest.mark.parametrize(('color', 'alpha'), [('#00000000', 0.0),
                                                  ('#00000040', 64 / 255),
                                                  ('#00000080', 128 / 255),
                                                  ('#000000C0', 192 / 255),
                                                  ('#000000FF', 1.0)])
    def test_make_cmap_alpha(self, color, alpha):
        """
        Test make cmap alpha.
        """
        cmap = self.segm.make_cmap(background_color=color)
        assert_allclose(cmap.colors[0], (0, 0, 0, alpha))

    def test_reassign_labels(self):
        """
        Test reassign labels.
        """
        segm = SegmentationImage(self.data.copy())
        segm.reassign_labels(labels=[1, 7], new_label=2)
        ref_data = np.array([[2, 2, 0, 0, 4, 4],
                             [0, 0, 0, 0, 0, 4],
                             [0, 0, 3, 3, 0, 0],
                             [2, 0, 0, 0, 0, 5],
                             [2, 2, 0, 5, 5, 5],
                             [2, 2, 0, 0, 5, 5]])
        assert_allclose(segm.data, ref_data)
        assert segm.n_labels == len(segm.slices) - segm.slices.count(None)

    @pytest.mark.parametrize('start_label', [1, 5])
    def test_relabel_consecutive(self, start_label):
        """
        Test relabel consecutive.
        """
        segm = SegmentationImage(self.data.copy())
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
        assert segm.n_labels == len(segm.slices) - segm.slices.count(None)

        # Test slices caching
        segm = SegmentationImage(self.data.copy())
        slc1 = segm.slices
        segm.relabel_consecutive()
        assert slc1 == segm.slices

    @pytest.mark.parametrize('start_label', [0, -1])
    def test_relabel_consecutive_start_invalid(self, start_label):
        """
        Test relabel consecutive start invalid.
        """
        segm = SegmentationImage(self.data.copy())
        match = 'start_label must be > 0'
        with pytest.raises(ValueError, match=match):
            segm.relabel_consecutive(start_label=start_label)

    def test_keep_labels(self):
        """
        Test keep labels.
        """
        ref_data = np.array([[0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0],
                             [0, 0, 3, 3, 0, 0],
                             [0, 0, 0, 0, 0, 5],
                             [0, 0, 0, 5, 5, 5],
                             [0, 0, 0, 0, 5, 5]])
        segm = SegmentationImage(self.data.copy())
        segm.keep_labels([5, 3])
        assert_allclose(segm.data, ref_data)

    def test_keep_labels_relabel(self):
        """
        Test keep labels relabel.
        """
        ref_data = np.array([[0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0],
                             [0, 0, 1, 1, 0, 0],
                             [0, 0, 0, 0, 0, 2],
                             [0, 0, 0, 2, 2, 2],
                             [0, 0, 0, 0, 2, 2]])
        segm = SegmentationImage(self.data.copy())
        segm.keep_labels([5, 3], relabel=True)
        assert_allclose(segm.data, ref_data)

    def test_remove_labels(self):
        """
        Test remove labels.
        """
        ref_data = np.array([[1, 1, 0, 0, 4, 4],
                             [0, 0, 0, 0, 0, 4],
                             [0, 0, 0, 0, 0, 0],
                             [7, 0, 0, 0, 0, 0],
                             [7, 7, 0, 0, 0, 0],
                             [7, 7, 0, 0, 0, 0]])
        segm = SegmentationImage(self.data.copy())
        segm.remove_labels(labels=[5, 3])
        assert_allclose(segm.data, ref_data)

        dtype = np.int32
        data2 = ref_data.copy().astype(dtype)
        segm2 = SegmentationImage(data2)
        segm2.remove_label(1)
        assert segm2.data.dtype == dtype

    def test_remove_labels_relabel(self):
        """
        Test remove labels relabel.
        """
        ref_data = np.array([[1, 1, 0, 0, 2, 2],
                             [0, 0, 0, 0, 0, 2],
                             [0, 0, 0, 0, 0, 0],
                             [3, 0, 0, 0, 0, 0],
                             [3, 3, 0, 0, 0, 0],
                             [3, 3, 0, 0, 0, 0]])
        segm = SegmentationImage(self.data.copy())
        segm.remove_labels(labels=[5, 3], relabel=True)
        assert_allclose(segm.data, ref_data)

    def test_remove_border_labels(self):
        """
        Test remove border labels.
        """
        ref_data = np.array([[0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0],
                             [0, 0, 3, 3, 0, 0],
                             [0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0]])
        segm = SegmentationImage(self.data.copy())
        segm.remove_border_labels(border_width=1)
        assert_allclose(segm.data, ref_data)

    def test_remove_border_labels_border_width(self):
        """
        Test remove border labels border width.
        """
        segm = SegmentationImage(self.data.copy())
        match = 'border_width must be smaller than half the array size'
        with pytest.raises(ValueError, match=match):
            segm.remove_border_labels(border_width=3)

    def test_remove_border_labels_no_remaining_segments(self):
        """
        Test remove border labels no remaining segments.
        """
        alt_data = self.data.copy()
        alt_data[alt_data == 3] = 0
        segm = SegmentationImage(alt_data)
        segm.remove_border_labels(border_width=1, relabel=True)
        assert segm.n_labels == 0

    def test_remove_masked_labels(self):
        """
        Test remove masked labels.
        """
        ref_data = np.array([[0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0],
                             [0, 0, 3, 3, 0, 0],
                             [7, 0, 0, 0, 0, 5],
                             [7, 7, 0, 5, 5, 5],
                             [7, 7, 0, 0, 5, 5]])
        segm = SegmentationImage(self.data.copy())
        mask = np.zeros(segm.data.shape, dtype=bool)
        mask[0, :] = True
        segm.remove_masked_labels(mask)
        assert_allclose(segm.data, ref_data)

    def test_remove_masked_labels_without_partial_overlap(self):
        """
        Test remove masked labels without partial overlap.
        """
        ref_data = np.array([[0, 0, 0, 0, 4, 4],
                             [0, 0, 0, 0, 0, 4],
                             [0, 0, 3, 3, 0, 0],
                             [7, 0, 0, 0, 0, 5],
                             [7, 7, 0, 5, 5, 5],
                             [7, 7, 0, 0, 5, 5]])
        segm = SegmentationImage(self.data.copy())
        mask = np.zeros(segm.data.shape, dtype=bool)
        mask[0, :] = True
        segm.remove_masked_labels(mask, partial_overlap=False)
        assert_allclose(segm.data, ref_data)

    def test_remove_masked_segments_mask_shape(self):
        """
        Test remove masked segments mask shape.
        """
        segm = SegmentationImage(np.ones((5, 5), dtype=int))
        mask = np.zeros((3, 3), dtype=bool)
        match = 'mask must have the same shape as the segmentation array'
        with pytest.raises(ValueError, match=match):
            segm.remove_masked_labels(mask)

    def test_make_source_mask(self):
        """
        Test make source mask.
        """
        segm_array = np.zeros((7, 7)).astype(int)
        segm_array[3, 3] = 1
        segm = SegmentationImage(segm_array)
        mask = segm.make_source_mask()
        assert_equal(mask, segm_array.astype(bool))

        mask = segm.make_source_mask(size=3)
        expected1 = np.array([[0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 1, 1, 1, 0, 0],
                              [0, 0, 1, 1, 1, 0, 0],
                              [0, 0, 1, 1, 1, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0]])
        assert_equal(mask.astype(int), expected1)

        mask = segm.make_source_mask(footprint=np.ones((3, 3)))
        assert_equal(mask.astype(int), expected1)

        footprint = circular_footprint(radius=3)
        mask = segm.make_source_mask(footprint=footprint)
        expected2 = np.array([[0, 0, 0, 1, 0, 0, 0],
                              [0, 1, 1, 1, 1, 1, 0],
                              [0, 1, 1, 1, 1, 1, 0],
                              [1, 1, 1, 1, 1, 1, 1],
                              [0, 1, 1, 1, 1, 1, 0],
                              [0, 1, 1, 1, 1, 1, 0],
                              [0, 0, 0, 1, 0, 0, 0]])
        assert_equal(mask.astype(int), expected2)

        mask = segm.make_source_mask(footprint=np.ones((3, 3)), size=5)
        assert_equal(mask, expected1)

    @pytest.mark.skipif(not HAS_MATPLOTLIB, reason='matplotlib is required')
    def test_imshow(self):
        """
        Test imshow.
        """
        from matplotlib.image import AxesImage

        axim = self.segm.imshow(figsize=(5, 5))
        assert isinstance(axim, AxesImage)

        axim, _ = self.segm.imshow_map(figsize=(5, 5))
        assert isinstance(axim, AxesImage)

    @pytest.mark.skipif(not HAS_RASTERIO, reason='rasterio is required')
    @pytest.mark.skipif(not HAS_SHAPELY, reason='shapely is required')
    def test_polygons(self):
        """
        Test polygons.
        """
        from shapely import Polygon

        polygons = self.segm.polygons
        assert len(polygons) == self.segm.n_labels
        assert isinstance(polygons[0], Polygon)

        data = np.zeros((5, 5), dtype=int)
        data[2, 2] = 10
        segm = SegmentationImage(data)
        polygons = segm.polygons
        assert len(polygons) == 1
        verts = np.array(polygons[0].exterior.coords)
        expected_verts = np.array([[1.5, 1.5], [1.5, 2.5], [2.5, 2.5],
                                   [2.5, 1.5], [1.5, 1.5]])
        assert_equal(verts, expected_verts)

    @pytest.mark.skipif(not HAS_RASTERIO, reason='rasterio is required')
    @pytest.mark.skipif(not HAS_SHAPELY, reason='shapely is required')
    def test_polygon_hole(self):

        """
        Test polygon hole.
        """
        data = np.zeros((11, 11), dtype=int)
        data[3:8, 3:8] = 10
        data[5, 5] = 0  # hole
        segm = SegmentationImage(data)
        polygons = segm.polygons
        assert len(polygons) == 1
        verts = np.array(polygons[0].exterior.coords)
        expected_verts = np.array([[2.5, 2.5], [2.5, 7.5], [7.5, 7.5],
                                   [7.5, 2.5], [2.5, 2.5]])
        assert_equal(verts, expected_verts)

    @pytest.mark.skipif(not HAS_RASTERIO, reason='rasterio is required')
    @pytest.mark.skipif(not HAS_REGIONS, reason='regions is required')
    @pytest.mark.skipif(not HAS_SHAPELY, reason='shapely is required')
    def test_regions(self):
        """
        Test regions.
        """
        from regions import PolygonPixelRegion, Regions

        regions = self.segm.to_regions()

        assert isinstance(regions, Regions)
        assert isinstance(regions[0], PolygonPixelRegion)
        assert len(regions) == self.segm.n_labels

        segm = self.segm.copy()
        segm.reassign_labels(labels=4, new_label=1)
        regions = segm.to_regions(group=True)
        assert isinstance(regions, list)
        assert isinstance(regions[0], Regions)
        assert isinstance(regions[1], PolygonPixelRegion)

        data = np.zeros((5, 5), dtype=int)
        data[2, 2] = 10
        segm = SegmentationImage(data)
        regions = segm.to_regions()
        assert len(regions) == 1
        verts = regions[0].vertices
        expected_xverts = np.array([1.5, 1.5, 2.5, 2.5])
        expected_yverts = np.array([1.5, 2.5, 2.5, 1.5])
        assert_equal(verts.x, expected_xverts)
        assert_equal(verts.y, expected_yverts)

    @pytest.mark.skipif(not HAS_RASTERIO, reason='rasterio is required')
    @pytest.mark.skipif(not HAS_SHAPELY, reason='shapely is required')
    @pytest.mark.skipif(not HAS_MATPLOTLIB, reason='matplotlib is required')
    def test_patches(self):
        """
        Test patches.
        """
        from matplotlib.patches import PathPatch

        patches = self.segm.to_patches(edgecolor='blue')
        assert isinstance(patches[0], PathPatch)
        assert patches[0].get_edgecolor() == (0, 0, 1, 1)

        scale = 2.0
        patches2 = self.segm.to_patches(scale=scale)
        v1 = patches[0].get_verts()
        v2 = patches2[0].get_verts()
        v3 = scale * (v1 + 0.5) - 0.5
        assert_allclose(v2, v3)

        patches = self.segm.plot_patches(edgecolor='red')
        assert isinstance(patches[0], PathPatch)
        assert patches[0].get_edgecolor() == (1, 0, 0, 1)

        patches = self.segm.plot_patches(labels=1)
        assert len(patches) == 1
        assert isinstance(patches, list)
        assert isinstance(patches[0], PathPatch)

        patches = self.segm.plot_patches(labels=(4, 7))
        assert len(patches) == 2
        assert isinstance(patches, list)
        assert isinstance(patches[0], PathPatch)

    @pytest.mark.skipif(not HAS_RASTERIO, reason='rasterio is required')
    @pytest.mark.skipif(not HAS_REGIONS, reason='regions is required')
    @pytest.mark.skipif(not HAS_SHAPELY, reason='shapely is required')
    @pytest.mark.skipif(not HAS_MATPLOTLIB, reason='matplotlib is required')
    def test_patches_corners(self):
        """
        Test that patches are generated for "invalid" Shapely polygons.

        This occurs when two pixels within a segment intersect only at a
        corner.
        """
        data = np.zeros((10, 10), dtype=np.uint32)
        data[5, 5] = 1
        data[4, 4] = 1
        data[3, 3] = 1
        segm = SegmentationImage(data)
        assert segm.n_labels == 1
        assert len(segm.segments) == 1
        assert len(segm.polygons) == 1
        assert len(segm.to_patches()) == 1
        assert len(segm.to_regions()) == 1

    @pytest.mark.skipif(not HAS_RASTERIO, reason='rasterio is required')
    @pytest.mark.skipif(not HAS_REGIONS, reason='regions is required')
    @pytest.mark.skipif(not HAS_SHAPELY, reason='shapely is required')
    @pytest.mark.skipif(not HAS_MATPLOTLIB, reason='matplotlib is required')
    def test_polygons_complex(self):
        """
        Test polygons, patches, and regions for segments that have holes
        and/or are non-contiguous.
        """
        from matplotlib.patches import PathPatch
        from regions import PolygonPixelRegion, Regions
        from shapely import MultiPolygon, Polygon

        image = np.zeros((150, 150), dtype=np.uint32)

        # Polygon with one hole
        image[10:90, 10:90] = 1
        image[30:70, 30:70] = 0

        # Simple Polygon
        image[15:25, 110:140] = 2
        image[25:55, 110:120] = 2

        # MultiPolygon
        image[100:120, 20:40] = 3
        image[105:120, 45:55] = 3
        image[114:130, 60:80] = 3

        # Single polygon with multiple holes
        image[85:145, 95:145] = 4
        image[105:115, 105:115] = 0
        image[125:135, 125:138] = 0
        image[120:125, 100:120] = 0

        # Simple Polygon
        image[5, 125:145] = 5

        segm = SegmentationImage(image)

        polygons = segm.polygons
        assert len(polygons) == 5
        for polygon in polygons:
            assert isinstance(polygon, (Polygon, MultiPolygon))
        assert isinstance(polygons[2], MultiPolygon)

        segments = segm.segments
        assert len(segments) == 5
        assert isinstance(segments[0], Segment)

        patches = segm.to_patches()
        assert len(patches) == 5
        for patch_ in patches:
            assert isinstance(patch_, PathPatch)

        regions = segm.to_regions()
        assert len(regions) == 7
        assert isinstance(regions, Regions)
        for region in regions:
            assert isinstance(region, PolygonPixelRegion)

        regions = segm.to_regions(group=True)
        assert len(regions) == 5
        assert isinstance(regions, list)
        for region in regions:
            assert isinstance(region, (Regions, PolygonPixelRegion))
        assert isinstance(regions[2], Regions)

        # Combine all segments into a single segment;
        # now have multipolygon objects, some with holes
        segm.reassign_labels(segm.labels, new_label=4)
        polygons = segm.polygons
        assert len(polygons) == 1
        assert isinstance(polygons[0], MultiPolygon)
        segments = segm.segments
        assert len(segments) == 1
        patches = segm.to_patches()
        assert len(patches) == 1
        assert isinstance(patches[0], PathPatch)
        regions = segm.to_regions()
        assert len(regions) == 7
        assert isinstance(regions, Regions)
        assert isinstance(regions[0], PolygonPixelRegion)
        regions = segm.to_regions(group=True)
        assert len(regions) == 1
        assert isinstance(regions, list)
        assert isinstance(regions[0], Regions)
        assert len(regions[0]) == 7

    def test_deblended_labels(self):
        """
        Test deblended labels.
        """
        data = np.array([[1, 1, 0, 0, 4, 4],
                         [0, 0, 0, 0, 0, 4],
                         [0, 0, 7, 8, 0, 0],
                         [6, 0, 0, 0, 0, 5],
                         [6, 6, 0, 5, 5, 5],
                         [6, 6, 0, 0, 5, 5]])
        segm = SegmentationImage(data)

        segm0 = segm.copy()
        assert segm0._deblend_label_map == {}
        assert segm0.deblended_labels.size == 0
        assert segm0.deblended_label_to_parent == {}
        assert segm0.parent_to_deblended_labels == {}

        deblend_map = {2: np.array([5, 6]), 3: np.array([7, 8])}
        segm._deblend_label_map = deblend_map
        assert_equal(segm._deblend_label_map, deblend_map)
        assert_equal(segm.deblended_labels, [5, 6, 7, 8])
        assert segm.deblended_label_to_parent == {5: 2, 6: 2, 7: 3, 8: 3}
        assert segm.parent_to_deblended_labels == deblend_map

        segm2 = segm.copy()
        segm2.relabel_consecutive()
        deblend_map = {2: [3, 4], 3: [5, 6]}
        assert_equal(segm2._deblend_label_map, deblend_map)
        assert_equal(segm2.deblended_labels, [3, 4, 5, 6])
        assert segm2.deblended_label_to_parent == {3: 2, 4: 2, 5: 3, 6: 3}
        assert_equal(segm2.parent_to_deblended_labels, deblend_map)

        segm3 = segm.copy()
        segm3.relabel_consecutive(start_label=10)
        deblend_map = {2: [12, 13], 3: [14, 15]}
        assert_equal(segm3._deblend_label_map, deblend_map)
        assert_equal(segm3.deblended_labels, [12, 13, 14, 15])
        assert segm3.deblended_label_to_parent == {12: 2, 13: 2, 14: 3, 15: 3}
        assert_equal(segm3.parent_to_deblended_labels, deblend_map)

        segm4 = segm.copy()
        segm4.reassign_label(5, 50)
        segm4.reassign_label(7, 70)
        deblend_map = {2: [50, 6], 3: [70, 8]}
        assert_equal(segm4._deblend_label_map, deblend_map)
        assert_equal(segm4.deblended_labels, [6, 8, 50, 70])
        assert segm4.deblended_label_to_parent == {50: 2, 6: 2, 70: 3, 8: 3}
        assert_equal(segm4.parent_to_deblended_labels, deblend_map)

        segm5 = segm.copy()
        segm5.reassign_label(5, 50, relabel=True)
        deblend_map = {2: [6, 3], 3: [4, 5]}
        assert_equal(segm5._deblend_label_map, deblend_map)
        assert_equal(segm5.deblended_labels, [3, 4, 5, 6])
        assert segm5.deblended_label_to_parent == {6: 2, 3: 2, 4: 3, 5: 3}
        assert_equal(segm5.parent_to_deblended_labels, deblend_map)


class CustomSegm(SegmentationImage):
    @lazyproperty
    def value(self):
        return np.median(self.data)


def test_subclass(segm_data):
    """
    Test that cached properties are reset in SegmentationImage
    subclasses.
    """
    segm = CustomSegm(segm_data)
    _ = segm.slices, segm.labels, segm.value, segm.areas

    data2 = np.array([[10, 10, 0, 40],
                      [0, 0, 0, 40],
                      [70, 70, 0, 0],
                      [70, 70, 0, 1]])
    segm.data = data2
    assert len(segm.__dict__) == 3
    assert_equal(segm.areas, [1, 2, 2, 4])


def test_segments_no_rasterio(segm_data, monkeypatch):
    """
    Test that segments property works without rasterio/shapely by
    creating Segment objects without polygon info.
    """
    import photutils.segmentation.core as core_mod

    monkeypatch.setattr(core_mod, 'HAS_RASTERIO', False)

    segm = SegmentationImage(segm_data)
    segments = segm.segments
    assert len(segments) == segm.n_labels
    assert isinstance(segments[0], Segment)
    # Without rasterio, segments should not have polygon attribute set
    assert segments[0].polygon is None


def test_reassign_labels_empty(segm_data):
    """
    Test reassign_labels with an empty labels array returns early.
    """
    segm = SegmentationImage(segm_data.copy())
    original = segm.data.copy()
    segm.reassign_labels(labels=[], new_label=99)
    assert_equal(segm.data, original)


def test_keep_label(segm_data):
    """
    Test that keep_label delegates to keep_labels correctly.
    """
    segm = SegmentationImage(segm_data.copy())
    segm.keep_label(3, relabel=True)
    assert segm.n_labels == 1
    assert_equal(segm.labels, [1])
    # Only label 3 (now relabeled to 1) should remain
    assert segm.data[2, 2] == 1
    assert segm.data[0, 0] == 0


@pytest.mark.skipif(not HAS_RASTERIO, reason='rasterio is required')
@pytest.mark.skipif(not HAS_SHAPELY, reason='shapely is required')
def test_geojson_polygons_int64_dtype():
    """
    Test _geojson_polygons with int64 dtype data that fits in int32.
    """
    data = np.array([[0, 0, 0],
                     [0, 1, 0],
                     [0, 0, 0]], dtype=np.int64)
    segm = SegmentationImage(data)
    polygons = segm._geojson_polygons
    assert 1 in polygons
    assert len(polygons[1]) >= 1


@pytest.mark.skipif(not HAS_RASTERIO, reason='rasterio is required')
@pytest.mark.skipif(not HAS_SHAPELY, reason='shapely is required')
def test_geojson_polygons_int64_out_of_range():
    """
    Test _geojson_polygons raises ValueError when int64 values exceed
    int32 range.
    """
    data = np.array([[0, 0, 0],
                     [0, np.iinfo(np.int32).max + 1, 0],
                     [0, 0, 0]], dtype=np.int64)
    segm = SegmentationImage.__new__(SegmentationImage)
    segm._data = data
    match = 'values outside the safe np.int32 range'
    with pytest.raises(ValueError, match=match):
        _ = segm._geojson_polygons


@pytest.mark.skipif(not HAS_RASTERIO, reason='rasterio is required')
@pytest.mark.skipif(not HAS_SHAPELY, reason='shapely is required')
def test_geojson_polygons_label_mismatch():
    """
    Test _geojson_polygons raises ValueError when polygon labels don't
    match segmentation labels.
    """
    data = np.array([[0, 0, 0],
                     [0, 1, 0],
                     [0, 0, 0]], dtype=np.int32)
    segm = SegmentationImage(data)

    # Mock rasterio.features.shapes to return a wrong label
    def fake_shapes(_data, **_kwargs):
        from shapely import Polygon
        from shapely.geometry import mapping

        poly = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        yield mapping(poly), 999  # wrong label

    with patch('rasterio.features.shapes', fake_shapes):
        match = 'labels do not match'
        with pytest.raises(ValueError, match=match):
            _ = segm._geojson_polygons


@pytest.mark.skipif(not HAS_RASTERIO, reason='rasterio is required')
@pytest.mark.skipif(not HAS_SHAPELY, reason='shapely is required')
def test_polygons_empty_geopolys():
    """
    Test polygons property raises ValueError when _geojson_polygons
    returns an empty polygon list for a label.
    """
    data = np.array([[0, 0, 0],
                     [0, 1, 0],
                     [0, 0, 0]], dtype=np.int32)
    segm = SegmentationImage(data)

    # Mock _geojson_polygons to return a dict with an empty list
    empty_dict = defaultdict(list)
    empty_dict[1] = []  # label 1 has no polygons

    with patch.object(type(segm), '_geojson_polygons',
                      new_callable=PropertyMock, return_value=empty_dict):
        match = 'Could not create a polygon for label'
        with pytest.raises(ValueError, match=match):
            _ = segm.polygons


@pytest.mark.skipif(not HAS_SHAPELY, reason='shapely is required')
@pytest.mark.skipif(not HAS_MATPLOTLIB, reason='matplotlib is required')
def test_convert_shapely_to_pathpatch_empty():
    """
    Test _convert_shapely_to_pathpatch returns None for empty geometry.
    """
    from shapely import Point

    data = np.array([[0, 0, 0],
                     [0, 1, 0],
                     [0, 0, 0]], dtype=np.int32)
    segm = SegmentationImage(data)

    # An empty geometry
    empty_geom = Point()  # empty point
    result = segm._convert_shapely_to_pathpatch(empty_geom)
    assert result is None


@pytest.mark.skipif(not HAS_SHAPELY, reason='shapely is required')
@pytest.mark.skipif(not HAS_MATPLOTLIB, reason='matplotlib is required')
def test_convert_shapely_to_pathpatch_empty_geom_collection():
    """
    Test _convert_shapely_to_pathpatch returns None for a non-Polygon
    geometry type that yields no polygons (empty all_vertices).
    """
    from shapely import GeometryCollection

    data = np.array([[0, 0, 0],
                     [0, 1, 0],
                     [0, 0, 0]], dtype=np.int32)
    segm = SegmentationImage(data)

    # A GeometryCollection that is not empty according to the
    # is_empty attribute but contains no polygon geometries.
    # We use a mock to make is_empty=False but geoms=[]
    geom = GeometryCollection()
    with patch.object(type(geom), 'is_empty',
                      new_callable=PropertyMock, return_value=False):
        result = segm._convert_shapely_to_pathpatch(geom)
    assert result is None


@pytest.mark.skipif(not HAS_MATPLOTLIB, reason='matplotlib is required')
def test_imshow_map_too_many_labels():
    """
    Test imshow_map warns when there are more labels than max_labels.
    """
    # Create segmentation with many labels
    data = np.arange(1, 31, dtype=int).reshape(5, 6)
    segm = SegmentationImage(data)

    match = 'The colorbar was not plotted'
    with pytest.warns(AstropyUserWarning, match=match):
        _im, cbar_info = segm.imshow_map(max_labels=5)
    assert cbar_info is None


@pytest.mark.skipif(not HAS_MATPLOTLIB, reason='matplotlib is required')
def test_imshow_map_cbar_labelsize(segm_data):
    """
    Test imshow_map with cbar_labelsize parameter.
    """
    segm = SegmentationImage(segm_data)
    _im, cbar_info = segm.imshow_map(cbar_labelsize=8)
    assert cbar_info is not None


class TestGetSegment:
    """
    Tests for get_segment and get_segments methods.
    """

    @pytest.fixture(autouse=True)
    def setup(self, segm_data):
        self.data = segm_data
        self.segm = SegmentationImage(self.data)

    def test_get_segment_basic(self):
        """
        Test that get_segment returns a valid Segment for each label.
        """
        for label in self.segm.labels:
            seg = self.segm.get_segment(label)
            assert isinstance(seg, Segment)
            assert seg.label == label

    def test_get_segment_matches_segments(self):
        """
        Test that get_segment returns the same data as indexing into the
        segments property.
        """
        for idx, label in enumerate(self.segm.labels):
            seg_new = self.segm.get_segment(label)
            seg_old = self.segm.segments[idx]
            assert seg_new.label == seg_old.label
            assert seg_new.slices == seg_old.slices
            assert seg_new.area == seg_old.area
            assert seg_new.bbox == seg_old.bbox
            assert_equal(seg_new.data, seg_old.data)

    def test_get_segment_invalid_label(self):
        """
        Test that get_segment raises ValueError for an invalid label.
        """
        match = 'is invalid'
        with pytest.raises(ValueError, match=match):
            self.segm.get_segment(99)
        with pytest.raises(ValueError, match=match):
            self.segm.get_segment(0)

    @pytest.mark.skipif(not HAS_RASTERIO, reason='rasterio is required')
    @pytest.mark.skipif(not HAS_SHAPELY, reason='shapely is required')
    def test_get_segment_polygon_matches(self):
        """
        Test that get_segment produces the same polygon as the
        segments property.
        """
        for idx, label in enumerate(self.segm.labels):
            seg_new = self.segm.get_segment(label)
            seg_old = self.segm.segments[idx]
            assert seg_new.polygon is not None
            assert seg_new.polygon.equals(seg_old.polygon)

    @pytest.mark.skipif(not HAS_RASTERIO, reason='rasterio is required')
    @pytest.mark.skipif(not HAS_SHAPELY, reason='shapely is required')
    def test_get_segment_polygon_multipolygon(self):
        """
        Test that get_segment returns a MultiPolygon for a
        non-contiguous segment.
        """
        from shapely import MultiPolygon

        data = np.zeros((10, 10), dtype=int)
        data[1:3, 1:3] = 1
        data[7:9, 7:9] = 1
        segm = SegmentationImage(data)
        seg = segm.get_segment(1)
        assert isinstance(seg.polygon, MultiPolygon)

    def test_get_segment_no_rasterio(self, monkeypatch):
        """
        Test that get_segment returns polygon=None without
        rasterio/shapely.
        """
        import photutils.segmentation.core as core_mod

        monkeypatch.setattr(core_mod, 'HAS_RASTERIO', False)

        segm = SegmentationImage(self.data.copy())
        seg = segm.get_segment(1)
        assert isinstance(seg, Segment)
        assert seg.polygon is None

    def test_get_segments_basic(self):
        """
        Test that get_segments returns a list of Segments in the
        correct order.
        """
        labels = [7, 3, 1]
        segs = self.segm.get_segments(labels)
        assert len(segs) == 3
        assert [s.label for s in segs] == labels

    def test_get_segments_single_label(self):
        """
        Test that get_segments works with a scalar label.
        """
        segs = self.segm.get_segments(5)
        assert len(segs) == 1
        assert segs[0].label == 5

    def test_get_segments_invalid_label(self):
        """
        Test that get_segments raises ValueError for invalid labels.
        """
        match = 'is invalid'
        with pytest.raises(ValueError, match=match):
            self.segm.get_segments(99)

        match = 'are invalid'
        with pytest.raises(ValueError, match=match):
            self.segm.get_segments([1, 99, 200])

    def test_get_segment_multiple_labels(self):
        """
        Test that get_segment raises TypeError for multiple labels.
        """
        match = 'label must be a scalar value'
        with pytest.raises(TypeError, match=match):
            self.segm.get_segment([1, 3])
        with pytest.raises(TypeError, match=match):
            self.segm.get_segment(np.array([1, 3]))

    @pytest.mark.skipif(not HAS_RASTERIO, reason='rasterio is required')
    @pytest.mark.skipif(not HAS_SHAPELY, reason='shapely is required')
    def test_get_segment_polygon_empty_geopolys(self):
        """
        Test that _make_polygon_for_label returns None when rasterio
        returns no polygons for the target label.
        """
        def fake_shapes(_data, **_kwargs):
            # Return no polygons at all
            return iter([])

        with patch('rasterio.features.shapes', fake_shapes):
            seg = self.segm.get_segment(1)
        assert seg.polygon is None

    def test_get_segments_matches_segments(self):
        """
        Test that get_segments results match the segments property.
        """
        labels = list(self.segm.labels)
        segs = self.segm.get_segments(labels)
        for seg_new, seg_old in zip(segs, self.segm.segments, strict=True):
            assert seg_new.label == seg_old.label
            assert seg_new.slices == seg_old.slices
            assert seg_new.area == seg_old.area
            assert seg_new.bbox == seg_old.bbox

    def test_get_segments_label_dtype(self):
        """
        Test that segment label dtype matches the segmentation image
        label dtype.
        """
        labels = [1, 5]
        segs = self.segm.get_segments(labels)
        expected_dtype = self.segm.labels.dtype
        for seg in segs:
            assert seg.label.dtype == expected_dtype


@pytest.mark.skipif(not HAS_RASTERIO, reason='rasterio is required')
@pytest.mark.skipif(not HAS_SHAPELY, reason='shapely is required')
class TestGetPolygon:
    """
    Tests for get_polygon and get_polygons methods.
    """

    @pytest.fixture(autouse=True)
    def setup(self, segm_data):
        self.data = segm_data
        self.segm = SegmentationImage(self.data)

    def test_get_polygon_basic(self):
        """
        Test that get_polygon returns a Shapely geometry for each label.
        """
        from shapely import MultiPolygon, Polygon

        for label in self.segm.labels:
            poly = self.segm.get_polygon(label)
            assert isinstance(poly, (Polygon, MultiPolygon))

    def test_get_polygon_matches_polygons(self):
        """
        Test that get_polygon matches the polygons property.
        """
        for idx, label in enumerate(self.segm.labels):
            poly_new = self.segm.get_polygon(label)
            poly_old = self.segm.polygons[idx]
            assert poly_new.equals(poly_old)

    def test_get_polygon_invalid_label(self):
        """
        Test that get_polygon raises ValueError for an invalid label.
        """
        match = 'is invalid'
        with pytest.raises(ValueError, match=match):
            self.segm.get_polygon(99)

    def test_get_polygon_multiple_labels(self):
        """
        Test that get_polygon raises TypeError for non-scalar input.
        """
        match = 'label must be a scalar value'
        with pytest.raises(TypeError, match=match):
            self.segm.get_polygon([1, 3])

    def test_get_polygon_no_rasterio(self, monkeypatch):
        """
        Test that get_polygon returns None without rasterio/shapely.
        """
        import photutils.segmentation.core as core_mod

        monkeypatch.setattr(core_mod, 'HAS_RASTERIO', False)
        segm = SegmentationImage(self.data.copy())
        assert segm.get_polygon(1) is None

    def test_get_polygons_basic(self):
        """
        Test that get_polygons returns a list in the correct order.
        """
        from shapely import MultiPolygon, Polygon

        labels = [7, 3, 1]
        polys = self.segm.get_polygons(labels)
        assert len(polys) == 3
        for poly in polys:
            assert isinstance(poly, (Polygon, MultiPolygon))

    def test_get_polygons_matches_polygons(self):
        """
        Test that get_polygons results match the polygons property.
        """
        labels = list(self.segm.labels)
        polys_new = self.segm.get_polygons(labels)
        polys_old = self.segm.polygons
        for p_new, p_old in zip(polys_new, polys_old, strict=True):
            assert p_new.equals(p_old)

    def test_get_polygons_invalid_label(self):
        """
        Test that get_polygons raises ValueError for invalid labels.
        """
        match = 'is invalid'
        with pytest.raises(ValueError, match=match):
            self.segm.get_polygons(99)

    def test_get_polygon_empty_geopolys(self):
        """
        Test that get_polygon returns None when rasterio yields no
        polygons.
        """
        def fake_shapes(_data, **_kwargs):
            return iter([])

        with patch('rasterio.features.shapes', fake_shapes):
            poly = self.segm.get_polygon(1)
        assert poly is None

    def test_make_polygon_none_slice(self):
        """
        Test that _make_polygon returns None when slc is None.
        """
        assert self.segm._make_polygon(99, None) is None


@pytest.mark.skipif(not HAS_RASTERIO, reason='rasterio is required')
@pytest.mark.skipif(not HAS_SHAPELY, reason='shapely is required')
@pytest.mark.skipif(not HAS_MATPLOTLIB, reason='matplotlib is required')
class TestGetPatch:
    """
    Tests for get_patch and get_patches methods.
    """

    @pytest.fixture(autouse=True)
    def setup(self, segm_data):
        self.data = segm_data
        self.segm = SegmentationImage(self.data)

    def test_get_patch_basic(self):
        """
        Test that get_patch returns a PathPatch for each label.
        """
        from matplotlib.patches import PathPatch

        for label in self.segm.labels:
            p = self.segm.get_patch(label)
            assert isinstance(p, PathPatch)

    def test_get_patch_kwargs(self):
        """
        Test that get_patch passes kwargs to PathPatch.
        """
        p = self.segm.get_patch(1, edgecolor='red', facecolor='blue')
        assert p.get_edgecolor()[0] == pytest.approx(1.0)  # red channel
        assert p.get_facecolor()[2] == pytest.approx(1.0)  # blue channel

    def test_get_patch_invalid_label(self):
        """
        Test that get_patch raises ValueError for an invalid label.
        """
        match = 'is invalid'
        with pytest.raises(ValueError, match=match):
            self.segm.get_patch(99)

    def test_get_patch_multiple_labels(self):
        """
        Test that get_patch raises TypeError for non-scalar input.
        """
        match = 'label must be a scalar value'
        with pytest.raises(TypeError, match=match):
            self.segm.get_patch([1, 3])

    def test_get_patches_basic(self):
        """
        Test that get_patches returns a list in the correct order.
        """
        from matplotlib.patches import PathPatch

        labels = [7, 3, 1]
        patches = self.segm.get_patches(labels)
        assert len(patches) == 3
        for p in patches:
            assert isinstance(p, PathPatch)

    def test_get_patches_matches_to_patches(self):
        """
        Test that get_patches results have the same path vertices as
        to_patches for the same labels.
        """
        labels = list(self.segm.labels)
        patches_new = self.segm.get_patches(labels)
        patches_old = self.segm.to_patches()
        for p_new, p_old in zip(patches_new, patches_old, strict=True):
            assert_equal(p_new.get_path().vertices, p_old.get_path().vertices)

    def test_get_patches_invalid_label(self):
        """
        Test that get_patches raises ValueError for invalid labels.
        """
        match = 'is invalid'
        with pytest.raises(ValueError, match=match):
            self.segm.get_patches(99)


@pytest.mark.skipif(not HAS_RASTERIO, reason='rasterio is required')
@pytest.mark.skipif(not HAS_SHAPELY, reason='shapely is required')
@pytest.mark.skipif(not HAS_REGIONS, reason='regions is required')
class TestGetRegion:
    """
    Tests for get_region and get_regions methods.
    """

    @pytest.fixture(autouse=True)
    def setup(self, segm_data):
        self.data = segm_data
        self.segm = SegmentationImage(self.data)

    def test_get_region_basic(self):
        """
        Test that get_region returns a PolygonPixelRegion for each label.
        """
        from regions import PolygonPixelRegion

        for label in self.segm.labels:
            region = self.segm.get_region(label)
            assert isinstance(region, PolygonPixelRegion)
            assert region.meta['label'] == label

    def test_get_region_matches_to_regions(self):
        """
        Test that get_region matches the to_regions output.
        """
        old_regions = self.segm.to_regions()
        label_to_old = {}
        for r in old_regions:
            lbl = r.meta['label']
            label_to_old.setdefault(lbl, r)

        for label in self.segm.labels:
            r_new = self.segm.get_region(label)
            r_old = label_to_old[label]
            assert_equal(r_new.vertices.x, r_old.vertices.x)
            assert_equal(r_new.vertices.y, r_old.vertices.y)

    def test_get_region_invalid_label(self):
        """
        Test that get_region raises ValueError for an invalid label.
        """
        match = 'is invalid'
        with pytest.raises(ValueError, match=match):
            self.segm.get_region(99)

    def test_get_region_multiple_labels(self):
        """
        Test that get_region raises TypeError for non-scalar input.
        """
        match = 'label must be a scalar value'
        with pytest.raises(TypeError, match=match):
            self.segm.get_region([1, 3])

    def test_get_region_multipolygon(self):
        """
        Test that get_region returns a Regions object for a
        non-contiguous (MultiPolygon) segment.
        """
        from regions import Regions

        data = np.zeros((10, 10), dtype=int)
        data[1:3, 1:3] = 1
        data[7:9, 7:9] = 1
        segm = SegmentationImage(data)
        region = segm.get_region(1)
        assert isinstance(region, Regions)

    def test_get_regions_basic(self):
        """
        Test that get_regions returns a list in the correct order.
        """
        from regions import PolygonPixelRegion

        labels = [7, 3, 1]
        regions = self.segm.get_regions(labels)
        assert len(regions) == 3
        for region, label in zip(regions, labels, strict=True):
            assert isinstance(region, PolygonPixelRegion)
            assert region.meta['label'] == label

    def test_get_regions_invalid_label(self):
        """
        Test that get_regions raises ValueError for invalid labels.
        """
        match = 'is invalid'
        with pytest.raises(ValueError, match=match):
            self.segm.get_regions(99)

    def test_to_regions_visual_kwargs(self):
        """
        Test that to_regions passes visual kwargs to the regions.
        """
        from regions import PolygonPixelRegion

        regions = self.segm.to_regions(edgecolor='red', linewidth=2)
        for region in regions:
            assert isinstance(region, PolygonPixelRegion)
            assert region.visual['edgecolor'] == 'red'
            assert region.visual['linewidth'] == 2

    def test_get_region_visual_kwargs(self):
        """
        Test that get_region passes visual kwargs to the region.
        """
        region = self.segm.get_region(1, edgecolor='blue', linewidth=3)
        assert region.visual['edgecolor'] == 'blue'
        assert region.visual['linewidth'] == 3

    def test_get_regions_visual_kwargs(self):
        """
        Test that get_regions passes visual kwargs to the regions.
        """
        regions = self.segm.get_regions([1, 3], color='green')
        for region in regions:
            assert region.visual['color'] == 'green'

    def test_to_regions_no_visual_kwargs(self):
        """
        Test that to_regions with no kwargs has no visual attributes.
        """
        regions = self.segm.to_regions()
        for region in regions:
            assert not region.visual


def test_segment_deprecations(segm_data):
    segment_map = SegmentationImage(segm_data)
    segments = segment_map.segments
    match = 'attribute was deprecated'
    with pytest.warns(AstropyDeprecationWarning, match=match):
        _ = segments[0].data_ma
