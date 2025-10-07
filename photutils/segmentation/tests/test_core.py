# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the core module.
"""

import numpy as np
import pytest
from astropy.utils import lazyproperty
from astropy.utils.exceptions import AstropyUserWarning
from numpy.testing import assert_allclose, assert_equal

from photutils.segmentation.core import Segment, SegmentationImage
from photutils.utils import circular_footprint
from photutils.utils._optional_deps import (HAS_MATPLOTLIB, HAS_RASTERIO,
                                            HAS_REGIONS, HAS_SHAPELY)


class TestSegmentationImage:
    def setup_class(self):
        self.data = np.array([[1, 1, 0, 0, 4, 4],
                              [0, 0, 0, 0, 0, 4],
                              [0, 0, 3, 3, 0, 0],
                              [7, 0, 0, 0, 0, 5],
                              [7, 7, 0, 5, 5, 5],
                              [7, 7, 0, 0, 5, 5]])
        self.segm = SegmentationImage(self.data)

    def test_array(self):
        assert_allclose(self.segm.data, self.segm.__array__())

    def test_copy(self):
        segm = SegmentationImage(self.data.copy())
        segm2 = segm.copy()
        assert segm.data is not segm2.data
        assert segm.labels is not segm2.labels
        segm.data[0, 0] = 100.0
        assert segm.data[0, 0] != segm2.data[0, 0]

    def test_slicing(self):
        segm2 = self.segm[1:5, 2:5]
        assert segm2.shape == (4, 3)
        assert_equal(segm2.labels, [3, 5])
        assert segm2.data.sum() == 16

        match = 'is not a valid 2D slice object'
        with pytest.raises(TypeError, match=match):
            self.segm[1]
        with pytest.raises(TypeError, match=match):
            self.segm[1:10]
        with pytest.raises(TypeError, match=match):
            self.segm[1:1, 2:4]

    def test_data_all_zeros(self):
        data = np.zeros((5, 5), dtype=int)
        segm = SegmentationImage(data)
        assert segm.max_label == 0
        assert not segm.is_consecutive
        assert segm.cmap is None
        match = 'segmentation image of all zeros'
        with pytest.warns(AstropyUserWarning, match=match):
            segm.relabel_consecutive()

    def test_data_reassignment(self):
        segm = SegmentationImage(self.data.copy())
        segm.data = self.data[0:3, :].copy()
        assert_equal(segm.labels, [1, 3, 4])

    def test_invalid_data(self):
        # is float dtype
        data = np.zeros((3, 3), dtype=float)
        match = 'data must be have integer type'
        with pytest.raises(TypeError, match=match):
            SegmentationImage(data)

        # contains a negative value
        data = np.arange(-1, 8).reshape(3, 3).astype(int)
        match = 'The segmentation image cannot contain negative integers'
        with pytest.raises(ValueError, match=match):
            SegmentationImage(data)

        # is not ndarray
        data = [[1, 1], [0, 1]]
        match = 'Input data must be a numpy array'
        with pytest.raises(TypeError, match=match):
            SegmentationImage(data)

    @pytest.mark.parametrize('label', [0, -1, 2])
    def test_invalid_label(self, label):
        # test with scalar labels
        match = 'is invalid'
        with pytest.raises(ValueError, match=match):
            self.segm.check_label(label)
        with pytest.raises(ValueError, match=match):
            self.segm.check_labels(label)

    def test_invalid_label_array(self):
        # test with array of labels
        match = 'are invalid'
        with pytest.raises(ValueError, match=match):
            self.segm.check_labels([0, -1, 2])

    def test_data_ma(self):
        assert isinstance(self.segm.data_ma, np.ma.MaskedArray)
        assert np.ma.count(self.segm.data_ma) == 18
        assert np.ma.count_masked(self.segm.data_ma) == 18

    def test_segments(self):
        assert isinstance(self.segm.segments[0], Segment)
        assert_allclose(self.segm.segments[0].data,
                        self.segm.segments[0].__array__())

        assert (self.segm.segments[0].data_ma.shape
                == self.segm.segments[0].data.shape)
        assert (self.segm.segments[0].data_ma.filled(0.0).sum()
                == self.segm.segments[0].data.sum())

        label = 4
        idx = self.segm.get_index(label)
        assert self.segm.segments[idx].label == label
        assert self.segm.segments[idx].area == self.segm.areas[idx]
        assert self.segm.segments[idx].slices == self.segm.slices[idx]
        assert self.segm.segments[idx].bbox == self.segm.bbox[idx]

    def test_repr_str(self):
        assert repr(self.segm) == str(self.segm)

        props = ['shape', 'nlabels']
        for prop in props:
            assert f'{prop}:' in repr(self.segm)

    def test_segment_repr_str(self):
        props = ['label', 'slices', 'area']
        for prop in props:
            assert f'{prop}:' in repr(self.segm.segments[0])

    def test_segment_data(self):
        assert_allclose(self.segm.segments[3].data.shape, (3, 3))
        assert_allclose(np.unique(self.segm.segments[3].data), [0, 5])

    def test_segment_make_cutout(self):
        cutout = self.segm.segments[3].make_cutout(self.data,
                                                   masked_array=False)
        assert not np.ma.is_masked(cutout)
        assert_allclose(cutout.shape, (3, 3))

        cutout = self.segm.segments[3].make_cutout(self.data,
                                                   masked_array=True)
        assert np.ma.is_masked(cutout)
        assert_allclose(cutout.shape, (3, 3))

    def test_segment_make_cutout_input(self):
        match = 'data must have the same shape as the segmentation array'
        with pytest.raises(ValueError, match=match):
            self.segm.segments[0].make_cutout(np.arange(10))

    def test_labels(self):
        assert_allclose(self.segm.labels, [1, 3, 4, 5, 7])

    def test_nlabels(self):
        assert self.segm.nlabels == 5

    def test_max_label(self):
        assert self.segm.max_label == 7

    def test_areas(self):
        expected = np.array([2, 2, 3, 6, 5])
        assert_allclose(self.segm.areas, expected)

        assert (self.segm.get_area(1)
                == self.segm.areas[self.segm.get_index(1)])
        assert_allclose(self.segm.get_areas(self.segm.labels),
                        self.segm.areas)

    def test_background_area(self):
        assert self.segm.background_area == 18

    def test_is_consecutive(self):
        assert not self.segm.is_consecutive

        data = np.array([[2, 2, 0], [0, 3, 3], [0, 0, 4]], dtype=np.int32)
        segm = SegmentationImage(data)
        dtype = segm.data.dtype
        assert not segm.is_consecutive  # does not start with label=1

        segm.relabel_consecutive(start_label=1)
        assert segm.is_consecutive
        assert segm.data.dtype == dtype

    def test_missing_labels(self):
        assert_allclose(self.segm.missing_labels, [2, 6])

    def test_check_labels(self):
        match = 'is invalid'
        with pytest.raises(ValueError, match=match):
            self.segm.check_label(2)
        with pytest.raises(ValueError, match=match):
            self.segm.check_labels([2])

        match = 'are invalid'
        with pytest.raises(ValueError, match=match):
            self.segm.check_labels([2, 6])

    def test_bbox_1d(self):
        segm = SegmentationImage(np.array([0, 0, 1, 1, 0, 2, 2, 0]))
        match = 'The "bbox" attribute requires a 2D segmentation image'
        with pytest.raises(ValueError, match=match):
            _ = segm.bbox

    @pytest.mark.skipif(not HAS_MATPLOTLIB, reason='matplotlib is required')
    def test_reset_cmap(self):
        segm = self.segm.copy()
        cmap = segm.cmap.copy()
        segm.reset_cmap(seed=123)
        assert not np.array_equal(cmap.colors, segm.cmap.colors)

    @pytest.mark.skipif(not HAS_MATPLOTLIB, reason='matplotlib is required')
    def test_make_cmap(self):
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
        cmap = self.segm.make_cmap(background_color=color)
        assert_allclose(cmap.colors[0], (0, 0, 0, alpha))

    def test_reassign_labels(self):
        segm = SegmentationImage(self.data.copy())
        segm.reassign_labels(labels=[1, 7], new_label=2)
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
        assert segm.nlabels == len(segm.slices) - segm.slices.count(None)

        # test slices caching
        segm = SegmentationImage(self.data.copy())
        slc1 = segm.slices
        segm.relabel_consecutive()
        assert slc1 == segm.slices

    @pytest.mark.parametrize('start_label', [0, -1])
    def test_relabel_consecutive_start_invalid(self, start_label):
        segm = SegmentationImage(self.data.copy())
        match = 'start_label must be > 0'
        with pytest.raises(ValueError, match=match):
            segm.relabel_consecutive(start_label=start_label)

    def test_keep_labels(self):
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
        segm = SegmentationImage(self.data.copy())
        match = 'border_width must be smaller than half the array size'
        with pytest.raises(ValueError, match=match):
            segm.remove_border_labels(border_width=3)

    def test_remove_border_labels_no_remaining_segments(self):
        alt_data = self.data.copy()
        alt_data[alt_data == 3] = 0
        segm = SegmentationImage(alt_data)
        segm.remove_border_labels(border_width=1, relabel=True)
        assert segm.nlabels == 0

    def test_remove_masked_labels(self):
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
        segm = SegmentationImage(np.ones((5, 5), dtype=int))
        mask = np.zeros((3, 3), dtype=bool)
        match = 'mask must have the same shape as the segmentation array'
        with pytest.raises(ValueError, match=match):
            segm.remove_masked_labels(mask)

    def test_make_source_mask(self):
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
        from matplotlib.image import AxesImage

        axim = self.segm.imshow(figsize=(5, 5))
        assert isinstance(axim, AxesImage)

        axim, _ = self.segm.imshow_map(figsize=(5, 5))
        assert isinstance(axim, AxesImage)

    @pytest.mark.skipif(not HAS_RASTERIO, reason='rasterio is required')
    @pytest.mark.skipif(not HAS_SHAPELY, reason='shapely is required')
    def test_polygons(self):
        from shapely.geometry.polygon import Polygon

        polygons = self.segm.polygons
        assert len(polygons) == self.segm.nlabels
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
        from regions import PolygonPixelRegion, Regions
        regions = self.segm.to_regions()

        assert isinstance(regions, Regions)
        assert isinstance(regions[0], PolygonPixelRegion)
        assert len(regions) == self.segm.nlabels

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
    @pytest.mark.skipif(not HAS_SHAPELY, reason='shapely is required')
    @pytest.mark.skipif(not HAS_MATPLOTLIB, reason='matplotlib is required')
    def test_patches_corners(self):
        """
        Regression test for a bug where patches were not generated for
        "invalid" Shapely polygons.

        This occurs when two pixels within a segment intersect only at a
        corner.
        """
        data = np.zeros((10, 10), dtype=np.uint32)
        data[5, 5] = 1
        data[4, 4] = 1
        data[3, 3] = 1
        segm = SegmentationImage(data)
        assert segm.nlabels == 1
        assert len(segm.segments) == 1
        assert len(segm.polygons) == 1
        assert len(segm.to_patches()) == 1
        assert len(segm.to_regions()) == 1

    @pytest.mark.skipif(not HAS_RASTERIO, reason='rasterio is required')
    @pytest.mark.skipif(not HAS_SHAPELY, reason='shapely is required')
    @pytest.mark.skipif(not HAS_MATPLOTLIB, reason='matplotlib is required')
    def test_polygons_complex(self):
        """
        Test polygons, patches, and regions for segments that have holes
        and/or are non-contiguous.
        """
        from matplotlib.patches import PathPatch
        from regions import PolygonPixelRegion, Regions
        from shapely.geometry import MultiPolygon, Polygon

        image = np.zeros((150, 150), dtype=np.uint32)

        # polygon with one hole
        image[10:90, 10:90] = 1
        image[30:70, 30:70] = 0

        # simple Polygon
        image[15:25, 110:140] = 2
        image[25:55, 110:120] = 2

        # MultiPolygon
        image[100:120, 20:40] = 3
        image[105:120, 45:55] = 3
        image[114:130, 60:80] = 3

        # single polygon with multiple holes
        image[85:145, 95:145] = 4
        image[105:115, 105:115] = 0
        image[125:135, 125:138] = 0
        image[120:125, 100:120] = 0

        # simple Polygon
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
        for patch in patches:
            assert isinstance(patch, PathPatch)

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

        # combine all segments into a single segment;
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
        assert segm0.deblended_labels_map == {}
        assert segm0.deblended_labels_inverse_map == {}

        deblend_map = {2: np.array([5, 6]), 3: np.array([7, 8])}
        segm._deblend_label_map = deblend_map
        assert_equal(segm._deblend_label_map, deblend_map)
        assert_equal(segm.deblended_labels, [5, 6, 7, 8])
        assert segm.deblended_labels_map == {5: 2, 6: 2, 7: 3, 8: 3}
        assert segm.deblended_labels_inverse_map == deblend_map

        segm2 = segm.copy()
        segm2.relabel_consecutive()
        deblend_map = {2: [3, 4], 3: [5, 6]}
        assert_equal(segm2._deblend_label_map, deblend_map)
        assert_equal(segm2.deblended_labels, [3, 4, 5, 6])
        assert segm2.deblended_labels_map == {3: 2, 4: 2, 5: 3, 6: 3}
        assert_equal(segm2.deblended_labels_inverse_map, deblend_map)

        segm3 = segm.copy()
        segm3.relabel_consecutive(start_label=10)
        deblend_map = {2: [12, 13], 3: [14, 15]}
        assert_equal(segm3._deblend_label_map, deblend_map)
        assert_equal(segm3.deblended_labels, [12, 13, 14, 15])
        assert segm3.deblended_labels_map == {12: 2, 13: 2, 14: 3, 15: 3}
        assert_equal(segm3.deblended_labels_inverse_map, deblend_map)

        segm4 = segm.copy()
        segm4.reassign_label(5, 50)
        segm4.reassign_label(7, 70)
        deblend_map = {2: [50, 6], 3: [70, 8]}
        assert_equal(segm4._deblend_label_map, deblend_map)
        assert_equal(segm4.deblended_labels, [6, 8, 50, 70])
        assert segm4.deblended_labels_map == {50: 2, 6: 2, 70: 3, 8: 3}
        assert_equal(segm4.deblended_labels_inverse_map, deblend_map)

        segm5 = segm.copy()
        segm5.reassign_label(5, 50, relabel=True)
        deblend_map = {2: [6, 3], 3: [4, 5]}
        assert_equal(segm5._deblend_label_map, deblend_map)
        assert_equal(segm5.deblended_labels, [3, 4, 5, 6])
        assert segm5.deblended_labels_map == {6: 2, 3: 2, 4: 3, 5: 3}
        assert_equal(segm5.deblended_labels_inverse_map, deblend_map)


class CustomSegm(SegmentationImage):
    @lazyproperty
    def value(self):
        return np.median(self.data)


def test_subclass():
    """
    Test that cached properties are reset in SegmentationImage
    subclasses.
    """
    data = np.array([[1, 1, 0, 0, 4, 4],
                     [0, 0, 0, 0, 0, 4],
                     [0, 0, 3, 3, 0, 0],
                     [7, 0, 0, 0, 0, 5],
                     [7, 7, 0, 5, 5, 5],
                     [7, 7, 0, 0, 5, 5]])
    segm = CustomSegm(data)
    _ = segm.slices, segm.labels, segm.value, segm.areas

    data2 = np.array([[10, 10, 0, 40],
                      [0, 0, 0, 40],
                      [70, 70, 0, 0],
                      [70, 70, 0, 1]])
    segm.data = data2
    assert len(segm.__dict__) == 3
    assert_equal(segm.areas, [1, 2, 2, 4])
