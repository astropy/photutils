# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for segmentation-based masking of aperture photometry, shared by
`AperturePhotometry` and `ApertureStats`.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from photutils.aperture._batch_photometry import (SHAPE_CIRCLE,
                                                  batch_aperture_sums)
from photutils.aperture._segmentation import (make_segmentation_exclusion,
                                              process_segmentation_inputs)
from photutils.aperture.circle import CircularAperture
from photutils.aperture.photometry import AperturePhotometry
from photutils.aperture.stats import ApertureStats
from photutils.aperture.tests.conftest import (NoBatchCircularAperture,
                                               make_scene)
from photutils.segmentation import SegmentationImage


class TestProcessSegmentationInputs:
    def test_method_none_returns_none(self):
        data, segm = make_scene()
        positions = [(21, 21)]
        result = process_segmentation_inputs(segm, None, 'none', positions,
                                             data.shape)
        assert result == (None, None)

    def test_invalid_method(self):
        data, segm = make_scene()
        match = 'mask_method must be one of'
        with pytest.raises(ValueError, match=match):
            process_segmentation_inputs(segm, None, 'invalid', [(21, 21)],
                                        data.shape)

    def test_missing_segmentation(self):
        match = 'segmentation_image must be input'
        with pytest.raises(ValueError, match=match):
            process_segmentation_inputs(None, None, 'mask', [(21, 21)],
                                        (50, 50))

    def test_segmentation_image_object(self):
        data, segm = make_scene()
        segm_obj = SegmentationImage(segm)
        out_segm, out_labels = process_segmentation_inputs(
            segm_obj, [1], 'mask', [(21, 21)], data.shape)
        assert out_segm.dtype == np.intp
        assert_allclose(out_labels, [1])

    def test_ndarray_input(self):
        data, segm = make_scene()
        out_segm, _ = process_segmentation_inputs(
            segm, [1], 'mask', [(21, 21)], data.shape)
        assert out_segm.dtype == np.intp

    def test_not_2d(self):
        match = 'segmentation_image must be a 2D array'
        with pytest.raises(ValueError, match=match):
            process_segmentation_inputs(np.zeros((3, 3, 3), dtype=int), [1],
                                        'mask', [(1, 1)], (3, 3))

    def test_wrong_shape(self):
        match = 'same shape as the data'
        with pytest.raises(ValueError, match=match):
            process_segmentation_inputs(np.zeros((10, 10), dtype=int), [1],
                                        'mask', [(1, 1)], (50, 50))

    def test_non_integer_dtype(self):
        match = 'integer data type'
        with pytest.raises(ValueError, match=match):
            process_segmentation_inputs(np.zeros((50, 50), dtype=float), [1],
                                        'mask', [(21, 21)], (50, 50))

    def test_labels_length_mismatch(self):
        data, segm = make_scene()
        match = 'labels must have the same length'
        with pytest.raises(ValueError, match=match):
            process_segmentation_inputs(segm, [1, 2], 'mask', [(21, 21)],
                                        data.shape)

    def test_labels_not_1d(self):
        data, segm = make_scene()
        match = 'labels must be a 1D array'
        with pytest.raises(ValueError, match=match):
            process_segmentation_inputs(segm, [[1, 2]], 'mask', [(21, 21)],
                                        data.shape)

    def test_labels_required(self):
        data, segm = make_scene()
        match = 'labels must be input when segmentation_image is input'
        with pytest.raises(ValueError, match=match):
            process_segmentation_inputs(segm, None, 'mask',
                                        [(21, 21), (28, 22)], data.shape)

    @pytest.mark.parametrize('use_segm_obj', [True, False])
    def test_label_not_in_image(self, use_segm_obj):
        data, segm = make_scene()
        segm_in = SegmentationImage(segm) if use_segm_obj else segm
        match = r'labels \[3\] are not present in the segmentation_image'
        with pytest.raises(ValueError, match=match):
            process_segmentation_inputs(segm_in, [1, 3], 'mask',
                                        [(21, 21), (28, 22)], data.shape)

    def test_invalid_labels_reported_once_and_sorted(self):
        data, segm = make_scene()
        match = r'labels \[-1, 5\] are not present'
        with pytest.raises(ValueError, match=match):
            process_segmentation_inputs(segm, [5, 1, -1, 5], 'mask',
                                        [(21, 21)] * 4, data.shape)

    @pytest.mark.parametrize('method', ['mask', 'source_only', 'correct'])
    def test_label_zero_allowed(self, method):
        # Label 0 disables the masking for that aperture and is not
        # required to be present in the image.
        data, segm = make_scene()
        _, out_labels = process_segmentation_inputs(
            segm, [0, 1], method, [(21, 21), (28, 22)], data.shape)
        assert_array_equal(out_labels, [0, 1])

    def test_background_only_labels_optional(self):
        data, segm = make_scene()
        out_segm, out_labels = process_segmentation_inputs(
            segm, None, 'background_only', [(21, 21), (28, 22)],
            data.shape)
        assert out_segm.dtype == np.intp
        assert out_labels.dtype == np.intp
        assert_array_equal(out_labels, [0, 0])

    def test_background_only_ignores_labels(self):
        # The labels are not used, so they need not be present in the
        # segmentation image.
        data, segm = make_scene()
        _, out_labels = process_segmentation_inputs(
            SegmentationImage(segm), [3], 'background_only', [(21, 21)],
            data.shape)
        assert_array_equal(out_labels, [0])

    @pytest.mark.parametrize(
        ('labels', 'match'),
        [([[1, 2]], 'labels must be a 1D array'),
         ([1, 2, 3], 'labels must have the same length')])
    def test_background_only_labels_shape_validated(self, labels, match):
        # The labels are not used, but their shape is still validated
        # because they are stored and sliced per aperture.
        data, segm = make_scene()
        with pytest.raises(ValueError, match=match):
            process_segmentation_inputs(SegmentationImage(segm), labels,
                                        'background_only', [(21, 21)],
                                        data.shape)


class TestAperturePhotometry:
    def test_batch_matches_mask_path(self):
        """
        Test that the batch driver for circular apertures matches the
        Python mask code path for segmentation masking.
        """
        data, segm = make_scene()
        aper = CircularAperture([(21, 21), (28, 22)], r=6)
        labels = [1, 2]
        result = AperturePhotometry(data, aper, segmentation_image=segm,
                                    labels=labels, mask_method='mask')
        for idx, label in enumerate(labels):
            manual_mask = (segm > 0) & (segm != label)
            ref = AperturePhotometry(
                data, CircularAperture(aper.positions[idx], r=6),
                mask=manual_mask)
            assert_allclose(result.flux[idx], ref.flux)

    def test_label_not_in_image(self):
        data, segm = make_scene()
        aper = CircularAperture([(21, 21), (28, 22)], r=6)
        match = 'not present in the segmentation_image'
        with pytest.raises(ValueError, match=match):
            AperturePhotometry(data, aper, segmentation_image=segm,
                               labels=[1, 3], mask_method='mask')

    def test_background_only_matches_manual(self):
        """
        Test that 'background_only' excludes every labeled pixel,
        regardless of the (optional) labels.
        """
        data, segm = make_scene()
        aper = CircularAperture([(21, 21), (28, 22)], r=6)
        ref = AperturePhotometry(data, aper, mask=segm > 0)
        for labels in (None, [1, 2], [0, 0]):
            result = AperturePhotometry(data, aper, segmentation_image=segm,
                                        labels=labels,
                                        mask_method='background_only')
            assert_allclose(result.flux, ref.flux)
            assert_allclose(result.area, ref.area)

    def test_background_only_mask_path_parity(self):
        """
        Test that the batch driver and the Python mask path agree for
        'background_only', including the flags.
        """
        data, segm = make_scene()
        error = np.full(data.shape, 0.5)
        positions = [(21, 21), (28, 22), (5, 5)]
        kwargs = {'error': error, 'segmentation_image': segm,
                  'mask_method': 'background_only'}
        batch = AperturePhotometry(data, CircularAperture(positions, r=6),
                                   **kwargs)
        nobatch = AperturePhotometry(
            data, NoBatchCircularAperture(positions, r=6), **kwargs)
        assert_allclose(batch.flux, nobatch.flux, rtol=1e-12)
        assert_allclose(batch.flux_err, nobatch.flux_err, rtol=1e-12)
        assert_allclose(batch.area, nobatch.area, rtol=1e-12)
        assert_array_equal(batch.flags, nobatch.flags)


class TestApertureStats:
    @pytest.mark.parametrize('method',
                             ['none', 'mask', 'source_only',
                              'background_only', 'correct'])
    def test_matches_aperture_photometry(self, method):
        data, segm = make_scene()
        aper = CircularAperture([(21, 21), (28, 22)], r=6)
        kwargs = {}
        if method != 'none':
            kwargs = {'segmentation_image': segm, 'labels': [1, 2],
                      'mask_method': method}
        phot = AperturePhotometry(data, aper, **kwargs)
        stats = ApertureStats(data, aper, **kwargs)
        assert_allclose(stats.sum, phot.flux, rtol=1e-10)

    def test_slicing_preserves_labels(self):
        data, segm = make_scene()
        aper = CircularAperture([(21, 21), (28, 22)], r=6)
        stats = ApertureStats(data, aper, segmentation_image=segm,
                              labels=[1, 2], mask_method='mask')
        sub = stats[1]
        assert_allclose(sub.sum, stats.sum[1])

    def test_copy_preserves_masking(self):
        data, segm = make_scene()
        aper = CircularAperture([(21, 21)], r=6)
        stats = ApertureStats(data, aper, segmentation_image=segm,
                              labels=[1], mask_method='mask')
        copied = stats.copy()
        assert_allclose(copied.sum, stats.sum)

    def test_no_segmentation_slicing(self):
        data, _ = make_scene()
        aper = CircularAperture([(21, 21), (28, 22)], r=6)
        stats = ApertureStats(data, aper)
        sub = stats[0]
        assert sub._seg_labels is None

    def test_label_not_in_image(self):
        data, segm = make_scene()
        aper = CircularAperture([(21, 21), (28, 22)], r=6)
        match = 'not present in the segmentation_image'
        with pytest.raises(ValueError, match=match):
            ApertureStats(data, aper, segmentation_image=segm,
                          labels=[1, 3], mask_method='mask')

    def test_background_only_without_labels(self):
        data, segm = make_scene()
        aper = CircularAperture([(21, 21), (28, 22)], r=6)
        stats = ApertureStats(data, aper, segmentation_image=segm,
                              mask_method='background_only')
        ref = ApertureStats(data, aper, mask=segm > 0)
        assert stats.labels is None
        assert_allclose(stats.sum, ref.sum)
        assert_allclose(stats.median, ref.median)

        sub = stats[1]
        assert sub.labels is None
        assert_allclose(sub.sum, stats.sum[1])

    def test_background_only_with_labels(self):
        # Input labels are echoed and sliced even though the method
        # does not use them.
        data, segm = make_scene()
        aper = CircularAperture([(21, 21), (28, 22)], r=6)
        stats = ApertureStats(data, aper,
                              segmentation_image=SegmentationImage(segm),
                              labels=[1, 2], mask_method='background_only')
        ref = ApertureStats(data, aper, mask=segm > 0)
        assert_allclose(stats.sum, ref.sum)
        sub = stats[1]
        assert sub.labels == 2
        assert_allclose(sub.sum, stats.sum[1])

        match = 'labels must have the same length'
        with pytest.raises(ValueError, match=match):
            ApertureStats(data, aper,
                          segmentation_image=SegmentationImage(segm),
                          labels=[1, 2, 3], mask_method='background_only')


class TestMakeSegmentationExclusion:
    def test_none_method(self):
        segm = np.array([[0, 1], [2, 1]])
        _, _, exclude, affected = make_segmentation_exclusion('none', segm, 1)
        assert not exclude.any()
        assert not affected.any()

    def test_label_zero(self):
        segm = np.array([[0, 1], [2, 1]])
        _, _, exclude, affected = make_segmentation_exclusion('mask', segm, 0)
        assert not exclude.any()
        assert not affected.any()

    def test_mask_method(self):
        segm = np.array([[0, 1], [2, 1]])
        _, _, exclude, affected = make_segmentation_exclusion('mask', segm, 1)
        expected = np.array([[False, False], [True, False]])
        assert_array_equal(exclude, expected)
        assert_array_equal(affected, expected)

    def test_source_only_method(self):
        segm = np.array([[0, 1], [2, 1]])
        _, _, exclude, affected = make_segmentation_exclusion(
            'source_only', segm, 1)
        expected = np.array([[True, False], [True, False]])
        assert_array_equal(exclude, expected)
        # Background exclusions are not marked as affected
        expected_affected = np.array([[False, False], [True, False]])
        assert_array_equal(affected, expected_affected)

    @pytest.mark.parametrize('label', [0, 1])
    def test_background_only_method(self, label):
        # Every labeled pixel is excluded and marked as affected. The
        # label is ignored, so label 0 does not disable the masking.
        segm = np.array([[0, 1], [2, 1]])
        _, _, exclude, affected = make_segmentation_exclusion(
            'background_only', segm, label)
        expected = np.array([[False, True], [True, True]])
        assert_array_equal(exclude, expected)
        assert_array_equal(affected, expected)

    def test_correct_replaces_neighbor(self):
        # 5x5 cutout, center (2, 2). A neighbor pixel at (1, 2) [x=1, y=2]
        # is mirrored from (3, 2) [x=3, y=2], a good background pixel.
        segm = np.zeros((5, 5), dtype=int)
        segm[2, 2] = 1  # target center
        segm[2, 1] = 2  # neighbor at x=1, y=2
        data = np.arange(25, dtype=float).reshape(5, 5)
        out_data, _, exclude, affected = make_segmentation_exclusion(
            'correct', segm, 1, data=data, cutout_xycen=(2, 2))
        # Neighbor replaced, not excluded
        assert not exclude[2, 1]
        # Replaced pixels are marked as affected
        assert affected[2, 1]
        # Mirror of (x=1, y=2) is (x=3, y=2) -> data[2, 3]
        assert out_data[2, 1] == data[2, 3]

    def test_correct_out_of_bounds_mirror(self):
        # Neighbor whose mirror falls outside the cutout is excluded.
        segm2 = np.zeros((5, 5), dtype=int)
        segm2[1, 1] = 1
        segm2[1, 0] = 2  # neighbor x=0,y=1 with mirror x=2*1-0=2 inside
        segm2[3, 3] = 2  # neighbor x=3,y=3 with mirror x=2*1-3=-1 outside
        data = np.ones((5, 5))
        _, _, exclude, affected = make_segmentation_exclusion(
            'correct', segm2, 1, data=data, cutout_xycen=(1, 1))
        assert exclude[3, 3]
        assert affected[3, 3]

    def test_correct_neighbor_mirror(self):
        # A neighbor whose mirror is also a neighbor must be excluded.
        segm = np.zeros((5, 5), dtype=int)
        segm[2, 2] = 1
        segm[2, 1] = 2  # neighbor x=1,y=2 with mirror x=3,y=2
        segm[2, 3] = 2  # neighbor x=3,y=2 with mirror x=1,y=2 (a neighbor)
        data = np.ones((5, 5))
        _, _, exclude, _ = make_segmentation_exclusion(
            'correct', segm, 1, data=data, cutout_xycen=(2, 2))
        assert exclude[2, 1]
        assert exclude[2, 3]

    def test_correct_masked_mirror(self):
        # A neighbor whose mirror is in base_mask must be excluded.
        segm = np.zeros((5, 5), dtype=int)
        segm[2, 2] = 1
        segm[2, 1] = 2  # neighbor x=1,y=2 with mirror x=3,y=2
        base_mask = np.zeros((5, 5), dtype=bool)
        base_mask[2, 3] = True  # mask the mirror pixel
        data = np.ones((5, 5))
        _, _, exclude, _ = make_segmentation_exclusion(
            'correct', segm, 1, data=data, base_mask=base_mask,
            cutout_xycen=(2, 2))
        assert exclude[2, 1]

    def test_correct_with_error(self):
        segm = np.zeros((5, 5), dtype=int)
        segm[2, 2] = 1
        segm[2, 1] = 2
        data = np.arange(25, dtype=float).reshape(5, 5)
        error = np.arange(25, dtype=float).reshape(5, 5) * 0.1
        _, out_error, _, _ = make_segmentation_exclusion(
            'correct', segm, 1, data=data, error=error, cutout_xycen=(2, 2))
        assert out_error[2, 1] == error[2, 3]


class TestBatchDriverSegmentation:
    def test_mask_method(self):
        rng = np.random.default_rng(0)
        data = rng.random((40, 40))
        error = rng.random((40, 40)) + 0.1
        mask = np.zeros((40, 40), dtype=np.uint8)
        positions = np.array([[20.0, 20.0]])
        params = np.array([8.0])

        segm = np.zeros((40, 40), dtype=np.intp)
        segm[18:23, 18:23] = 1
        segm[18:23, 23:28] = 2
        labels = np.array([1], dtype=np.intp)

        sums = batch_aperture_sums(
            data, error, mask, positions, SHAPE_CIRCLE, params, 8.0, 8.0,
            0.0, 0.0, 1, 8, segm, labels, 1)[0]

        # Reference via global mask
        manual_mask = ((segm > 0) & (segm != 1)).astype(np.uint8)
        ref = batch_aperture_sums(
            data, error, manual_mask, positions, SHAPE_CIRCLE, params,
            8.0, 8.0, 0.0, 0.0, 1, 8)[0]
        assert_allclose(sums, ref)

    def test_source_only_method(self):
        rng = np.random.default_rng(1)
        data = rng.random((40, 40))
        error = rng.random((40, 40)) + 0.1
        mask = np.zeros((40, 40), dtype=np.uint8)
        positions = np.array([[20.0, 20.0]])
        params = np.array([8.0])

        segm = np.zeros((40, 40), dtype=np.intp)
        segm[18:23, 18:23] = 1
        labels = np.array([1], dtype=np.intp)

        sums = batch_aperture_sums(
            data, error, mask, positions, SHAPE_CIRCLE, params, 8.0, 8.0,
            0.0, 0.0, 1, 8, segm, labels, 2)[0]

        manual_mask = (segm != 1).astype(np.uint8)
        ref = batch_aperture_sums(
            data, error, manual_mask, positions, SHAPE_CIRCLE, params,
            8.0, 8.0, 0.0, 0.0, 1, 8)[0]
        assert_allclose(sums, ref)

    @pytest.mark.parametrize('label', [0, 1])
    def test_background_only_method(self, label):
        # Method 4 excludes every labeled pixel. The label is ignored,
        # so label 0 does not disable the masking.
        rng = np.random.default_rng(4)
        data = rng.random((40, 40))
        error = rng.random((40, 40)) + 0.1
        mask = np.zeros((40, 40), dtype=np.uint8)
        positions = np.array([[20.0, 20.0]])
        params = np.array([8.0])

        segm = np.zeros((40, 40), dtype=np.intp)
        segm[18:23, 18:23] = 1
        segm[18:23, 23:28] = 2
        labels = np.array([label], dtype=np.intp)

        sums = batch_aperture_sums(
            data, error, mask, positions, SHAPE_CIRCLE, params, 8.0, 8.0,
            0.0, 0.0, 1, 8, segm, labels, 4)[0]

        manual_mask = (segm > 0).astype(np.uint8)
        ref = batch_aperture_sums(
            data, error, manual_mask, positions, SHAPE_CIRCLE, params,
            8.0, 8.0, 0.0, 0.0, 1, 8)[0]
        assert_allclose(sums, ref)

    def test_label0_disables(self):
        rng = np.random.default_rng(2)
        data = rng.random((40, 40))
        error = rng.random((40, 40)) + 0.1
        mask = np.zeros((40, 40), dtype=np.uint8)
        positions = np.array([[20.0, 20.0]])
        params = np.array([8.0])

        segm = np.zeros((40, 40), dtype=np.intp)
        segm[18:23, 18:23] = 1
        segm[18:23, 23:28] = 2
        labels = np.array([0], dtype=np.intp)

        sums = batch_aperture_sums(
            data, error, mask, positions, SHAPE_CIRCLE, params, 8.0, 8.0,
            0.0, 0.0, 1, 8, segm, labels, 1)[0]
        ref = batch_aperture_sums(
            data, error, mask, positions, SHAPE_CIRCLE, params, 8.0, 8.0,
            0.0, 0.0, 1, 8)[0]
        assert_allclose(sums, ref)

    def test_correct_method_matches_mask_path(self):
        # The batch 'correct' kernel (seg_method=3) must exactly match
        # the Python mask-path 'correct' implementation.
        rng = np.random.default_rng(3)
        data = rng.normal(10.0, 1.0, (60, 60))
        error = rng.random((60, 60)) + 0.5
        mask = np.zeros((60, 60), dtype=bool)
        mask[22, 30] = True
        segm = np.zeros((60, 60), dtype=np.intp)
        segm[18:25, 18:25] = 1  # target
        segm[18:25, 25:31] = 2  # bright neighbor
        data[18:25, 25:31] += 200.0
        positions = [(21.0, 21.0)]
        aper = CircularAperture(positions, r=8)
        labels = np.array([1], dtype=np.intp)

        batch = aper._photometry(
            data, error=error, mask=mask, segmentation_image=segm,
            labels=labels, mask_method='correct')
        mask_sum, mask_err, _area, *_ = aper._mask_photometry(
            data, error=error, mask=mask, method='exact', subpixels=5,
            segmentation=segm, labels=labels,
            mask_method='correct')
        assert_allclose(batch.flux, mask_sum, rtol=1e-12)
        assert_allclose(batch.flux_err, mask_err, rtol=1e-12)
