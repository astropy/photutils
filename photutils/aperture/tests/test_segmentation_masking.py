# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for segmentation-based masking of aperture photometry, shared
by `aperture_photometry`, `PixelAperture.do_photometry`, and
`ApertureStats`.
"""

import astropy.units as u
import numpy as np
import pytest
from numpy.testing import assert_allclose

from photutils.aperture._batch_photometry import (SHAPE_CIRCLE,
                                                  batch_aperture_sums)
from photutils.aperture._segmentation import (make_segmentation_exclusion,
                                              process_segmentation_inputs)
from photutils.aperture.circle import CircularAperture
from photutils.aperture.photometry import aperture_photometry
from photutils.aperture.polygon import PolygonAperture
from photutils.aperture.stats import ApertureStats
from photutils.segmentation import SegmentationImage


def make_scene():
    """
    Build a deterministic scene with a target source (label 1) and a
    bright neighbor source (label 2), on a nonzero background.
    """
    data = np.ones((50, 50))
    segm = np.zeros((50, 50), dtype=int)

    # Target source (label 1)
    data[18:25, 18:25] = 10.0
    segm[18:25, 18:25] = 1

    # Bright neighbor source (label 2)
    data[20:25, 26:32] = 100.0
    segm[20:25, 26:32] = 2

    return data, segm


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

    def test_labels_required(self):
        data, segm = make_scene()
        match = 'labels must be input when segmentation_image is input'
        with pytest.raises(ValueError, match=match):
            process_segmentation_inputs(segm, None, 'mask',
                                        [(21, 21), (28, 22)], data.shape)


class TestAperturePhotometry:
    @pytest.mark.parametrize('use_segm_obj', [True, False])
    def test_mask_method_matches_manual(self, use_segm_obj):
        data, segm = make_scene()
        aper = CircularAperture([(21, 21)], r=8)
        segm_in = SegmentationImage(segm) if use_segm_obj else segm

        phot = aperture_photometry(data, aper, segmentation_image=segm_in,
                                   labels=[1], mask_method='mask')
        manual_mask = (segm > 0) & (segm != 1)
        ref = aperture_photometry(data, aper, mask=manual_mask)
        assert_allclose(phot['aperture_sum'], ref['aperture_sum'])

    def test_source_only_matches_manual(self):
        data, segm = make_scene()
        aper = CircularAperture([(21, 21)], r=8)
        phot = aperture_photometry(data, aper, segmentation_image=segm,
                                   labels=[1],
                                   mask_method='source_only')
        manual_mask = segm != 1
        ref = aperture_photometry(data, aper, mask=manual_mask)
        assert_allclose(phot['aperture_sum'], ref['aperture_sum'])

    def test_none_method_ignores_segmentation(self):
        data, segm = make_scene()
        aper = CircularAperture([(21, 21)], r=8)
        phot = aperture_photometry(data, aper, segmentation_image=segm,
                                   mask_method='none')
        ref = aperture_photometry(data, aper)
        assert_allclose(phot['aperture_sum'], ref['aperture_sum'])

    def test_missing_segmentation_raises(self):
        data, _ = make_scene()
        aper = CircularAperture([(21, 21)], r=8)
        match = 'segmentation_image must be input'
        with pytest.raises(ValueError, match=match):
            aperture_photometry(data, aper, mask_method='mask')

    def test_mask_excludes_neighbor_flux(self):
        data, segm = make_scene()
        aper = CircularAperture([(21, 21)], r=10)
        none_phot = aperture_photometry(data, aper,
                                        mask_method='none')
        mask_phot = aperture_photometry(data, aper, segmentation_image=segm,
                                        labels=[1], mask_method='mask')
        # Masking the bright neighbor reduces the sum
        assert mask_phot['aperture_sum'][0] < none_phot['aperture_sum'][0]

    def test_label0_disables_masking(self):
        data, segm = make_scene()
        aper = CircularAperture([(21, 21)], r=8)
        phot = aperture_photometry(data, aper, segmentation_image=segm,
                                   labels=[0], mask_method='mask')
        ref = aperture_photometry(data, aper)
        assert_allclose(phot['aperture_sum'], ref['aperture_sum'])

    def test_correct_matches_mask_path(self):
        # The 'correct' method uses the Cython batch driver for circular
        # apertures. Verify it exactly matches the Python mask code path
        # and differs from 'none' for a scene with a neighbor.
        data, segm = make_scene()
        error = np.ones_like(data) * 2.0
        mask = np.zeros(data.shape, dtype=bool)
        mask[22, 30] = True
        aper = CircularAperture([(21, 21)], r=10)
        none_phot = aperture_photometry(data, aper,
                                        mask_method='none')

        labels = [1]
        batch_sum, batch_err = aper.do_photometry(
            data, error=error, mask=mask, segmentation_image=segm,
            labels=labels, mask_method='correct')
        mask_sum, mask_err, _area = aper._do_mask_photometry(
            data, error=error, mask=mask, method='exact', subpixels=5,
            segmentation=segm.astype(np.intp), labels=labels,
            mask_method='correct')
        assert_allclose(batch_sum, mask_sum, rtol=1e-12)
        assert_allclose(batch_err, mask_err, rtol=1e-12)
        assert batch_sum[0] != none_phot['aperture_sum'][0]

    def test_polygon_mask_path(self):
        # Polygon apertures have no batch driver, exercising the mask
        # code path for segmentation masking.
        data, segm = make_scene()
        offsets = np.array([[-7, -7], [9, -7], [9, 9], [-7, 9]])
        aper = PolygonAperture((21, 21), offsets)
        phot = aperture_photometry(data, aper, segmentation_image=segm,
                                   labels=[1], mask_method='mask')
        manual_mask = (segm > 0) & (segm != 1)
        ref = aperture_photometry(data, aper, mask=manual_mask)
        assert_allclose(phot['aperture_sum'], ref['aperture_sum'])

    def test_with_error(self):
        data, segm = make_scene()
        error = np.ones_like(data) * 2.0
        aper = CircularAperture([(21, 21)], r=8)
        phot = aperture_photometry(data, aper, error=error,
                                   segmentation_image=segm, labels=[1],
                                   mask_method='mask')
        manual_mask = (segm > 0) & (segm != 1)
        ref = aperture_photometry(data, aper, error=error, mask=manual_mask)
        assert_allclose(phot['aperture_sum'], ref['aperture_sum'])
        assert_allclose(phot['aperture_sum_err'], ref['aperture_sum_err'])

    def test_with_units(self):
        data, segm = make_scene()
        unit = u.Jy
        aper = CircularAperture([(21, 21)], r=8)
        phot = aperture_photometry(data * unit, aper,
                                   segmentation_image=segm, labels=[1],
                                   mask_method='mask')
        manual_mask = (segm > 0) & (segm != 1)
        ref = aperture_photometry(data * unit, aper, mask=manual_mask)
        assert_allclose(phot['aperture_sum'].value, ref['aperture_sum'].value)
        assert phot['aperture_sum'].unit == unit

    def test_labels_required(self):
        data, segm = make_scene()
        aper = CircularAperture([(21, 21)], r=8)
        match = 'labels must be input when segmentation_image is input'
        with pytest.raises(ValueError, match=match):
            aperture_photometry(data, aper, segmentation_image=segm,
                                mask_method='mask')


class TestDoPhotometry:
    def test_batch_matches_mask_path(self):
        # do_photometry uses the batch driver for circular apertures;
        # compare to the equivalent manual global mask.
        data, segm = make_scene()
        aper = CircularAperture([(21, 21), (28, 22)], r=6)
        labels = [1, 2]
        sums, _ = aper.do_photometry(data, segmentation_image=segm,
                                     labels=labels,
                                     mask_method='mask')
        for idx, label in enumerate(labels):
            manual_mask = (segm > 0) & (segm != label)
            ref, _ = CircularAperture(aper.positions[idx], r=6).do_photometry(
                data, mask=manual_mask)
            assert_allclose(sums[idx], ref[0])


class TestApertureStats:
    @pytest.mark.parametrize('method',
                             ['none', 'mask', 'source_only', 'correct'])
    def test_matches_aperture_photometry(self, method):
        data, segm = make_scene()
        aper = CircularAperture([(21, 21), (28, 22)], r=6)
        kwargs = {}
        if method != 'none':
            kwargs = {'segmentation_image': segm, 'labels': [1, 2],
                      'mask_method': method}
        phot = aperture_photometry(data, aper, **kwargs)
        stats = ApertureStats(data, aper, **kwargs)
        assert_allclose(stats.sum, phot['aperture_sum'], rtol=1e-10)

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


class TestMakeSegmentationExclusion:
    def test_none_method(self):
        segm = np.array([[0, 1], [2, 1]])
        _, _, exclude = make_segmentation_exclusion('none', segm, 1)
        assert not exclude.any()

    def test_label_zero(self):
        segm = np.array([[0, 1], [2, 1]])
        _, _, exclude = make_segmentation_exclusion('mask', segm, 0)
        assert not exclude.any()

    def test_mask_method(self):
        segm = np.array([[0, 1], [2, 1]])
        _, _, exclude = make_segmentation_exclusion('mask', segm, 1)
        expected = np.array([[False, False], [True, False]])
        assert_allclose(exclude, expected)

    def test_source_only_method(self):
        segm = np.array([[0, 1], [2, 1]])
        _, _, exclude = make_segmentation_exclusion('source_only', segm, 1)
        expected = np.array([[True, False], [True, False]])
        assert_allclose(exclude, expected)

    def test_correct_replaces_neighbor(self):
        # 5x5 cutout, center (2, 2). A neighbor pixel at (1, 2) [x=1, y=2]
        # is mirrored from (3, 2) [x=3, y=2], a good background pixel.
        segm = np.zeros((5, 5), dtype=int)
        segm[2, 2] = 1  # target center
        segm[2, 1] = 2  # neighbor at x=1, y=2
        data = np.arange(25, dtype=float).reshape(5, 5)
        out_data, _, exclude = make_segmentation_exclusion(
            'correct', segm, 1, data=data, cutout_xycen=(2, 2))
        # neighbor replaced, not excluded
        assert not exclude[2, 1]
        # mirror of (x=1, y=2) is (x=3, y=2) -> data[2, 3]
        assert out_data[2, 1] == data[2, 3]

    def test_correct_out_of_bounds_mirror(self):
        # Neighbor whose mirror falls outside the cutout is excluded.
        segm2 = np.zeros((5, 5), dtype=int)
        segm2[1, 1] = 1
        segm2[1, 0] = 2  # neighbor x=0,y=1; mirror x=2*1-0=2 in bounds
        segm2[3, 3] = 2  # neighbor x=3,y=3; mirror x=2*1-3=-1 out of bounds
        data = np.ones((5, 5))
        _, _, exclude = make_segmentation_exclusion(
            'correct', segm2, 1, data=data, cutout_xycen=(1, 1))
        assert exclude[3, 3]

    def test_correct_neighbor_mirror(self):
        # A neighbor whose mirror is also a neighbor must be excluded.
        segm = np.zeros((5, 5), dtype=int)
        segm[2, 2] = 1
        segm[2, 1] = 2  # neighbor x=1,y=2; mirror x=3,y=2
        segm[2, 3] = 2  # neighbor x=3,y=2; mirror x=1,y=2 (both neighbors)
        data = np.ones((5, 5))
        _, _, exclude = make_segmentation_exclusion(
            'correct', segm, 1, data=data, cutout_xycen=(2, 2))
        assert exclude[2, 1]
        assert exclude[2, 3]

    def test_correct_masked_mirror(self):
        # A neighbor whose mirror is in base_mask must be excluded.
        segm = np.zeros((5, 5), dtype=int)
        segm[2, 2] = 1
        segm[2, 1] = 2  # neighbor x=1,y=2; mirror x=3,y=2
        base_mask = np.zeros((5, 5), dtype=bool)
        base_mask[2, 3] = True  # mask the mirror pixel
        data = np.ones((5, 5))
        _, _, exclude = make_segmentation_exclusion(
            'correct', segm, 1, data=data, base_mask=base_mask,
            cutout_xycen=(2, 2))
        assert exclude[2, 1]

    def test_correct_with_error(self):
        segm = np.zeros((5, 5), dtype=int)
        segm[2, 2] = 1
        segm[2, 1] = 2
        data = np.arange(25, dtype=float).reshape(5, 5)
        error = np.arange(25, dtype=float).reshape(5, 5) * 0.1
        _, out_error, _ = make_segmentation_exclusion(
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
            1, 8, segm, labels, 1)[0]

        # Reference via global mask
        manual_mask = ((segm > 0) & (segm != 1)).astype(np.uint8)
        ref = batch_aperture_sums(
            data, error, manual_mask, positions, SHAPE_CIRCLE, params,
            8.0, 8.0, 1, 8)[0]
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
            1, 8, segm, labels, 2)[0]

        manual_mask = (segm != 1).astype(np.uint8)
        ref = batch_aperture_sums(
            data, error, manual_mask, positions, SHAPE_CIRCLE, params,
            8.0, 8.0, 1, 8)[0]
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
            1, 8, segm, labels, 1)[0]
        ref = batch_aperture_sums(
            data, error, mask, positions, SHAPE_CIRCLE, params, 8.0, 8.0,
            1, 8)[0]
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

        batch_sum, batch_err = aper.do_photometry(
            data, error=error, mask=mask, segmentation_image=segm,
            labels=labels, mask_method='correct')
        mask_sum, mask_err, _area = aper._do_mask_photometry(
            data, error=error, mask=mask, method='exact', subpixels=5,
            segmentation=segm, labels=labels,
            mask_method='correct')
        assert_allclose(batch_sum, mask_sum, rtol=1e-12)
        assert_allclose(batch_err, mask_err, rtol=1e-12)
