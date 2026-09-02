# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the per-source flag counts returned by the batch drivers.
"""

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from photutils.aperture import CircularAperture
from photutils.aperture._batch_photometry import (FLAG_COL_BBOX_CLIPPED,
                                                  FLAG_COL_MASKED,
                                                  FLAG_COL_N_PIXELS,
                                                  FLAG_COL_NONFINITE_DATA,
                                                  FLAG_COL_NONFINITE_ERROR,
                                                  FLAG_COL_SEG,
                                                  FLAG_COL_SEG_MASKED,
                                                  FLAG_COL_UNCORRECTED,
                                                  FLAG_COL_UNCORRECTED_MASKED,
                                                  FLAG_COL_VALID, N_FLAG_COLS,
                                                  SHAPE_CIRCLE,
                                                  batch_aperture_sums)
from photutils.aperture._batch_stats import batch_aperture_gather
from photutils.aperture.tests.conftest import UNIT_SHAPE


def _sums_fcounts(data, positions, radius, *, error=None, mask=None,
                  use_exact=1, subpixels=5, segmentation=None, labels=None,
                  seg_method=0):
    """
    Run ``batch_aperture_sums`` and return the flag-count array.
    """
    positions = np.ascontiguousarray(np.atleast_2d(positions),
                                     dtype=np.float64)
    params = np.array([radius], dtype=np.float64)
    result = batch_aperture_sums(
        np.ascontiguousarray(data, dtype=np.float64), error, mask,
        positions, SHAPE_CIRCLE, params, radius, radius, 0.0, 0.0,
        use_exact, subpixels, segmentation, labels, seg_method)
    return result.flag_counts


def _gather_fcounts(data, positions, radius, *, mask=None,
                    segmentation=None, labels=None, seg_method=0):
    """
    Run ``batch_aperture_gather`` and return the flag-count array.
    """
    positions = np.ascontiguousarray(np.atleast_2d(positions),
                                     dtype=np.float64)
    params = np.array([radius], dtype=np.float64)
    result = batch_aperture_gather(
        np.ascontiguousarray(data, dtype=np.float64), mask, positions,
        SHAPE_CIRCLE, params, radius, radius, 0.0, 0.0, None,
        segmentation, labels, seg_method)
    return result[-1]


class TestPixelCounts:
    """
    Tests for the per-source pixel counts returned by the batch
    drivers.
    """

    def test_interior_source(self, unit_data):
        """
        Test that a clean interior source has only n_pixels/valid counts.
        """
        data = unit_data
        fc = _sums_fcounts(data, (12.0, 12.0), 3.0)[0]
        assert fc[FLAG_COL_N_PIXELS] > 0
        assert fc[FLAG_COL_VALID] == fc[FLAG_COL_N_PIXELS]
        assert fc[FLAG_COL_MASKED] == 0
        assert fc[FLAG_COL_NONFINITE_DATA] == 0
        assert fc[FLAG_COL_NONFINITE_ERROR] == 0
        assert fc[FLAG_COL_SEG] == 0
        assert fc[FLAG_COL_UNCORRECTED] == 0
        assert fc[FLAG_COL_BBOX_CLIPPED] == 0

    def test_no_bbox_overlap(self, unit_data):
        """
        Test that a source with no bounding-box overlap has all-zero counts.
        """
        data = unit_data
        fc = _sums_fcounts(data, (100.0, 100.0), 3.0)[0]
        assert_array_equal(fc, 0)
        fc = _gather_fcounts(data, (100.0, 100.0), 3.0)[0]
        assert_array_equal(fc, 0)

    @pytest.mark.parametrize('use_exact', [0, 1])
    def test_masked_pixel_membership(self, unit_data, use_exact):
        """
        Test that only masked pixels with nonzero overlap fraction are
        counted.
        """
        data = unit_data

        # Pixel inside the aperture
        mask = np.zeros(UNIT_SHAPE, dtype=np.uint8)
        mask[12, 12] = 1
        fc = _sums_fcounts(data, (12.0, 12.0), 3.0, mask=mask,
                           use_exact=use_exact)[0]
        assert fc[FLAG_COL_MASKED] == 1
        assert fc[FLAG_COL_VALID] == fc[FLAG_COL_N_PIXELS] - 1

        # Pixel inside the bounding box but outside the aperture (bbox
        # corner)
        mask = np.zeros(UNIT_SHAPE, dtype=np.uint8)
        mask[9, 9] = 1
        fc = _sums_fcounts(data, (12.0, 12.0), 3.0, mask=mask,
                           use_exact=use_exact)[0]
        assert fc[FLAG_COL_MASKED] == 0
        assert fc[FLAG_COL_VALID] == fc[FLAG_COL_N_PIXELS]

    def test_mask_plane_bits(self, unit_data):
        """
        Test that mask-plane bit 1 counts as masked and bit 2 as non-finite
        data, with bit 1 taking precedence.
        """
        data = unit_data
        mask = np.zeros(UNIT_SHAPE, dtype=np.uint8)
        mask[12, 12] = 1  # input-masked
        mask[12, 13] = 2  # non-finite data
        mask[13, 12] = 3  # both; masked wins
        for func in (_sums_fcounts, _gather_fcounts):
            fc = func(data, (12.0, 12.0), 3.0, mask=mask)[0]
            assert fc[FLAG_COL_MASKED] == 2
            assert fc[FLAG_COL_NONFINITE_DATA] == 1
            assert fc[FLAG_COL_VALID] == fc[FLAG_COL_N_PIXELS] - 3

    def test_gather_matches_sums_center(self):
        """
        Test that the gather kernel flag counts match the photometry kernel
        with the "center" method.
        """
        rng = np.random.default_rng(1)
        data = rng.normal(size=UNIT_SHAPE)
        data[11, 12] = np.nan
        user_mask = np.zeros(UNIT_SHAPE, dtype=bool)
        user_mask[12, 13] = True

        xy = np.array([(12.0, 12.0), (0.5, 3.0), (24.0, 24.0)])

        # 2-bit plane: bit 1 = user mask, bit 2 = non-finite data
        plane = user_mask.astype(np.uint8)
        plane |= (~np.isfinite(data) & ~user_mask) * np.uint8(2)

        fc_sums = _sums_fcounts(data, xy, 3.0, mask=plane, use_exact=0,
                                subpixels=1)
        fc_gather = _gather_fcounts(data, xy, 3.0, mask=plane)
        # The gather kernel never reads error values
        cols = [FLAG_COL_N_PIXELS, FLAG_COL_MASKED, FLAG_COL_NONFINITE_DATA,
                FLAG_COL_SEG, FLAG_COL_UNCORRECTED, FLAG_COL_VALID,
                FLAG_COL_BBOX_CLIPPED]
        assert_array_equal(fc_gather[:, cols], fc_sums[:, cols])


class TestNonFiniteCounts:
    """
    Tests for the non-finite data and error indicator counts.
    """

    def test_nonfinite_data_unmasked(self, unit_data):
        """
        Test that unmasked non-finite data values are detected (and still
        contribute) in the photometry kernel.

        Unmasked non-finite contributions are detected from the accumulated
        sums as a 0/1 indicator.
        """
        data = unit_data
        data[12, 12] = np.nan
        data[12, 13] = np.inf
        fc = _sums_fcounts(data, (12.0, 12.0), 3.0)[0]
        assert fc[FLAG_COL_NONFINITE_DATA] == 1
        assert fc[FLAG_COL_VALID] == fc[FLAG_COL_N_PIXELS]

    def test_nonfinite_error(self, unit_data):
        """
        Test that non-finite error values among contributing pixels are
        counted.
        """
        data = unit_data
        error = np.ones(UNIT_SHAPE)
        error[12, 12] = np.nan
        fc = _sums_fcounts(data, (12.0, 12.0), 3.0, error=error)[0]
        assert fc[FLAG_COL_NONFINITE_ERROR] == 1

        # A masked pixel with non-finite error is not counted
        mask = np.zeros(UNIT_SHAPE, dtype=np.uint8)
        mask[12, 12] = 1
        fc = _sums_fcounts(data, (12.0, 12.0), 3.0, error=error, mask=mask)[0]
        assert fc[FLAG_COL_NONFINITE_ERROR] == 0


class TestBboxClipped:
    """
    Tests for the bounding-box-clipped candidate indicator.
    """

    @pytest.mark.parametrize('use_exact', [0, 1])
    def test_indicator(self, unit_data, use_exact):
        """
        Test the bbox-clipped indicator for interior and edge sources.
        """
        data = unit_data

        # Interior source
        fc = _sums_fcounts(data, (12.0, 12.0), 3.0, use_exact=use_exact)[0]
        assert fc[FLAG_COL_BBOX_CLIPPED] == 0

        # Aperture straddling the left edge
        fc = _sums_fcounts(data, (0.0, 12.0), 3.0, use_exact=use_exact)[0]
        assert fc[FLAG_COL_BBOX_CLIPPED] == 1
        assert fc[FLAG_COL_N_PIXELS] > 0

        # Clipped bbox whose "center"-method weights are all inside the data
        # (the caller resolves the precise outside-weight test)
        fc = _sums_fcounts(data, (0.2, 12.0), 0.8, use_exact=0,
                           subpixels=1)[0]
        assert fc[FLAG_COL_BBOX_CLIPPED] == 1
        assert fc[FLAG_COL_N_PIXELS] > 0

    def test_parity_with_mask(self, unit_data):
        """
        Test that n_pixels matches the nonzero in-data aperture-mask weights
        and that the bbox-clipped indicator matches the overlap slices, for
        randomized positions including edge and rounding cases.
        """
        rng = np.random.default_rng(0)
        data = unit_data
        n_src = 50
        xy = rng.uniform(-6.0, 30.0, size=(n_src, 2))
        # Include exact half-integer rounding cases
        xy[:5] = [(0.5, 0.5), (-0.5, 12.0), (24.5, 24.5), (0.0, 0.0),
                  (12.5, -0.5)]
        radius = 2.5

        for use_exact, method in ((1, 'exact'), (0, 'center')):
            fcs = _sums_fcounts(data, xy, radius, use_exact=use_exact,
                                subpixels=1)
            apertures = CircularAperture(xy, r=radius)
            masks = apertures.to_mask(method=method)
            for fc, apermask in zip(fcs, masks, strict=True):
                slc_large, slc_small = apermask.get_overlap_slices(UNIT_SHAPE)
                if slc_large is None:
                    assert_array_equal(fc, 0)
                    continue
                n_in = np.count_nonzero(apermask.data[slc_small])
                assert fc[FLAG_COL_N_PIXELS] == n_in
                full = (slice(0, apermask.data.shape[0]),
                        slice(0, apermask.data.shape[1]))
                assert fc[FLAG_COL_BBOX_CLIPPED] == int(slc_small != full)


class TestSegmentationCounts:
    """
    Tests for the segmentation neighbor and uncorrected pixel counts.
    """

    def test_seg_counts(self, unit_data):
        """
        Test the segmentation-affected pixel counts for the mask,
        source_only, and correct methods.
        """
        data = unit_data
        segm = np.zeros(UNIT_SHAPE, dtype=np.intp)
        segm[10:15, 10:15] = 1
        segm[12, 14] = 2  # neighbor pixel inside the aperture
        labels = np.array([1], dtype=np.intp)

        # Method 1 ('mask'): neighbor pixels are excluded
        fc = _sums_fcounts(data, (12.0, 12.0), 3.0, segmentation=segm,
                           labels=labels, seg_method=1)[0]
        assert fc[FLAG_COL_SEG] == 1
        assert fc[FLAG_COL_UNCORRECTED] == 0
        assert fc[FLAG_COL_VALID] == fc[FLAG_COL_N_PIXELS] - 1

        # Method 2 ('source_only'): background exclusions are not counted
        # as neighbor pixels
        fc = _sums_fcounts(data, (12.0, 12.0), 3.0, segmentation=segm,
                           labels=labels, seg_method=2)[0]
        assert fc[FLAG_COL_SEG] == 1

        # Method 3 ('correct'): the neighbor pixel is corrected (mirror
        # pixel is valid)
        fc = _sums_fcounts(data, (12.0, 12.0), 3.0, segmentation=segm,
                           labels=labels, seg_method=3)[0]
        assert fc[FLAG_COL_SEG] == 1
        assert fc[FLAG_COL_UNCORRECTED] == 0
        assert fc[FLAG_COL_VALID] == fc[FLAG_COL_N_PIXELS]

        # Method 3 with the mirror pixel also a neighbor: uncorrectable
        segm2 = segm.copy()
        segm2[12, 10] = 2  # mirror of (12, 14) across (12, 12)
        fc = _sums_fcounts(data, (12.0, 12.0), 3.0, segmentation=segm2,
                           labels=labels, seg_method=3)[0]
        assert fc[FLAG_COL_SEG] == 2
        assert fc[FLAG_COL_UNCORRECTED] == 2

        # Method 3 with a masked mirror pixel: uncorrectable
        mask = np.zeros(UNIT_SHAPE, dtype=np.uint8)
        mask[12, 10] = 1
        fc = _sums_fcounts(data, (12.0, 12.0), 3.0, mask=mask,
                           segmentation=segm, labels=labels, seg_method=3)[0]
        assert fc[FLAG_COL_SEG] == 1
        assert fc[FLAG_COL_UNCORRECTED] == 1


class TestSegMaskedCounts:
    """
    Tests for the masked neighbor-segment pixel counts.

    Masked pixels never reach the segmentation branch of the per-pixel
    loop, so they are counted in separate columns for callers that
    treat the mask and neighbor overlays independently.
    """

    @staticmethod
    def _inputs():
        """
        Return the data, mask, segmentation, and labels of a source with
        one unmasked and one masked neighbor-segment pixel.
        """
        data = np.ones((11, 11))
        segm = np.zeros((11, 11), dtype=np.intp)
        segm[4:7, 4:7] = 1
        segm[5, 8] = 2  # neighbor pixel, unmasked
        # Neighbor pixel that is masked. Its mirror across the center is
        # (5, 8), which is also a neighbor, so neither pixel can be
        # corrected by the 'correct' method.
        segm[5, 2] = 2
        mask = np.zeros((11, 11), dtype=np.uint8)
        mask[5, 2] = 1
        labels = np.array([1], dtype=np.intp)
        return data, mask, segm, labels

    def test_n_flag_cols(self):
        """
        Test the flag-count column layout.
        """
        assert FLAG_COL_SEG_MASKED == 8
        assert FLAG_COL_UNCORRECTED_MASKED == 9
        assert N_FLAG_COLS == 10

    @pytest.mark.parametrize('func', [_sums_fcounts, _gather_fcounts])
    @pytest.mark.parametrize(('seg_method', 'n_seg', 'n_uncorr',
                              'n_seg_masked', 'n_uncorr_masked'),
                             [(3, 1, 1, 1, 1),
                              (1, 1, 0, 1, 0),
                              (0, 0, 0, 0, 0)])
    def test_seg_masked_columns(self, func, seg_method, n_seg, n_uncorr,
                                n_seg_masked, n_uncorr_masked):
        """
        Test that masked neighbor-segment pixels are counted separately
        from the unmasked ones by both batch drivers.
        """
        data, mask, segm, labels = self._inputs()
        fc = func(data, (5.0, 5.0), 4.0, mask=mask, segmentation=segm,
                  labels=labels, seg_method=seg_method)[0]
        assert fc.shape == (N_FLAG_COLS,)
        assert fc[FLAG_COL_MASKED] == 1
        assert fc[FLAG_COL_SEG] == n_seg
        assert fc[FLAG_COL_UNCORRECTED] == n_uncorr
        assert fc[FLAG_COL_SEG_MASKED] == n_seg_masked
        assert fc[FLAG_COL_UNCORRECTED_MASKED] == n_uncorr_masked
