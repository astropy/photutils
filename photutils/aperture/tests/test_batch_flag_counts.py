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
                                                  FLAG_COL_NONFINITE_DATA,
                                                  FLAG_COL_NONFINITE_ERROR,
                                                  FLAG_COL_NPIX, FLAG_COL_SEG,
                                                  FLAG_COL_UNCORRECTED,
                                                  FLAG_COL_VALID, SHAPE_CIRCLE,
                                                  batch_aperture_sums)
from photutils.aperture._batch_stats import batch_aperture_gather

SHAPE = (25, 25)


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
        positions, SHAPE_CIRCLE, params, radius, radius, use_exact,
        subpixels, segmentation, labels, seg_method)
    return result[-1]


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
        SHAPE_CIRCLE, params, radius, radius, None, segmentation,
        labels, seg_method)
    return result[-1]


def test_interior_source():
    """
    Test that a clean interior source has only npix/valid counts.
    """
    data = np.ones(SHAPE)
    fc = _sums_fcounts(data, (12.0, 12.0), 3.0)[0]
    assert fc[FLAG_COL_NPIX] > 0
    assert fc[FLAG_COL_VALID] == fc[FLAG_COL_NPIX]
    assert fc[FLAG_COL_MASKED] == 0
    assert fc[FLAG_COL_NONFINITE_DATA] == 0
    assert fc[FLAG_COL_NONFINITE_ERROR] == 0
    assert fc[FLAG_COL_SEG] == 0
    assert fc[FLAG_COL_UNCORRECTED] == 0
    assert fc[FLAG_COL_BBOX_CLIPPED] == 0


def test_no_bbox_overlap():
    """
    Test that a source with no bounding-box overlap has all-zero counts.
    """
    data = np.ones(SHAPE)
    fc = _sums_fcounts(data, (100.0, 100.0), 3.0)[0]
    assert_array_equal(fc, 0)
    fc = _gather_fcounts(data, (100.0, 100.0), 3.0)[0]
    assert_array_equal(fc, 0)


@pytest.mark.parametrize('use_exact', [0, 1])
def test_masked_pixel_membership(use_exact):
    """
    Test that only masked pixels with nonzero overlap fraction are
    counted.
    """
    data = np.ones(SHAPE)

    # Pixel inside the aperture
    mask = np.zeros(SHAPE, dtype=np.uint8)
    mask[12, 12] = 1
    fc = _sums_fcounts(data, (12.0, 12.0), 3.0, mask=mask,
                       use_exact=use_exact)[0]
    assert fc[FLAG_COL_MASKED] == 1
    assert fc[FLAG_COL_VALID] == fc[FLAG_COL_NPIX] - 1

    # Pixel inside the bounding box but outside the aperture (bbox
    # corner)
    mask = np.zeros(SHAPE, dtype=np.uint8)
    mask[9, 9] = 1
    fc = _sums_fcounts(data, (12.0, 12.0), 3.0, mask=mask,
                       use_exact=use_exact)[0]
    assert fc[FLAG_COL_MASKED] == 0
    assert fc[FLAG_COL_VALID] == fc[FLAG_COL_NPIX]


def test_mask_plane_bits():
    """
    Test that mask-plane bit 1 counts as masked and bit 2 as non-finite
    data, with bit 1 taking precedence.
    """
    data = np.ones(SHAPE)
    mask = np.zeros(SHAPE, dtype=np.uint8)
    mask[12, 12] = 1  # input-masked
    mask[12, 13] = 2  # non-finite data
    mask[13, 12] = 3  # both; masked wins
    for func in (_sums_fcounts, _gather_fcounts):
        fc = func(data, (12.0, 12.0), 3.0, mask=mask)[0]
        assert fc[FLAG_COL_MASKED] == 2
        assert fc[FLAG_COL_NONFINITE_DATA] == 1
        assert fc[FLAG_COL_VALID] == fc[FLAG_COL_NPIX] - 3


def test_nonfinite_data_unmasked():
    """
    Test that unmasked non-finite data values are detected (and still
    contribute) in the photometry kernel.

    Unmasked non-finite contributions are detected from the accumulated
    sums as a 0/1 indicator.
    """
    data = np.ones(SHAPE)
    data[12, 12] = np.nan
    data[12, 13] = np.inf
    fc = _sums_fcounts(data, (12.0, 12.0), 3.0)[0]
    assert fc[FLAG_COL_NONFINITE_DATA] == 1
    assert fc[FLAG_COL_VALID] == fc[FLAG_COL_NPIX]


def test_nonfinite_error():
    """
    Test that non-finite error values among contributing pixels are
    counted.
    """
    data = np.ones(SHAPE)
    error = np.ones(SHAPE)
    error[12, 12] = np.nan
    fc = _sums_fcounts(data, (12.0, 12.0), 3.0, error=error)[0]
    assert fc[FLAG_COL_NONFINITE_ERROR] == 1

    # A masked pixel with non-finite error is not counted
    mask = np.zeros(SHAPE, dtype=np.uint8)
    mask[12, 12] = 1
    fc = _sums_fcounts(data, (12.0, 12.0), 3.0, error=error, mask=mask)[0]
    assert fc[FLAG_COL_NONFINITE_ERROR] == 0


@pytest.mark.parametrize('use_exact', [0, 1])
def test_bbox_clipped_indicator(use_exact):
    """
    Test the bbox-clipped indicator for interior and edge sources.
    """
    data = np.ones(SHAPE)

    # Interior source
    fc = _sums_fcounts(data, (12.0, 12.0), 3.0, use_exact=use_exact)[0]
    assert fc[FLAG_COL_BBOX_CLIPPED] == 0

    # Aperture straddling the left edge
    fc = _sums_fcounts(data, (0.0, 12.0), 3.0, use_exact=use_exact)[0]
    assert fc[FLAG_COL_BBOX_CLIPPED] == 1
    assert fc[FLAG_COL_NPIX] > 0

    # Clipped bbox whose "center"-method weights are all inside the data
    # (the caller resolves the precise outside-weight test)
    fc = _sums_fcounts(data, (0.2, 12.0), 0.8, use_exact=0,
                       subpixels=1)[0]
    assert fc[FLAG_COL_BBOX_CLIPPED] == 1
    assert fc[FLAG_COL_NPIX] > 0


def test_bbox_clipped_parity_with_aperture_mask():
    """
    Test that npix matches the nonzero in-data aperture-mask weights
    and that the bbox-clipped indicator matches the overlap slices, for
    randomized positions including edge and rounding cases.
    """
    rng = np.random.default_rng(0)
    data = np.ones(SHAPE)
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
            slc_large, slc_small = apermask.get_overlap_slices(SHAPE)
            if slc_large is None:
                assert_array_equal(fc, 0)
                continue
            n_in = np.count_nonzero(apermask.data[slc_small])
            assert fc[FLAG_COL_NPIX] == n_in
            full = (slice(0, apermask.data.shape[0]),
                    slice(0, apermask.data.shape[1]))
            assert fc[FLAG_COL_BBOX_CLIPPED] == int(slc_small != full)


def test_seg_counts():
    """
    Test the segmentation-affected pixel counts for the mask,
    source_only, and correct methods.
    """
    data = np.ones(SHAPE)
    segm = np.zeros(SHAPE, dtype=np.intp)
    segm[10:15, 10:15] = 1
    segm[12, 14] = 2  # neighbor pixel inside the aperture
    labels = np.array([1], dtype=np.intp)

    # Method 1 ('mask'): neighbor pixels are excluded
    fc = _sums_fcounts(data, (12.0, 12.0), 3.0, segmentation=segm,
                       labels=labels, seg_method=1)[0]
    assert fc[FLAG_COL_SEG] == 1
    assert fc[FLAG_COL_UNCORRECTED] == 0
    assert fc[FLAG_COL_VALID] == fc[FLAG_COL_NPIX] - 1

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
    assert fc[FLAG_COL_VALID] == fc[FLAG_COL_NPIX]

    # Method 3 with the mirror pixel also a neighbor: uncorrectable
    segm2 = segm.copy()
    segm2[12, 10] = 2  # mirror of (12, 14) across (12, 12)
    fc = _sums_fcounts(data, (12.0, 12.0), 3.0, segmentation=segm2,
                       labels=labels, seg_method=3)[0]
    assert fc[FLAG_COL_SEG] == 2
    assert fc[FLAG_COL_UNCORRECTED] == 2

    # Method 3 with a masked mirror pixel: uncorrectable
    mask = np.zeros(SHAPE, dtype=np.uint8)
    mask[12, 10] = 1
    fc = _sums_fcounts(data, (12.0, 12.0), 3.0, mask=mask,
                       segmentation=segm, labels=labels, seg_method=3)[0]
    assert fc[FLAG_COL_SEG] == 1
    assert fc[FLAG_COL_UNCORRECTED] == 1


def test_gather_matches_sums_center():
    """
    Test that the gather kernel flag counts match the photometry kernel
    with the "center" method.
    """
    rng = np.random.default_rng(1)
    data = rng.normal(size=SHAPE)
    data[11, 12] = np.nan
    user_mask = np.zeros(SHAPE, dtype=bool)
    user_mask[12, 13] = True

    xy = np.array([(12.0, 12.0), (0.5, 3.0), (24.0, 24.0)])

    # 2-bit plane: bit 1 = user mask, bit 2 = non-finite data
    plane = user_mask.astype(np.uint8)
    plane |= (~np.isfinite(data) & ~user_mask) * np.uint8(2)

    fc_sums = _sums_fcounts(data, xy, 3.0, mask=plane, use_exact=0,
                            subpixels=1)
    fc_gather = _gather_fcounts(data, xy, 3.0, mask=plane)
    # The gather kernel never reads error values
    cols = [FLAG_COL_NPIX, FLAG_COL_MASKED, FLAG_COL_NONFINITE_DATA,
            FLAG_COL_SEG, FLAG_COL_UNCORRECTED, FLAG_COL_VALID,
            FLAG_COL_BBOX_CLIPPED]
    assert_array_equal(fc_gather[:, cols], fc_sums[:, cols])
