# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the aperture photometry quality flags.
"""

import astropy.units as u
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from photutils.aperture import (APERTURE_FLAGS, CircularAnnulus,
                                CircularAperture, EllipticalAperture,
                                RectangularAperture, aperture_photometry)
from photutils.aperture.flags import _counts_to_flag_bits

SHAPE = (25, 25)

APERTURE_FACTORIES = [
    lambda xy: CircularAperture(xy, r=3.0),
    lambda xy: CircularAnnulus(xy, r_in=2.0, r_out=4.0),
    lambda xy: EllipticalAperture(xy, a=4.0, b=2.5, theta=0.5),
    lambda xy: RectangularAperture(xy, w=5.0, h=4.0, theta=0.3),
]

METHODS = [('exact', 5), ('center', 1), ('subpixel', 4)]


class _NoBatchCircularAperture(CircularAperture):
    """
    A CircularAperture subclass that does not opt in to the batch
    Cython driver, forcing the mask-based code path.
    """


def _flags(aperture, data, **kwargs):
    return aperture.photometry(data, **kwargs).flags


def test_no_flags():
    """
    Test that a clean interior source has no flags set.
    """
    data = np.ones(SHAPE)
    aper = CircularAperture((12, 12), r=3.0)
    assert_array_equal(_flags(aper, data), [0])


@pytest.mark.parametrize('factory', APERTURE_FACTORIES)
@pytest.mark.parametrize(('method', 'subpixels'), METHODS)
def test_no_overlap(factory, method, subpixels):
    """
    Test that fully off-image apertures set no_overlap and no_pixels.
    """
    data = np.ones(SHAPE)
    aper = factory([(-50.0, 12.0), (12.0, 12.0)])
    result = aper.photometry(data, method=method, subpixels=subpixels)
    expected = APERTURE_FLAGS.NO_OVERLAP | APERTURE_FLAGS.NO_PIXELS
    assert result.flags[0] == expected
    assert np.isnan(result.aperture_sum[0])
    assert result.flags[1] == 0


@pytest.mark.parametrize('factory', APERTURE_FACTORIES)
@pytest.mark.parametrize(('method', 'subpixels'), METHODS)
def test_partial_overlap(factory, method, subpixels):
    """
    Test that apertures straddling an edge set partial_overlap.
    """
    data = np.ones(SHAPE)
    aper = factory([(0.0, 12.0), (12.0, 24.5), (12.0, 12.0)])
    flags = _flags(aper, data, method=method, subpixels=subpixels)
    assert flags[0] == APERTURE_FLAGS.PARTIAL_OVERLAP
    assert flags[1] == APERTURE_FLAGS.PARTIAL_OVERLAP
    assert flags[2] == 0


def test_no_overlap_close_to_edge():
    """
    Test an aperture whose bounding box overlaps the data but whose
    nonzero-weight pixels are all outside (precise no_overlap).
    """
    data = np.ones(SHAPE)
    # bbox column 0 overlaps the data, but with the "center" method
    # the pixel center at x=0 (distance 1.2) is outside the circle
    aper = CircularAperture((-1.2, 12.0), r=0.8)
    result = aper.photometry(data, method='center')
    assert result.flags[0] == (APERTURE_FLAGS.NO_OVERLAP
                               | APERTURE_FLAGS.NO_PIXELS)
    # The sum is 0.0 (not NaN) here: the bounding box overlaps the
    # data, but no weighted pixel falls inside
    assert result.aperture_sum[0] == 0.0

    # With the exact method, the aperture area extends into the data
    result = aper.photometry(data, method='exact')
    assert result.flags[0] == APERTURE_FLAGS.PARTIAL_OVERLAP


def test_partial_overlap_clipped_bbox_only():
    """
    Test that a clipped bounding box alone does not set
    partial_overlap when all nonzero weights are inside the data.
    """
    data = np.ones(SHAPE)
    # bbox extends to column -1, but the pixel center at x=-1
    # (distance 1.2 > r=0.8) is outside the circle
    aper = CircularAperture((0.2, 12.0), r=0.8)
    assert_array_equal(_flags(aper, data, method='center'), [0])

    # The exact-method footprint does extend outside
    flags = _flags(aper, data, method='exact')
    assert_array_equal(flags, [APERTURE_FLAGS.PARTIAL_OVERLAP])


def test_no_pixels_tiny_aperture():
    """
    Test that a tiny aperture with the "center" method sets no_pixels.
    """
    data = np.ones(SHAPE)
    # Nearest pixel centers are sqrt(0.5) ~ 0.707 away
    aper = CircularAperture((12.5, 12.5), r=0.4)
    result = aper.photometry(data, method='center')
    assert result.flags[0] == APERTURE_FLAGS.NO_PIXELS
    assert result.aperture_sum[0] == 0.0

    # The exact method has a nonzero footprint
    assert_array_equal(_flags(aper, data, method='exact'), [0])


@pytest.mark.parametrize(('method', 'subpixels'), METHODS)
def test_masked_pixels(method, subpixels):
    """
    Test the masked_pixels and all_masked flags.
    """
    data = np.ones(SHAPE)

    # One masked pixel inside the aperture
    mask = np.zeros(SHAPE, dtype=bool)
    mask[12, 12] = True
    aper = CircularAperture((12, 12), r=3.0)
    flags = _flags(aper, data, mask=mask, method=method,
                   subpixels=subpixels)
    assert flags[0] == APERTURE_FLAGS.MASKED_PIXELS

    # Masked pixel inside the bounding box but outside the aperture
    mask = np.zeros(SHAPE, dtype=bool)
    mask[9, 9] = True
    flags = _flags(aper, data, mask=mask, method=method,
                   subpixels=subpixels)
    assert flags[0] == 0

    # Fully masked aperture
    mask = np.zeros(SHAPE, dtype=bool)
    mask[8:17, 8:17] = True
    flags = _flags(aper, data, mask=mask, method=method,
                   subpixels=subpixels)
    assert flags[0] == (APERTURE_FLAGS.MASKED_PIXELS
                        | APERTURE_FLAGS.ALL_MASKED)


def test_non_finite_data():
    """
    Test the non_finite_data flag.

    In photometry, unmasked non-finite values contribute to the sum
    (making it NaN). They do not set all_masked.
    """
    data = np.ones(SHAPE)
    data[12, 12] = np.nan
    aper = CircularAperture((12, 12), r=3.0)
    result = aper.photometry(data)
    assert result.flags[0] == APERTURE_FLAGS.NON_FINITE_DATA
    assert np.isnan(result.aperture_sum[0])

    # An all-NaN aperture is still non_finite_data only (the pixels
    # contribute, so all_masked is not set)
    data = np.full(SHAPE, np.nan)
    result = aper.photometry(data)
    assert result.flags[0] == APERTURE_FLAGS.NON_FINITE_DATA

    # A masked non-finite pixel counts only as masked
    data = np.ones(SHAPE)
    data[12, 12] = np.nan
    mask = np.zeros(SHAPE, dtype=bool)
    mask[12, 12] = True
    flags = _flags(aper, data, mask=mask)
    assert flags[0] == APERTURE_FLAGS.MASKED_PIXELS


def test_non_finite_error():
    """
    Test the non_finite_error flag.
    """
    data = np.ones(SHAPE)
    error = np.ones(SHAPE)
    error[12, 12] = np.inf
    aper = CircularAperture((12, 12), r=3.0)
    flags = _flags(aper, data, error=error)
    assert flags[0] == APERTURE_FLAGS.NON_FINITE_ERROR

    # Without an error array, the flag is never set
    assert_array_equal(_flags(aper, data), [0])


@pytest.mark.parametrize('mask_method', ['mask', 'source_only', 'correct'])
def test_neighbor_pixels(mask_method):
    """
    Test the neighbor_pixels flag for all segmentation mask methods.
    """
    data = np.ones(SHAPE)
    segm = np.zeros(SHAPE, dtype=int)
    segm[10:15, 10:15] = 1
    segm[12, 14] = 2  # neighbor pixel inside the aperture
    aper = CircularAperture((12, 12), r=3.0)
    flags = _flags(aper, data, segmentation_image=segm, labels=1,
                   mask_method=mask_method)
    assert flags[0] == APERTURE_FLAGS.NEIGHBOR_PIXELS

    # Without any neighbor pixels inside the aperture, no flag is set
    segm2 = np.zeros(SHAPE, dtype=int)
    segm2[10:15, 10:15] = 1
    flags = _flags(aper, data, segmentation_image=segm2, labels=1,
                   mask_method=mask_method)
    assert flags[0] == 0


def test_uncorrected_pixels():
    """
    Test the uncorrected_pixels flag with mask_method='correct'.
    """
    data = np.ones(SHAPE)
    segm = np.zeros(SHAPE, dtype=int)
    segm[10:15, 10:15] = 1
    segm[12, 14] = 2  # neighbor; its mirror (12, 10) is source pixel
    segm[12, 10] = 2  # make the mirror a neighbor too: uncorrectable
    aper = CircularAperture((12, 12), r=3.0)
    flags = _flags(aper, data, segmentation_image=segm, labels=1,
                   mask_method='correct')
    assert flags[0] == (APERTURE_FLAGS.NEIGHBOR_PIXELS
                        | APERTURE_FLAGS.UNCORRECTED_PIXELS)

    # A correctable neighbor does not set uncorrected_pixels
    segm[12, 10] = 0
    flags = _flags(aper, data, segmentation_image=segm, labels=1,
                   mask_method='correct')
    assert flags[0] == APERTURE_FLAGS.NEIGHBOR_PIXELS


def test_flag_combinations():
    """
    Test that multiple conditions combine bitwise.
    """
    data = np.ones(SHAPE)
    data[12, 6] = np.nan  # outside the r=3 aperture at (0, 12)
    mask = np.zeros(SHAPE, dtype=bool)
    mask[10, 1] = True
    aper = CircularAperture((0.0, 12.0), r=3.0)  # straddles left edge
    flags = _flags(aper, data, mask=mask)
    assert flags[0] == (APERTURE_FLAGS.PARTIAL_OVERLAP
                        | APERTURE_FLAGS.MASKED_PIXELS)


@pytest.mark.parametrize(('method', 'subpixels'), METHODS)
def test_mask_path_parity(method, subpixels):
    """
    Test that the mask-based code path produces flags identical to the
    batch Cython driver.
    """
    rng = np.random.default_rng(0)
    data = rng.normal(1.0, 0.1, size=SHAPE)
    data[13, 12] = np.nan
    error = np.ones(SHAPE)
    error[10, 12] = np.inf
    mask = np.zeros(SHAPE, dtype=bool)
    mask[12, 12] = True

    xy = [(12.0, 12.0), (0.0, 12.0), (-50.0, 12.0), (24.5, 24.5),
          (0.2, 12.0), (-1.2, 12.0)]
    batch_aper = CircularAperture(xy, r=3.0)
    nobatch_aper = _NoBatchCircularAperture(xy, r=3.0)

    kwargs = {'error': error, 'mask': mask, 'method': method,
              'subpixels': subpixels}
    result_batch = batch_aper.photometry(data, **kwargs)
    result_nobatch = nobatch_aper.photometry(data, **kwargs)
    assert_array_equal(result_batch.flags, result_nobatch.flags)
    assert_allclose(result_batch.aperture_sum,
                    result_nobatch.aperture_sum, rtol=1e-12,
                    equal_nan=True)


def test_mask_path_parity_segmentation():
    """
    Test batch/mask-path flag parity with segmentation masking.
    """
    data = np.ones(SHAPE)
    segm = np.zeros(SHAPE, dtype=int)
    segm[10:15, 10:15] = 1
    segm[12, 14] = 2
    segm[12, 10] = 2
    segm[3:7, 3:7] = 3
    xy = [(12.0, 12.0), (5.0, 5.0)]
    labels = [1, 3]

    for mask_method in ('mask', 'source_only', 'correct'):
        flags_batch = CircularAperture(xy, r=3.0).photometry(
            data, segmentation_image=segm, labels=labels,
            mask_method=mask_method).flags
        flags_nobatch = _NoBatchCircularAperture(xy, r=3.0).photometry(
            data, segmentation_image=segm, labels=labels,
            mask_method=mask_method).flags
        assert_array_equal(flags_batch, flags_nobatch)


def test_aperture_photometry_flags_column():
    """
    Test the flags column in the aperture_photometry table.
    """
    data = np.ones(SHAPE)
    xy = [(12.0, 12.0), (-50.0, 12.0), (0.0, 12.0)]
    aper = CircularAperture(xy, r=3.0)
    tbl = aperture_photometry(data, aper)
    expected = [0, APERTURE_FLAGS.NO_OVERLAP | APERTURE_FLAGS.NO_PIXELS,
                APERTURE_FLAGS.PARTIAL_OVERLAP]
    assert_array_equal(tbl['flags'], expected)
    assert 'decode_aperture_flags' in tbl['flags'].info.description

    # Multiple apertures use suffixed column names
    aper2 = CircularAperture(xy, r=4.0)
    tbl = aperture_photometry(data, [aper, aper2])
    assert_array_equal(tbl['flags_0'], expected)
    assert_array_equal(tbl['flags_1'], expected)


def test_flags_with_units():
    """
    Test that flags are also set for Quantity inputs.
    """
    data = np.ones(SHAPE) * u.Jy
    data[12, 12] = np.nan * u.Jy
    aper = CircularAperture((12, 12), r=3.0)
    result = aper.photometry(data)
    assert result.flags[0] == APERTURE_FLAGS.NON_FINITE_DATA
    tbl = aperture_photometry(data, aper)
    assert tbl['flags'][0] == APERTURE_FLAGS.NON_FINITE_DATA


def test_legacy_tuple_unpacking():
    """
    Test that the legacy 2-tuple unpacking of the photometry results
    is unchanged by the flags attribute.
    """
    data = np.ones(SHAPE)
    aper = CircularAperture((12, 12), r=3.0)
    result = aper.photometry(data)
    aperture_sum, _aperture_sum_err = result
    assert len(result) == 2
    assert_array_equal(result[0], aperture_sum)
    assert result.flags is not None


def test_counts_to_flag_bits_scalar_inputs():
    """
    Test that _counts_to_flag_bits accepts 1D count rows and scalar
    overlap/w_out values.
    """
    counts = np.zeros(8, dtype=int)
    flags = _counts_to_flag_bits(counts, np.False_, np.False_)
    assert flags[0] == (APERTURE_FLAGS.NO_OVERLAP
                        | APERTURE_FLAGS.NO_PIXELS)


def test_resolve_outside_weights_no_mask_overlap():
    """
    Test that _resolve_outside_weights returns True for a candidate
    whose aperture mask does not actually overlap the data.

    In practice, the caller only ever marks a source as a candidate when
    its aperture mask does overlap the data, so the mask's bounding box
    is never actually clipped away entirely. This directly exercises
    that defensive branch.
    """
    aper = CircularAperture((-5.0, 12.0), r=3.0)
    shape = (25, 25)
    mask = aper.to_mask(method='exact', subpixels=5)
    assert mask.get_overlap_slices(shape) == (None, None)

    candidates = np.array([True])
    w_out = aper._resolve_outside_weights(shape, method='exact',
                                          subpixels=5,
                                          candidates=candidates)
    assert w_out[0]
