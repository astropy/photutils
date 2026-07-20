# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the ApertureStats quality flags.
"""

from unittest.mock import patch

import numpy as np
import pytest
from astropy.stats import SigmaClip
from numpy.testing import assert_array_equal

from photutils.aperture import APERTURE_FLAGS, ApertureStats, CircularAperture

SHAPE = (25, 25)


def _stats_flags(data, aperture, **kwargs):
    return ApertureStats(data, aperture, **kwargs).flags


def _force_mask_path():
    """
    Return a context manager that disables the fast Cython batch path.
    """
    return patch.object(ApertureStats, '_batch_inputs',
                        property(lambda _self: None))


@pytest.fixture(params=[True, False], ids=['fast', 'maskpath'])
def maybe_mask_path(request):
    """
    Run a test with both the fast batch path and the mask-based path.
    """
    if request.param:
        yield
    else:
        with _force_mask_path():
            yield


@pytest.mark.usefixtures('maybe_mask_path')
def test_no_flags():
    """
    Test that a clean interior source has no flags set.
    """
    data = np.ones(SHAPE)
    aper = CircularAperture((12, 12), r=3.0)
    stats = ApertureStats(data, aper)
    assert stats.flags == 0  # scalar aperture gives a scalar flag


@pytest.mark.usefixtures('maybe_mask_path')
def test_overlap_flags():
    """
    Test the no_overlap, partial_overlap, and no_pixels flags.
    """
    data = np.ones(SHAPE)
    aper = CircularAperture([(12.0, 12.0), (0.0, 12.0), (-50.0, 12.0)],
                            r=3.0)
    stats = ApertureStats(data, aper)
    assert stats.flags[0] == 0
    assert stats.flags[1] == APERTURE_FLAGS.PARTIAL_OVERLAP
    assert stats.flags[2] == (APERTURE_FLAGS.NO_OVERLAP
                              | APERTURE_FLAGS.NO_PIXELS)


@pytest.mark.usefixtures('maybe_mask_path')
def test_no_pixels_center_footprint():
    """
    Test that an empty "center" footprint sets no_pixels even when the
    default exact-method sum footprint is populated (the per-footprint
    flag bits are combined with OR).
    """
    data = np.ones(SHAPE)
    # Nearest pixel centers are sqrt(0.5) ~ 0.707 away
    aper = CircularAperture((12.5, 12.5), r=0.4)
    stats = ApertureStats(data, aper)
    assert stats.flags == APERTURE_FLAGS.NO_PIXELS
    assert stats.sum > 0.0  # the exact-method sum is finite
    assert np.isnan(stats.mean)  # center statistics are undefined


@pytest.mark.usefixtures('maybe_mask_path')
def test_masked_pixels():
    """
    Test the masked_pixels and all_masked flags.
    """
    data = np.ones(SHAPE)
    aper = CircularAperture((12, 12), r=3.0)

    mask = np.zeros(SHAPE, dtype=bool)
    mask[12, 12] = True
    assert _stats_flags(data, aper,
                        mask=mask) == APERTURE_FLAGS.MASKED_PIXELS

    # Masked pixel inside the bounding box but outside the aperture
    mask = np.zeros(SHAPE, dtype=bool)
    mask[9, 9] = True
    assert _stats_flags(data, aper, mask=mask) == 0

    # Fully masked aperture
    mask = np.zeros(SHAPE, dtype=bool)
    mask[8:17, 8:17] = True
    assert _stats_flags(data, aper, mask=mask) == (
        APERTURE_FLAGS.MASKED_PIXELS | APERTURE_FLAGS.ALL_MASKED)


@pytest.mark.usefixtures('maybe_mask_path')
def test_masked_pixels_sum_footprint_only():
    """
    Test that a masked boundary pixel touched only by the exact-method
    sum footprint (its center lies exactly on the aperture boundary)
    sets masked_pixels in sum_flags but not in the value-statistics
    flags.
    """
    data = np.ones(SHAPE)
    aper = CircularAperture((12, 12), r=3.0)
    mask = np.zeros(SHAPE, dtype=bool)
    mask[15, 12] = True  # center at distance exactly 3.0 (boundary)
    stats = ApertureStats(data, aper, mask=mask)
    assert stats.sum_flags == APERTURE_FLAGS.MASKED_PIXELS
    # The pixel is excluded from the center (value-statistics) footprint
    assert stats.flags == 0
    stats = ApertureStats(data, aper, mask=mask, sum_method='center')
    assert stats.flags == 0
    assert stats.sum_flags == 0


@pytest.mark.usefixtures('maybe_mask_path')
def test_non_finite_data():
    """
    Test the non_finite_data flag.

    In ApertureStats, non-finite data values are automatically masked,
    so they contribute to all_masked (but not to masked_pixels, which
    reflects only the input mask).
    """
    data = np.ones(SHAPE)
    data[12, 12] = np.nan
    aper = CircularAperture((12, 12), r=3.0)
    assert _stats_flags(data, aper) == APERTURE_FLAGS.NON_FINITE_DATA

    # All-NaN aperture: auto-masked, so also all_masked
    data = np.full(SHAPE, np.nan)
    assert _stats_flags(data, aper) == (APERTURE_FLAGS.NON_FINITE_DATA
                                        | APERTURE_FLAGS.ALL_MASKED)

    # A pixel that is both input-masked and non-finite counts only as
    # masked
    data = np.ones(SHAPE)
    data[12, 12] = np.nan
    mask = np.zeros(SHAPE, dtype=bool)
    mask[12, 12] = True
    assert _stats_flags(data, aper,
                        mask=mask) == APERTURE_FLAGS.MASKED_PIXELS


@pytest.mark.usefixtures('maybe_mask_path')
def test_non_finite_error():
    """
    Test the non_finite_error flag (evaluated on the sum footprint).
    """
    data = np.ones(SHAPE)
    error = np.ones(SHAPE)
    error[12, 12] = np.nan
    aper = CircularAperture((12, 12), r=3.0)
    stats = ApertureStats(data, aper, error=error)
    # non_finite_error is a sum-footprint flag, not a value-statistics
    # flag, so it appears in sum_flags but not flags
    assert stats.sum_flags == APERTURE_FLAGS.NON_FINITE_ERROR
    assert stats.flags == 0
    assert ApertureStats(data, aper).sum_flags == 0


@pytest.mark.usefixtures('maybe_mask_path')
def test_sigma_clipped():
    """
    Test the sigma_clipped flag.
    """
    rng = np.random.default_rng(0)
    data = rng.normal(1.0, 0.1, size=SHAPE)
    data[12, 12] = 1000.0  # outlier
    aper = CircularAperture((12, 12), r=3.0)
    sigclip = SigmaClip(sigma=3.0, maxiters=10)
    flags = _stats_flags(data, aper, sigma_clip=sigclip)
    assert flags == APERTURE_FLAGS.SIGMA_CLIPPED

    # Sigma-clip flags apply only to the value statistics, not sum_flags
    stats = ApertureStats(data, aper, sigma_clip=sigclip)
    assert not stats.sum_flags & APERTURE_FLAGS.SIGMA_CLIPPED

    # Without clipping, no flag is set
    assert _stats_flags(data, aper) == 0


def test_all_clipped():
    """
    Test the all_clipped flag using a SigmaClip subclass that rejects
    every pixel (only reachable via the mask-based path).
    """
    class _ClipAll(SigmaClip):
        def __call__(self, data, **kwargs):  # noqa: ARG002
            return np.ma.masked_array(data,
                                      mask=np.ones(data.shape, dtype=bool))

    data = np.ones(SHAPE)
    aper = CircularAperture((12, 12), r=3.0)
    # A callable cenfunc is not supported by the fast clipping kernel,
    # forcing the mask-based path
    sigclip = _ClipAll(cenfunc=np.ma.median)
    stats = ApertureStats(data, aper, sigma_clip=sigclip)
    assert stats.flags == (APERTURE_FLAGS.SIGMA_CLIPPED
                           | APERTURE_FLAGS.ALL_CLIPPED)
    assert np.isnan(stats.mean)


@pytest.mark.usefixtures('maybe_mask_path')
def test_too_few_pixels():
    """
    Test the too_few_pixels flag with ddof.
    """
    data = np.ones(SHAPE)
    # The r=1.1 center footprint contains exactly 5 pixels
    aper = CircularAperture((12, 12), r=1.1)

    stats = ApertureStats(data, aper, ddof=5)
    assert stats.flags == APERTURE_FLAGS.TOO_FEW_PIXELS
    assert np.isnan(stats.var)
    assert np.isnan(stats.std)

    assert _stats_flags(data, aper, ddof=1) == 0
    assert _stats_flags(data, aper, ddof=0) == 0


@pytest.mark.parametrize('mask_method', ['mask', 'source_only', 'correct'])
@pytest.mark.usefixtures('maybe_mask_path')
def test_neighbor_pixels(mask_method):
    """
    Test the neighbor_pixels flag for all segmentation mask methods.
    """
    data = np.ones(SHAPE)
    segm = np.zeros(SHAPE, dtype=int)
    segm[10:15, 10:15] = 1
    segm[12, 14] = 2  # neighbor pixel inside the aperture
    aper = CircularAperture((12, 12), r=3.0)
    flags = _stats_flags(data, aper, segmentation_image=segm, labels=1,
                         mask_method=mask_method)
    assert flags == APERTURE_FLAGS.NEIGHBOR_PIXELS


@pytest.mark.usefixtures('maybe_mask_path')
def test_uncorrected_pixels():
    """
    Test the uncorrected_pixels flag with mask_method='correct'.
    """
    data = np.ones(SHAPE)
    segm = np.zeros(SHAPE, dtype=int)
    segm[10:15, 10:15] = 1
    segm[12, 14] = 2
    segm[12, 10] = 2  # the mirror is also a neighbor: uncorrectable
    aper = CircularAperture((12, 12), r=3.0)
    flags = _stats_flags(data, aper, segmentation_image=segm, labels=1,
                         mask_method='correct')
    assert flags == (APERTURE_FLAGS.NEIGHBOR_PIXELS
                     | APERTURE_FLAGS.UNCORRECTED_PIXELS)


def test_fast_mask_path_parity():
    """
    Test that the fast batch path and the mask-based path produce
    identical flags for a mix of conditions.
    """
    rng = np.random.default_rng(1)
    data = rng.normal(1.0, 0.1, size=SHAPE)
    data[13, 12] = np.nan
    data[5, 5] = 100.0  # sigma-clip outlier
    error = np.ones(SHAPE)
    error[10, 12] = np.inf
    mask = np.zeros(SHAPE, dtype=bool)
    mask[12, 12] = True

    xy = [(12.0, 12.0), (5.0, 5.0), (0.0, 12.0), (-50.0, 12.0),
          (24.5, 24.5), (0.2, 12.0), (12.5, 3.5)]
    aper = CircularAperture(xy, r=3.0)

    for kwargs in ({}, {'error': error, 'mask': mask},
                   {'sigma_clip': SigmaClip(sigma=3.0)},
                   {'sum_method': 'center'},
                   {'ddof': 1}):
        fast = ApertureStats(data, aper, **kwargs)
        assert fast._batch_inputs is not None
        with _force_mask_path():
            slow = ApertureStats(data, aper, **kwargs)
            assert slow._batch_inputs is None
            assert_array_equal(fast.flags, slow.flags)
            assert_array_equal(fast.sum_flags, slow.sum_flags)


def test_flags_in_table_and_properties():
    """
    Test that flags appears in the properties list and the default
    to_table() columns.
    """
    data = np.ones(SHAPE)
    aper = CircularAperture([(12, 12), (-50, 12)], r=3.0)
    stats = ApertureStats(data, aper)
    assert 'flags' in stats.properties
    assert 'sum_flags' in stats.properties
    tbl = stats.to_table()
    assert 'flags' in tbl.colnames
    assert 'sum_flags' in tbl.colnames
    expected = [0, APERTURE_FLAGS.NO_OVERLAP | APERTURE_FLAGS.NO_PIXELS]
    assert_array_equal(tbl['flags'], expected)
    assert_array_equal(tbl['sum_flags'], expected)


def test_flags_slicing():
    """
    Test that flags are sliced correctly by __getitem__.
    """
    data = np.ones(SHAPE)
    mask = np.zeros(SHAPE, dtype=bool)
    mask[12, 12] = True
    aper = CircularAperture([(12, 12), (0.0, 12.0)], r=3.0)
    stats = ApertureStats(data, aper, mask=mask)
    flags = stats.flags  # evaluate before slicing
    assert stats[0].flags == flags[0]
    assert stats[1].flags == flags[1]

    # Also test slicing before evaluation
    stats2 = ApertureStats(data, aper, mask=mask)
    assert stats2[1].flags == flags[1]


def test_decode_flags():
    """
    Test the decode_flags convenience method.
    """
    data = np.ones(SHAPE)
    mask = np.zeros(SHAPE, dtype=bool)
    mask[12, 12] = True
    aper = CircularAperture([(12.0, 12.0), (0.0, 12.0)], r=3.0)
    stats = ApertureStats(data, aper, mask=mask)
    decoded = stats.decode_flags()
    assert decoded == [['masked_pixels'], ['partial_overlap']]

    decoded = stats.decode_flags(return_bit_values=True)
    assert decoded == [[APERTURE_FLAGS.MASKED_PIXELS],
                       [APERTURE_FLAGS.PARTIAL_OVERLAP]]

    # Decode the sum flags via the column keyword
    decoded = stats.decode_flags(column='sum_flags')
    assert decoded == [['masked_pixels'], ['partial_overlap']]

    match = "column must be 'flags' or 'sum_flags'"
    with pytest.raises(ValueError, match=match):
        stats.decode_flags(column='invalid')

    # Scalar ApertureStats also returns a list of lists
    stats = ApertureStats(data, CircularAperture((0.0, 12.0), r=3.0))
    assert stats.decode_flags() == [['partial_overlap']]
    assert stats.decode_flags(column='sum_flags') == [['partial_overlap']]


def test_flags_docstring():
    """
    Test that the flags docstring placeholder was substituted.
    """
    for prop in (ApertureStats.flags, ApertureStats.sum_flags):
        docstring = prop.__doc__
        assert '<flag_descriptions>' not in docstring
        assert "**1** (``'no_overlap'``)" in docstring
