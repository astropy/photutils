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
from photutils.aperture.tests.conftest import UNIT_SHAPE


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


def _single_pixel_data():
    """
    Return data with a single bright pixel, giving a source whose
    covariance matrix is singular (zero spatial extent).
    """
    data = np.zeros(UNIT_SHAPE)
    data[12, 12] = 100.0
    return data


def _stats_with_injected_covariance(cov_xx, cov_yy, cov_xy):
    """
    Return a length-1 array ``ApertureStats`` whose central moments
    are overridden so that the source covariance matrix has the given
    entries (with unit total weight).

    This exercises the singular-covariance criterion directly for cases
    that are awkward to realize from pixel data (rank-1 degeneracy and a
    covariance matrix that is not positive semidefinite).
    """
    data = _single_pixel_data()
    aper = CircularAperture([(12.0, 12.0)], r=5.0)
    stats = ApertureStats(data, aper)
    moments = np.zeros((1, 4, 4))
    moments[0, 0, 0] = 1.0  # total weight (m00)
    moments[0, 2, 0] = cov_xx  # normalized -> covariance_xx
    moments[0, 0, 2] = cov_yy  # normalized -> covariance_yy
    moments[0, 1, 1] = cov_xy  # normalized -> covariance_xy
    stats.moments_central = moments
    return stats


class TestOverlapFlags:
    """
    Tests for the no_overlap, partial_overlap, and no_pixels flags.
    """

    @pytest.mark.usefixtures('maybe_mask_path')
    def test_no_flags(self, unit_data):
        """
        Test that a clean interior source has no flags set.
        """
        data = unit_data
        aper = CircularAperture((12, 12), r=3.0)
        stats = ApertureStats(data, aper)
        assert stats.flags == 0  # scalar aperture gives a scalar flag

    @pytest.mark.usefixtures('maybe_mask_path')
    def test_overlap_flags(self, unit_data):
        """
        Test the no_overlap, partial_overlap, and no_pixels flags.
        """
        data = unit_data
        aper = CircularAperture([(12.0, 12.0), (0.0, 12.0), (-50.0, 12.0)],
                                r=3.0)
        stats = ApertureStats(data, aper)
        assert stats.flags[0] == 0
        assert stats.flags[1] == APERTURE_FLAGS.PARTIAL_OVERLAP
        assert stats.flags[2] == (APERTURE_FLAGS.NO_OVERLAP
                                  | APERTURE_FLAGS.NO_PIXELS)

    @pytest.mark.usefixtures('maybe_mask_path')
    def test_no_pixels(self, unit_data):
        """
        Test that an empty "center" footprint sets no_pixels even when the
        default exact-method sum footprint is populated (the per-footprint
        flag bits are combined with OR).
        """
        data = unit_data
        # Nearest pixel centers are sqrt(0.5) ~ 0.707 away
        aper = CircularAperture((12.5, 12.5), r=0.4)
        stats = ApertureStats(data, aper)
        assert stats.flags == APERTURE_FLAGS.NO_PIXELS
        assert stats.sum > 0.0  # the exact-method sum is finite
        assert np.isnan(stats.mean)  # center statistics are undefined


class TestMaskedAndNonFiniteFlags:
    """
    Tests for the masked_pixels, all_masked, and non-finite flags.
    """

    @pytest.mark.usefixtures('maybe_mask_path')
    def test_masked_pixels(self, unit_data):
        """
        Test the masked_pixels and all_masked flags.
        """
        data = unit_data
        aper = CircularAperture((12, 12), r=3.0)

        mask = np.zeros(UNIT_SHAPE, dtype=bool)
        mask[12, 12] = True
        assert _stats_flags(data, aper,
                            mask=mask) == APERTURE_FLAGS.MASKED_PIXELS

        # Masked pixel inside the bounding box but outside the aperture
        mask = np.zeros(UNIT_SHAPE, dtype=bool)
        mask[9, 9] = True
        assert _stats_flags(data, aper, mask=mask) == 0

        # Fully masked aperture
        mask = np.zeros(UNIT_SHAPE, dtype=bool)
        mask[8:17, 8:17] = True
        assert _stats_flags(data, aper, mask=mask) == (
            APERTURE_FLAGS.MASKED_PIXELS | APERTURE_FLAGS.ALL_MASKED)

    @pytest.mark.usefixtures('maybe_mask_path')
    def test_sum_footprint_masked(self, unit_data, unit_mask):
        """
        Test that a masked boundary pixel touched only by the exact-method
        sum footprint (its center lies exactly on the aperture boundary)
        sets masked_pixels in the merged flags.
        """
        data = unit_data
        aper = CircularAperture((12, 12), r=3.0)
        mask = unit_mask
        mask[15, 12] = True  # center at distance exactly 3.0 (boundary)
        stats = ApertureStats(data, aper, mask=mask)
        assert stats.flags == APERTURE_FLAGS.MASKED_PIXELS
        # With sum_method='center' both footprints exclude the pixel
        stats = ApertureStats(data, aper, mask=mask, sum_method='center')
        assert stats.flags == 0

    @pytest.mark.usefixtures('maybe_mask_path')
    def test_non_finite_data(self, unit_mask):
        """
        Test the non_finite_data flag.

        In ApertureStats, non-finite data values are automatically masked,
        so they contribute to all_masked (but not to masked_pixels, which
        reflects only the input mask).
        """
        data = np.ones(UNIT_SHAPE)
        data[12, 12] = np.nan
        aper = CircularAperture((12, 12), r=3.0)
        assert _stats_flags(data, aper) == APERTURE_FLAGS.NON_FINITE_DATA

        # All-NaN aperture: auto-masked, so also all_masked
        data = np.full(UNIT_SHAPE, np.nan)
        assert _stats_flags(data, aper) == (APERTURE_FLAGS.NON_FINITE_DATA
                                            | APERTURE_FLAGS.ALL_MASKED)

        # A pixel that is both input-masked and non-finite counts only as
        # masked
        data = np.ones(UNIT_SHAPE)
        data[12, 12] = np.nan
        mask = unit_mask
        mask[12, 12] = True
        assert _stats_flags(data, aper,
                            mask=mask) == APERTURE_FLAGS.MASKED_PIXELS

    @pytest.mark.usefixtures('maybe_mask_path')
    def test_non_finite_error(self, unit_data):
        """
        Test the non_finite_error flag (evaluated on the sum footprint).
        """
        data = unit_data
        error = np.ones(UNIT_SHAPE)
        error[12, 12] = np.nan
        aper = CircularAperture((12, 12), r=3.0)
        stats = ApertureStats(data, aper, error=error)
        assert stats.flags == APERTURE_FLAGS.NON_FINITE_ERROR
        assert ApertureStats(data, aper).flags == 0


class TestSigmaClipFlags:
    """
    Tests for the sigma_clipped, all_clipped, and too_few_pixels flags.
    """

    @pytest.mark.usefixtures('maybe_mask_path')
    def test_sigma_clipped(self):
        """
        Test the sigma_clipped flag.
        """
        rng = np.random.default_rng(0)
        data = rng.normal(1.0, 0.1, size=UNIT_SHAPE)
        data[12, 12] = 1000.0  # outlier
        aper = CircularAperture((12, 12), r=3.0)
        sigclip = SigmaClip(sigma=3.0, maxiters=10)
        flags = _stats_flags(data, aper, sigma_clip=sigclip)
        assert flags == APERTURE_FLAGS.SIGMA_CLIPPED

        # Without clipping, the sigma_clipped flag is not set (the
        # unclipped outlier makes the source point-like, so the
        # always-evaluated singular_covariance bit may be set)
        assert not _stats_flags(data, aper) & APERTURE_FLAGS.SIGMA_CLIPPED

    def test_all_clipped(self, unit_data):
        """
        Test the all_clipped flag using a SigmaClip subclass that rejects
        every pixel (only reachable via the mask-based path).
        """
        class _ClipAll(SigmaClip):
            def __call__(self, data, **kwargs):  # noqa: ARG002
                return np.ma.masked_array(data,
                                          mask=np.ones(data.shape, dtype=bool))

        data = unit_data
        aper = CircularAperture((12, 12), r=3.0)
        # A callable cenfunc is not supported by the fast clipping kernel,
        # forcing the mask-based path
        sigclip = _ClipAll(cenfunc=np.ma.median)
        stats = ApertureStats(data, aper, sigma_clip=sigclip)
        assert stats.flags == (APERTURE_FLAGS.SIGMA_CLIPPED
                               | APERTURE_FLAGS.ALL_CLIPPED)
        assert np.isnan(stats.mean)

    @pytest.mark.usefixtures('maybe_mask_path')
    def test_too_few_pixels(self, unit_data):
        """
        Test the too_few_pixels flag with ddof.
        """
        data = unit_data
        # The r=1.1 center footprint contains exactly 5 pixels
        aper = CircularAperture((12, 12), r=1.1)

        stats = ApertureStats(data, aper, ddof=5)
        assert stats.flags == APERTURE_FLAGS.TOO_FEW_PIXELS
        assert np.isnan(stats.var)
        assert np.isnan(stats.std)

        assert _stats_flags(data, aper, ddof=1) == 0
        assert _stats_flags(data, aper, ddof=0) == 0


class TestSegmentationFlags:
    """
    Tests for the neighbor_pixels and uncorrected_pixels flags.
    """

    @pytest.mark.parametrize('mask_method', ['mask', 'source_only', 'correct'])
    @pytest.mark.usefixtures('maybe_mask_path')
    def test_neighbor_pixels(self, unit_data, mask_method):
        """
        Test the neighbor_pixels flag for all segmentation mask methods.
        """
        data = unit_data
        segm = np.zeros(UNIT_SHAPE, dtype=int)
        segm[10:15, 10:15] = 1
        segm[12, 14] = 2  # neighbor pixel inside the aperture
        aper = CircularAperture((12, 12), r=3.0)
        flags = _stats_flags(data, aper, segmentation_image=segm, labels=1,
                             mask_method=mask_method)
        assert flags == APERTURE_FLAGS.NEIGHBOR_PIXELS

    @pytest.mark.usefixtures('maybe_mask_path')
    def test_uncorrected_pixels(self, unit_data):
        """
        Test the uncorrected_pixels flag with mask_method='correct'.
        """
        data = unit_data
        segm = np.zeros(UNIT_SHAPE, dtype=int)
        segm[10:15, 10:15] = 1
        segm[12, 14] = 2
        segm[12, 10] = 2  # the mirror is also a neighbor: uncorrectable
        aper = CircularAperture((12, 12), r=3.0)
        flags = _stats_flags(data, aper, segmentation_image=segm, labels=1,
                             mask_method='correct')
        assert flags == (APERTURE_FLAGS.NEIGHBOR_PIXELS
                         | APERTURE_FLAGS.UNCORRECTED_PIXELS)


class TestMaskPathParity:
    """
    The mask-based code path must set the same flags as the batch
    driver.
    """

    def test_parity(self, unit_mask):
        """
        Test that the fast batch path and the mask-based path produce
        identical flags for a mix of conditions.
        """
        rng = np.random.default_rng(1)
        data = rng.normal(1.0, 0.1, size=UNIT_SHAPE)
        data[13, 12] = np.nan
        data[5, 5] = 100.0  # sigma-clip outlier
        error = np.ones(UNIT_SHAPE)
        error[10, 12] = np.inf
        mask = unit_mask
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


class TestFlagsAPI:
    """
    Tests for how the flags are exposed by ApertureStats.
    """

    def test_flags_in_table_and_properties(self, unit_data):
        """
        Test that flags appears in the properties list and the default
        to_table() columns.
        """
        data = unit_data
        aper = CircularAperture([(12, 12), (-50, 12)], r=3.0)
        stats = ApertureStats(data, aper)
        assert 'flags' in stats.properties
        assert 'sum_flags' not in stats.properties
        tbl = stats.to_table()
        assert 'flags' in tbl.colnames
        assert 'sum_flags' not in tbl.colnames
        expected = [0, APERTURE_FLAGS.NO_OVERLAP | APERTURE_FLAGS.NO_PIXELS]
        assert_array_equal(tbl['flags'], expected)

    def test_flags_deterministic(self, unit_data, unit_mask):
        """
        Test that flags is stable regardless of property access order
        and includes the shape bits without prior shape access.
        """
        data = unit_data.copy()
        data[12, 12] = -100.0  # non-positive net flux for source 1
        mask = unit_mask
        mask[18, 18] = True
        aper = CircularAperture([(12.0, 12.0), (18.0, 18.0)], r=3.0)
        stats = ApertureStats(data, aper, mask=mask)
        flags1 = stats.flags.copy()
        assert (flags1[0] & APERTURE_FLAGS.UNDEFINED_SHAPE) != 0
        assert (flags1[1] & APERTURE_FLAGS.MASKED_PIXELS) != 0
        _ = stats.semimajor_axis
        _ = stats.eccentricity
        _ = stats.sum
        assert_array_equal(stats.flags, flags1)

    def test_flags_slicing(self, unit_data, unit_mask):
        """
        Test that flags are sliced correctly by __getitem__.
        """
        data = unit_data
        mask = unit_mask
        mask[12, 12] = True
        aper = CircularAperture([(12, 12), (0.0, 12.0)], r=3.0)
        stats = ApertureStats(data, aper, mask=mask)
        flags = stats.flags  # evaluate before slicing
        assert stats[0].flags == flags[0]
        assert stats[1].flags == flags[1]

        # Also test slicing before evaluation
        stats2 = ApertureStats(data, aper, mask=mask)
        assert stats2[1].flags == flags[1]

    def test_decode_flags(self, unit_data, unit_mask):
        """
        Test the decode_flags convenience method.
        """
        data = unit_data
        mask = unit_mask
        mask[12, 12] = True
        aper = CircularAperture([(12.0, 12.0), (0.0, 12.0)], r=3.0)
        stats = ApertureStats(data, aper, mask=mask)
        decoded = stats.decode_flags()
        assert decoded == [['masked_pixels'], ['partial_overlap']]

        decoded = stats.decode_flags(return_bit_values=True)
        assert decoded == [[APERTURE_FLAGS.MASKED_PIXELS],
                           [APERTURE_FLAGS.PARTIAL_OVERLAP]]

        # Scalar ApertureStats also returns a list of lists
        stats = ApertureStats(data, CircularAperture((0.0, 12.0), r=3.0))
        assert stats.decode_flags() == [['partial_overlap']]

    def test_decode_flags_no_column_keyword(self, unit_data):
        """
        Test that decode_flags no longer accepts a column keyword.
        """
        data = unit_data
        aper = CircularAperture((12.0, 12.0), r=3.0)
        stats = ApertureStats(data, aper)
        match = 'unexpected keyword'
        with pytest.raises(TypeError, match=match):
            stats.decode_flags(column='flags')
        assert stats.decode_flags() == [[]]

    def test_flags_docstring(self):
        """
        Test that the flags docstring placeholder was substituted.
        """
        docstring = ApertureStats.flags.__doc__
        assert '<flag_descriptions>' not in docstring
        assert "**1** (``'no_overlap'``)" in docstring


class TestSingularCovariance:
    """
    Tests for the singular_covariance flag, which is always evaluated
    by the flags property.
    """

    @pytest.mark.usefixtures('maybe_mask_path')
    def test_set_without_shape_access(self):
        """
        Test that the singular_covariance bit is set without any prior
        covariance-derived property access.
        """
        data = _single_pixel_data()
        aper = CircularAperture((12.0, 12.0), r=5.0)
        stats = ApertureStats(data, aper)
        assert (stats.flags & APERTURE_FLAGS.SINGULAR_COVARIANCE) != 0
        assert 'singular_covariance' in stats.decode_flags()[0]

    @pytest.mark.usefixtures('maybe_mask_path')
    def test_array_and_guards(self):
        """
        Test that the singular_covariance bit is set for an array of sources
        and that the covariance computation is guarded against sources with
        no overlap (undefined moments).
        """
        data = _single_pixel_data()
        data[6, 6] = 100.0
        yy, xx = np.mgrid[0:25, 0:25]
        data = data + 50.0 * np.exp(-((xx - 18)**2 + (yy - 18)**2)
                                    / (2 * 2.5**2))
        aper = CircularAperture([(6.0, 6.0), (18.0, 18.0), (-50.0, 12.0)],
                                r=4.0)
        stats = ApertureStats(data, aper)
        flags = stats.flags
        covar_flag = APERTURE_FLAGS.SINGULAR_COVARIANCE
        assert (flags[0] & covar_flag) != 0  # singular point source
        assert (flags[1] & covar_flag) == 0  # extended source
        assert (flags[2] & covar_flag) == 0  # no-overlap: not flagged singular
        assert (flags[2] & APERTURE_FLAGS.NO_OVERLAP) != 0

    @pytest.mark.usefixtures('maybe_mask_path')
    def test_extended_source_not_flagged(self):
        """
        Test that a well-resolved source with a non-singular covariance
        matrix is not flagged as singular.
        """
        yy, xx = np.mgrid[0:25, 0:25]
        data = 100.0 * np.exp(-((xx - 12)**2 + (yy - 12)**2) / (2 * 3.0**2))
        aper = CircularAperture((12.0, 12.0), r=6.0)
        stats = ApertureStats(data, aper)
        assert (stats.flags & APERTURE_FLAGS.SINGULAR_COVARIANCE) == 0

    @pytest.mark.usefixtures('maybe_mask_path')
    def test_in_default_table(self):
        """
        Test that the singular_covariance bit is reflected in the default
        to_table() output.
        """
        data = _single_pixel_data()
        aper = CircularAperture((12.0, 12.0), r=5.0)
        stats = ApertureStats(data, aper)
        tbl = stats.to_table()
        assert (tbl['flags'][0] & APERTURE_FLAGS.SINGULAR_COVARIANCE) != 0

    @pytest.mark.usefixtures('maybe_mask_path')
    def test_to_table_flags_only(self):
        """
        Test that requesting only the 'flags' column reports the
        singular_covariance bit.
        """
        data = _single_pixel_data()
        aper = CircularAperture((12.0, 12.0), r=5.0)
        stats = ApertureStats(data, aper)
        tbl = stats.to_table(columns=['flags'])
        assert (tbl['flags'][0] & APERTURE_FLAGS.SINGULAR_COVARIANCE) != 0

    @pytest.mark.usefixtures('maybe_mask_path')
    def test_in_properties(self):
        """
        Test that the singular_covariance bit is reflected in the properties
        list even if no shape properties are accessed.
        """
        data = _single_pixel_data()
        aper = CircularAperture((12.0, 12.0), r=5.0)
        stats = ApertureStats(data, aper)
        assert 'flags' in stats.properties

    @pytest.mark.usefixtures('maybe_mask_path')
    def test_slicing(self):
        """
        Test that the singular_covariance bit is preserved when slicing an
        array of ApertureStats objects.
        """
        data = _single_pixel_data()
        data[6, 6] = 100.0
        aper = CircularAperture([(6.0, 6.0), (12.0, 12.0)], r=4.0)
        stats = ApertureStats(data, aper)
        covar_flag = APERTURE_FLAGS.SINGULAR_COVARIANCE
        assert (stats[0].flags & covar_flag) != 0
        assert (stats[1].flags & covar_flag) != 0

    def test_rank1_degeneracy(self):
        """
        Test that a rank-1 degenerate source is flagged as singular.

        A rank-1 degenerate source has one unresolved axis and one extended
        axis, so its covariance matrix has one tiny eigenvalue. The
        singularity criterion is that the minor-axis variance is below a
        floor of ``1/12`` (the variance of a uniform distribution over a
        unit pixel). This is a more robust criterion than the determinant
        test, which can be fooled by a rank-1 source with a large major-axis
        variance.
        """
        # Minor-axis variance 0.05 < 1/12, major-axis variance 0.5
        stats = _stats_with_injected_covariance(cov_xx=0.5, cov_yy=0.05,
                                                cov_xy=0.0)
        delta = 1.0 / 12
        assert delta**2 < 0.5 * 0.05  # determinant test alone would miss it
        assert delta > 0.05  # minor-axis variance below the floor
        assert stats._singular_covariance_mask[0]
        assert (stats.flags[0] & APERTURE_FLAGS.SINGULAR_COVARIANCE) != 0

    def test_not_positive_semidefinite(self):
        """
        Test that a covariance matrix that is not positive semidefinite is
        flagged as singular.
        """
        stats = _stats_with_injected_covariance(cov_xx=1.0, cov_yy=1.0,
                                                cov_xy=2.0)
        assert (1.0 * 1.0 - 2.0**2) < 0  # negative determinant
        assert stats._singular_covariance_mask[0]
        assert (stats.flags[0] & APERTURE_FLAGS.SINGULAR_COVARIANCE) != 0


class TestUndefinedShape:
    """
    Tests for the undefined_shape flag, which marks sources whose net
    flux is not positive and is always evaluated by the flags property.
    """

    @pytest.mark.usefixtures('maybe_mask_path')
    def test_set_without_moment_access(self):
        """
        Test that the undefined_shape bit is set without any prior
        moment-derived property access.
        """
        data = np.zeros(UNIT_SHAPE)
        aper = CircularAperture((12.0, 12.0), r=5.0)
        stats = ApertureStats(data, aper)
        assert (stats.flags & APERTURE_FLAGS.UNDEFINED_SHAPE) != 0
        assert 'undefined_shape' in stats.decode_flags()[0]
        assert np.all(np.isnan(stats.centroid))

    @pytest.mark.usefixtures('maybe_mask_path')
    @pytest.mark.parametrize('value', [0.0, -1.0])
    def test_non_positive_flux(self, value):
        """
        Test that both zero and negative net flux set the
        undefined_shape bit.
        """
        data = np.full(UNIT_SHAPE, value)
        aper = CircularAperture((12.0, 12.0), r=5.0)
        stats = ApertureStats(data, aper)
        assert (stats.flags & APERTURE_FLAGS.UNDEFINED_SHAPE) != 0

    @pytest.mark.usefixtures('maybe_mask_path')
    def test_positive_flux_not_flagged(self):
        """
        Test that a source with positive net flux is not flagged.
        """
        data = np.ones(UNIT_SHAPE)
        aper = CircularAperture((12.0, 12.0), r=5.0)
        stats = ApertureStats(data, aper)
        assert (stats.flags & APERTURE_FLAGS.UNDEFINED_SHAPE) == 0

    @pytest.mark.usefixtures('maybe_mask_path')
    def test_array_and_guards(self):
        """
        Test the undefined_shape bit for an array of sources, and that
        sources with no valid pixels (no overlap or fully masked) are
        not flagged (they are reported by the overlap and masking
        bits).
        """
        data = np.zeros(UNIT_SHAPE)
        data[16:21, 16:21] = 50.0  # positive-flux source at (18, 18)
        mask = np.zeros(UNIT_SHAPE, dtype=bool)
        mask[0:12, 0:12] = True  # fully mask the third aperture
        aper = CircularAperture([(6.0, 18.0), (18.0, 18.0), (6.0, 6.0),
                                 (-50.0, 12.0)], r=4.0)
        stats = ApertureStats(data, aper, mask=mask)
        flags = stats.flags
        shape_flag = APERTURE_FLAGS.UNDEFINED_SHAPE
        assert (flags[0] & shape_flag) != 0  # zero-flux source
        assert (flags[1] & shape_flag) == 0  # positive-flux source
        assert (flags[2] & shape_flag) == 0  # fully masked: not flagged
        assert (flags[2] & APERTURE_FLAGS.ALL_MASKED) != 0
        assert (flags[3] & shape_flag) == 0  # no overlap: not flagged
        assert (flags[3] & APERTURE_FLAGS.NO_OVERLAP) != 0

    @pytest.mark.usefixtures('maybe_mask_path')
    def test_in_default_table(self):
        """
        Test that the undefined_shape bit is reflected in the default
        to_table() output, which evaluates the moment-derived columns.
        """
        data = np.zeros(UNIT_SHAPE)
        aper = CircularAperture((12.0, 12.0), r=5.0)
        stats = ApertureStats(data, aper)
        tbl = stats.to_table()
        assert (tbl['flags'][0] & APERTURE_FLAGS.UNDEFINED_SHAPE) != 0

    @pytest.mark.usefixtures('maybe_mask_path')
    def test_to_table_flags_only(self):
        """
        Test that requesting only the 'flags' column reports the
        undefined_shape bit.
        """
        data = np.zeros(UNIT_SHAPE)
        aper = CircularAperture((12.0, 12.0), r=5.0)
        stats = ApertureStats(data, aper)
        tbl = stats.to_table(columns=['flags'])
        assert (tbl['flags'][0] & APERTURE_FLAGS.UNDEFINED_SHAPE) != 0

    @pytest.mark.usefixtures('maybe_mask_path')
    def test_slicing(self):
        """
        Test that the undefined_shape bit is preserved when slicing an
        ApertureStats object.
        """
        data = np.zeros(UNIT_SHAPE)
        data[18, 18] = 100.0
        aper = CircularAperture([(6.0, 6.0), (18.0, 18.0)], r=4.0)
        stats = ApertureStats(data, aper)
        shape_flag = APERTURE_FLAGS.UNDEFINED_SHAPE
        assert (stats[0].flags & shape_flag) != 0
        assert (stats[1].flags & shape_flag) == 0

    @pytest.mark.usefixtures('maybe_mask_path')
    def test_with_singular_covariance(self):
        """
        Test that a single negative pixel sets both the undefined_shape
        and singular_covariance bits once the covariance is computed.
        """
        data = np.zeros(UNIT_SHAPE)
        data[12, 12] = -100.0
        aper = CircularAperture((12.0, 12.0), r=5.0)
        stats = ApertureStats(data, aper)
        flags = stats.flags
        assert (flags & APERTURE_FLAGS.UNDEFINED_SHAPE) != 0
        assert (flags & APERTURE_FLAGS.SINGULAR_COVARIANCE) != 0
