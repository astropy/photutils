# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the peakfinder module.
"""

import astropy.units as u
import numpy as np
import pytest
from astropy.tests.helper import assert_quantity_allclose
from numpy.testing import assert_array_equal, assert_equal

from photutils.centroids import centroid_com
from photutils.datasets import make_gwcs, make_wcs
from photutils.detection import find_peaks
from photutils.utils._optional_deps import HAS_GWCS
from photutils.utils.exceptions import NoDetectionsWarning


class TestFindPeaks:
    def test_box_size(self, data):
        """
        Test with box_size.
        """
        tbl = find_peaks(data, 0.1, box_size=3)
        assert tbl['id'][0] == 1
        assert len(tbl) == 25
        columns = ['id', 'x_peak', 'y_peak', 'peak_value']
        assert all(column in tbl.colnames for column in columns)
        assert np.min(tbl['x_peak']) > 0
        assert np.max(tbl['x_peak']) < 101
        assert np.min(tbl['y_peak']) > 0
        assert np.max(tbl['y_peak']) < 101
        assert np.max(tbl['peak_value']) < 13.2

        # Test with units
        unit = u.Jy
        tbl2 = find_peaks(data << unit, 0.1 << unit, box_size=3)
        columns = ['id', 'x_peak', 'y_peak']
        for column in columns:
            assert_equal(tbl[column], tbl2[column])
        col = 'peak_value'
        assert tbl2[col].unit == unit
        assert_equal(tbl[col], tbl2[col].value)

    def test_footprint(self, data):
        """
        Test with footprint.
        """
        tbl0 = find_peaks(data, 0.1, box_size=3)
        tbl1 = find_peaks(data, 0.1, footprint=np.ones((3, 3)))
        assert_array_equal(tbl0, tbl1)

    def test_mask(self, data):
        """
        Test with mask.
        """
        mask = np.zeros(data.shape, dtype=bool)
        mask[0:50, :] = True
        tbl0 = find_peaks(data, 0.1, box_size=3)
        tbl1 = find_peaks(data, 0.1, box_size=3, mask=mask)
        assert len(tbl1) < len(tbl0)

    def test_mask_int(self, data):
        """
        Test that an integer mask gives the same result as a boolean
        mask.
        """
        bool_mask = np.zeros(data.shape, dtype=bool)
        bool_mask[0:50, :] = True
        int_mask = bool_mask.astype(int)

        tbl_bool = find_peaks(data, 0.1, box_size=3, mask=bool_mask)
        tbl_int = find_peaks(data, 0.1, box_size=3, mask=int_mask)
        assert_array_equal(tbl_bool, tbl_int)

    def test_maskshape(self, data):
        """
        Test if mask shape doesn't match data shape.
        """
        match = 'data and mask must have the same shape'
        with pytest.raises(ValueError, match=match):
            find_peaks(data, 0.1, mask=np.ones((5, 5)))

    def test_thresholdshape(self, data):
        """
        Test if threshold shape doesn't match data shape.
        """
        match = 'threshold array must have the same shape as the input data'
        with pytest.raises(ValueError, match=match):
            find_peaks(data, np.ones((2, 2)))

    def test_n_peaks(self, data):
        """
        Test n_peaks.
        """
        tbl = find_peaks(data, 0.1, box_size=3, n_peaks=1)
        assert len(tbl) == 1

    def test_border_width(self, data):
        """
        Test border exclusion.
        """
        tbl0 = find_peaks(data, 0.1, box_size=3)
        tbl1 = find_peaks(data, 0.1, box_size=3, border_width=0)
        tbl2 = find_peaks(data, 0.1, box_size=3, border_width=(0, 0))
        assert len(tbl0) == len(tbl1)
        assert len(tbl1) == len(tbl2)

        tbl3 = find_peaks(data, 0.1, box_size=3, border_width=25)
        tbl4 = find_peaks(data, 0.1, box_size=3, border_width=(25, 25))
        assert len(tbl3) == len(tbl4)
        assert len(tbl3) < len(tbl0)

        tbl0 = find_peaks(data, 0.1, box_size=3, border_width=(34, 0))
        tbl1 = find_peaks(data, 0.1, box_size=3, border_width=(0, 36))
        assert np.min(tbl0['y_peak']) >= 34
        assert np.min(tbl1['x_peak']) >= 36

        match = 'border_width must be >= 0'
        with pytest.raises(ValueError, match=match):
            find_peaks(data, 0.1, box_size=3, border_width=-1)
        match = 'border_width must have integer values'
        with pytest.raises(ValueError, match=match):
            find_peaks(data, 0.1, box_size=3, border_width=3.1)

    def test_border_width_excludes_all(self, data):
        """
        Test that a border_width encompassing the entire image returns
        None with a NoDetectionsWarning.
        """
        match = 'No local peaks were found'
        with pytest.warns(NoDetectionsWarning, match=match):
            tbl = find_peaks(data, 0.1, box_size=3, border_width=100)
        assert tbl is None

    def test_box_size_int(self, data):
        """
        Test noninteger box_size.
        """
        tbl1 = find_peaks(data, 0.1, box_size=5.0)
        tbl2 = find_peaks(data, 0.1, box_size=5.5)
        assert_array_equal(tbl1, tbl2)

    def test_centroid_func_callable(self, data):
        """
        Test that centroid_func is callable.
        """
        match = 'centroid_func must be a callable object'
        with pytest.raises(TypeError, match=match):
            find_peaks(data, 0.1, box_size=2, centroid_func=True)

    def test_centroid_func_with_error(self, data):
        """
        Test find_peaks with a centroid_func and an error array.
        """
        error = np.ones_like(data) * 0.1
        tbl = find_peaks(data, 0.1, box_size=3, centroid_func=centroid_com,
                         error=error)
        assert 'x_centroid' in tbl.colnames
        assert 'y_centroid' in tbl.colnames
        assert len(tbl) > 0

    def test_centroid_func_with_footprint(self, data):
        """
        Test find_peaks with a footprint and centroid_func.

        Even-sized footprint dimensions should be rounded up to odd for
        the centroid box_size.
        """
        footprint = np.ones((4, 4), dtype=bool)
        tbl = find_peaks(data, 0.1, footprint=footprint,
                         centroid_func=centroid_com)
        assert 'x_centroid' in tbl.colnames
        assert 'y_centroid' in tbl.colnames
        assert len(tbl) > 0

    def test_error_without_centroid_func(self, data):
        """
        Test that error is silently ignored when centroid_func is None.
        """
        error = np.ones_like(data) * 0.1
        tbl1 = find_peaks(data, 0.1, box_size=3)
        tbl2 = find_peaks(data, 0.1, box_size=3, error=error)
        assert_array_equal(tbl1, tbl2)

    def test_wcs(self, data):
        """
        Test with astropy WCS.
        """
        columns = ['skycoord_peak', 'skycoord_centroid']

        fits_wcs = make_wcs(data.shape)
        tbl = find_peaks(data, 1, wcs=fits_wcs, centroid_func=centroid_com)
        for column in columns:
            assert column in tbl.colnames
        assert tbl.colnames == ['id', 'x_peak', 'y_peak', 'skycoord_peak',
                                'peak_value', 'x_centroid', 'y_centroid',
                                'skycoord_centroid']

    @pytest.mark.skipif(not HAS_GWCS, reason='gwcs is required')
    def test_gwcs(self, data):
        """
        Test with gwcs.
        """
        columns = ['skycoord_peak', 'skycoord_centroid']

        gwcs_obj = make_gwcs(data.shape)
        tbl = find_peaks(data, 1, wcs=gwcs_obj, centroid_func=centroid_com)
        for column in columns:
            assert column in tbl.colnames

    @pytest.mark.skipif(not HAS_GWCS, reason='gwcs is required')
    def test_wcs_values(self, data):
        """
        Test that WCS and GWCS give the same sky coordinates.
        """
        fits_wcs = make_wcs(data.shape)
        gwcs_obj = make_gwcs(data.shape)
        tbl1 = find_peaks(data, 1, wcs=fits_wcs, centroid_func=centroid_com)
        tbl2 = find_peaks(data, 1, wcs=gwcs_obj, centroid_func=centroid_com)
        columns = ['skycoord_peak', 'skycoord_centroid']
        for column in columns:
            assert_quantity_allclose(tbl1[column].ra, tbl2[column].ra)
            assert_quantity_allclose(tbl1[column].dec, tbl2[column].dec)

    def test_constant_array(self):
        """
        Test for empty output table when data is constant.
        """
        data = np.ones((10, 10))
        match = 'Input data is constant'
        with pytest.warns(NoDetectionsWarning, match=match):
            tbl = find_peaks(data, 0.0)
        assert tbl is None

    def test_no_peaks(self, data):
        """
        Tests for when no peaks are found.
        """
        fits_wcs = make_wcs(data.shape)

        match = 'No local peaks were found'
        with pytest.warns(NoDetectionsWarning, match=match):
            tbl = find_peaks(data, 10000)
        assert tbl is None

        with pytest.warns(NoDetectionsWarning, match=match):
            tbl = find_peaks(data, 100000, centroid_func=centroid_com)
        assert tbl is None

        with pytest.warns(NoDetectionsWarning, match=match):
            tbl = find_peaks(data, 100000, wcs=fits_wcs)
        assert tbl is None

        with pytest.warns(NoDetectionsWarning, match=match):
            tbl = find_peaks(data, 100000, wcs=fits_wcs,
                             centroid_func=centroid_com)
        assert tbl is None

    def test_data_nans(self, data):
        """
        Test that data with NaNs does not issue Runtime warning.
        """
        data = np.copy(data)
        data[50:, :] = np.nan
        find_peaks(data, 0.1)

    def test_data_not_mutated(self, data):
        """
        Test that input data is not mutated by find_peaks.
        """
        data_copy = data.copy()
        find_peaks(data, 0.1, box_size=3)
        assert_equal(data, data_copy)

    def test_data_not_mutated_with_mask(self, data):
        """
        Test that input data is not mutated when a mask is used.
        """
        data_copy = data.copy()
        mask = np.zeros(data.shape, dtype=bool)
        mask[0:50] = True
        find_peaks(data, 0.1, box_size=3, mask=mask)
        assert_equal(data, data_copy)

    @pytest.mark.parametrize('box_size', [3, 5, 7, 11])
    def test_box_size_min_separation(self, box_size):
        """
        Test that box_size imposes a minimum separation of ``box_size //
        2 + 1`` pixels between detected peaks.
        """
        min_sep = box_size // 2 + 1
        size = 10 * box_size
        img = np.zeros((size, size))

        # Place two peaks exactly at the minimum allowed separation
        cy = size // 2
        cx1 = size // 2
        cx2 = cx1 + min_sep
        img[cy, cx1] = 10.0
        img[cy, cx2] = 10.0

        tbl = find_peaks(img, 1.0, box_size=box_size)
        assert len(tbl) == 2

    @pytest.mark.parametrize('box_size', [3, 5, 7, 11])
    def test_box_size_below_min_separation(self, box_size):
        """
        Test that peaks separated by less than ``box_size // 2 + 1``
        pixels are merged (only the brighter one survives).
        """
        min_sep = box_size // 2 + 1
        size = 10 * box_size
        img = np.zeros((size, size))

        # Place two peaks one pixel closer than the minimum separation;
        # only the brighter peak should survive
        cy = size // 2
        cx1 = size // 2
        cx2 = cx1 + min_sep - 1
        img[cy, cx1] = 10.0
        img[cy, cx2] = 8.0

        tbl = find_peaks(img, 1.0, box_size=box_size)
        assert len(tbl) == 1
        assert tbl['peak_value'][0] == 10.0

    def test_min_separation(self, data):
        """
        Test that min_separation enforces minimum Euclidean distance.
        """
        tbl0 = find_peaks(data, 0.1, box_size=3)
        tbl1 = find_peaks(data, 0.1, box_size=3, min_separation=10)
        assert len(tbl1) <= len(tbl0)

        # Check that all pairs of peaks are at least min_separation
        # apart
        if len(tbl1) > 1:
            x = np.array(tbl1['x_peak'], dtype=float)
            y = np.array(tbl1['y_peak'], dtype=float)
            for i in range(len(tbl1)):
                for j in range(i + 1, len(tbl1)):
                    dist = np.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2)
                    assert dist > 10

    def test_min_separation_two_peaks(self):
        """
        Test min_separation with two peaks at known separation.
        """
        data = np.zeros((100, 100))
        data[50, 30] = 10.0
        data[50, 50] = 8.0

        # Separation is 20 pixels; min_separation=15 should keep both
        tbl = find_peaks(data, 1.0, box_size=3, min_separation=15)
        assert len(tbl) == 2

        # min_separation=25 should keep only the brightest
        tbl = find_peaks(data, 1.0, box_size=3, min_separation=25)
        assert len(tbl) == 1
        assert tbl['peak_value'][0] == 10.0

    def test_min_separation_plateau(self):
        """
        Test that min_separation treats plateaus identically to an
        equivalent circular footprint (all equal-valued plateau pixels
        are local maxima).
        """
        data = np.zeros((50, 50))
        data[20:30, 20:30] = 10.0  # 10x10 plateau (diagonal ~12.7 px)

        for radius in (5, 15):
            idx = np.arange(-radius, radius + 1)
            xx, yy = np.meshgrid(idx, idx)
            fp = np.array((xx**2 + yy**2) <= radius**2, dtype=int)
            tbl_ref = find_peaks(data, 1.0, footprint=fp)
            tbl_fast = find_peaks(data, 1.0, min_separation=radius)

            assert len(tbl_ref) == len(tbl_fast)
            assert_array_equal(tbl_ref['x_peak'], tbl_fast['x_peak'])
            assert_array_equal(tbl_ref['y_peak'], tbl_fast['y_peak'])

    def test_min_separation_with_units(self):
        """
        Test min_separation with Quantity data.
        """
        unit = u.Jy
        data = np.zeros((100, 100))
        data[50, 30] = 10.0
        data[50, 50] = 8.0

        tbl = find_peaks(data << unit, 1.0 << unit, box_size=3,
                         min_separation=25)
        assert len(tbl) == 1
        assert tbl['peak_value'][0].value == 10.0
        assert tbl['peak_value'][0].unit == unit

    def test_min_separation_with_npeaks(self):
        """
        Test that min_separation and n_peaks work together.
        """
        data = np.zeros((100, 100))
        data[20, 20] = 10.0
        data[20, 60] = 8.0
        data[60, 20] = 6.0
        data[60, 60] = 4.0

        # All peaks are well-separated;
        # n_peaks=2 should keep brightest 2
        tbl = find_peaks(data, 1.0, min_separation=5, n_peaks=2)
        assert len(tbl) == 2
        assert tbl['peak_value'][0] == 10.0
        assert tbl['peak_value'][1] == 8.0

    def test_min_separation_negative(self, data):
        """
        Test that negative min_separation raises ValueError.
        """
        match = 'min_separation must be >= 0'
        with pytest.raises(ValueError, match=match):
            find_peaks(data, 0.1, min_separation=-1)

    def test_min_separation_zero(self, data):
        """
        Test that min_separation=0 gives the same result as None.
        """
        tbl0 = find_peaks(data, 0.1, box_size=3)
        tbl1 = find_peaks(data, 0.1, box_size=3, min_separation=0)
        assert_array_equal(tbl0, tbl1)

    def test_min_separation_with_footprint(self, data):
        """
        Test that min_separation takes priority over footprint.
        """
        footprint = np.ones((3, 3))
        tbl = find_peaks(data, 0.1, footprint=footprint, min_separation=10)
        assert len(tbl) > 0

        # Check minimum separation is enforced
        if len(tbl) > 1:
            x = np.array(tbl['x_peak'], dtype=float)
            y = np.array(tbl['y_peak'], dtype=float)
            for i in range(len(tbl)):
                for j in range(i + 1, len(tbl)):
                    dist = np.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2)
                    assert dist > 10

    def test_min_separation_matches_circular_footprint(self):
        """
        Test that min_separation produces the same peaks as an
        equivalent circular footprint passed to maximum_filter.
        """
        rng = np.random.default_rng(42)
        data = rng.standard_normal((200, 200))
        data[50, 50] = 20.0
        data[120, 130] = 18.0
        data[30, 170] = 15.0
        threshold = 3.0

        for radius in (5, 10, 25, 50):
            # Reference: actual circular footprint (slow but correct)
            idx = np.arange(-radius, radius + 1)
            xx, yy = np.meshgrid(idx, idx)
            fp = np.array((xx**2 + yy**2) <= radius**2, dtype=int)
            tbl_ref = find_peaks(data, threshold, footprint=fp)

            tbl_fast = find_peaks(data, threshold, min_separation=radius)

            if tbl_ref is None:
                assert tbl_fast is None
            else:
                ref_xy = set(zip(tbl_ref['x_peak'].tolist(),
                                 tbl_ref['y_peak'].tolist(),
                                 strict=True))
                fast_xy = set(zip(tbl_fast['x_peak'].tolist(),
                                  tbl_fast['y_peak'].tolist(),
                                  strict=True))
                assert ref_xy == fast_xy

    def test_min_separation_rejects_non_maxima(self):
        """
        Test that min_separation rejects peaks that are not the true
        local maximum within the circular region.

        This test would fail with a greedy KD-tree-only approach that
        uses a small box_size for initial peak detection, because such
        an approach would keep faint peaks that are not the maximum
        within a circle of min_separation (due to non-peak pixels with
        higher values in the neighborhood).
        """
        data = np.zeros((100, 100))

        # Bright peak with a declining gradient
        data[50, 50] = 100.0
        for i in range(1, 30):
            data[50, 50 + i] = 100.0 - 2 * i  # 98, 96, ..., 42

        # Faint peak at (50, 85), which is 35 px from the bright peak.
        # The gradient pixel at (50, 65) = 100 - 2*15 = 70, which is
        # within radius 20 of (50, 85) and brighter (70 > 45).
        data[50, 85] = 45.0

        # With min_separation=20: (50, 85) is NOT the local max within a
        # circle of radius 20 because (50, 65)=70 > 45.
        tbl = find_peaks(data, 1.0, min_separation=20)
        assert len(tbl) == 1
        assert tbl['x_peak'][0] == 50
        assert tbl['y_peak'][0] == 50

    def test_min_separation_keeps_nearby_true_maxima(self):
        """
        Test that two equal-valued peaks within min_separation of each
        other are both retained, matching the circular footprint result.
        """
        radius = 12
        data = np.zeros((100, 100))

        # Two equal-valued peaks separated by less than min_separation
        # (dist = 11 px < radius = 12 px). Because they are equal,
        # each is tied for the max in its own circle, so both should
        # be detected (same behavior as maximum_filter with a circular
        # footprint).
        data[50, 40] = 10.0
        data[50, 51] = 10.0

        # Reference: actual circular footprint
        idx = np.arange(-radius, radius + 1)
        xx, yy = np.meshgrid(idx, idx)
        fp = np.array((xx**2 + yy**2) <= radius**2, dtype=int)
        tbl_ref = find_peaks(data, 1.0, footprint=fp)
        tbl_fast = find_peaks(data, 1.0, min_separation=radius)

        assert len(tbl_ref) == 2
        assert len(tbl_fast) == 2
        assert_array_equal(tbl_ref['x_peak'], tbl_fast['x_peak'])
        assert_array_equal(tbl_ref['y_peak'], tbl_fast['y_peak'])

    def test_nan_no_false_peaks(self):
        """
        Test that NaN pixels do not produce false peaks when the fill
        value (nanmin) happens to be a local maximum.
        """
        data = np.full((50, 50), 5.0)
        data[25, 25] = 10.0  # one real peak
        data[10, 10] = np.nan  # NaN pixel (fill value = 5.0 = background)
        data[10, 11] = np.nan

        tbl = find_peaks(data, 6.0, box_size=3)
        assert len(tbl) == 1
        assert tbl['x_peak'][0] == 25
        assert tbl['y_peak'][0] == 25

    def test_nan_adjacent_to_peak(self):
        """
        Test that NaN pixels adjacent to a real peak do not cause the
        peak to be lost or duplicated.
        """
        data = np.zeros((50, 50))
        data[25, 25] = 10.0
        data[25, 26] = np.nan
        data[24, 25] = np.nan

        tbl = find_peaks(data, 1.0, box_size=3)
        assert len(tbl) == 1
        assert tbl['x_peak'][0] == 25
        assert tbl['y_peak'][0] == 25

    def test_all_negative_data(self):
        """
        Test peak detection with all-negative data.

        Peaks near the border may be suppressed because maximum_filter
        uses cval=0.0, but interior peaks should be detected correctly.
        """
        data = np.full((50, 50), -10.0)
        data[25, 25] = -1.0  # brightest pixel, well inside border

        tbl = find_peaks(data, -5.0, box_size=3)
        assert tbl is not None
        assert len(tbl) == 1
        assert tbl['x_peak'][0] == 25
        assert tbl['y_peak'][0] == 25

    def test_all_negative_border_suppression(self):
        """
        Test that all-negative data near the border is suppressed by
        cval=0.0 in maximum_filter.
        """
        data = np.full((50, 50), -10.0)
        # Peak at border and one well inside
        data[0, 0] = -1.0
        data[25, 25] = -1.0

        # The peak at (0,0) is above the threshold but cval=0.0 means
        # the border region has a "virtual" maximum of 0.0, which is
        # greater than -1.0, so this pixel won't be detected as a peak.
        tbl = find_peaks(data, -5.0, box_size=3)
        assert tbl is not None
        # The border peak should not be among the results
        assert not any((tbl['x_peak'] == 0) & (tbl['y_peak'] == 0))
        # The interior peak should be detected
        assert any((tbl['x_peak'] == 25) & (tbl['y_peak'] == 25))

    def test_min_separation_with_centroid_func(self, data):
        """
        Test that min_separation works with centroid_func.

        The centroid box_size defaults to box_size (3) when
        min_separation is used.
        """
        tbl = find_peaks(data, 0.1, min_separation=10,
                         centroid_func=centroid_com)
        assert 'x_centroid' in tbl.colnames
        assert 'y_centroid' in tbl.colnames
        assert len(tbl) > 0
