# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the peakfinder module.
"""

import astropy.units as u
import numpy as np
import pytest
from astropy.tests.helper import assert_quantity_allclose
from numpy.testing import assert_array_equal, assert_equal
from scipy.spatial import cKDTree

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

    def test_masked_pixels_do_not_suppress_peaks(self):
        """
        Test that a masked pixel brighter than a nearby true peak does
        not suppress that peak.
        """
        data = np.zeros((20, 20))
        data[10, 10] = 100.0  # bad pixel (masked)
        data[10, 12] = 50.0  # real peak, 2 pixels away
        mask = np.zeros(data.shape, dtype=bool)
        mask[10, 10] = True

        tbl = find_peaks(data, 1.0, box_size=5, mask=mask)
        assert len(tbl) == 1
        assert tbl['x_peak'][0] == 12
        assert tbl['y_peak'][0] == 10

    def test_masked_pixels_min_separation(self):
        """
        Test that masked pixels do not suppress nearby peaks with the
        fast circular (min_separation) peak detection.
        """
        data = np.zeros((40, 40))
        data[20, 20] = 100.0  # bad pixel (masked)
        data[20, 24] = 50.0  # real peak, 4 pixels away
        mask = np.zeros(data.shape, dtype=bool)
        mask[20, 20] = True

        tbl = find_peaks(data, 1.0, min_separation=6, mask=mask)
        assert len(tbl) == 1
        assert tbl['x_peak'][0] == 24
        assert tbl['y_peak'][0] == 20

    def test_mask_equivalent_to_nan(self):
        """
        Test that masking a pixel gives the same result as setting it
        to NaN.
        """
        rng = np.random.default_rng(0)
        data = rng.normal(0.0, 1.0, (50, 50))
        data[25, 25] = 100.0
        data[25, 27] = 50.0
        mask = np.zeros(data.shape, dtype=bool)
        mask[25, 25] = True

        data_nan = data.copy()
        data_nan[25, 25] = np.nan

        tbl_mask = find_peaks(data, 10.0, box_size=5, mask=mask)
        tbl_nan = find_peaks(data_nan, 10.0, box_size=5)
        assert_array_equal(tbl_mask, tbl_nan)

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

    def test_n_peaks_integral_float(self, data):
        """
        Test that an integral float n_peaks works like the integer.
        """
        tbl1 = find_peaks(data, 0.1, box_size=3, n_peaks=5)
        tbl2 = find_peaks(data, 0.1, box_size=3, n_peaks=5.0)
        assert_array_equal(tbl1, tbl2)

    @pytest.mark.parametrize('n_peaks', [0, -1, 5.5])
    def test_n_peaks_invalid(self, data, n_peaks):
        """
        Test that nonpositive or nonintegral n_peaks values raise a
        ValueError.
        """
        match = 'n_peaks must be'
        with pytest.raises(ValueError, match=match):
            find_peaks(data, 0.1, box_size=3, n_peaks=n_peaks)

    def test_n_peaks_bool(self, data):
        """
        Test that a boolean n_peaks raises a TypeError.
        """
        match = 'n_peaks must be an integer'
        with pytest.raises(TypeError, match=match):
            find_peaks(data, 0.1, box_size=3, n_peaks=True)

    def test_empty_data(self):
        """
        Test that empty input data raises a ValueError.
        """
        match = 'data must not be empty'
        with pytest.raises(ValueError, match=match):
            find_peaks(np.empty((0, 0)), 0.0)

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

        The error array is ignored because centroid_com does not accept
        an error keyword.
        """
        error = np.ones_like(data) * 0.1
        tbl = find_peaks(data, 0.1, box_size=3, centroid_func=centroid_com,
                         error=error)
        assert 'x_centroid' in tbl.colnames
        assert 'y_centroid' in tbl.colnames
        assert len(tbl) > 0

    def test_centroid_func_with_error_keyword(self, data):
        """
        Test that find_peaks passes the error array to a centroid
        function that accepts an error keyword.
        """
        received_errors = []

        def my_centroid(data, mask=None, error=None):
            received_errors.append(error)
            return centroid_com(data, mask=mask)

        error = np.ones_like(data) * 0.1
        tbl = find_peaks(data, 0.1, box_size=3, centroid_func=my_centroid,
                         error=error)
        assert 'x_centroid' in tbl.colnames
        assert len(received_errors) == len(tbl)
        assert all(err is not None for err in received_errors)

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

        # Place two peaks one pixel closer than the minimum separation.
        # Only the brighter peak should survive
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

        # Separation is 20 pixels, so min_separation=15 should keep both
        tbl = find_peaks(data, 1.0, box_size=3, min_separation=15)
        assert len(tbl) == 2

        # min_separation=25 should keep only the brightest
        tbl = find_peaks(data, 1.0, box_size=3, min_separation=25)
        assert len(tbl) == 1
        assert tbl['peak_value'][0] == 10.0

    @pytest.mark.parametrize('min_separation', [1, 1.5, 5, 15])
    def test_min_separation_plateau(self, min_separation):
        """
        Test that a plateau of connected equal-valued pixels gives one
        peak at its center, even if the plateau is larger than
        min_separation.
        """
        data = np.zeros((50, 50))
        data[20:31, 18:29] = 10.0  # 11x11 plateau

        tbl = find_peaks(data, 1.0, min_separation=min_separation)
        assert len(tbl) == 1
        assert tbl['x_peak'][0] == 23
        assert tbl['y_peak'][0] == 25

    def test_min_separation_plateau_tie(self):
        """
        Test that the first pixel in row-major order is kept when
        plateau pixels are equally close to the plateau centroid.
        """
        data = np.zeros((20, 20))
        data[10:12, 5:7] = 10.0  # 2x2 plateau

        tbl = find_peaks(data, 1.0, min_separation=3)
        assert len(tbl) == 1
        assert tbl['x_peak'][0] == 5
        assert tbl['y_peak'][0] == 10

    def test_min_separation_plateau_concave(self):
        """
        Test that the peak of a plateau is one of its pixels when the
        plateau centroid is not in the plateau.
        """
        data = np.zeros((30, 30))
        data[10, 10:21] = 10.0
        data[10:21, 10] = 10.0  # L-shaped plateau

        tbl = find_peaks(data, 1.0, min_separation=2)
        assert len(tbl) == 1
        assert data[tbl['y_peak'][0], tbl['x_peak'][0]] == 10.0

    def test_min_separation_plateau_exact_tie(self):
        """
        Test that the first pixel in row-major order is kept when
        plateau pixels are equally close to a plateau centroid that is
        not exactly represented as a float.
        """
        plateau = np.array([[1, 1, 0],
                            [0, 1, 0],
                            [1, 0, 0],
                            [1, 1, 0],
                            [1, 1, 1]])
        data = np.zeros((20, 20))
        data[5:10, 8:11] = 10.0 * plateau

        # the (2, 0) and (3, 1) plateau pixels are equally close to the
        # centroid at (y, x) = (7/3, 2/3)
        tbl = find_peaks(data, 1.0, min_separation=3)
        assert len(tbl) == 1
        assert tbl['x_peak'][0] == 8
        assert tbl['y_peak'][0] == 7

    def test_min_separation_plateau_mask(self):
        """
        Test that a mask splits a plateau into separate plateaus.
        """
        data = np.zeros((30, 30))
        data[10:15, 10:21] = 9.0
        mask = np.zeros(data.shape, dtype=bool)
        mask[10:15, 15] = True

        tbl = find_peaks(data, 1.0, min_separation=3, mask=mask)
        assert_array_equal(tbl['x_peak'], [12, 18])
        assert_array_equal(tbl['y_peak'], [12, 12])

        # the two plateaus have the same size
        tbl = find_peaks(data, 1.0, min_separation=30, mask=mask)
        assert_array_equal(tbl['x_peak'], [12])
        assert_array_equal(tbl['y_peak'], [12])

    def test_min_separation_plateau_threshold(self):
        """
        Test that only the plateau pixels above a threshold array are
        used to find the plateau peak.
        """
        data = np.zeros((30, 30))
        data[10:15, 10:21] = 9.0
        threshold = np.ones(data.shape)
        threshold[:, :14] = 100.0

        tbl = find_peaks(data, threshold, min_separation=3)
        assert_array_equal(tbl['x_peak'], [17])
        assert_array_equal(tbl['y_peak'], [12])

    def test_min_separation_plateau_border(self):
        """
        Test that only the plateau pixels inside the border are used to
        find the plateau peak.
        """
        data = np.zeros((30, 30))
        data[10:15, 10:21] = 9.0

        tbl = find_peaks(data, 1.0, min_separation=3, border_width=12)
        assert_array_equal(tbl['x_peak'], [14])
        assert_array_equal(tbl['y_peak'], [13])

    def test_min_separation_saturated_star(self):
        """
        Test that a saturated star gives one peak near its center.
        """
        yy, xx = np.mgrid[:101, :101]
        data = 5.0e5 * np.exp(-((xx - 50.3)**2 + (yy - 49.6)**2) / 18.0)
        data = np.minimum(data, 65535.0)
        assert np.count_nonzero(data == 65535.0) > 100

        tbl = find_peaks(data, 500.0, min_separation=4)
        assert len(tbl) == 1
        assert tbl['x_peak'][0] == 50
        assert tbl['y_peak'][0] == 50

    def test_min_separation_plateaus_separate(self):
        """
        Test that separate plateaus each give one peak.
        """
        data = np.zeros((60, 60))
        data[10:15, 10:15] = 10.0
        data[40:45, 30:35] = 10.0
        data[20:23, 40:43] = 8.0

        tbl = find_peaks(data, 1.0, min_separation=3)
        assert len(tbl) == 3
        assert_array_equal(tbl['x_peak'], [12, 41, 32])
        assert_array_equal(tbl['y_peak'], [12, 21, 42])

    def test_min_separation_plateau_preferred(self):
        """
        Test that the peak of a plateau is kept instead of a nearby
        equal-valued isolated pixel, regardless of their order in the
        image.
        """
        for y_pixel in (8, 32):
            data = np.zeros((41, 41))
            data[17:24, 17:24] = 10.0  # 7x7 plateau centered at (20, 20)
            data[y_pixel, 20] = 10.0  # not connected to the plateau

            tbl = find_peaks(data, 1.0, min_separation=15)
            assert len(tbl) == 1
            assert tbl['x_peak'][0] == 20
            assert tbl['y_peak'][0] == 20

            tbl = find_peaks(data, 1.0, min_separation=5)
            assert len(tbl) == 2

    def test_min_separation_subpixel(self):
        """
        Test that adjacent pixels are separate peaks for a
        min_separation smaller than 1 pixel.
        """
        data = np.zeros((20, 20))
        data[10, 5:8] = [10.0, 10.0, 7.0]

        tbl = find_peaks(data, 1.0, min_separation=0.5)
        assert len(tbl) == 3

    def test_min_separation_diagonal_plateau(self):
        """
        Test that equal-valued diagonal neighbors are one plateau only
        if they are within min_separation of each other.
        """
        data = np.zeros((20, 20))
        data[10, 10] = 10.0
        data[11, 11] = 10.0

        tbl = find_peaks(data, 1.0, min_separation=1.2)
        assert len(tbl) == 2

        tbl = find_peaks(data, 1.0, min_separation=1.5)
        assert len(tbl) == 1

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

        # All peaks are well-separated, so n_peaks=2 should keep the
        # brightest 2.
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

    @pytest.mark.parametrize('offset', [0.0, -30.0])
    @pytest.mark.parametrize('radius', [0.5, 1, 1.5, 2.05, 2.5, 4, 4.55, 5,
                                        7.3, 10, 12.5, 25, 50])
    def test_min_separation_matches_circular_footprint(self, radius,
                                                       offset):
        """
        Test that min_separation produces a subset of the peaks found
        by an equivalent circular footprint. The min_separation result
        is at least as strict, because equal-valued peaks are further
        pruned.

        The radii cover both the footprint filter used for small radii
        and the box filters used for large radii. The fractional radii
        are typical of the default star finder path (min_separation
        = 2.5 * fwhm). The negative offset tests the zero padding at
        the image borders.
        """
        rng = np.random.default_rng(42)
        data = rng.standard_normal((200, 200)) + offset
        data[50, 50] = 20.0
        data[120, 130] = 18.0
        data[30, 170] = 15.0
        threshold = 3.0 + offset

        # Reference uses the actual circular footprint (slow but correct)
        half = int(radius)
        idx = np.arange(-half, half + 1)
        xx, yy = np.meshgrid(idx, idx)
        fp = (xx**2 + yy**2) <= radius**2
        tbl_ref = find_peaks(data, threshold, footprint=fp)

        tbl_fast = find_peaks(data, threshold, min_separation=radius)

        # The min_separation result is a subset of the footprint
        # result, because equal-valued peaks may be removed.
        ref_xy = set(zip(tbl_ref['x_peak'].tolist(),
                         tbl_ref['y_peak'].tolist(),
                         strict=True))
        fast_xy = set(zip(tbl_fast['x_peak'].tolist(),
                          tbl_fast['y_peak'].tolist(),
                          strict=True))
        assert fast_xy <= ref_xy

        # All surviving peaks must satisfy min_separation
        if len(tbl_fast) > 1:
            x = np.array(tbl_fast['x_peak'], dtype=float)
            y = np.array(tbl_fast['y_peak'], dtype=float)
            for i in range(len(tbl_fast)):
                for j in range(i + 1, len(tbl_fast)):
                    dist = np.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2)
                    assert dist > radius

    def test_min_separation_diagonal_neighbors(self):
        """
        Test that diagonal neighbors, which are separated by more than
        1 pixel, are both peaks for ``min_separation=1``.
        """
        data = np.zeros((9, 9))
        data[3, 3] = 5.0
        data[4, 4] = 6.0
        tbl = find_peaks(data, 1.0, min_separation=1)
        assert len(tbl) == 2

        tbl = find_peaks(data, 1.0, min_separation=1.5)
        assert len(tbl) == 1
        assert tbl['peak_value'][0] == 6.0

    def test_min_separation_rejects_non_maxima(self):
        """
        Test that min_separation rejects peaks that are not the true
        local maximum within the circular region.

        This test would fail with a greedy KD-tree-only approach that
        uses a small box_size for initial peak detection, because such
        an approach would keep faint peaks that are not the maximum
        within a circle of min_separation (due to nonpeak pixels with
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

    def test_min_separation_equal_peaks_pruned(self):
        """
        Test that two equal-valued peaks within min_separation are
        pruned by the greedy spacing step, keeping only the first one
        in row-major order.
        """
        radius = 12
        data = np.zeros((100, 100))

        # Two equal-valued peaks separated by 11 px < radius = 12 px
        data[50, 40] = 10.0
        data[50, 51] = 10.0

        tbl = find_peaks(data, 1.0, min_separation=radius)
        assert len(tbl) == 1
        assert tbl['x_peak'][0] == 40
        assert tbl['y_peak'][0] == 50

    @pytest.mark.parametrize(('radius', 'n_peaks'), [(4.9, 2), (5, 1)])
    def test_min_separation_equal_peaks_boundary(self, radius, n_peaks):
        """
        Test that equal-valued peaks separated by exactly
        min_separation are too close, as for peaks with different
        values.
        """
        data = np.zeros((40, 40))
        data[20, 10] = 10.0
        data[23, 14] = 10.0  # 5 pixels from the first peak

        tbl = find_peaks(data, 1.0, min_separation=radius)
        assert len(tbl) == n_peaks

    @pytest.mark.parametrize('dtype', [np.uint16, np.int32, np.float32])
    def test_min_separation_equal_peaks_dtype(self, dtype):
        """
        Test that equal-valued peaks are reduced for integer and
        unsigned integer data.
        """
        data = np.zeros((40, 40), dtype=dtype)
        data[10, [10, 13]] = 500
        data[20:23, 30] = 100
        data[30, [10, 14]] = 7

        tbl = find_peaks(data, 1, min_separation=5)
        assert_array_equal(tbl['x_peak'], [10, 30, 10])
        assert_array_equal(tbl['y_peak'], [10, 21, 30])
        assert_array_equal(tbl['peak_value'], [500, 100, 7])

    def test_min_separation_equal_peaks_chain(self):
        """
        Test that the greedy spacing step keeps a peak that is too
        close only to a peak that was itself removed.
        """
        data = np.zeros((20, 40))
        data[10, [10, 14, 18]] = 10.0

        tbl = find_peaks(data, 1.0, min_separation=5)
        assert_array_equal(tbl['x_peak'], [10, 18])

    @pytest.mark.parametrize('radius', [3, 8, 20])
    def test_min_separation_many_pairs(self, radius, monkeypatch):
        """
        Test that the two ways of finding the equal-valued peaks that
        are too close give the same peaks.

        A sparse image of ones has many pairs of equal-valued peaks.
        """
        from photutils.detection import peakfinder

        rng = np.random.default_rng(0)
        data = (rng.random((150, 150)) < 0.1).astype(float)

        monkeypatch.setattr(peakfinder, '_MAX_PAIRS_PER_PEAK', np.inf)
        tbl1 = find_peaks(data, 0.5, min_separation=radius)
        monkeypatch.setattr(peakfinder, '_MAX_PAIRS_PER_PEAK', 0)
        tbl2 = find_peaks(data, 0.5, min_separation=radius)

        assert len(tbl1) > 10
        assert_array_equal(tbl1['x_peak'], tbl2['x_peak'])
        assert_array_equal(tbl1['y_peak'], tbl2['y_peak'])

        xypos = np.transpose((tbl1['x_peak'], tbl1['y_peak']))
        assert len(cKDTree(xypos).query_pairs(radius)) == 0

    def test_min_separation_integer_data(self):
        """
        Test that all peaks are separated by more than min_separation
        for integer-valued data, which has many equal-valued peaks.
        """
        rng = np.random.default_rng(0)
        data = rng.poisson(5, (200, 200)).astype(float)
        radius = 5

        tbl = find_peaks(data, 8, min_separation=radius)
        xypos = np.transpose((tbl['x_peak'], tbl['y_peak']))
        assert len(cKDTree(xypos).query_pairs(radius)) == 0

        # the peaks stay in row-major order
        idx = np.lexsort((tbl['x_peak'], tbl['y_peak']))
        assert_array_equal(idx, np.arange(len(tbl)))

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

    @pytest.mark.parametrize('axis', ['x', 'y'])
    def test_min_separation_noninteger_radius_regression(self, axis):
        """
        Regression test for noninteger min_separation with peaks
        separated by 12 pixels (< min_separation = 12.5).

        With a noninteger radius the circular footprint had an
        even number of elements, causing an asymmetric half-pixel
        shift that allowed the fainter peak to survive depending on
        which side of the brighter peak it was placed.
        """
        data = np.zeros((100, 100))
        if axis == 'x':
            data[50, 40] = 15.0  # brighter
            data[50, 52] = 10.0  # fainter, 12 px away
        else:
            data[40, 50] = 15.0  # brighter
            data[52, 50] = 10.0  # fainter, 12 px away

        # 12 < 12.5, so only the brighter peak should survive
        tbl = find_peaks(data, 1.0, min_separation=12.5)
        assert len(tbl) == 1
        assert tbl['peak_value'][0] == 15.0

    def test_min_separation_equal_at_boundary(self):
        """
        Test that two peaks separated by more than min_separation are
        both retained.
        """
        data = np.zeros((100, 100))
        data[50, 30] = 10.0
        data[50, 44] = 8.0  # distance = 14 > 13

        tbl = find_peaks(data, 1.0, min_separation=13)
        assert len(tbl) == 2
