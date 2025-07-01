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

        # test with units
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

    def test_maskshape(self, data):
        """
        Test if make shape doesn't match data shape.
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

    def test_npeaks(self, data):
        """
        Test npeaks.
        """
        tbl = find_peaks(data, 0.1, box_size=3, npeaks=1)
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

    def test_box_size_int(self, data):
        """
        Test non-integer box_size.
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
