# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the peakfinder module.
"""

import numpy as np
import pytest
from astropy.tests.helper import assert_quantity_allclose
from numpy.testing import assert_array_equal

from photutils.centroids import centroid_com
from photutils.datasets import make_4gaussians_image, make_gwcs, make_wcs
from photutils.detection.peakfinder import find_peaks
from photutils.utils._optional_deps import HAS_GWCS, HAS_SCIPY
from photutils.utils.exceptions import NoDetectionsWarning

PEAKDATA = np.array([[1, 0, 0], [0, 0, 0], [0, 0, 1]]).astype(float)
PEAKREF1 = np.array([[0, 0], [2, 2]])

IMAGE = make_4gaussians_image()
FITSWCS = make_wcs(IMAGE.shape)


@pytest.mark.skipif(not HAS_SCIPY, reason='scipy is required')
class TestFindPeaks:
    def test_box_size(self):
        """Test with box_size."""
        tbl = find_peaks(PEAKDATA, 0.1, box_size=3)
        assert_array_equal(tbl['x_peak'], PEAKREF1[:, 1])
        assert_array_equal(tbl['y_peak'], PEAKREF1[:, 0])
        assert_array_equal(tbl['peak_value'], [1.0, 1.0])

    def test_footprint(self):
        """Test with footprint."""
        tbl = find_peaks(PEAKDATA, 0.1, footprint=np.ones((3, 3)))
        assert_array_equal(tbl['x_peak'], PEAKREF1[:, 1])
        assert_array_equal(tbl['y_peak'], PEAKREF1[:, 0])
        assert_array_equal(tbl['peak_value'], [1.0, 1.0])

    def test_mask(self):
        """Test with mask."""
        mask = np.zeros(PEAKDATA.shape, dtype=bool)
        mask[0, 0] = True
        tbl = find_peaks(PEAKDATA, 0.1, box_size=3, mask=mask)
        assert len(tbl) == 1
        assert_array_equal(tbl['x_peak'], PEAKREF1[1, 0])
        assert_array_equal(tbl['y_peak'], PEAKREF1[1, 1])
        assert_array_equal(tbl['peak_value'], 1.0)

    def test_maskshape(self):
        """Test if make shape doesn't match data shape."""
        with pytest.raises(ValueError):
            find_peaks(PEAKDATA, 0.1, mask=np.ones((5, 5)))

    def test_thresholdshape(self):
        """Test if threshold shape doesn't match data shape."""
        with pytest.raises(ValueError):
            find_peaks(PEAKDATA, np.ones((2, 2)))

    def test_npeaks(self):
        """Test npeaks."""
        tbl = find_peaks(PEAKDATA, 0.1, box_size=3, npeaks=1)
        assert_array_equal(tbl['x_peak'], PEAKREF1[1, 1])
        assert_array_equal(tbl['y_peak'], PEAKREF1[1, 0])

    def test_border_width(self):
        """Test border exclusion."""
        with pytest.warns(NoDetectionsWarning,
                          match='No local peaks were found'):
            tbl = find_peaks(PEAKDATA, 0.1, box_size=3, border_width=3)
            assert tbl is None

    def test_box_size_int(self):
        """Test non-integer box_size."""
        tbl1 = find_peaks(PEAKDATA, 0.1, box_size=5.0)
        tbl2 = find_peaks(PEAKDATA, 0.1, box_size=5.5)
        assert_array_equal(tbl1, tbl2)

    def test_centroid_func_callable(self):
        """Test that centroid_func is callable."""
        with pytest.raises(TypeError):
            find_peaks(PEAKDATA, 0.1, box_size=2, centroid_func=True)

    def test_wcs(self):
        """Test with astropy WCS."""
        columns = ['skycoord_peak', 'skycoord_centroid']

        tbl = find_peaks(IMAGE, 100, wcs=FITSWCS, centroid_func=centroid_com)
        for column in columns:
            assert column in tbl.colnames

    @pytest.mark.skipif(not HAS_GWCS, reason='gwcs is required')
    def test_gwcs(self):
        """Test with gwcs."""
        columns = ['skycoord_peak', 'skycoord_centroid']

        gwcs_obj = make_gwcs(IMAGE.shape)
        tbl = find_peaks(IMAGE, 100, wcs=gwcs_obj, centroid_func=centroid_com)
        for column in columns:
            assert column in tbl.colnames

    @pytest.mark.skipif(not HAS_GWCS, reason='gwcs is required')
    def test_wcs_values(self):
        gwcs_obj = make_gwcs(IMAGE.shape)
        tbl1 = find_peaks(IMAGE, 100, wcs=FITSWCS, centroid_func=centroid_com)
        tbl2 = find_peaks(IMAGE, 100, wcs=gwcs_obj,
                          centroid_func=centroid_com)
        columns = ['skycoord_peak', 'skycoord_centroid']
        for column in columns:
            assert_quantity_allclose(tbl1[column].ra, tbl2[column].ra)
            assert_quantity_allclose(tbl1[column].dec, tbl2[column].dec)

    def test_constant_array(self):
        """Test for empty output table when data is constant."""
        data = np.ones((10, 10))
        with pytest.warns(NoDetectionsWarning, match='Input data is constant'):
            tbl = find_peaks(data, 0.0)
            assert tbl is None

    def test_no_peaks(self):
        """
        Tests for when no peaks are found.
        """
        with pytest.warns(NoDetectionsWarning,
                          match='No local peaks were found'):
            tbl = find_peaks(IMAGE, 10000)
            assert tbl is None
        with pytest.warns(NoDetectionsWarning,
                          match='No local peaks were found'):
            tbl = find_peaks(IMAGE, 100000, centroid_func=centroid_com)
            assert tbl is None
        with pytest.warns(NoDetectionsWarning,
                          match='No local peaks were found'):
            tbl = find_peaks(IMAGE, 100000, wcs=FITSWCS)
            assert tbl is None
        with pytest.warns(NoDetectionsWarning,
                          match='No local peaks were found'):
            tbl = find_peaks(IMAGE, 100000, wcs=FITSWCS,
                             centroid_func=centroid_com)
            assert tbl is None

    def test_data_nans(self):
        """Test that data with NaNs does not issue Runtime warning."""
        data = np.copy(PEAKDATA)
        data[1, 1] = np.nan
        find_peaks(data, 0.0)
