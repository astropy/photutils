# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
from numpy.testing import assert_array_equal, assert_allclose
import pytest
import warnings

from ..core import detect_threshold, find_peaks
from ...centroids import centroid_com
from ...datasets import make_4gaussians_image, make_wcs

try:
    import scipy    # noqa
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

DATA = np.array([[0, 1, 0], [0, 2, 0], [0, 0, 0]]).astype(np.float)
REF1 = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])

PEAKDATA = np.array([[1, 0, 0], [0, 0, 0], [0, 0, 1]]).astype(np.float)
PEAKREF1 = np.array([[0, 0], [2, 2]])


@pytest.mark.skipif('not HAS_SCIPY')
class TestDetectThreshold:
    def test_snr(self):
        """Test basic snr."""

        threshold = detect_threshold(DATA, snr=0.1)
        ref = 0.4 * np.ones((3, 3))
        assert_allclose(threshold, ref)

    def test_snr_zero(self):
        """Test snr=0."""

        threshold = detect_threshold(DATA, snr=0.0)
        ref = (1. / 3.) * np.ones((3, 3))
        assert_allclose(threshold, ref)

    def test_background(self):
        threshold = detect_threshold(DATA, snr=1.0, background=1)
        ref = (5. / 3.) * np.ones((3, 3))
        assert_allclose(threshold, ref)

    def test_background_image(self):
        background = np.ones((3, 3))
        threshold = detect_threshold(DATA, snr=1.0, background=background)
        ref = (5. / 3.) * np.ones((3, 3))
        assert_allclose(threshold, ref)

    def test_background_badshape(self):
        wrong_shape = np.zeros((2, 2))
        with pytest.raises(ValueError):
            detect_threshold(DATA, snr=2., background=wrong_shape)

    def test_error(self):
        threshold = detect_threshold(DATA, snr=1.0, error=1)
        ref = (4. / 3.) * np.ones((3, 3))
        assert_allclose(threshold, ref)

    def test_error_image(self):
        error = np.ones((3, 3))
        threshold = detect_threshold(DATA, snr=1.0, error=error)
        ref = (4. / 3.) * np.ones((3, 3))
        assert_allclose(threshold, ref)

    def test_error_badshape(self):
        wrong_shape = np.zeros((2, 2))
        with pytest.raises(ValueError):
            detect_threshold(DATA, snr=2., error=wrong_shape)

    def test_background_error(self):
        threshold = detect_threshold(DATA, snr=2.0, background=10., error=1.)
        ref = 12. * np.ones((3, 3))
        assert_allclose(threshold, ref)

    def test_background_error_images(self):
        background = np.ones((3, 3)) * 10.
        error = np.ones((3, 3))
        threshold = detect_threshold(DATA, snr=2.0, background=background,
                                     error=error)
        ref = 12. * np.ones((3, 3))
        assert_allclose(threshold, ref)

    def test_mask_value(self):
        """Test detection with mask_value."""

        threshold = detect_threshold(DATA, snr=1.0, mask_value=0.0)
        ref = 2. * np.ones((3, 3))
        assert_array_equal(threshold, ref)

    def test_image_mask(self):
        """
        Test detection with image_mask.
        Set sigma=10 and iters=1 to prevent sigma clipping after
        applying the mask.
        """

        mask = REF1.astype(np.bool)
        threshold = detect_threshold(DATA, snr=1., error=0, mask=mask,
                                     sigclip_sigma=10, sigclip_iters=1)
        ref = (1. / 8.) * np.ones((3, 3))
        assert_array_equal(threshold, ref)

    def test_image_mask_override(self):
        """Test that image_mask overrides mask_value."""

        mask = REF1.astype(np.bool)
        threshold = detect_threshold(DATA, snr=0.1, error=0, mask_value=0.0,
                                     mask=mask, sigclip_sigma=10,
                                     sigclip_iters=1)
        ref = np.ones((3, 3))
        assert_array_equal(threshold, ref)


@pytest.mark.skipif('not HAS_SCIPY')
class TestFindPeaks:
    def test_box_size(self):
        """Test with box_size."""

        tbl = find_peaks(PEAKDATA, 0.1, box_size=3)
        assert_array_equal(tbl['x_peak'], PEAKREF1[:, 1])
        assert_array_equal(tbl['y_peak'], PEAKREF1[:, 0])
        assert_array_equal(tbl['peak_value'], [1., 1.])

    def test_footprint(self):
        """Test with footprint."""

        tbl = find_peaks(PEAKDATA, 0.1, footprint=np.ones((3, 3)))
        assert_array_equal(tbl['x_peak'], PEAKREF1[:, 1])
        assert_array_equal(tbl['y_peak'], PEAKREF1[:, 0])
        assert_array_equal(tbl['peak_value'], [1., 1.])

    def test_centroid_func_and_subpixel(self):
        with pytest.raises(ValueError):
            find_peaks(PEAKDATA, 0.1, centroid_func=centroid_com,
                       subpixel=True)

    def test_subpixel_regionsize(self):
        """Test that data cutout has at least 6 values."""

        with pytest.raises(ValueError):
            find_peaks(PEAKDATA, 0.1, box_size=2, subpixel=True)

    def test_mask(self):
        """Test with mask."""

        mask = np.zeros_like(PEAKDATA, dtype=bool)
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

        tbl = find_peaks(PEAKDATA, 0.1, box_size=3, border_width=3)
        assert len(tbl) == 0

    def test_box_size_int(self):
        """Test non-integer box_size."""

        tbl1 = find_peaks(PEAKDATA, 0.1, box_size=5.)
        tbl2 = find_peaks(PEAKDATA, 0.1, box_size=5.5)
        assert_array_equal(tbl1, tbl2)

    def test_centroid_func_callable(self):
        """Test that centroid_func is callable."""

        with pytest.raises(ValueError):
            find_peaks(PEAKDATA, 0.1, box_size=2, centroid_func=True)

    def test_wcs(self):
        """Test with WCS."""

        data = make_4gaussians_image()
        wcs = make_wcs(data.shape)
        cols = ['skycoord_peak', 'skycoord_centroid']

        tbl = find_peaks(data, 100, wcs=wcs, centroid_func=centroid_com)
        for col in cols:
            assert col in tbl.colnames

        tbl = find_peaks(data, 100, wcs=wcs, subpixel=True)
        for col in cols:
            assert col in tbl.colnames

    def test_constant_array(self):
        """Test for empty output table when data is constant."""

        data = np.ones((10, 10))
        tbl = find_peaks(data, 0.)
        assert len(tbl) == 0
        assert set(tbl.colnames) == {'x_peak', 'y_peak', 'peak_value'}

    def test_no_peaks(self):
        """
        Test for an empty output table with the expected column names
        when no peaks are found.
        """

        data = make_4gaussians_image()
        wcs = make_wcs(data.shape)

        tbl1 = find_peaks(data, 100)
        tbl2 = find_peaks(data, 10000)
        assert set(tbl1.colnames) == set(tbl2.colnames)

        tbl1 = find_peaks(data, 100, centroid_func=centroid_com)
        tbl2 = find_peaks(data, 100000, centroid_func=centroid_com)
        assert set(tbl1.colnames) == set(tbl2.colnames)

        tbl1 = find_peaks(data, 100, subpixel=True)
        tbl2 = find_peaks(data, 100000, subpixel=True)
        assert set(tbl1.colnames) == set(tbl2.colnames)

        tbl1 = find_peaks(data, 100, wcs=wcs)
        tbl2 = find_peaks(data, 100000, wcs=wcs)
        assert set(tbl1.colnames) == set(tbl2.colnames)

        tbl1 = find_peaks(data, 100, wcs=wcs, centroid_func=centroid_com)
        tbl2 = find_peaks(data, 100000, wcs=wcs, centroid_func=centroid_com)
        assert set(tbl1.colnames) == set(tbl2.colnames)

        tbl1 = find_peaks(data, 100, wcs=wcs, subpixel=True)
        tbl2 = find_peaks(data, 100000, wcs=wcs, subpixel=True)
        assert set(tbl1.colnames) == set(tbl2.colnames)

    def test_data_nans(self):
        """Test that data with NaNs does not issue Runtime warning."""

        data = np.copy(PEAKDATA)
        data[1, 1] = np.nan

        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            find_peaks(data, 0.)
