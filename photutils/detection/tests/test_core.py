# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the core module.
"""

import warnings

from astropy.tests.helper import assert_quantity_allclose
# from astropy.tests.helper import catch_warnings
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest

from ..core import detect_threshold, find_peaks
from ...centroids import centroid_com
from ...datasets import make_4gaussians_image, make_gwcs, make_wcs
# from ...utils.exceptions import NoDetectionsWarning

try:
    import scipy  # noqa
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    # shared WCS interface requires gwcs >= 0.10
    import gwcs  # noqa
    HAS_GWCS = True
except ImportError:
    HAS_GWCS = False

DATA = np.array([[0, 1, 0], [0, 2, 0], [0, 0, 0]]).astype(float)
REF1 = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])

PEAKDATA = np.array([[1, 0, 0], [0, 0, 0], [0, 0, 1]]).astype(float)
PEAKREF1 = np.array([[0, 0], [2, 2]])

IMAGE = make_4gaussians_image()
FITSWCS = make_wcs(IMAGE.shape)


@pytest.mark.skipif('not HAS_SCIPY')
class TestDetectThreshold:
    def test_nsigma(self):
        """Test basic nsigma."""

        threshold = detect_threshold(DATA, nsigma=0.1)
        ref = 0.4 * np.ones((3, 3))
        assert_allclose(threshold, ref)

    def test_nsigma_zero(self):
        """Test nsigma=0."""

        threshold = detect_threshold(DATA, nsigma=0.0)
        ref = (1. / 3.) * np.ones((3, 3))
        assert_allclose(threshold, ref)

    def test_background(self):
        threshold = detect_threshold(DATA, nsigma=1.0, background=1)
        ref = (5. / 3.) * np.ones((3, 3))
        assert_allclose(threshold, ref)

    def test_background_image(self):
        background = np.ones((3, 3))
        threshold = detect_threshold(DATA, nsigma=1.0, background=background)
        ref = (5. / 3.) * np.ones((3, 3))
        assert_allclose(threshold, ref)

    def test_background_badshape(self):
        wrong_shape = np.zeros((2, 2))
        with pytest.raises(ValueError):
            detect_threshold(DATA, nsigma=2., background=wrong_shape)

    def test_error(self):
        threshold = detect_threshold(DATA, nsigma=1.0, error=1)
        ref = (4. / 3.) * np.ones((3, 3))
        assert_allclose(threshold, ref)

    def test_error_image(self):
        error = np.ones((3, 3))
        threshold = detect_threshold(DATA, nsigma=1.0, error=error)
        ref = (4. / 3.) * np.ones((3, 3))
        assert_allclose(threshold, ref)

    def test_error_badshape(self):
        wrong_shape = np.zeros((2, 2))
        with pytest.raises(ValueError):
            detect_threshold(DATA, nsigma=2., error=wrong_shape)

    def test_background_error(self):
        threshold = detect_threshold(DATA, nsigma=2.0, background=10.,
                                     error=1.)
        ref = 12. * np.ones((3, 3))
        assert_allclose(threshold, ref)

    def test_background_error_images(self):
        background = np.ones((3, 3)) * 10.
        error = np.ones((3, 3))
        threshold = detect_threshold(DATA, nsigma=2.0, background=background,
                                     error=error)
        ref = 12. * np.ones((3, 3))
        assert_allclose(threshold, ref)

    def test_mask_value(self):
        """Test detection with mask_value."""

        threshold = detect_threshold(DATA, nsigma=1.0, mask_value=0.0)
        ref = 2. * np.ones((3, 3))
        assert_array_equal(threshold, ref)

    def test_image_mask(self):
        """
        Test detection with image_mask.
        Set sigma=10 and iters=1 to prevent sigma clipping after
        applying the mask.
        """

        mask = REF1.astype(bool)
        threshold = detect_threshold(DATA, nsigma=1., error=0, mask=mask,
                                     sigclip_sigma=10, sigclip_iters=1)
        ref = (1. / 8.) * np.ones((3, 3))
        assert_array_equal(threshold, ref)

    def test_image_mask_override(self):
        """Test that image_mask overrides mask_value."""

        mask = REF1.astype(bool)
        threshold = detect_threshold(DATA, nsigma=0.1, error=0, mask_value=0.0,
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
        tbl = find_peaks(PEAKDATA, 0.1, box_size=3, border_width=3)
        assert tbl is None
        # temporarily disable this test due to upstream
        # "Distutils was imported before Setuptools" warning
        # with catch_warnings(NoDetectionsWarning) as warning_lines:
        #     tbl = find_peaks(PEAKDATA, 0.1, box_size=3, border_width=3)
        #     assert tbl is None
        #     assert len(warning_lines) > 0
        #     assert ('No local peaks were found.' in
        #             str(warning_lines[0].message))

    def test_box_size_int(self):
        """Test non-integer box_size."""

        tbl1 = find_peaks(PEAKDATA, 0.1, box_size=5.)
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

    @pytest.mark.skipif('not HAS_GWCS')
    def test_gwcs(self):
        """Test with gwcs."""

        columns = ['skycoord_peak', 'skycoord_centroid']

        gwcs_obj = make_gwcs(IMAGE.shape)
        tbl = find_peaks(IMAGE, 100, wcs=gwcs_obj, centroid_func=centroid_com)
        for column in columns:
            assert column in tbl.colnames

    @pytest.mark.skipif('not HAS_GWCS')
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
        tbl = find_peaks(data, 0.)
        assert tbl is None
        # temporarily disable this test due to upstream
        # "Distutils was imported before Setuptools" warning
        # with catch_warnings(NoDetectionsWarning) as warning_lines:
        #     tbl = find_peaks(data, 0.)
        #     assert tbl is None
        #     assert len(warning_lines) > 0
        #     assert ('Input data is constant.' in
        #             str(warning_lines[0].message))

    def test_no_peaks(self):
        """
        Test for an empty output table with the expected column names
        when no peaks are found.
        """

        tbl = find_peaks(IMAGE, 10000)
        assert tbl is None

        tbl = find_peaks(IMAGE, 100000, centroid_func=centroid_com)
        assert tbl is None

        tbl = find_peaks(IMAGE, 100000, wcs=FITSWCS)
        assert tbl is None

        tbl = find_peaks(IMAGE, 100000, wcs=FITSWCS,
                         centroid_func=centroid_com)
        assert tbl is None

    def test_data_nans(self):
        """Test that data with NaNs does not issue Runtime warning."""

        data = np.copy(PEAKDATA)
        data[1, 1] = np.nan

        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            find_peaks(data, 0.)
