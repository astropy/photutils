# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from astropy.tests.helper import pytest
import numpy as np
from numpy.testing import assert_array_equal
from ..core import detect_sources, find_peaks

try:
    import scipy
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    import skimage
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False


DATA = np.array([[0, 1, 0], [0, 2, 0], [0, 0, 0]]).astype(np.float)
REF1 = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
REF2 = np.array([[0, 1, 0], [0, 1, 0], [0, 0, 0]])
REF3 = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])

PEAKDATA = np.array([[1, 0, 0], [0, 0, 0], [0, 0, 1]]).astype(np.float)
PEAKREF1 = np.array([[0, 0], [2, 2]])
PEAKREF2 = np.array([]).reshape(0, 2)


@pytest.mark.skipif('not HAS_SCIPY')
class TestDetectSources(object):
    def test_detection(self):
        """Test basic detection."""
        segm = detect_sources(DATA, 0.1, 2)
        assert_array_equal(segm, REF2)

    def test_small_sources(self):
        """Test detection where sources are smaller than npixels size."""
        segm = detect_sources(DATA, 0.1, 5)
        assert_array_equal(segm, REF1)

    def test_zerothresh(self):
        """Test detection with zero snr_threshold."""
        segm = detect_sources(DATA, 0.0, 2)
        assert_array_equal(segm, REF2)

    def test_zerodet(self):
        """Test detection with large snr_threshold giving no detections."""
        segm = detect_sources(DATA, 10.0, 2)
        assert_array_equal(segm, REF1)

    def test_filter1(self):
        """Test detection with filter_fwhm."""
        segm = detect_sources(DATA, 0.1, 2, filter_fwhm=1.)
        assert_array_equal(segm, REF2)

    def test_filter2(self):
        """Test detection for small filter_fwhm."""
        segm = detect_sources(DATA, 1, 1, filter_fwhm=0.5)
        assert_array_equal(segm, REF3)

    def test_npix1_error(self):
        """Test if AssertionError raises if npixel is non-integer."""
        with pytest.raises(AssertionError):
            detect_sources(DATA, 1, 0.1)

    def test_npix2_error(self):
        """Test if AssertionError raises if npixel is negative."""
        with pytest.raises(AssertionError):
            detect_sources(DATA, 1, -1)

    def test_mask_val(self):
        """Test detection with mask_val."""
        segm = detect_sources(DATA, 0.1, 1, mask_val=0.0)
        assert_array_equal(segm, REF3)

    def test_image_mask(self):
        """
        Test detection with image_mask.
        sig=10 and iters=1 to prevent sigma clipping after applying the mask.
        """

        mask = REF3.astype(np.bool)
        segm = detect_sources(DATA, 0.1, 1, mask=mask, sig=10, iters=1)
        assert_array_equal(segm, REF2)

    def test_image_mask_override(self):
        """Test that image_mask overrides mask_val."""
        mask = REF3.astype(np.bool)
        segm = detect_sources(DATA, 0.1, 1, mask_val=0.0, mask=mask,
                              sig=10, iters=1)
        assert_array_equal(segm, REF2)


@pytest.mark.skipif('not HAS_SCIPY')
@pytest.mark.skipif('not HAS_SKIMAGE')
class TestFindPeaks(object):
    def test_find_peaks(self):
        """Test basic peak detection."""
        segm = find_peaks(PEAKDATA, 0., min_distance=1, exclude_border=False)
        assert_array_equal(segm, PEAKREF1)

    def test_exclude_border(self):
        """Test exclude_border."""
        segm = find_peaks(PEAKDATA, 0., min_distance=1, exclude_border=True)
        assert_array_equal(segm, PEAKREF2)

    def test_zerodet(self):
        """Test with large snr_threshold giving no sources."""
        segm = find_peaks(PEAKDATA, 0., min_distance=1, exclude_border=True)
        assert_array_equal(segm, PEAKREF2)
