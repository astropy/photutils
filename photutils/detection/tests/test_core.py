# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from astropy.tests.helper import pytest
import numpy as np
from numpy.testing import assert_array_equal, assert_allclose
from ..core import detect_threshold, detect_sources, find_peaks

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
class TestDetectThreshold(object):
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

    def test_mask_val(self):
        """Test detection with mask_val."""
        threshold = detect_threshold(DATA, snr=1.0, mask_val=0.0)
        ref = 2. * np.ones((3, 3))
        assert_array_equal(threshold, ref)

    def test_image_mask(self):
        """
        Test detection with image_mask.
        sig=10 and iters=1 to prevent sigma clipping after applying the mask.
        """

        mask = REF3.astype(np.bool)
        threshold = detect_threshold(DATA, snr=1., error=0, mask=mask,
                                     sigclip_sigma=10, sigclip_iters=1)
        ref = (1. / 8.) * np.ones((3, 3))
        assert_array_equal(threshold, ref)

    def test_image_mask_override(self):
        """Test that image_mask overrides mask_val."""
        mask = REF3.astype(np.bool)
        threshold = detect_threshold(DATA, snr=0.1, error=0, mask_val=0.0,
                                     mask=mask, sigclip_sigma=10,
                                     sigclip_iters=1)
        ref = (1. / 8.) * np.ones((3, 3))
        assert_array_equal(threshold, ref)


@pytest.mark.skipif('not HAS_SCIPY')
class TestDetectSources(object):
    def test_detection(self):
        """Test basic detection."""
        segm = detect_sources(DATA, threshold=0.9, npixels=2)
        assert_array_equal(segm, REF2)

    def test_small_sources(self):
        """Test detection where sources are smaller than npixels size."""
        segm = detect_sources(DATA, threshold=0.9, npixels=5)
        assert_array_equal(segm, REF1)

    def test_zerothresh(self):
        """Test detection with zero threshold."""
        segm = detect_sources(DATA, threshold=0., npixels=2)
        assert_array_equal(segm, REF2)

    def test_zerodet(self):
        """Test detection with large snr_threshold giving no detections."""
        segm = detect_sources(DATA, threshold=7, npixels=2)
        assert_array_equal(segm, REF1)

    def test_8connectivity(self):
        """Test detection with connectivity=8."""
        data = np.eye(3)
        segm = detect_sources(data, threshold=0.9, npixels=1, connectivity=8)
        assert_array_equal(segm, data)

    def test_4connectivity(self):
        """Test detection with connectivity=4."""
        data = np.eye(3)
        ref = np.diag([1, 2, 3])
        segm = detect_sources(data, threshold=0.9, npixels=1, connectivity=4)
        assert_array_equal(segm, ref)

    def test_filter_kernel(self):
        """Test detection with filter_kernel."""
        kernel = np.ones((3, 3))
        threshold = 1.5
        segm = detect_sources(DATA, threshold, npixels=1,
                              filter_kernel=kernel)
        assert_array_equal(segm, kernel)

    def test_npixels_nonint(self):
        """Test if AssertionError raises if npixel is non-integer."""
        with pytest.raises(ValueError):
            detect_sources(DATA, threshold=1, npixels=0.1)

    def test_npixels_negative(self):
        """Test if AssertionError raises if npixel is negative."""
        with pytest.raises(ValueError):
            detect_sources(DATA, threshold=1, npixels=-1)

    def test_filtering(self):
        from astropy.convolution import Gaussian2DKernel
        FWHM2SIGMA = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        filter_kernel = Gaussian2DKernel(2.*FWHM2SIGMA, x_size=3, y_size=3)
        segm = detect_sources(DATA, 0.1, npixels=1,
                              filter_kernel=filter_kernel.array)
        assert_array_equal(segm, np.ones((3, 3)))

    def test_filtering_kernel(self):
        from astropy.convolution import Gaussian2DKernel
        FWHM2SIGMA = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        filter_kernel = Gaussian2DKernel(2.*FWHM2SIGMA, x_size=3, y_size=3)
        segm = detect_sources(DATA, 0.1, npixels=1,
                              filter_kernel=filter_kernel)
        assert_array_equal(segm, np.ones((3, 3)))


@pytest.mark.skipif('not HAS_SCIPY')
@pytest.mark.skipif('not HAS_SKIMAGE')
class TestFindPeaks(object):
    def test_find_peaks(self):
        """Test basic peak detection."""
        coords = find_peaks(PEAKDATA, 0.1, min_separation=1,
                            exclude_border=False)
        assert_array_equal(coords, PEAKREF1)

    def test_segment_image(self):
        segm = PEAKDATA.copy()
        coords = find_peaks(PEAKDATA, 0.1, min_separation=1,
                            exclude_border=False, segment_image=segm)
        assert_array_equal(coords, PEAKREF1)

    def test_segment_image_npeaks(self):
        segm = PEAKDATA.copy()
        coords = find_peaks(PEAKDATA, 0.1, min_separation=1,
                            exclude_border=False, segment_image=segm,
                            npeaks=1)
        assert_array_equal(coords, np.array([PEAKREF1[1]]))

    def test_segment_image_shape(self):
        segm = np.zeros((2, 2))
        with pytest.raises(ValueError):
            find_peaks(PEAKDATA, 0.1, segment_image=segm)

    def test_exclude_border(self):
        """Test exclude_border."""
        coords = find_peaks(PEAKDATA, 0.1, min_separation=1,
                            exclude_border=True)
        assert_array_equal(coords, PEAKREF2)

    def test_zerodet(self):
        """Test with large threshold giving no sources."""
        coords = find_peaks(PEAKDATA, 5., min_separation=1,
                            exclude_border=True)
        assert_array_equal(coords, PEAKREF2)

    def test_min_separation_int(self):
        with pytest.raises(ValueError):
            find_peaks(PEAKDATA, 0.1, min_separation=0.5)
