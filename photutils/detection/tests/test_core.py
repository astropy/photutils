# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import numpy as np
from numpy.testing import assert_array_equal, assert_allclose
from astropy.tests.helper import pytest, catch_warnings
from astropy.utils.exceptions import AstropyUserWarning
from astropy.convolution import Gaussian2DKernel
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

    def test_mask_value(self):
        """Test detection with mask_value."""
        threshold = detect_threshold(DATA, snr=1.0, mask_value=0.0)
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
        """Test that image_mask overrides mask_value."""
        mask = REF3.astype(np.bool)
        threshold = detect_threshold(DATA, snr=0.1, error=0, mask_value=0.0,
                                     mask=mask, sigclip_sigma=10,
                                     sigclip_iters=1)
        ref = np.ones((3, 3))
        assert_array_equal(threshold, ref)


@pytest.mark.skipif('not HAS_SCIPY')
class TestDetectSources(object):
    def setup_class(self):
        FWHM2SIGMA = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        filter_kernel = Gaussian2DKernel(2.*FWHM2SIGMA, x_size=3, y_size=3)
        filter_kernel.normalize()
        self.filter_kernel = filter_kernel

    def test_detection(self):
        """Test basic detection."""
        segm = detect_sources(DATA, threshold=0.9, npixels=2)
        assert_array_equal(segm.data, REF2)

    def test_small_sources(self):
        """Test detection where sources are smaller than npixels size."""
        segm = detect_sources(DATA, threshold=0.9, npixels=5)
        assert_array_equal(segm.data, REF1)

    def test_zerothresh(self):
        """Test detection with zero threshold."""
        segm = detect_sources(DATA, threshold=0., npixels=2)
        assert_array_equal(segm.data, REF2)

    def test_zerodet(self):
        """Test detection with large snr_threshold giving no detections."""
        segm = detect_sources(DATA, threshold=7, npixels=2)
        assert_array_equal(segm.data, REF1)

    def test_8connectivity(self):
        """Test detection with connectivity=8."""
        data = np.eye(3)
        segm = detect_sources(data, threshold=0.9, npixels=1, connectivity=8)
        assert_array_equal(segm.data, data)

    def test_4connectivity(self):
        """Test detection with connectivity=4."""
        data = np.eye(3)
        ref = np.diag([1, 2, 3])
        segm = detect_sources(data, threshold=0.9, npixels=1, connectivity=4)
        assert_array_equal(segm.data, ref)

    def test_basic_filter_kernel(self):
        """Test detection with filter_kernel."""
        kernel = np.ones((3, 3)) / 9.
        threshold = 0.3
        expected = np.ones((3, 3))
        expected[2] = 0
        segm = detect_sources(DATA, threshold, npixels=1,
                              filter_kernel=kernel)
        assert_array_equal(segm.data, expected)

    def test_npixels_nonint(self):
        """Test if error raises if npixel is non-integer."""
        with pytest.raises(ValueError):
            detect_sources(DATA, threshold=1, npixels=0.1)

    def test_npixels_negative(self):
        """Test if error raises if npixel is negative."""
        with pytest.raises(ValueError):
            detect_sources(DATA, threshold=1, npixels=-1)

    def test_connectivity_invalid(self):
        """Test if error raises if connectivity is invalid."""
        with pytest.raises(ValueError):
            detect_sources(DATA, threshold=1, npixels=1, connectivity=10)

    def test_filter_kernel_array(self):
        segm = detect_sources(DATA, 0.1, npixels=1,
                              filter_kernel=self.filter_kernel.array)
        assert_array_equal(segm.data, np.ones((3, 3)))

    def test_filter_kernel(self):
        segm = detect_sources(DATA, 0.1, npixels=1,
                              filter_kernel=self.filter_kernel)
        assert_array_equal(segm.data, np.ones((3, 3)))

    def test_unnormalized_filter_kernel(self):
        with catch_warnings(AstropyUserWarning) as warning_lines:
            detect_sources(DATA, 0.1, npixels=1,
                           filter_kernel=self.filter_kernel*10.)
            assert warning_lines[0].category == AstropyUserWarning
            assert ('The kernel is not normalized.'
                    in str(warning_lines[0].message))


@pytest.mark.skipif('not HAS_SCIPY')
@pytest.mark.skipif('not HAS_SKIMAGE')
class TestFindPeaks(object):
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

    def test_subpixel_regionsize(self):
        """Test that data cutout has at least 6 values."""
        tbl = find_peaks(PEAKDATA, 0.1, box_size=2, subpixel=True)
        assert np.all(np.isnan(tbl['x_centroid']))
        assert np.all(np.isnan(tbl['y_centroid']))
        assert np.all(np.isnan(tbl['fit_peak_value']))

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

    def test_npeaks(self):
        """Test npeaks."""
        tbl = find_peaks(PEAKDATA, 0.1, box_size=3, npeaks=1)
        assert_array_equal(tbl['x_peak'], PEAKREF1[1, 1])
        assert_array_equal(tbl['y_peak'], PEAKREF1[1, 0])

    def test_border_width(self):
        """Test border exclusion."""
        tbl = find_peaks(PEAKDATA, 0.1, box_size=3, border_width=3)
        assert_array_equal(len(tbl), 0)

    def test_zerodet(self):
        """Test with large threshold giving no sources."""
        tbl = find_peaks(PEAKDATA, 5., box_size=3, border_width=3)
        assert_array_equal(len(tbl), 0)

    def test_constant_data(self):
        """Test constant data."""
        tbl = find_peaks(np.ones((5, 5)), 0.1, box_size=3.)
        assert_array_equal(len(tbl), 0)

    def test_box_size_int(self):
        """Test non-integer box_size."""
        tbl1 = find_peaks(PEAKDATA, 0.1, box_size=5.)
        tbl2 = find_peaks(PEAKDATA, 0.1, box_size=5.5)
        assert_array_equal(tbl1, tbl2)

    def test_wcs(self):
        """Test with WCS."""
        from photutils.datasets import make_4gaussians_image
        from astropy.wcs import WCS
        hdu = make_4gaussians_image(hdu=True, wcs=True)
        wcs = WCS(hdu.header)
        tbl = find_peaks(hdu.data, 100, wcs=wcs, subpixel=True)
        cols = ['icrs_ra_peak', 'icrs_dec_peak', 'icrs_ra_centroid',
                'icrs_dec_centroid']
        for col in cols:
            assert col in tbl.colnames
