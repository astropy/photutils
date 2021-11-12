# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the detect module.
"""

from astropy.convolution import Gaussian2DKernel
from astropy.stats import gaussian_fwhm_to_sigma
from astropy.tests.helper import catch_warnings
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest

from ...utils.exceptions import NoDetectionsWarning
from ..detect import detect_threshold, detect_sources, make_source_mask
from ...datasets import make_4gaussians_image
from ...utils._optional_deps import HAS_SCIPY  # noqa


DATA = np.array([[0, 1, 0], [0, 2, 0], [0, 0, 0]]).astype(float)
REF1 = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])


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
class TestDetectSources:
    def setup_class(self):
        self.data = np.array([[0, 1, 0], [0, 2, 0],
                              [0, 0, 0]]).astype(float)
        self.refdata = np.array([[0, 1, 0], [0, 1, 0], [0, 0, 0]])

        fwhm2sigma = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        kernel = Gaussian2DKernel(2. * fwhm2sigma, x_size=3, y_size=3)
        kernel.normalize()
        self.kernel = kernel

    def test_detection(self):
        """Test basic detection."""

        segm = detect_sources(self.data, threshold=0.9, npixels=2)
        assert_array_equal(segm.data, self.refdata)

    def test_small_sources(self):
        """Test detection where sources are smaller than npixels size."""

        with catch_warnings(NoDetectionsWarning) as warning_lines:
            detect_sources(self.data, threshold=0.9, npixels=5)
            assert warning_lines[0].category == NoDetectionsWarning
            assert 'No sources were found.' in str(warning_lines[0].message)

    def test_npixels(self):
        """
        Test removal of sources whose size is less than npixels.
        Regression tests for #663.
        """

        data = np.zeros((8, 8))
        data[0:4, 0] = 1
        data[0, 0:4] = 1
        data[3, 3:] = 2
        data[3:, 3] = 2

        segm = detect_sources(data, 0, npixels=8)
        assert segm.nlabels == 1
        segm = detect_sources(data, 0, npixels=9)
        assert segm.nlabels == 1

        data = np.zeros((8, 8))
        data[0:4, 0] = 1
        data[0, 0:4] = 1
        data[3, 2:] = 2
        data[3:, 2] = 2
        data[5:, 3] = 2

        npixels = np.arange(9, 14)
        for npixels in np.arange(9, 14):
            segm = detect_sources(data, 0, npixels=npixels)
            assert segm.nlabels == 1
            assert segm.areas[0] == 13

        with catch_warnings(NoDetectionsWarning) as warning_lines:
            detect_sources(data, 0, npixels=14)
            assert warning_lines[0].category == NoDetectionsWarning
            assert 'No sources were found.' in str(warning_lines[0].message)

    def test_zerothresh(self):
        """Test detection with zero threshold."""

        segm = detect_sources(self.data, threshold=0., npixels=2)
        assert_array_equal(segm.data, self.refdata)

    def test_zerodet(self):
        """Test detection with large threshold giving no detections."""

        with catch_warnings(NoDetectionsWarning) as warning_lines:
            detect_sources(self.data, threshold=7, npixels=2)
            assert warning_lines[0].category == NoDetectionsWarning
            assert 'No sources were found.' in str(warning_lines[0].message)

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

    def test_basic_kernel(self):
        """Test detection with kernel."""

        kernel = np.ones((3, 3)) / 9.
        threshold = 0.3
        expected = np.ones((3, 3))
        expected[2] = 0
        segm = detect_sources(self.data, threshold, npixels=1, kernel=kernel)
        assert_array_equal(segm.data, expected)

    def test_npixels_nonint(self):
        """Test if error raises if npixel is non-integer."""

        with pytest.raises(ValueError):
            detect_sources(self.data, threshold=1, npixels=0.1)

    def test_npixels_negative(self):
        """Test if error raises if npixel is negative."""

        with pytest.raises(ValueError):
            detect_sources(self.data, threshold=1, npixels=-1)

    def test_connectivity_invalid(self):
        """Test if error raises if connectivity is invalid."""

        with pytest.raises(ValueError):
            detect_sources(self.data, threshold=1, npixels=1, connectivity=10)

    def test_kernel_array(self):
        segm = detect_sources(self.data, 0.1, npixels=1,
                              kernel=self.kernel.array)
        assert_array_equal(segm.data, np.ones((3, 3)))

    def test_kernel(self):
        segm = detect_sources(self.data, 0.1, npixels=1, kernel=self.kernel)
        assert_array_equal(segm.data, np.ones((3, 3)))

    def test_mask(self):
        data = np.zeros((11, 11))
        data[3:8, 3:8] = 5.
        mask = np.zeros(data.shape, dtype=bool)
        mask[4:6, 4:6] = True
        segm1 = detect_sources(data, 1., 1.)
        segm2 = detect_sources(data, 1., 1., mask=mask)
        assert segm2.areas[0] == segm1.areas[0] - mask.sum()

    def test_mask_shape(self):
        with pytest.raises(ValueError):
            detect_sources(self.data, 1., 1., mask=np.ones((5, 5)))


@pytest.mark.skipif('not HAS_SCIPY')
class TestMakeSourceMask:
    def setup_class(self):
        self.data = make_4gaussians_image()

    def test_dilate_size(self):
        mask1 = make_source_mask(self.data, 5, 10)
        mask2 = make_source_mask(self.data, 5, 10, dilate_size=20)
        assert np.count_nonzero(mask2) > np.count_nonzero(mask1)

    def test_kernel(self):
        mask1 = make_source_mask(self.data, 5, 10, filter_fwhm=2,
                                 filter_size=3)
        sigma = 2 * gaussian_fwhm_to_sigma
        kernel = Gaussian2DKernel(sigma, x_size=3, y_size=3)
        mask2 = make_source_mask(self.data, 5, 10, kernel=kernel)
        assert_allclose(mask1, mask2)

    def test_no_detections(self):
        with catch_warnings(NoDetectionsWarning) as warning_lines:
            mask = make_source_mask(self.data, 100, 100)
            assert np.count_nonzero(mask) == 0
            assert warning_lines[0].category == NoDetectionsWarning
