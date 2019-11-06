# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the detect module.
"""

from astropy.convolution import Gaussian2DKernel
from astropy.stats import gaussian_fwhm_to_sigma
from astropy.tests.helper import catch_warnings
from astropy.utils.exceptions import AstropyUserWarning
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest

from ...utils.exceptions import NoDetectionsWarning
from ..detect import detect_sources, make_source_mask
from ...datasets import make_4gaussians_image

try:
    import scipy  # noqa
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


@pytest.mark.skipif('not HAS_SCIPY')
class TestDetectSources:
    def setup_class(self):
        self.data = np.array([[0, 1, 0], [0, 2, 0],
                              [0, 0, 0]]).astype(float)
        self.refdata = np.array([[0, 1, 0], [0, 1, 0], [0, 0, 0]])

        fwhm2sigma = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        filter_kernel = Gaussian2DKernel(2. * fwhm2sigma, x_size=3, y_size=3)
        filter_kernel.normalize()
        self.filter_kernel = filter_kernel

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

    def test_basic_filter_kernel(self):
        """Test detection with filter_kernel."""

        kernel = np.ones((3, 3)) / 9.
        threshold = 0.3
        expected = np.ones((3, 3))
        expected[2] = 0
        segm = detect_sources(self.data, threshold, npixels=1,
                              filter_kernel=kernel)
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

    def test_filter_kernel_array(self):
        segm = detect_sources(self.data, 0.1, npixels=1,
                              filter_kernel=self.filter_kernel.array)
        assert_array_equal(segm.data, np.ones((3, 3)))

    def test_filter_kernel(self):
        segm = detect_sources(self.data, 0.1, npixels=1,
                              filter_kernel=self.filter_kernel)
        assert_array_equal(segm.data, np.ones((3, 3)))

    def test_unnormalized_filter_kernel(self):
        with catch_warnings(AstropyUserWarning) as warning_lines:
            detect_sources(self.data, 0.1, npixels=1,
                           filter_kernel=self.filter_kernel*10.)
            assert warning_lines[0].category == AstropyUserWarning
            assert ('The kernel is not normalized.'
                    in str(warning_lines[0].message))

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
        mask2 = make_source_mask(self.data, 5, 10, filter_kernel=kernel)
        assert_allclose(mask1, mask2)

    def test_no_detections(self):
        mask = make_source_mask(self.data, 100, 100)
        assert np.count_nonzero(mask) == 0
