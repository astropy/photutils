# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
from numpy.testing import assert_array_equal, assert_allclose
import pytest

from astropy.tests.helper import catch_warnings
from astropy.utils.exceptions import AstropyUserWarning
from astropy.convolution import Gaussian2DKernel
from astropy.stats import gaussian_fwhm_to_sigma

from ..detect import detect_sources, make_source_mask
from ...datasets import make_4gaussians_image

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


@pytest.mark.skipif('not HAS_SCIPY')
class TestDetectSources(object):
    def setup_class(self):
        self.data = np.array([[0, 1, 0], [0, 2, 0],
                              [0, 0, 0]]).astype(np.float)
        self.ref1 = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
        self.ref2 = np.array([[0, 1, 0], [0, 1, 0], [0, 0, 0]])

        fwhm2sigma = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        filter_kernel = Gaussian2DKernel(2. * fwhm2sigma, x_size=3, y_size=3)
        filter_kernel.normalize()
        self.filter_kernel = filter_kernel

    def test_detection(self):
        """Test basic detection."""

        segm = detect_sources(self.data, threshold=0.9, npixels=2)
        assert_array_equal(segm.data, self.ref2)

    def test_small_sources(self):
        """Test detection where sources are smaller than npixels size."""

        segm = detect_sources(self.data, threshold=0.9, npixels=5)
        assert_array_equal(segm.data, self.ref1)

    def test_zerothresh(self):
        """Test detection with zero threshold."""

        segm = detect_sources(self.data, threshold=0., npixels=2)
        assert_array_equal(segm.data, self.ref2)

    def test_zerodet(self):
        """Test detection with large snr_threshold giving no detections."""

        segm = detect_sources(self.data, threshold=7, npixels=2)
        assert_array_equal(segm.data, self.ref1)

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


@pytest.mark.skipif('not HAS_SCIPY')
class TestMakeSourceMask(object):
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
