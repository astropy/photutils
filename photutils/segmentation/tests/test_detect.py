# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the detect module.
"""

import astropy.units as u
import numpy as np
import pytest
from astropy.stats import SigmaClip
from numpy.testing import assert_allclose, assert_array_equal

from photutils.segmentation.detect import detect_sources, detect_threshold
from photutils.segmentation.utils import make_2dgaussian_kernel
from photutils.utils._optional_deps import HAS_SCIPY
from photutils.utils.exceptions import NoDetectionsWarning

DATA = np.array([[0, 1, 0], [0, 2, 0], [0, 0, 0]]).astype(float)
REF1 = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])


@pytest.mark.skipif(not HAS_SCIPY, reason='scipy is required')
class TestDetectThreshold:
    def test_nsigma(self):
        """Test basic nsigma."""
        threshold = detect_threshold(DATA, nsigma=0.1)
        ref = 0.4 * np.ones((3, 3))
        assert_allclose(threshold, ref)

        threshold = detect_threshold(DATA << u.uJy, nsigma=0.1)
        assert isinstance(threshold, u.Quantity)
        assert_allclose(threshold.value, ref)

    def test_nsigma_zero(self):
        """Test nsigma=0."""
        threshold = detect_threshold(DATA, nsigma=0.0)
        ref = (1.0 / 3.0) * np.ones((3, 3))
        assert_allclose(threshold, ref)

    def test_background(self):
        threshold = detect_threshold(DATA, nsigma=1.0, background=1)
        ref = (5.0 / 3.0) * np.ones((3, 3))
        assert_allclose(threshold, ref)

    def test_background_image(self):
        background = np.ones((3, 3))
        threshold = detect_threshold(DATA, nsigma=1.0, background=background)
        ref = (5.0 / 3.0) * np.ones((3, 3))
        assert_allclose(threshold, ref)

        threshold = detect_threshold(DATA << u.Jy, nsigma=1.0,
                                     background=background << u.Jy)
        assert isinstance(threshold, u.Quantity)
        assert_allclose(threshold.value, ref)

    def test_background_badshape(self):
        wrong_shape = np.zeros((2, 2))
        with pytest.raises(ValueError):
            detect_threshold(DATA, nsigma=2.0, background=wrong_shape)

    def test_error(self):
        threshold = detect_threshold(DATA, nsigma=1.0, error=1)
        ref = (4.0 / 3.0) * np.ones((3, 3))
        assert_allclose(threshold, ref)

    def test_error_image(self):
        error = np.ones((3, 3))
        threshold = detect_threshold(DATA, nsigma=1.0, error=error)
        ref = (4.0 / 3.0) * np.ones((3, 3))
        assert_allclose(threshold, ref)

        threshold = detect_threshold(DATA << u.Jy, nsigma=1.0,
                                     error=error << u.Jy)
        assert isinstance(threshold, u.Quantity)
        assert_allclose(threshold.value, ref)

    def test_error_badshape(self):
        wrong_shape = np.zeros((2, 2))
        with pytest.raises(ValueError):
            detect_threshold(DATA, nsigma=2.0, error=wrong_shape)

    def test_background_error(self):
        threshold = detect_threshold(DATA, nsigma=2.0, background=10.0,
                                     error=1.0)
        ref = 12.0 * np.ones((3, 3))
        assert_allclose(threshold, ref)

        threshold = detect_threshold(DATA << u.Jy, nsigma=2.0,
                                     background=10.0 * u.Jy,
                                     error=1.0 * u.Jy)
        assert isinstance(threshold, u.Quantity)
        assert_allclose(threshold.value, ref)

        with pytest.raises(ValueError):
            detect_threshold(DATA << u.Jy, nsigma=2.0, background=10.0,
                             error=1.0 * u.Jy)
        with pytest.raises(ValueError):
            detect_threshold(DATA << u.Jy, nsigma=2.0, background=10.0 * u.m,
                             error=1.0 * u.Jy)

    def test_background_error_images(self):
        background = np.ones((3, 3)) * 10.0
        error = np.ones((3, 3))
        threshold = detect_threshold(DATA, nsigma=2.0, background=background,
                                     error=error)
        ref = 12.0 * np.ones((3, 3))
        assert_allclose(threshold, ref)

    def test_image_mask(self):
        """
        Test detection with image_mask.
        Set sigma=10 and iters=1 to prevent sigma clipping after
        applying the mask.
        """
        mask = REF1.astype(bool)
        sigma_clip = SigmaClip(sigma=10, maxiters=1)
        threshold = detect_threshold(DATA, nsigma=1.0, error=0, mask=mask,
                                     sigma_clip=sigma_clip)
        ref = (1.0 / 8.0) * np.ones((3, 3))
        assert_array_equal(threshold, ref)

    def test_invalid_sigma_clip(self):
        with pytest.raises(TypeError):
            detect_threshold(DATA, 1.0, sigma_clip=10)


@pytest.mark.skipif(not HAS_SCIPY, reason='scipy is required')
class TestDetectSources:
    def setup_class(self):
        self.data = np.array([[0, 1, 0], [0, 2, 0],
                              [0, 0, 0]]).astype(float)
        self.refdata = np.array([[0, 1, 0], [0, 1, 0], [0, 0, 0]])

        kernel = make_2dgaussian_kernel(2.0, size=3)
        self.kernel = kernel

    def test_detection(self):
        """Test basic detection."""
        segm = detect_sources(self.data, threshold=0.9, npixels=2)
        assert_array_equal(segm.data, self.refdata)

        segm = detect_sources(self.data << u.uJy, threshold=0.9 * u.uJy,
                              npixels=2)
        assert_array_equal(segm.data, self.refdata)

        with pytest.raises(ValueError):
            detect_sources(self.data << u.uJy, threshold=0.9, npixels=2)
        with pytest.raises(ValueError):
            detect_sources(self.data, threshold=0.9 * u.Jy, npixels=2)
        with pytest.raises(ValueError):
            detect_sources(self.data << u.uJy, threshold=0.9 * u.m, npixels=2)

    def test_small_sources(self):
        """Test detection where sources are smaller than npixels size."""
        with pytest.warns(NoDetectionsWarning, match='No sources were found'):
            detect_sources(self.data, threshold=0.9, npixels=5)

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

        with pytest.warns(NoDetectionsWarning, match='No sources were found'):
            detect_sources(data, 0, npixels=14)

    def test_zerothresh(self):
        """Test detection with zero threshold."""
        segm = detect_sources(self.data, threshold=0.0, npixels=2)
        assert_array_equal(segm.data, self.refdata)

    def test_zerodet(self):
        """Test detection with large threshold giving no detections."""
        with pytest.warns(NoDetectionsWarning, match='No sources were found'):
            detect_sources(self.data, threshold=7, npixels=2)

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

    def test_mask(self):
        data = np.zeros((11, 11))
        data[3:8, 3:8] = 5.0
        mask = np.zeros(data.shape, dtype=bool)
        mask[4:6, 4:6] = True
        segm1 = detect_sources(data, 1.0, 1.0)
        segm2 = detect_sources(data, 1.0, 1.0, mask=mask)
        assert segm2.areas[0] == segm1.areas[0] - mask.sum()

        # mask with all True
        with pytest.raises(ValueError):
            mask = np.ones(data.shape, dtype=bool)
            detect_sources(data, 1.0, 1.0, mask=mask)

    def test_mask_shape(self):
        with pytest.raises(ValueError):
            detect_sources(self.data, 1.0, 1.0, mask=np.ones((5, 5)))
