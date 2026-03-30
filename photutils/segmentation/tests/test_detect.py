# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the detect module.
"""

import astropy.units as u
import numpy as np
import pytest
from astropy.stats import SigmaClip
from numpy.testing import assert_allclose, assert_equal

from photutils.segmentation.detect import detect_sources, detect_threshold
from photutils.segmentation.utils import make_2dgaussian_kernel
from photutils.utils.exceptions import NoDetectionsWarning

DATA = np.array([[0, 1, 0], [0, 2, 0], [0, 0, 0]]).astype(float)
REF1 = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])


class TestDetectThreshold:
    def test_nsigma(self):
        """
        Test basic n_sigma.
        """
        threshold = detect_threshold(DATA, n_sigma=0.1)
        ref = 0.4 * np.ones((3, 3))
        assert_allclose(threshold, ref)

        threshold = detect_threshold(DATA << u.uJy, n_sigma=0.1)
        assert isinstance(threshold, u.Quantity)
        assert_allclose(threshold.value, ref)

    def test_nsigma_zero(self):
        """
        Test n_sigma=0.
        """
        threshold = detect_threshold(DATA, n_sigma=0.0)
        ref = (1.0 / 3.0) * np.ones((3, 3))
        assert_allclose(threshold, ref)

    def test_background(self):
        """
        Test background.
        """
        threshold = detect_threshold(DATA, n_sigma=1.0, background=1)
        ref = (5.0 / 3.0) * np.ones((3, 3))
        assert_allclose(threshold, ref)

    def test_background_image(self):
        """
        Test background image.
        """
        background = np.ones((3, 3))
        threshold = detect_threshold(DATA, n_sigma=1.0, background=background)
        ref = (5.0 / 3.0) * np.ones((3, 3))
        assert_allclose(threshold, ref)

        threshold = detect_threshold(DATA << u.Jy, n_sigma=1.0,
                                     background=background << u.Jy)
        assert isinstance(threshold, u.Quantity)
        assert_allclose(threshold.value, ref)

    def test_background_badshape(self):
        """
        Test background badshape.
        """
        wrong_shape = np.zeros((2, 2))
        match = 'input background is 2D, then it must have the same shape'
        with pytest.raises(ValueError, match=match):
            detect_threshold(DATA, n_sigma=2.0, background=wrong_shape)

    def test_error(self):
        """
        Test error.
        """
        threshold = detect_threshold(DATA, n_sigma=1.0, error=1)
        ref = (4.0 / 3.0) * np.ones((3, 3))
        assert_allclose(threshold, ref)

    def test_error_image(self):
        """
        Test error image.
        """
        error = np.ones((3, 3))
        threshold = detect_threshold(DATA, n_sigma=1.0, error=error)
        ref = (4.0 / 3.0) * np.ones((3, 3))
        assert_allclose(threshold, ref)

        threshold = detect_threshold(DATA << u.Jy, n_sigma=1.0,
                                     error=error << u.Jy)
        assert isinstance(threshold, u.Quantity)
        assert_allclose(threshold.value, ref)

    def test_error_badshape(self):
        """
        Test error badshape.
        """
        wrong_shape = np.zeros((2, 2))
        match = 'If input error is 2D, then it must have the same shape'
        with pytest.raises(ValueError, match=match):
            detect_threshold(DATA, n_sigma=2.0, error=wrong_shape)

    def test_background_error(self):
        """
        Test background error.
        """
        threshold = detect_threshold(DATA, n_sigma=2.0, background=10.0,
                                     error=1.0)
        ref = 12.0 * np.ones((3, 3))
        assert_allclose(threshold, ref)

        threshold = detect_threshold(DATA << u.Jy, n_sigma=2.0,
                                     background=10.0 * u.Jy,
                                     error=1.0 * u.Jy)
        assert isinstance(threshold, u.Quantity)
        assert_allclose(threshold.value, ref)

        match = 'must all have the same units'
        with pytest.raises(ValueError, match=match):
            detect_threshold(DATA << u.Jy, n_sigma=2.0, background=10.0,
                             error=1.0 * u.Jy)
        with pytest.raises(ValueError, match=match):
            detect_threshold(DATA << u.Jy, n_sigma=2.0, background=10.0 * u.m,
                             error=1.0 * u.Jy)

    def test_background_error_images(self):
        """
        Test background error images.
        """
        background = np.ones((3, 3)) * 10.0
        error = np.ones((3, 3))
        threshold = detect_threshold(DATA, n_sigma=2.0, background=background,
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
        threshold = detect_threshold(DATA, n_sigma=1.0, error=0, mask=mask,
                                     sigma_clip=sigma_clip)
        ref = (1.0 / 8.0) * np.ones((3, 3))
        assert_equal(threshold, ref)

    def test_invalid_sigma_clip(self):
        """
        Test invalid sigma clip.
        """
        match = 'sigma_clip must be a SigmaClip object'
        with pytest.raises(TypeError, match=match):
            detect_threshold(DATA, 1.0, sigma_clip=10)


class TestDetectSources:
    def setup_class(self):
        self.data = np.array([[0, 1, 0], [0, 2, 0],
                              [0, 0, 0]]).astype(float)
        self.refdata = np.array([[0, 1, 0], [0, 1, 0], [0, 0, 0]])

        kernel = make_2dgaussian_kernel(2.0, size=3)
        self.kernel = kernel

    def test_detection(self):
        """
        Test basic detection.
        """
        segm = detect_sources(self.data, threshold=0.9, n_pixels=2)
        assert_equal(segm.data, self.refdata)

        assert segm.data.dtype == np.int32
        assert segm.labels.dtype == np.int32

        segm = detect_sources(self.data << u.uJy, threshold=0.9 * u.uJy,
                              n_pixels=2)
        assert_equal(segm.data, self.refdata)

        match = 'must all have the same units'
        with pytest.raises(ValueError, match=match):
            detect_sources(self.data << u.uJy, threshold=0.9, n_pixels=2)
        with pytest.raises(ValueError, match=match):
            detect_sources(self.data, threshold=0.9 * u.Jy, n_pixels=2)
        with pytest.raises(ValueError, match=match):
            detect_sources(self.data << u.uJy, threshold=0.9 * u.m, n_pixels=2)

    def test_small_sources(self):
        """
        Test detection where sources are smaller than n_pixels size.
        """
        match = 'No sources were found'
        with pytest.warns(NoDetectionsWarning, match=match):
            detect_sources(self.data, threshold=0.9, n_pixels=5)

    def test_n_pixels(self):
        """
        Test removal of sources whose size is less than n_pixels.

        Regression tests for #663.
        """
        data = np.zeros((8, 8))
        data[0:4, 0] = 1
        data[0, 0:4] = 1
        data[3, 3:] = 2
        data[3:, 3] = 2

        segm = detect_sources(data, 0, n_pixels=4)
        assert segm.n_labels == 2
        assert segm.data.dtype == np.int32

        # Removal of labels with size less than n_pixels
        # dtype should still be np.int32
        segm = detect_sources(data, 0, n_pixels=8)
        assert segm.n_labels == 1
        assert segm.data.dtype == np.int32

        segm = detect_sources(data, 0, n_pixels=9)
        assert segm.n_labels == 1
        assert segm.data.dtype == np.int32

        data = np.zeros((8, 8))
        data[0:4, 0] = 1
        data[0, 0:4] = 1
        data[3, 2:] = 2
        data[3:, 2] = 2
        data[5:, 3] = 2

        n_pixels = np.arange(9, 14)
        for n_pixels in np.arange(9, 14):
            segm = detect_sources(data, 0, n_pixels=n_pixels)
            assert segm.n_labels == 1
            assert segm.areas[0] == 13

        match = 'No sources were found'
        with pytest.warns(NoDetectionsWarning, match=match):
            detect_sources(data, 0, n_pixels=14)

    def test_zerothresh(self):
        """
        Test detection with zero threshold.
        """
        segm = detect_sources(self.data, threshold=0.0, n_pixels=2)
        assert_equal(segm.data, self.refdata)

    def test_zerodet(self):
        """
        Test detection with large threshold giving no detections.
        """
        match = 'No sources were found'
        with pytest.warns(NoDetectionsWarning, match=match):
            detect_sources(self.data, threshold=7, n_pixels=2)

    def test_8connectivity(self):
        """
        Test detection with connectivity=8.
        """
        data = np.eye(3)
        segm = detect_sources(data, threshold=0.9, n_pixels=1, connectivity=8)
        assert_equal(segm.data, data)

    def test_4connectivity(self):
        """
        Test detection with connectivity=4.
        """
        data = np.eye(3)
        ref = np.diag([1, 2, 3])
        segm = detect_sources(data, threshold=0.9, n_pixels=1, connectivity=4)
        assert_equal(segm.data, ref)

    def test_n_pixels_nonint(self):
        """
        Test if an error is raised when npixel is noninteger.
        """
        match = 'n_pixels must be a positive integer'
        with pytest.raises(ValueError, match=match):
            detect_sources(self.data, threshold=1, n_pixels=0.1)

    def test_n_pixels_negative(self):
        """
        Test if an error is raised when npixel is negative.
        """
        match = 'n_pixels must be a positive integer'
        with pytest.raises(ValueError, match=match):
            detect_sources(self.data, threshold=1, n_pixels=-1)

    @pytest.mark.parametrize('connectivity', [0, -1, 3, 6, 10])
    def test_connectivity_invalid(self, connectivity):
        """
        Test if an error is raised when connectivity is invalid.
        """
        match = f'Invalid connectivity={connectivity}'
        with pytest.raises(ValueError, match=match):
            detect_sources(self.data, threshold=1, n_pixels=1,
                           connectivity=connectivity)

    def test_mask(self):
        """
        Test mask.
        """
        data = np.zeros((11, 11))
        data[3:8, 3:8] = 5.0
        mask = np.zeros(data.shape, dtype=bool)
        mask[4:6, 4:6] = True
        segm1 = detect_sources(data, 1.0, 1.0)
        segm2 = detect_sources(data, 1.0, 1.0, mask=mask)
        assert segm2.areas[0] == segm1.areas[0] - mask.sum()

        # Mask with all True
        mask = np.ones(data.shape, dtype=bool)
        match = 'mask must not be True for every pixel'
        with pytest.raises(ValueError, match=match):
            detect_sources(data, 1.0, 1.0, mask=mask)

    def test_mask_shape(self):
        """
        Test mask shape.
        """
        match = 'mask must have the same shape as the input image'
        with pytest.raises(ValueError, match=match):
            detect_sources(self.data, 1.0, 1.0, mask=np.ones((5, 5)))
