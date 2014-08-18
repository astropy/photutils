# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from astropy.tests.helper import pytest
import numpy as np
from numpy.testing import assert_array_equal, assert_allclose
from ..lacosmic import lacosmic

try:
    import scipy
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

IMG = np.random.RandomState(1234567890).randn(5, 5) * 0.5
x, y = [0, 3, 4, 2, 1], [0, 2, 1, 4, 2]
CR = np.zeros(IMG.shape)
CR[y, x] = 10.
MASK_REF = CR.astype(np.bool)
CR_IMG = IMG + CR


@pytest.mark.skipif('not HAS_SCIPY')
class TestLACosmic(object):
    def test_lacosmic(self):
        """Test basic lacosmic."""
        crclean_img, crmask_img = lacosmic(CR_IMG, 3, 2, 2, gain=1,
                                           readnoise=0)
        assert_allclose(crclean_img, IMG, atol=0.76)
        assert_array_equal(crmask_img, MASK_REF)

    def test_background_scalar(self):
        """Test lacosmic with a background scalar."""
        crclean_img, crmask_img = lacosmic(CR_IMG, 3, 1, 1, gain=1,
                                           readnoise=0, background=10)
        assert_allclose(crclean_img, IMG, atol=0.76)
        assert_array_equal(crmask_img, MASK_REF)

    def test_background_maxiter(self):
        """Test lacosmic with a background scalar."""
        crclean_img, crmask_img = lacosmic(CR_IMG, 3, 1, 1, gain=1,
                                           readnoise=0, background=10,
                                           maxiter=1)
        assert_allclose(crclean_img, IMG, atol=0.76)
        assert_array_equal(crmask_img, MASK_REF)

    def test_background_image(self):
        """Test lacosmic with a 2D background image."""
        bkgrd_img = np.ones(IMG.shape) * 10.
        crclean_img, crmask_img = lacosmic(CR_IMG, 3, 1, 1, gain=1,
                                           readnoise=0, background=bkgrd_img)
        assert_allclose(crclean_img, IMG, atol=0.76)
        assert_array_equal(crmask_img, MASK_REF)

    def test_mask_image(self):
        """Test lacosmic with an input mask image."""
        mask = MASK_REF.copy()
        mask[2, 1] = False
        mask_ref2 = np.logical_and(MASK_REF, ~mask)
        crclean_img, crmask_img = lacosmic(CR_IMG, 3, 2, 2, gain=1,
                                           readnoise=0, mask=mask)
        assert_allclose(crclean_img * mask_ref2, IMG * mask_ref2, atol=0.76)
        assert_array_equal(crmask_img, mask_ref2)

    def test_error_image(self):
        """Test lacosmic with an input error image."""
        error_img = np.sqrt(IMG.clip(min=0.001))
        crclean_img, crmask_img = lacosmic(CR_IMG, 3, 2, 2, gain=1,
                                           readnoise=0, uncertainty=error_img)
        assert_allclose(crclean_img, IMG, atol=0.76)
        assert_array_equal(crmask_img, MASK_REF)

    def test_large_cosmics(self):
        """Test lacosmic cleaning with large cosmic rays."""
        test_img = np.ones((7, 7))
        test_img[1:6, 1:6] = 100.
        mask_ref2 = np.zeros((7, 7), dtype=np.bool)
        mask_ref2[1:6, 1:6] = True
        crclean_img, crmask_img = lacosmic(test_img, 3, 2, 2, gain=1,
                                           readnoise=0)
        assert_array_equal(crmask_img, mask_ref2)

    def test_error_image_size(self):
        """
        Test if AssertionError raises if shape of error_img doesn't
        match image.  """
        error_img = np.zeros((7, 7))
        with pytest.raises(AssertionError):
            crclean_img, crmask_img = lacosmic(CR_IMG, 3, 2, 2, gain=1,
                                               readnoise=0,
                                               uncertainty=error_img)
