# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the depths module.
"""

import astropy.units as u
import numpy as np
import pytest
from astropy.convolution import convolve
from astropy.tests.helper import assert_quantity_allclose
from astropy.utils.exceptions import (AstropyDeprecationWarning,
                                      AstropyUserWarning)
from numpy.testing import assert_allclose

from photutils.datasets import make_100gaussians_image
from photutils.segmentation import SourceFinder, make_2dgaussian_kernel
from photutils.utils._optional_deps import HAS_SKIMAGE
from photutils.utils.depths import ImageDepth

bool_vals = (True, False)


@pytest.mark.skipif(not HAS_SKIMAGE, reason='skimage is required')
class TestImageDepth:
    def setup_class(self):
        bkg = 5.0
        data = make_100gaussians_image() - bkg
        kernel = make_2dgaussian_kernel(3.0, size=5)
        convolved_data = convolve(data, kernel)

        n_pixels = 10
        threshold = 3.2
        finder = SourceFinder(n_pixels=n_pixels, progress_bar=False)
        segment_map = finder(convolved_data, threshold)
        self.data = data
        self.mask = segment_map.make_source_mask()

    @pytest.mark.parametrize('units', bool_vals)
    @pytest.mark.parametrize('overlap', bool_vals)
    def test_image_depth(self, units, overlap):
        """
        Test ImageDepth with various unit and overlap settings.
        """
        radius = 4
        depth = ImageDepth(radius, n_sigma=5.0, n_apertures=100, n_iters=2,
                           mask_pad=5, overlap=overlap, seed=123,
                           zeropoint=23.9, progress_bar=False)

        if overlap:
            exp_limits = (72.65695364143787, 19.246807037943814)
        else:
            exp_limits = (71.07332848526178, 19.27073336332396)

        data = self.data
        fluxlim = exp_limits[0]
        if units:
            data = self.data * u.Jy
            fluxlim *= u.Jy

        limits = depth(data, self.mask)
        assert_allclose(limits[1], exp_limits[1])

        if not units:
            assert_allclose(limits[0], fluxlim)
        else:
            assert_quantity_allclose(limits[0], fluxlim)

    def test_mask_none(self):
        """
        Test ImageDepth with mask=None.
        """
        radius = 4
        depth = ImageDepth(radius, n_sigma=5.0, n_apertures=100, n_iters=2,
                           mask_pad=5, overlap=True, seed=123, zeropoint=23.9,
                           progress_bar=False)
        limits = depth(self.data, mask=None)
        assert_allclose(limits, (79.348118, 19.151158))

    def test_many_apertures(self):
        """
        Test ImageDepth with too many apertures.
        """
        radius = 4
        depth = ImageDepth(radius, n_sigma=5.0, n_apertures=5000, n_iters=2,
                           mask_pad=5, overlap=True, seed=123, zeropoint=23.9,
                           progress_bar=False)
        mask = np.zeros(self.data.shape)
        mask[:, 20:] = True
        match = 'Too many apertures for given unmasked area'
        with pytest.raises(ValueError, match=match):
            depth(self.data, mask)

        depth = ImageDepth(radius, n_sigma=5.0, n_apertures=250, n_iters=2,
                           mask_pad=5, overlap=False, seed=123, zeropoint=23.9,
                           progress_bar=False)
        mask = np.zeros(self.data.shape)
        mask[:, 100:] = True
        match = r'Unable to generate .* non-overlapping apertures'
        with pytest.warns(AstropyUserWarning, match=match):
            depth(self.data, mask)

        # Test for zero non-overlapping apertures before slow loop
        radius = 5
        depth = ImageDepth(radius, n_sigma=5.0, n_apertures=100, n_iters=2,
                           overlap=False, seed=123, zeropoint=23.9,
                           progress_bar=False)
        mask = np.zeros(self.data.shape)
        mask[:, 40:] = True
        match = r'Unable to generate .* non-overlapping apertures'
        with pytest.warns(AstropyUserWarning, match=match):
            depth(self.data, mask)

    def test_zero_data(self):
        """
        Test ImageDepth with all-zero data.
        """
        radius = 4
        depth = ImageDepth(radius, n_apertures=500, n_iters=2,
                           overlap=True, seed=123, progress_bar=False)
        data = np.zeros((300, 400))
        mask = None
        match = 'One or more flux_limit values was zero'
        with pytest.warns(AstropyUserWarning, match=match):
            limits = depth(data, mask)
        assert_allclose(limits, (0.0, np.inf))

    def test_all_masked(self):
        """
        Test ImageDepth when all pixels are masked.
        """
        radius = 4
        depth = ImageDepth(radius, n_apertures=500, n_iters=1, mask_pad=5,
                           overlap=True, seed=123, progress_bar=False)
        data = np.zeros(self.data.shape)
        mask = np.zeros(data.shape, dtype=bool)
        mask[:, 10:] = True
        match = 'There are no unmasked pixel values'
        with pytest.raises(ValueError, match=match):
            depth(data, mask)

    def test_mask_not_modified(self):
        """
        Test that the input mask is not modified in place.
        """
        radius = 4
        depth = ImageDepth(radius, n_sigma=5.0, n_apertures=100, n_iters=2,
                           mask_pad=5, overlap=True, seed=123,
                           zeropoint=23.9, progress_bar=False)
        mask_orig = np.zeros(self.data.shape, dtype=bool)
        mask_copy = mask_orig.copy()
        depth(self.data, mask_orig)
        np.testing.assert_array_equal(mask_orig, mask_copy)

        # Also when mask has no True pixels (border-only masking path)
        mask_empty = np.zeros(self.data.shape, dtype=bool)
        mask_empty_copy = mask_empty.copy()
        depth(self.data, mask_empty)
        np.testing.assert_array_equal(mask_empty, mask_empty_copy)

    def test_inputs(self):
        """
        Test ImageDepth with invalid input parameters.
        """
        match = 'aper_radius must be > 0'
        with pytest.raises(ValueError, match=match):
            ImageDepth(0.0, n_sigma=5.0, n_apertures=500, n_iters=2,
                       overlap=True, seed=123, zeropoint=23.9,
                       progress_bar=False)

        match = 'aper_radius must be > 0'
        with pytest.raises(ValueError, match=match):
            ImageDepth(-12.4, n_sigma=5.0, n_apertures=500, n_iters=2,
                       overlap=True, seed=123, zeropoint=23.9,
                       progress_bar=False)

        match = 'mask_pad must be >= 0'
        with pytest.raises(ValueError, match=match):
            ImageDepth(12.4, n_sigma=5.0, n_apertures=500, n_iters=2,
                       mask_pad=-7.1, overlap=True, seed=123, zeropoint=23.9,
                       progress_bar=False)

        match = 'sigma_clip must be a callable'
        with pytest.raises(TypeError, match=match):
            ImageDepth(4.0, n_sigma=5.0, n_apertures=500, n_iters=2,
                       sigma_clip='not_callable', progress_bar=False)

    def test_repr(self):
        """
        Test ImageDepth __repr__ output.
        """
        depth = ImageDepth(aper_radius=4, n_sigma=5.0, n_apertures=100,
                           n_iters=2, overlap=False, seed=123,
                           zeropoint=23.9, progress_bar=False)
        cls_repr = repr(depth)
        assert cls_repr.startswith(f'{depth.__class__.__name__}')

    def test_progress_bar(self):
        """
        Test running ImageDepth with progress_bar=True.
        """
        radius = 4
        depth = ImageDepth(radius, n_sigma=5.0, n_apertures=100, n_iters=1,
                           mask_pad=5, overlap=True, seed=123,
                           zeropoint=23.9, progress_bar=True)
        limits = depth(self.data, self.mask)
        assert np.isfinite(limits[0])
        assert np.isfinite(limits[1])

    def test_deprecation(self):
        """
        Test ImageDepth deprecation warnings.
        """
        depth = ImageDepth(aper_radius=4, n_sigma=5.0, n_apertures=100,
                           n_iters=2, mask_pad=5, overlap=True, seed=123,
                           zeropoint=23.9, progress_bar=False)
        match = 'attribute was deprecated'
        with pytest.warns(AstropyDeprecationWarning, match=match):
            _ = depth.napers
