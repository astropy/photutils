# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the depths module.
"""

import astropy.units as u
import numpy as np
import pytest
from astropy.convolution import convolve
from astropy.tests.helper import assert_quantity_allclose
from astropy.utils.exceptions import AstropyUserWarning
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

        npixels = 10
        threshold = 3.2
        finder = SourceFinder(npixels=npixels, progress_bar=False)
        segment_map = finder(convolved_data, threshold)
        self.data = data
        self.mask = segment_map.make_source_mask()

    @pytest.mark.parametrize('units', bool_vals)
    @pytest.mark.parametrize('overlap', bool_vals)
    def test_image_depth(self, units, overlap):
        radius = 4
        depth = ImageDepth(radius, nsigma=5.0, napers=100, niters=2,
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
        radius = 4
        depth = ImageDepth(radius, nsigma=5.0, napers=100, niters=2,
                           mask_pad=5, overlap=True, seed=123, zeropoint=23.9,
                           progress_bar=False)
        limits = depth(self.data, mask=None)
        assert_allclose(limits, (79.348118, 19.151158))

    def test_many_apertures(self):
        radius = 4
        depth = ImageDepth(radius, nsigma=5.0, napers=5000, niters=2,
                           mask_pad=5, overlap=True, seed=123, zeropoint=23.9,
                           progress_bar=False)
        mask = np.zeros(self.data.shape)
        mask[:, 20:] = True
        match = 'Too many apertures for given unmasked area'
        with pytest.raises(ValueError, match=match):
            depth(self.data, mask)

        depth = ImageDepth(radius, nsigma=5.0, napers=250, niters=2,
                           mask_pad=5, overlap=False, seed=123, zeropoint=23.9,
                           progress_bar=False)
        mask = np.zeros(self.data.shape)
        mask[:, 100:] = True
        match = r'Unable to generate .* non-overlapping apertures'
        with pytest.warns(AstropyUserWarning, match=match):
            depth(self.data, mask)

        # test for zero non-overlapping apertures before slow loop
        radius = 5
        depth = ImageDepth(radius, nsigma=5.0, napers=100, niters=2,
                           overlap=False, seed=123, zeropoint=23.9,
                           progress_bar=False)
        mask = np.zeros(self.data.shape)
        mask[:, 40:] = True
        match = r'Unable to generate .* non-overlapping apertures'
        with pytest.warns(AstropyUserWarning, match=match):
            depth(self.data, mask)

    def test_zero_data(self):
        radius = 4
        depth = ImageDepth(radius, napers=500, niters=2,
                           overlap=True, seed=123, progress_bar=False)
        data = np.zeros((300, 400))
        mask = None
        match = 'One or more flux_limit values was zero'
        with pytest.warns(AstropyUserWarning, match=match):
            limits = depth(data, mask)
        assert_allclose(limits, (0.0, np.inf))

    def test_all_masked(self):
        radius = 4
        depth = ImageDepth(radius, napers=500, niters=1, mask_pad=5,
                           overlap=True, seed=123, progress_bar=False)
        data = np.zeros(self.data.shape)
        mask = np.zeros(data.shape, dtype=bool)
        mask[:, 10:] = True
        match = 'There are no unmasked pixel values'
        with pytest.raises(ValueError, match=match):
            depth(data, mask)

    def test_inputs(self):
        match = 'aper_radius must be > 0'
        with pytest.raises(ValueError, match=match):
            ImageDepth(0.0, nsigma=5.0, napers=500, niters=2,
                       overlap=True, seed=123, zeropoint=23.9,
                       progress_bar=False)

        match = 'aper_radius must be > 0'
        with pytest.raises(ValueError, match=match):
            ImageDepth(-12.4, nsigma=5.0, napers=500, niters=2,
                       overlap=True, seed=123, zeropoint=23.9,
                       progress_bar=False)

        match = 'mask_pad must be >= 0'
        with pytest.raises(ValueError, match=match):
            ImageDepth(12.4, nsigma=5.0, napers=500, niters=2,
                       mask_pad=-7.1, overlap=True, seed=123, zeropoint=23.9,
                       progress_bar=False)

    def test_repr(self):
        depth = ImageDepth(aper_radius=4, nsigma=5.0, napers=100, niters=2,
                           overlap=False, seed=123, zeropoint=23.9,
                           progress_bar=False)
        cls_repr = repr(depth)
        assert cls_repr.startswith(f'{depth.__class__.__name__}')
