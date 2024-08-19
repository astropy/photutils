# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the integrator module.
"""

import numpy as np
import pytest
from astropy.io import fits
from numpy.testing import assert_allclose

from photutils.datasets import get_path
from photutils.isophote.integrator import (BILINEAR, MEAN, MEDIAN,
                                           NEAREST_NEIGHBOR)
from photutils.isophote.sample import EllipseSample


@pytest.mark.remote_data
class TestData:
    def setup_class(self):
        path = get_path('isophote/synth_highsnr.fits',
                        location='photutils-datasets', cache=True)
        hdu = fits.open(path)
        self.data = hdu[0].data
        hdu.close()

    def make_sample(self, masked=False, sma=40.0, integrmode=BILINEAR):
        if masked:
            data = np.ma.masked_values(self.data, 200.0, atol=10.0, rtol=0.0)
        else:
            data = self.data
        sample = EllipseSample(data, sma, integrmode=integrmode)
        s = sample.extract()

        assert len(s) == 3
        assert len(s[0]) == len(s[1])
        assert len(s[0]) == len(s[2])

        return s, sample


@pytest.mark.remote_data
class TestUnmasked(TestData):
    def test_bilinear(self):
        s, sample = self.make_sample()

        assert len(s[0]) == 225
        # intensities
        assert_allclose(np.mean(s[2]), 200.76, atol=0.01)
        assert_allclose(np.std(s[2]), 21.55, atol=0.01)
        # radii
        assert_allclose(np.max(s[1]), 40.0, atol=0.01)
        assert_allclose(np.min(s[1]), 32.0, atol=0.01)

        assert sample.total_points == 225
        assert sample.actual_points == 225

    def test_bilinear_small(self):
        # small radius forces sub-pixel sampling
        s, sample = self.make_sample(sma=10.0)

        # intensities
        assert_allclose(np.mean(s[2]), 1045.4, atol=0.1)
        assert_allclose(np.std(s[2]), 143.0, atol=0.1)
        # radii
        assert_allclose(np.max(s[1]), 10.0, atol=0.1)
        assert_allclose(np.min(s[1]), 8.0, atol=0.1)

        assert sample.total_points == 57
        assert sample.actual_points == 57

    def test_nearest_neighbor(self):
        s, sample = self.make_sample(integrmode=NEAREST_NEIGHBOR)

        assert len(s[0]) == 225
        # intensities
        assert_allclose(np.mean(s[2]), 201.1, atol=0.1)
        assert_allclose(np.std(s[2]), 21.8, atol=0.1)
        # radii
        assert_allclose(np.max(s[1]), 40.0, atol=0.01)
        assert_allclose(np.min(s[1]), 32.0, atol=0.01)

        assert sample.total_points == 225
        assert sample.actual_points == 225

    def test_mean(self):
        s, sample = self.make_sample(integrmode=MEAN)

        assert len(s[0]) == 64
        # intensities
        assert_allclose(np.mean(s[2]), 199.9, atol=0.1)
        assert_allclose(np.std(s[2]), 21.3, atol=0.1)
        # radii
        assert_allclose(np.max(s[1]), 40.0, atol=0.01)
        assert_allclose(np.min(s[1]), 32.0, atol=0.01)

        assert_allclose(sample.sector_area, 12.4, atol=0.1)
        assert sample.total_points == 64
        assert sample.actual_points == 64

    def test_mean_small(self):
        s, sample = self.make_sample(sma=5.0, integrmode=MEAN)

        assert len(s[0]) == 29
        # intensities
        assert_allclose(np.mean(s[2]), 2339.0, atol=0.1)
        assert_allclose(np.std(s[2]), 284.7, atol=0.1)
        # radii
        assert_allclose(np.max(s[1]), 5.0, atol=0.01)
        assert_allclose(np.min(s[1]), 4.0, atol=0.01)

        assert_allclose(sample.sector_area, 2.0, atol=0.1)
        assert sample.total_points == 29
        assert sample.actual_points == 29

    def test_median(self):
        s, sample = self.make_sample(integrmode=MEDIAN)

        assert len(s[0]) == 64
        # intensities
        assert_allclose(np.mean(s[2]), 199.9, atol=0.1)
        assert_allclose(np.std(s[2]), 21.3, atol=0.1)
        # radii
        assert_allclose(np.max(s[1]), 40.0, atol=0.01)
        assert_allclose(np.min(s[1]), 32.01, atol=0.01)

        assert_allclose(sample.sector_area, 12.4, atol=0.1)
        assert sample.total_points == 64
        assert sample.actual_points == 64


@pytest.mark.remote_data
class TestMasked(TestData):
    def test_bilinear(self):
        s, sample = self.make_sample(masked=True, integrmode=BILINEAR)

        assert len(s[0]) == 157
        # intensities
        assert_allclose(np.mean(s[2]), 201.52, atol=0.01)
        assert_allclose(np.std(s[2]), 25.21, atol=0.01)
        # radii
        assert_allclose(np.max(s[1]), 40.0, atol=0.01)
        assert_allclose(np.min(s[1]), 32.0, atol=0.01)

        assert sample.total_points == 225
        assert sample.actual_points == 157

    def test_mean(self):
        s, sample = self.make_sample(masked=True, integrmode=MEAN)

        assert len(s[0]) == 51
        # intensities
        assert_allclose(np.mean(s[2]), 199.9, atol=0.1)
        assert_allclose(np.std(s[2]), 24.12, atol=0.1)
        # radii
        assert_allclose(np.max(s[1]), 40.0, atol=0.01)
        assert_allclose(np.min(s[1]), 32.0, atol=0.01)

        assert_allclose(sample.sector_area, 12.4, atol=0.1)
        assert sample.total_points == 64
        assert sample.actual_points == 51
