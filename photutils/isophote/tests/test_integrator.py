# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
import numpy.ma as ma
import pytest

from astropy.io import fits
from astropy.tests.helper import remote_data

from ..sample import Sample
from ..integrator import NEAREST_NEIGHBOR, BI_LINEAR, MEAN, MEDIAN
from ...datasets import get_path


@remote_data
class TestData(object):

    def setup_class(self):
        path = get_path('isophote/synth_highsnr.fits',
                        location='photutils-datasets', cache=True)
        hdu = fits.open(path)
        self.data = hdu[0].data
        hdu.close()


@remote_data
class TestUnmasked(TestData):

    def _init_test(self, sma=40., integrmode=BI_LINEAR):

        sample = Sample(self.data, sma, integrmode=integrmode)

        s = sample.extract()

        assert len(s) == 3
        assert len(s[0]) == len(s[1])
        assert len(s[0]) == len(s[2])

        return s, sample

    def test_bilinear(self):

        s, sample = self._init_test()

        assert len(s[0]) == 225
        # intensities
        assert np.mean(s[2]) == pytest.approx(200.76, abs=0.01)
        assert np.std(s[2])  == pytest.approx(21.55,  abs=0.01)
        # radii
        assert np.max(s[1]) == pytest.approx(40.0, abs=0.01)
        assert np.min(s[1]) == pytest.approx(32.0,  abs=0.01)

        assert sample.total_points  == 225
        assert sample.actual_points == 225

    def test_bilinear_small(self):

        # small radius forces sub-pixel sampling
        s, sample = self._init_test(sma=10.)

        # intensities
        assert np.mean(s[2]) == pytest.approx(1045.4, abs=0.1)
        assert np.std(s[2])  == pytest.approx(143.0,  abs=0.1)
        # radii
        assert np.max(s[1]) == pytest.approx(10.0, abs=0.1)
        assert np.min(s[1]) == pytest.approx(8.0,  abs=0.1)

        assert sample.total_points  == 57
        assert sample.actual_points == 57

    def test_nearest_neighbor(self):

        s, sample = self._init_test(integrmode=NEAREST_NEIGHBOR)

        assert len(s[0]) == 225
        # intensities
        assert np.mean(s[2]) == pytest.approx(201.1, abs=0.1)
        assert np.std(s[2])  == pytest.approx(21.8,  abs=0.1)
        # radii
        assert np.max(s[1]) == pytest.approx(40.0, abs=0.01)
        assert np.min(s[1]) == pytest.approx(32.0, abs=0.01)

        assert sample.total_points  == 225
        assert sample.actual_points == 225

    def test_mean(self):

        s, sample = self._init_test(integrmode=MEAN)

        assert len(s[0]) == 64
        # intensities
        assert np.mean(s[2]) == pytest.approx(199.9, abs=0.1)
        assert np.std(s[2])  == pytest.approx(21.3,  abs=0.1)
        # radii
        assert np.max(s[1]) == pytest.approx(40.0, abs=0.01)
        assert np.min(s[1]) == pytest.approx(32.0, abs=0.01)

        assert sample.sector_area == pytest.approx(12.4, abs=0.1)
        assert sample.total_points  == 64
        assert sample.actual_points == 64

    def test_mean_small(self):

        s, sample = self._init_test(sma=5., integrmode=MEAN)

        assert len(s[0]) == 29
        # intensities
        assert np.mean(s[2]) == pytest.approx(2339.0, abs=0.1)
        assert np.std(s[2])  == pytest.approx(284.7,  abs=0.1)
        # radii
        assert np.max(s[1]) == pytest.approx(5.0, abs=0.01)
        assert np.min(s[1]) == pytest.approx(4.0, abs=0.01)

        assert sample.sector_area == pytest.approx(2.0, abs=0.1)
        assert sample.total_points  == 29
        assert sample.actual_points == 29

    def test_median(self):

        s, sample = self._init_test(integrmode=MEDIAN)

        assert len(s[0]) == 64
        # intensities
        assert np.mean(s[2]) == pytest.approx(199.9, abs=0.1)
        assert np.std(s[2])  == pytest.approx(21.3,  abs=0.1)
        # radii
        assert np.max(s[1]) == pytest.approx(40.0,  abs=0.01)
        assert np.min(s[1]) == pytest.approx(32.01, abs=0.01)

        assert sample.sector_area == pytest.approx(12.4, abs=0.1)
        assert sample.total_points  == 64
        assert sample.actual_points == 64


@remote_data
class TestMasked(TestData):

    def _init_test(self, sma=40., integrmode=BI_LINEAR):

        data = ma.masked_values(self.data, 200., atol=10.0, rtol=0.)

        sample = Sample(data, sma, integrmode=integrmode)

        s = sample.extract()

        assert len(s) == 3
        assert len(s[0]) == len(s[1])
        assert len(s[0]) == len(s[2])

        return s, sample

    def test_bilinear(self):

        s, sample = self._init_test()

        assert len(s[0]) == 157
        # intensities
        assert np.mean(s[2]) == pytest.approx(201.52, abs=0.01)
        assert np.std(s[2])  == pytest.approx(25.21,  abs=0.01)
        # radii
        assert np.max(s[1]) == pytest.approx(40.0, abs=0.01)
        assert np.min(s[1]) == pytest.approx(32.0,  abs=0.01)

        assert sample.total_points  == 225
        assert sample.actual_points == 157

    def test_mean(self):

        s, sample = self._init_test(integrmode=MEAN)

        assert len(s[0]) == 51
        # intensities
        assert np.mean(s[2]) == pytest.approx(199.9, abs=0.1)
        assert np.std(s[2])  == pytest.approx(24.12,  abs=0.1)
        # radii
        assert np.max(s[1]) == pytest.approx(40.0, abs=0.01)
        assert np.min(s[1]) == pytest.approx(32.0, abs=0.01)

        assert sample.sector_area == pytest.approx(12.4, abs=0.1)
        assert sample.total_points  == 64
        assert sample.actual_points == 51

