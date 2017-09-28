from __future__ import (absolute_import, division, print_function, unicode_literals)

import pytest

import numpy as np
from astropy.io import fits

from photutils.isophote.sample import Sample
from photutils.isophote.fitter import Fitter
from photutils.isophote.isophote import Isophote, IsophoteList
from photutils.isophote.tests.test_data import TEST_DATA

from .make_test_data import make_test_image

try:
    import scipy
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


@pytest.mark.skipif('not HAS_SCIPY')
class TestIsophote(object):

    def test_fit(self):

        # low noise image, fitted perfectly by sample.
        test_data = make_test_image(noise=1.e-10, random_state=123)
        sample = Sample(test_data, 40)
        fitter = Fitter(sample)
        iso = fitter.fit(maxit=400)

        assert iso.valid
        assert iso.stop_code == 0 or iso.stop_code == 2

        # fitted values
        assert iso.intens <= 201.
        assert iso.intens >= 199.
        assert iso.int_err <= 0.0010
        assert iso.int_err >= 0.0009
        assert iso.pix_stddev <= 0.03
        assert iso.pix_stddev >= 0.02
        assert abs(iso.grad) <= 4.25
        assert abs(iso.grad) >= 4.20

        # integrals
        assert iso.tflux_e <= 1.85E6
        assert iso.tflux_e >= 1.82E6
        assert iso.tflux_c <= 2.025E6
        assert iso.tflux_c >= 2.022E6

        # deviations from perfect ellipticity
        assert abs(iso.a3) <= 0.01
        assert abs(iso.b3) <= 0.01
        assert abs(iso.a4) <= 0.01
        assert abs(iso.b4) <= 0.01

    def test_m51(self):

        image = fits.open(TEST_DATA + "M51.fits")
        test_data = image[0].data

        sample = Sample(test_data, 21.44)
        fitter = Fitter(sample)
        iso = fitter.fit()

        assert iso.valid
        assert iso.stop_code == 0 or iso.stop_code == 2

        # geometry
        g = iso.sample.geometry
        assert g.x0 >=  (257 - 1.5)   # position within 1.5 pixel
        assert g.x0 <=  (257 + 1.5)
        assert g.y0 >=  (259 - 1.5)
        assert g.y0 <=  (259 + 2.0)
        assert g.eps >= (0.19 - 0.05) # eps within 0.05
        assert g.eps <= (0.19 + 0.05)
        assert g.pa >=  (0.62 - 0.05) # pa within 5 deg
        assert g.pa <=  (0.62 + 0.05)

        # fitted values
        assert iso.intens     == pytest.approx(682.9, abs=0.1)
        assert iso.rms        == pytest.approx(83.27, abs=0.01)
        assert iso.int_err    == pytest.approx(7.63, abs=0.01)
        assert iso.pix_stddev == pytest.approx(117.8, abs=0.1)
        assert iso.grad       == pytest.approx(-36.08, abs=0.1)

        # integrals
        assert iso.tflux_e <= 1.20E6
        assert iso.tflux_e >= 1.19E6
        assert iso.tflux_c <= 1.38E6
        assert iso.tflux_c >= 1.36E6

        # deviations from perfect ellipticity
        assert abs(iso.a3) <= 0.05
        assert abs(iso.b3) <= 0.05
        assert abs(iso.a4) <= 0.05
        assert abs(iso.b4) <= 0.05

    def test_m51_niter(self):
        # compares with old STSDAS task. In this task, the
        # default for the starting value of SMA is 10; it
        # fits with 20 iterations.
        image = fits.open(TEST_DATA + "M51.fits")
        test_data = image[0].data

        sample = Sample(test_data, 10)
        fitter = Fitter(sample)
        iso = fitter.fit()

        assert iso.valid
        assert iso.niter == 50


class TestIsophoteList(object):

    def _build_list(self, sma0):
        test_data = make_test_image(random_state=123)
        iso_list = []
        for k in range(10):
            sample = Sample(test_data, float(k + sma0))
            sample.update()
            iso_list.append(Isophote(sample, k, True, 0))
        result = IsophoteList(iso_list)
        return result

    def test_basic_list(self):
        result = self._build_list(10.)

        # make sure it can be indexed as a list.
        assert isinstance(result[0], Isophote)

        array = np.array([])
        # make sure the important arrays contain floats.
        # especially the sma array, which is derived
        # from a property in the Isophote class.
        assert type(result.sma) == type(array)
        assert isinstance(result.sma[0], float)

        assert type(result.intens) == type(array)
        assert isinstance(result.intens[0], float)

        assert type(result.rms) == type(array)
        assert type(result.int_err) == type(array)
        assert type(result.pix_stddev) == type(array)
        assert type(result.grad) == type(array)
        assert type(result.grad_error) == type(array)
        assert type(result.grad_r_error) == type(array)
        assert type(result.sarea) == type(array)
        assert type(result.niter) == type(array)
        assert type(result.ndata) == type(array)
        assert type(result.nflag) == type(array)
        assert type(result.valid) == type(array)
        assert type(result.stop_code) == type(array)
        assert type(result.tflux_c) == type(array)
        assert type(result.tflux_e) == type(array)
        assert type(result.npix_c) == type(array)
        assert type(result.npix_e) == type(array)
        assert type(result.a3) == type(array)
        assert type(result.a4) == type(array)
        assert type(result.b3) == type(array)
        assert type(result.b4) == type(array)

        samples = result.sample
        assert isinstance(samples, list)
        assert isinstance(samples[0], Sample)

        iso = result.get_closest(13.6)
        assert isinstance(iso, Isophote)
        assert iso.sma == pytest.approx(14., abs=0.000001)

    def test_extend(self):

        # the extend method shouldn't return anything,
        # and should modify the first list in place.
        inner_list = self._build_list(10.)
        outer_list = self._build_list(100.)
        dummy = inner_list.extend(outer_list)
        assert not dummy
        assert len(inner_list) == 20

        # the __iadd__ operator should behave like the
        # extend method.
        inner_list = self._build_list(10.)
        outer_list = self._build_list(100.)
        inner_list += outer_list
        assert len(inner_list) == 20

        # the __add__ operator should create a new IsophoteList
        # instance with the result, and should not modify
        # the operands.
        inner_list = self._build_list(10.)
        outer_list = self._build_list(100.)
        result = inner_list + outer_list
        assert isinstance(result, IsophoteList)
        assert len(inner_list) == 10
        assert len(outer_list) == 10
        assert len(result) == 20

    def test_slicing(self):
        iso_list = self._build_list(10.)
        assert len(iso_list) == 10
        assert len(iso_list[1:-1]) == 8
        assert len(iso_list[2:-2]) == 6

    def test_combined(self):

        # combine extend with slicing.
        inner_list = self._build_list(10.)
        outer_list = self._build_list(100.)
        sublist = inner_list[2:-2]
        dummy = sublist.extend(outer_list)
        assert not dummy
        assert len(sublist) == 16

        # try one more slice.
        even_outer_list = self._build_list(200.)
        sublist.extend(even_outer_list[3:-3])
        assert len(sublist) == 20

        # combine __add__ with slicing.
        sublist = inner_list[2:-2]
        result = sublist + outer_list
        assert isinstance(result, IsophoteList)
        assert len(sublist) == 6
        assert len(result) == 16

        result = inner_list[2:-2] + outer_list
        assert isinstance(result, IsophoteList)
        assert len(result) == 16

    def test_sort(self):
        inner_list = self._build_list(10.)
        outer_list = self._build_list(100.)

        result = outer_list[2:-2] + inner_list

        assert result[-1].sma < result[0].sma
        result.sort()
        assert result[-1].sma > result[0].sma



