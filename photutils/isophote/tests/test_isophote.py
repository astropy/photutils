# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
from numpy.testing import assert_allclose
import pytest

from astropy.io import fits

from .make_test_data import make_test_image
from ..fitter import EllipseFitter
from ..isophote import Isophote, IsophoteList
from ..sample import EllipseSample
from ...datasets import get_path

try:
    import scipy    # noqa
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


@pytest.mark.remote_data
@pytest.mark.skipif('not HAS_SCIPY')
class TestIsophote:
    def setup_class(self):
        path = get_path('isophote/M51.fits', location='photutils-datasets',
                        cache=True)
        hdu = fits.open(path)
        self.data = hdu[0].data
        hdu.close()

    def test_fit(self):
        # low noise image, fitted perfectly by sample
        data = make_test_image(noise=1.e-10, random_state=123)
        sample = EllipseSample(data, 40)
        fitter = EllipseFitter(sample)
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
        sample = EllipseSample(self.data, 21.44)
        fitter = EllipseFitter(sample)
        iso = fitter.fit()

        assert iso.valid
        assert iso.stop_code == 0 or iso.stop_code == 2

        # geometry
        g = iso.sample.geometry
        assert g.x0 >= (257 - 1.5)   # position within 1.5 pixel
        assert g.x0 <= (257 + 1.5)
        assert g.y0 >= (259 - 1.5)
        assert g.y0 <= (259 + 2.0)
        assert g.eps >= (0.19 - 0.05)  # eps within 0.05
        assert g.eps <= (0.19 + 0.05)
        assert g.pa >= (0.62 - 0.05)  # pa within 5 deg
        assert g.pa <= (0.62 + 0.05)

        # fitted values
        assert_allclose(iso.intens, 682.9, atol=0.1)
        assert_allclose(iso.rms, 83.27, atol=0.01)
        assert_allclose(iso.int_err, 7.63, atol=0.01)
        assert_allclose(iso.pix_stddev, 117.8, atol=0.1)
        assert_allclose(iso.grad, -36.08, atol=0.1)

        # integrals
        assert iso.tflux_e <= 1.20e6
        assert iso.tflux_e >= 1.19e6
        assert iso.tflux_c <= 1.38e6
        assert iso.tflux_c >= 1.36e6

        # deviations from perfect ellipticity
        assert abs(iso.a3) <= 0.05
        assert abs(iso.b3) <= 0.05
        assert abs(iso.a4) <= 0.05
        assert abs(iso.b4) <= 0.05

    def test_m51_niter(self):
        # compares with old STSDAS task. In this task, the
        # default for the starting value of SMA is 10; it
        # fits with 20 iterations.
        sample = EllipseSample(self.data, 10)
        fitter = EllipseFitter(sample)
        iso = fitter.fit()

        assert iso.valid
        assert iso.niter == 50


class TestIsophoteList:
    def setup_class(self):
        data = make_test_image(random_state=123)
        self.slen = 5
        self.isolist_sma10 = self.build_list(data, sma0=10., slen=self.slen)
        self.isolist_sma100 = self.build_list(data, sma0=100., slen=self.slen)
        self.isolist_sma200 = self.build_list(data, sma0=200., slen=self.slen)

    @staticmethod
    def build_list(data, sma0, slen=5):
        iso_list = []
        for k in range(slen):
            sample = EllipseSample(data, float(k + sma0))
            sample.update()
            iso_list.append(Isophote(sample, k, True, 0))
        result = IsophoteList(iso_list)
        return result

    def test_basic_list(self):
        # make sure it can be indexed as a list.
        result = self.isolist_sma10[:]
        assert isinstance(result[0], Isophote)

        # make sure the important arrays contain floats.
        # especially the sma array, which is derived
        # from a property in the Isophote class.
        assert isinstance(result.sma, np.ndarray)
        assert isinstance(result.sma[0], float)
        assert isinstance(result.intens, np.ndarray)
        assert isinstance(result.intens[0], float)
        assert isinstance(result.rms, np.ndarray)
        assert isinstance(result.int_err, np.ndarray)
        assert isinstance(result.pix_stddev, np.ndarray)
        assert isinstance(result.grad, np.ndarray)
        assert isinstance(result.grad_error, np.ndarray)
        assert isinstance(result.grad_r_error, np.ndarray)
        assert isinstance(result.sarea, np.ndarray)
        assert isinstance(result.niter, np.ndarray)
        assert isinstance(result.ndata, np.ndarray)
        assert isinstance(result.nflag, np.ndarray)
        assert isinstance(result.valid, np.ndarray)
        assert isinstance(result.stop_code, np.ndarray)
        assert isinstance(result.tflux_c, np.ndarray)
        assert isinstance(result.tflux_e, np.ndarray)
        assert isinstance(result.npix_c, np.ndarray)
        assert isinstance(result.npix_e, np.ndarray)
        assert isinstance(result.a3, np.ndarray)
        assert isinstance(result.a4, np.ndarray)
        assert isinstance(result.b3, np.ndarray)
        assert isinstance(result.b4, np.ndarray)

        samples = result.sample
        assert isinstance(samples, list)
        assert isinstance(samples[0], EllipseSample)

        iso = result.get_closest(13.6)
        assert isinstance(iso, Isophote)
        assert_allclose(iso.sma, 14., atol=1e-6)

    def test_extend(self):
        # the extend method shouldn't return anything,
        # and should modify the first list in place.
        inner_list = self.isolist_sma10[:]
        outer_list = self.isolist_sma100[:]
        assert len(inner_list) == self.slen
        assert len(outer_list) == self.slen
        dummy = inner_list.extend(outer_list)
        assert not dummy
        assert len(inner_list) == 2 * self.slen

        # the __iadd__ operator should behave like the
        # extend method.
        inner_list = self.isolist_sma10[:]
        outer_list = self.isolist_sma100[:]
        inner_list += outer_list
        assert len(inner_list) == 2 * self.slen

        # the __add__ operator should create a new IsophoteList
        # instance with the result, and should not modify
        # the operands.
        inner_list = self.isolist_sma10[:]
        outer_list = self.isolist_sma100[:]
        result = inner_list + outer_list
        assert isinstance(result, IsophoteList)
        assert len(inner_list) == self.slen
        assert len(outer_list) == self.slen
        assert len(result) == 2 * self.slen

    def test_slicing(self):
        iso_list = self.isolist_sma10[:]
        assert len(iso_list) == self.slen
        assert len(iso_list[1:-1]) == self.slen - 2
        assert len(iso_list[2:-2]) == self.slen - 4

    def test_combined(self):
        # combine extend with slicing.
        inner_list = self.isolist_sma10[:]
        outer_list = self.isolist_sma100[:]
        sublist = inner_list[2:-2]
        dummy = sublist.extend(outer_list)
        assert not dummy
        assert len(sublist) == 2*self.slen - 4

        # try one more slice.
        even_outer_list = self.isolist_sma200
        sublist.extend(even_outer_list[1:-1])
        assert len(sublist) == 2*self.slen - 4 + 3

        # combine __add__ with slicing.
        sublist = inner_list[2:-2]
        result = sublist + outer_list
        assert isinstance(result, IsophoteList)
        assert len(sublist) == self.slen - 4
        assert len(result) == 2*self.slen - 4

        result = inner_list[2:-2] + outer_list
        assert isinstance(result, IsophoteList)
        assert len(result) == 2*self.slen - 4

    def test_sort(self):
        inner_list = self.isolist_sma10[:]
        outer_list = self.isolist_sma100[:]
        result = outer_list[2:-2] + inner_list

        assert result[-1].sma < result[0].sma
        result.sort()
        assert result[-1].sma > result[0].sma
