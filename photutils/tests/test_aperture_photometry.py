# Licensed under a 3-clause BSD style license - see LICENSE.rst

# The tests in this file test the accuracy of the photometric results.
# Here we test directly with aperture objects since we are checking the
# algorithms in aperture_photometry, not in the wrappers.

import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal_nulp

from ..aperture import CircularAperture,\
                       CircularAnnulus, \
                       EllipticalAperture, \
                       EllipticalAnnulus, \
                       aperture_photometry


APERTURES = [CircularAperture(3.),
             CircularAnnulus(3., 5.),
             EllipticalAperture(3., 5., 1.),
             EllipticalAnnulus(3., 5., 4., 1.)]


@pytest.mark.parametrize(('aperture'), APERTURES)
def test_outside_array(aperture):
    data = np.ones((10, 10), dtype=np.float)
    flux = aperture_photometry(data, -60., 60., aperture)
    assert flux == 0.  # aperture is fully outside array


@pytest.mark.parametrize(('aperture'), APERTURES)
def test_inside_array_simple(aperture):
    data = np.ones((40, 40), dtype=np.float)
    flux = aperture_photometry(data, 20., 20., aperture, subpixels=10)
    true_flux = aperture.area()
    assert abs((flux - true_flux) / true_flux) < 0.01


class BaseTestErrorGain(object):

    def test_scalar_error_no_gain(self):

        # Scalar error, no gain.
        error = 1.
        flux, fluxerr = aperture_photometry(self.data, self.xc, self.yc, self.aperture,
                                            error=error)
        assert abs((flux - self.true_flux) / self.true_flux) < 0.01

        # Error should be exact to machine precision for apertures
        # with defined area.
        true_error = error * np.sqrt(self.area)
        assert_array_almost_equal_nulp(fluxerr, true_error, 1)

    def test_scalar_error_scalar_gain(self):

        # Scalar error, scalar gain.
        error = 1.
        gain = 1.
        flux, fluxerr = aperture_photometry(self.data, self.xc, self.yc, self.aperture,
                                            error=error, gain=gain)

        assert abs((flux - self.true_flux) / self.true_flux) < 0.01
        true_error = np.sqrt(error ** 2 * self.area + flux)
        assert_array_almost_equal_nulp(fluxerr, true_error, 1)

    def test_scalar_error_array_gain(self):

        # Scalar error, Array gain.
        error = 1.
        gain = np.ones(self.data.shape, dtype=np.float)
        flux, fluxerr = aperture_photometry(self.data, self.xc, self.yc, self.aperture,
                                            error=error, gain=gain)
        assert abs((flux - self.true_flux) / self.true_flux) < 0.01
        true_error = np.sqrt(error ** 2 * self.area + flux)
        assert abs((fluxerr - true_error) / true_error) < 0.01

    def test_array_error_no_gain(self):

        # Array error, no gain.
        error = np.ones(self.data.shape, dtype=np.float)
        flux, fluxerr = aperture_photometry(self.data, self.xc, self.yc, self.aperture,
                                            error=error)
        assert abs((flux - self.true_flux) / self.true_flux) < 0.01
        true_error = np.sqrt(self.area)
        assert abs((fluxerr - true_error) / true_error) < 0.01

    def test_array_error_scalar_gain(self):

        # Array error, scalar gain.
        error = np.ones(self.data.shape, dtype=np.float)
        gain = 1.
        flux, fluxerr = aperture_photometry(self.data, self.xc, self.yc, self.aperture,
                                            error=error, gain=gain)
        assert abs((flux - self.true_flux) / self.true_flux) < 0.01
        true_error = np.sqrt(self.area + flux)
        assert abs((fluxerr - true_error) / true_error) < 0.01

    def test_array_error_array_gain(self):

        # Array error, Array gain.
        error = np.ones(self.data.shape, dtype=np.float)
        gain = np.ones(self.data.shape, dtype=np.float)
        flux, fluxerr = aperture_photometry(self.data, self.xc, self.yc, self.aperture,
                                            error=error, gain=gain)
        assert abs((flux - self.true_flux) / self.true_flux) < 0.01
        true_error = np.sqrt(self.area + flux)
        assert abs((fluxerr - true_error) / true_error) < 0.01


class TestErrorGainCircular(BaseTestErrorGain):

    def setup_class(self):
        self.data = np.ones((40, 40), dtype=np.float)
        self.xc = 20.
        self.yc = 20.
        r = 10.
        self.aperture = CircularAperture(r)
        self.area = np.pi * r * r
        self.true_flux = self.area


class TestErrorGainCircularAnnulus(BaseTestErrorGain):

    def setup_class(self):
        self.data = np.ones((40, 40), dtype=np.float)
        self.xc = 20.
        self.yc = 20.
        r_in = 8.
        r_out = 10.
        self.aperture = CircularAnnulus(r_in, r_out)
        self.area = np.pi * (r_out * r_out - r_in * r_in)
        self.true_flux = self.area


class TestErrorGainElliptical(BaseTestErrorGain):

    def setup_class(self):
        self.data = np.ones((40, 40), dtype=np.float)
        self.xc = 20.
        self.yc = 20.
        a = 10.
        b = 5.
        theta = -np.pi / 4.
        self.aperture = EllipticalAperture(a, b, theta)
        self.area = np.pi * a * b
        self.true_flux = self.area


class TestErrorGainEllipticalAnnulus(BaseTestErrorGain):

    def setup_class(self):
        self.data = np.ones((40, 40), dtype=np.float)
        self.xc = 20.
        self.yc = 20.
        a_in = 5.
        a_out = 8.
        b_out = 6.
        theta = -np.pi / 4.
        self.aperture = EllipticalAnnulus(a_in, a_out, b_out, theta)
        self.area = np.pi * (a_out * b_out) - np.pi * (a_in * b_out * a_in / a_out)
        self.true_flux = self.area
