# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the harmonics module.
"""

import numpy as np
from astropy.modeling.models import Gaussian2D
from numpy.testing import assert_allclose
from scipy.optimize import leastsq

from photutils.isophote.ellipse import Ellipse
from photutils.isophote.fitter import EllipseFitter
from photutils.isophote.geometry import EllipseGeometry
from photutils.isophote.harmonics import (first_and_second_harmonic_function,
                                          fit_first_and_second_harmonics,
                                          fit_upper_harmonic)
from photutils.isophote.sample import EllipseSample
from photutils.isophote.tests.make_test_data import make_test_image


def test_harmonics_1():
    # this is an almost as-is example taken from stackoverflow
    npts = 100  # number of data points
    theta = np.linspace(0, 4 * np.pi, npts)

    # create artificial data with noise:
    # mean = 0.5, amplitude = 3.0, phase = 0.1, noise-std = 0.01
    rng = np.random.default_rng(0)
    data = 3.0 * np.sin(theta + 0.1) + 0.5 + 0.01 * rng.standard_normal(npts)

    # first guesses for harmonic parameters
    guess_mean = np.mean(data)
    guess_std = 3 * np.std(data) / 2**0.5
    guess_phase = 0

    # Minimize the difference between the actual data and our "guessed"
    # parameters
    def optimize_func(x):
        return x[0] * np.sin(theta + x[1]) + x[2] - data

    est_std, est_phase, est_mean = leastsq(
        optimize_func, [guess_std, guess_phase, guess_mean])[0]

    # recreate the fitted curve using the optimized parameters
    data_fit = est_std * np.sin(theta + est_phase) + est_mean
    residual = data - data_fit

    assert_allclose(np.mean(residual), 0.0, atol=0.001)
    assert_allclose(np.std(residual), 0.01, atol=0.01)


def test_harmonics_2():
    # this uses the actual functional form used for fitting ellipses
    npts = 100
    theta = np.linspace(0, 4 * np.pi, npts)

    y0_0 = 100.0
    a1_0 = 10.0
    b1_0 = 5.0
    a2_0 = 8.0
    b2_0 = 2.0
    rng = np.random.default_rng(0)
    data = (y0_0 + a1_0 * np.sin(theta) + b1_0 * np.cos(theta)
            + a2_0 * np.sin(2 * theta) + b2_0 * np.cos(2 * theta)
            + 0.01 * rng.standard_normal(npts))

    harmonics = fit_first_and_second_harmonics(theta, data)
    y0, a1, b1, a2, b2 = harmonics[0]
    data_fit = (y0 + a1 * np.sin(theta) + b1 * np.cos(theta)
                + a2 * np.sin(2 * theta) + b2 * np.cos(2 * theta)
                + 0.01 * rng.standard_normal(npts))
    residual = data - data_fit

    assert_allclose(np.mean(residual), 0.0, atol=0.01)
    assert_allclose(np.std(residual), 0.015, atol=0.01)


def test_harmonics_3():
    """
    Tests an upper harmonic fit.
    """
    npts = 100
    theta = np.linspace(0, 4 * np.pi, npts)
    y0_0 = 100.0
    a1_0 = 10.0
    b1_0 = 5.0
    order = 3
    rng = np.random.default_rng(0)
    data = (y0_0 + a1_0 * np.sin(order * theta) + b1_0 * np.cos(order * theta)
            + 0.01 * rng.standard_normal(npts))

    harmonic = fit_upper_harmonic(theta, data, order)
    y0, a1, b1 = harmonic[0]
    rng = np.random.default_rng(0)
    data_fit = (y0 + a1 * np.sin(order * theta) + b1 * np.cos(order * theta)
                + 0.01 * rng.standard_normal(npts))
    residual = data - data_fit

    assert_allclose(np.mean(residual), 0.0, atol=0.01)
    assert_allclose(np.std(residual), 0.015, atol=0.014)


class TestFitEllipseSamples:
    def setup_class(self):
        # major axis parallel to X image axis
        self.data1 = make_test_image(seed=0)

        # major axis tilted 45 deg wrt X image axis
        self.data2 = make_test_image(pa=np.pi / 4, seed=0)

    def test_fit_ellipsesample_1(self):
        sample = EllipseSample(self.data1, 40.0)
        s = sample.extract()

        harmonics = fit_first_and_second_harmonics(s[0], s[2])
        y0, a1, b1, a2, b2 = harmonics[0]

        assert_allclose(np.mean(y0), 200.019, atol=0.001)
        assert_allclose(np.mean(a1), -0.000138, atol=0.001)
        assert_allclose(np.mean(b1), 0.000254, atol=0.001)
        assert_allclose(np.mean(a2), -5.658e-05, atol=0.001)
        assert_allclose(np.mean(b2), -0.00911, atol=0.001)

        # check that harmonics subtract nicely
        model = first_and_second_harmonic_function(
            s[0], np.array([y0, a1, b1, a2, b2]))
        residual = s[2] - model

        assert_allclose(np.mean(residual), 0.0, atol=0.001)
        assert_allclose(np.std(residual), 0.015, atol=0.01)

    def test_fit_ellipsesample_2(self):
        # initial guess is rounder than actual image
        sample = EllipseSample(self.data1, 40.0, eps=0.1)
        s = sample.extract()

        harmonics = fit_first_and_second_harmonics(s[0], s[2])
        y0, a1, b1, a2, b2 = harmonics[0]

        assert_allclose(np.mean(y0), 188.686, atol=0.001)
        assert_allclose(np.mean(a1), 0.000283, atol=0.001)
        assert_allclose(np.mean(b1), 0.00692, atol=0.001)
        assert_allclose(np.mean(a2), -0.000215, atol=0.001)
        assert_allclose(np.mean(b2), 10.153, atol=0.001)

    def test_fit_ellipsesample_3(self):
        # initial guess for center is offset
        sample = EllipseSample(self.data1, x0=220.0, y0=210.0, sma=40.0)
        s = sample.extract()

        harmonics = fit_first_and_second_harmonics(s[0], s[2])
        y0, a1, b1, a2, b2 = harmonics[0]

        assert_allclose(np.mean(y0), 152.660, atol=0.001)
        assert_allclose(np.mean(a1), 55.338, atol=0.001)
        assert_allclose(np.mean(b1), 33.091, atol=0.001)
        assert_allclose(np.mean(a2), 33.036, atol=0.001)
        assert_allclose(np.mean(b2), -14.306, atol=0.001)

    def test_fit_ellipsesample_4(self):
        sample = EllipseSample(self.data2, 40.0, eps=0.4)
        s = sample.extract()

        harmonics = fit_first_and_second_harmonics(s[0], s[2])
        y0, a1, b1, a2, b2 = harmonics[0]

        assert_allclose(np.mean(y0), 245.102, atol=0.001)
        assert_allclose(np.mean(a1), -0.003108, atol=0.001)
        assert_allclose(np.mean(b1), -0.0578, atol=0.001)
        assert_allclose(np.mean(a2), 28.781, atol=0.001)
        assert_allclose(np.mean(b2), -63.184, atol=0.001)

    def test_fit_upper_harmonics(self):
        data = make_test_image(noise=1.0e-10, seed=0)
        sample = EllipseSample(data, 40)
        fitter = EllipseFitter(sample)
        iso = fitter.fit(maxit=400)

        assert_allclose(iso.a3, 6.825e-7, atol=1.0e-8)
        assert_allclose(iso.b3, -1.68e-6, atol=1.0e-8)
        assert_allclose(iso.a4, 4.36e-6, atol=1.0e-8)
        assert_allclose(iso.b4, -4.73e-5, atol=1.0e-7)

        assert_allclose(iso.a3_err, 8.152e-6, atol=1.0e-7)
        assert_allclose(iso.b3_err, 8.115e-6, atol=1.0e-7)
        assert_allclose(iso.a4_err, 7.501e-6, atol=1.0e-7)
        assert_allclose(iso.b4_err, 7.473e-6, atol=1.0e-7)


def test_upper_harmonics_sign():
    """
    Regression test for #1486/#1501.
    """
    angle = 40.0 * np.pi / 180.0
    g1 = Gaussian2D(100.0, 75, 75, 15, 3, theta=angle)
    g2 = Gaussian2D(100.0, 75, 75, 10, 8, theta=angle)

    ny = nx = 150
    y, x = np.mgrid[0:ny, 0:nx]
    data = g1(x, y) + g2(x, y)
    geometry = EllipseGeometry(x0=75, y0=75, sma=20, eps=0.9, pa=angle)
    ellipse = Ellipse(data, geometry)
    isolist = ellipse.fit_image()

    # test image is "disky: disky isophotes have b4 > 0
    # (boxy isophotes have b4 < 0)
    assert np.all(isolist.b4[30:] > 0)
    assert isolist.a3[-1] < 0
    assert isolist.a4[-1] < 0
    assert isolist.b3[-1] > 0
    assert isolist.b4[-1] > 0
