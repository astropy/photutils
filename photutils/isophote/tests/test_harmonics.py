# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
from numpy.testing import assert_allclose
import pytest

from .make_test_data import make_test_image
from ..harmonics import (fit_first_and_second_harmonics, fit_upper_harmonic,
                         first_and_second_harmonic_function)
from ..sample import EllipseSample

try:
    from scipy.optimize import leastsq    # noqa
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


@pytest.mark.skipif('not HAS_SCIPY')
def test_harmonics_1():
    # this is an almost as-is example taken from stackoverflow
    N = 100  # number of data points
    t = np.linspace(0, 4*np.pi, N)

    # create artificial data with noise:
    # mean = 0.5, amplitude = 3., phase = 0.1, noise-std = 0.01
    data = 3.0 * np.sin(t + 0.1) + 0.5 + 0.01 * np.random.randn(N)

    # first guesses for harmonic parameters
    guess_mean = np.mean(data)
    guess_std = 3 * np.std(data) / 2**0.5
    guess_phase = 0

    # Minimize the difference between the actual data and our "guessed"
    # parameters
    # optimize_func = lambda x: x[0] * np.sin(t + x[1]) + x[2] - data
    def optimize_func(x):
        return x[0] * np.sin(t + x[1]) + x[2] - data

    est_std, est_phase, est_mean = leastsq(
        optimize_func, [guess_std, guess_phase, guess_mean])[0]

    # recreate the fitted curve using the optimized parameters
    data_fit = est_std * np.sin(t + est_phase) + est_mean
    residual = data - data_fit

    assert_allclose(np.mean(residual), 0., atol=0.001)
    assert_allclose(np.std(residual), 0.01, atol=0.01)


@pytest.mark.skipif('not HAS_SCIPY')
def test_harmonics_2():
    # this uses the actual functional form used for fitting ellipses
    N = 100
    E = np.linspace(0, 4*np.pi, N)

    y0_0 = 100.
    a1_0 = 10.
    b1_0 = 5.
    a2_0 = 8.
    b2_0 = 2.
    data = (y0_0 + a1_0*np.sin(E) + b1_0*np.cos(E) + a2_0*np.sin(2*E) +
            b2_0*np.cos(2*E) + 0.01*np.random.randn(N))

    harmonics = fit_first_and_second_harmonics(E, data)
    y0, a1, b1, a2, b2 = harmonics[0]
    data_fit = (y0 + a1*np.sin(E) + b1*np.cos(E) + a2*np.sin(2*E) +
                b2*np.cos(2*E) + 0.01*np.random.randn(N))
    residual = data - data_fit

    assert_allclose(np.mean(residual), 0., atol=0.01)
    assert_allclose(np.std(residual), 0.015, atol=0.01)


@pytest.mark.skipif('not HAS_SCIPY')
def test_harmonics_3():
    """Tests an upper harmonic fit."""

    N = 100
    E = np.linspace(0, 4*np.pi, N)
    y0_0 = 100.
    a1_0 = 10.
    b1_0 = 5.
    order = 3
    data = (y0_0 + a1_0*np.sin(order*E) + b1_0*np.cos(order*E) +
            0.01*np.random.randn(N))

    harmonic = fit_upper_harmonic(E, data, order)
    y0, a1, b1 = harmonic[0]
    data_fit = (y0 + a1*np.sin(order*E) + b1*np.cos(order*E) +
                0.01*np.random.randn(N))
    residual = data - data_fit

    assert_allclose(np.mean(residual), 0., atol=0.01)
    assert_allclose(np.std(residual), 0.015, atol=0.01)


@pytest.mark.skipif('not HAS_SCIPY')
class TestFitEllipseSamples:
    def setup_class(self):
        # major axis parallel to X image axis
        self.data1 = make_test_image(random_state=123)

        # major axis tilted 45 deg wrt X image axis
        self.data2 = make_test_image(pa=np.pi/4, random_state=123)

    def test_fit_ellipsesample_1(self):
        sample = EllipseSample(self.data1, 40.)
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

        assert_allclose(np.mean(residual), 0., atol=0.001)
        assert_allclose(np.std(residual), 0.015, atol=0.01)

    def test_fit_ellipsesample_2(self):
        # initial guess is rounder than actual image
        sample = EllipseSample(self.data1, 40., eps=0.1)
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
        sample = EllipseSample(self.data1, x0=220., y0=210., sma=40.)
        s = sample.extract()

        harmonics = fit_first_and_second_harmonics(s[0], s[2])
        y0, a1, b1, a2, b2 = harmonics[0]

        assert_allclose(np.mean(y0), 152.660, atol=0.001)
        assert_allclose(np.mean(a1), 55.338, atol=0.001)
        assert_allclose(np.mean(b1), 33.091, atol=0.001)
        assert_allclose(np.mean(a2), 33.036, atol=0.001)
        assert_allclose(np.mean(b2), -14.306, atol=0.001)

    def test_fit_ellipsesample_4(self):
        sample = EllipseSample(self.data2, 40., eps=0.4)
        s = sample.extract()

        harmonics = fit_first_and_second_harmonics(s[0], s[2])
        y0, a1, b1, a2, b2 = harmonics[0]

        assert_allclose(np.mean(y0), 245.102, atol=0.001)
        assert_allclose(np.mean(a1), -0.003108, atol=0.001)
        assert_allclose(np.mean(b1), -0.0578, atol=0.001)
        assert_allclose(np.mean(a2), 28.781, atol=0.001)
        assert_allclose(np.mean(b2), -63.184, atol=0.001)
