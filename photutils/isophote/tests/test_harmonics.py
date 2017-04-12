from __future__ import (absolute_import, division, print_function, unicode_literals)

import numpy as np

import unittest

from photutils.isophote import build_test_data
from photutils.isophote.sample import Sample
from photutils.isophote.harmonics import fit_1st_and_2nd_harmonics, fit_upper_harmonic, first_and_2nd_harmonic_function


class TestHarmonics(unittest.TestCase):

    def test_harmonics_1(self):

        try:
            from scipy.optimize import leastsq
        except ImportError:
            return

        # this is an almost as-is example taken from stackoverflow

        N = 100 # number of data points
        t = np.linspace(0, 4*np.pi, N)

        # create artificial data with noise:
        # mean = 0.5, amplitude = 3., phase = 0.1, noise-std = 0.01
        data = 3.0 * np.sin(t + 0.1) + 0.5 + 0.01 * np.random.randn(N)

        # first guesses for harmonic parameters
        guess_mean = np.mean(data)
        guess_std = 3 * np.std(data)/(2**0.5)
        guess_phase = 0

        # Minimize the difference between the actual data and our "guessed" parameters
        optimize_func = lambda x: x[0] * np.sin(t+x[1]) + x[2] - data

        est_std, est_phase, est_mean = leastsq(optimize_func, [guess_std, guess_phase, guess_mean])[0]

        # recreate the fitted curve using the optimized parameters
        data_fit = est_std * np.sin(t+est_phase) + est_mean
        residual = data - data_fit

        self.assertAlmostEqual(np.mean(residual), 0.000000, 6)
        self.assertAlmostEqual(np.std(residual),  0.01, 2)

    def test_harmonics_2(self):

        # this uses the actual functional form used for fitting ellipses

        N = 100
        E = np.linspace(0, 4*np.pi, N)

        y0_0 = 100.
        a1_0 = 10.
        b1_0 = 5.
        a2_0 = 8.
        b2_0 = 2.
        data = y0_0 + a1_0 * np.sin(E) + b1_0 * np.cos(E) + a2_0 * np.sin(2*E) + b2_0 * np.cos(2*E) + 0.01 * np.random.randn(N)

        harmonics = fit_1st_and_2nd_harmonics(E, data)
        y0, a1, b1, a2, b2 = harmonics[0]
        data_fit = y0 + a1*np.sin(E) + b1*np.cos(E) + a2*np.sin(2*E) + b2* np.cos(2*E) + 0.01 * np.random.randn(N)
        residual = data - data_fit

        self.assertAlmostEqual(np.mean(residual), 0.000, 2)
        self.assertAlmostEqual(np.std(residual),  0.015, 2)

    def test_harmonics_3(self):

        # tests an upper harmonic fit

        N = 100
        E = np.linspace(0, 4*np.pi, N)

        y0_0 = 100.
        a1_0 = 10.
        b1_0 = 5.
        order = 3
        data = y0_0 + a1_0 * np.sin(order*E) + b1_0 * np.cos(order*E) + 0.01 * np.random.randn(N)

        harmonic = fit_upper_harmonic(E, data, order)
        y0, a1, b1 = harmonic[0]
        data_fit = y0 + a1*np.sin(order*E) + b1*np.cos(order*E) + 0.01 * np.random.randn(N)
        residual = data - data_fit

        self.assertAlmostEqual(np.mean(residual), 0.000, 2)
        self.assertAlmostEqual(np.std(residual),  0.015, 2)

    def test_fit_sample_1(self):

        # major axis parallel to X image axis
        test_data = build_test_data.build()

        sample = Sample(test_data, 40.)
        s = sample.extract()

        harmonics = fit_1st_and_2nd_harmonics(s[0], s[2])
        y0, a1, b1, a2, b2 = harmonics[0]

        self.assertAlmostEqual(y0, 200.019, 3)
        self.assertAlmostEqual(a1, -0.000138, 3)
        self.assertAlmostEqual(b1, 0.000254, 3)
        self.assertAlmostEqual(a2, -5.658e-05, 3)
        self.assertAlmostEqual(b2, -0.00911, 3)

        # check that harmonics subtract nicely
        model = first_and_2nd_harmonic_function(s[0], np.array([y0, a1, b1, a2, b2]))
        residual = s[2] - model

        self.assertTrue(np.mean(residual) < 1.E-4)
        self.assertAlmostEqual(np.std(residual),  0.015, 2)

    def test_fit_sample_2(self):

        # major axis tilted 45 deg wrt X image axis
        test_data = build_test_data.build(pa=np.pi/4)

        sample = Sample(test_data, 40., eps=0.4)
        s = sample.extract()

        harmonics = fit_1st_and_2nd_harmonics(s[0], s[2])
        y0, a1, b1, a2, b2 = harmonics[0]

        self.assertAlmostEqual(y0, 245.102, 3)
        self.assertAlmostEqual(a1, -0.00310, 3)
        self.assertAlmostEqual(b1, -0.0578, 3)
        # these drive the correction for position angle
        self.assertAlmostEqual(a2, 28.781, 3)
        self.assertAlmostEqual(b2, -63.184, 3)

    def test_fit_sample_3(self):

        test_data = build_test_data.build()

        # initial guess is rounder than actual image
        sample = Sample(test_data, 40., eps=0.1)
        s = sample.extract()

        harmonics = fit_1st_and_2nd_harmonics(s[0], s[2])
        y0, a1, b1, a2, b2 = harmonics[0]

        self.assertAlmostEqual(y0, 188.686, 3)
        self.assertAlmostEqual(a1, 0.000283, 3)
        self.assertAlmostEqual(b1, 0.00692, 3)
        self.assertAlmostEqual(a2, -0.000215, 3)
        # this drives the correction for ellipticity
        self.assertAlmostEqual(b2, 10.153, 3)

    def test_fit_sample_4(self):

        test_data = build_test_data.build()

        # initial guess for center is offset
        sample = Sample(test_data, x0=220., y0=210., sma=40.)
        s = sample.extract()

        harmonics = fit_1st_and_2nd_harmonics(s[0], s[2])
        y0, a1, b1, a2, b2 = harmonics[0]

        self.assertAlmostEqual(y0, 152.660, 3)
        self.assertAlmostEqual(a1, 55.338, 3)
        self.assertAlmostEqual(b1, 33.091, 3)
        self.assertAlmostEqual(a2, 33.036, 3)
        self.assertAlmostEqual(b2, -14.306, 3)


