from __future__ import (absolute_import, division, print_function, unicode_literals)

import unittest

import numpy as np
from astropy.io import fits

from photutils.isophote import build_test_data
from photutils.isophote.build_test_data import DEFAULT_POS
from photutils.isophote.geometry import Geometry, DEFAULT_EPS
from photutils.isophote.integrator import MEAN, MEDIAN
from photutils.isophote.harmonics import fit_1st_and_2nd_harmonics
from photutils.isophote.sample import Sample, CentralSample
from photutils.isophote.isophote import Isophote
from photutils.isophote.fitter import Fitter, CentralFitter
from photutils.isophote.tests.test_data import TEST_DATA


class TestFitter(unittest.TestCase):

    def test_gradient(self):

        test_data = build_test_data.build()

        sample = Sample(test_data, 40.)
        sample.update()

        self.assertAlmostEqual(sample.mean, 200.02, 2)
        self.assertAlmostEqual(sample.gradient, -4.222, 3)
        self.assertAlmostEqual(sample.gradient_error, 0.0003, 1)
        self.assertAlmostEqual(sample.gradient_relative_error, 7.45e-05, 2)
        self.assertAlmostEqual(sample.sector_area, 2.00, 2)

    def test_fitting_raw(self):
        # this test performs a raw (no Fitter), 1-step
        # correction in one single ellipse coefficient.

        test_data = build_test_data.build()

        # pick first guess ellipse that is off in just
        # one of the parameters (eps).
        sample = Sample(test_data, 40., eps=2*DEFAULT_EPS)
        sample.update()
        s = sample.extract()

        harmonics = fit_1st_and_2nd_harmonics(s[0], s[2])
        y0, a1, b1, a2, b2 = harmonics[0]

        # when eps is off, b2 is the largest (in absolute value).
        self.assertGreater(abs(b2), abs(a1))
        self.assertGreater(abs(b2), abs(b1))
        self.assertGreater(abs(b2), abs(a2))

        correction = b2 * 2. * (1. - sample.geometry.eps) / sample.geometry.sma / sample.gradient
        new_eps = sample.geometry.eps - correction

        # got closer to test data (eps=0.2)
        self.assertAlmostEqual(new_eps, 0.21, 2)

    def test_fitting_small_radii(self):

        test_data = build_test_data.build()

        sample = Sample(test_data, 2.)
        fitter = Fitter(sample)

        isophote = fitter.fit()

        self.assertIsInstance(isophote, Isophote)
        self.assertEqual(isophote.ndata, 13)

    def test_fitting_eps(self):

        test_data = build_test_data.build()

        # initial guess is off in the eps parameter
        sample = Sample(test_data, 40., eps=2*DEFAULT_EPS)
        fitter = Fitter(sample)

        isophote = fitter.fit()

        self.assertIsInstance(isophote, Isophote)
        g = isophote.sample.geometry
        self.assertGreaterEqual(g.eps, 0.19)
        self.assertLessEqual(g.eps, 0.21)

    def test_fitting_pa(self):

        test_data = build_test_data.build(pa=np.pi/4, noise=0.01)

        # initial guess is off in the pa parameter
        sample = Sample(test_data, 40)
        fitter = Fitter(sample)

        isophote = fitter.fit()

        g = isophote.sample.geometry
        self.assertGreaterEqual(g.pa, np.pi/4 - 0.05)
        self.assertLessEqual(g.pa, np.pi/4 + 0.05)

    def test_fitting_xy(self):

        pos_ = DEFAULT_POS - 5
        test_data = build_test_data.build(x0=pos_, y0=pos_)

        # initial guess is off in the x0 and y0 parameters
        sample = Sample(test_data, 40)
        fitter = Fitter(sample)

        isophote = fitter.fit()

        g = isophote.sample.geometry
        self.assertGreaterEqual(g.x0, pos_ - 1)
        self.assertLessEqual(g.x0,    pos_ + 1)
        self.assertGreaterEqual(g.y0, pos_ - 1)
        self.assertLessEqual(g.y0,    pos_ + 1)

    def test_fitting_all(self):

        # build test image that is off from the defaults
        # assumed by the Sample constructor.
        POS = DEFAULT_POS - 5
        ANGLE = np.pi / 4
        EPS = 2 * DEFAULT_EPS
        test_data = build_test_data.build(x0=POS, y0=POS, eps=EPS, pa=ANGLE)

        sma = 60.

        # initial guess is off in all parameters. We find that the initial
        # guesses, especially for position angle, must be kinda close to the
        # actual value. 20% off max seems to work in this case of high SNR.
        sample = Sample(test_data, sma, position_angle=(1.2 * ANGLE))

        fitter = Fitter(sample)
        isophote = fitter.fit()

        self.assertEqual(isophote.stop_code, 0)

        g = isophote.sample.geometry
        self.assertGreaterEqual(g.x0, POS - 1.5)      # position within 1.5 pixel
        self.assertLessEqual(g.x0,    POS + 1.5)
        self.assertGreaterEqual(g.y0, POS - 1.5)
        self.assertLessEqual(g.y0,    POS + 1.5)
        self.assertGreaterEqual(g.eps, EPS - 0.01)    # eps within 0.01
        self.assertLessEqual(g.eps,    EPS + 0.01)
        self.assertGreaterEqual(g.pa, ANGLE - 0.05)   # pa within 5 deg
        self.assertLessEqual(g.pa,    ANGLE + 0.05)

        sample_m = Sample(test_data, sma, position_angle=(1.2 * ANGLE), integrmode=MEAN)

        fitter_m = Fitter(sample_m)
        isophote_m = fitter_m.fit()

        self.assertEqual(isophote_m.stop_code, 0)

    def test_m51(self):
        image = fits.open(TEST_DATA + "M51.fits")
        test_data = image[0].data

        #
        # # here we evaluate the detailed convergency behavior
        # # for a particular ellipse where we can see the eps
        # # parameter jumping back and forth.
        # sample = Sample(test_data, 13.31000001, eps=0.16, position_angle=((-37.5+90)/180.*np.pi))
        # sample.update()
        # fitter = Fitter(sample)
        # isophote = fitter.fit()
        #

        # we start the fit with initial values taken from
        # previous isophote, as determined by the old code.

        # sample taken in high SNR region
        sample = Sample(test_data, 21.44, eps=0.18, position_angle=(36./180.*np.pi))
        fitter = Fitter(sample)
        isophote = fitter.fit()

        self.assertEqual(isophote.ndata, 119)
        self.assertAlmostEqual(isophote.intens, 685.4, 1)

        # last sample taken by the original code, before turning inwards.
        sample = Sample(test_data, 61.16, eps=0.219, position_angle=((77.5+90)/180*np.pi))
        fitter = Fitter(sample)
        isophote = fitter.fit()

        self.assertEqual(isophote.ndata, 382)
        self.assertAlmostEqual(isophote.intens, 155.0, 1)

    def test_m51_central(self):
        image = fits.open(TEST_DATA + "M51.fits")
        test_data = image[0].data

        # this code finds central x and y offset by about 0.1 pixel wrt the
        # spp code. In here we use as input the position computed by this
        # code, thus this test is checking just the extraction algorithm.
        g = Geometry(257.02, 258.1, 0.0, 0.0, 0.0, 0.1, False)
        sample = CentralSample(test_data, 0.0, geometry=g)
        fitter = CentralFitter(sample)

        isophote = fitter.fit()

        # the central pixel intensity is about 3% larger than
        # found by the spp code.
        self.assertEqual(isophote.ndata, 1)
        self.assertLessEqual(isophote.intens, 7560.)
        self.assertGreaterEqual(isophote.intens, 7550.)
