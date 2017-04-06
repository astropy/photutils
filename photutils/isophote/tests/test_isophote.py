from __future__ import (absolute_import, division, print_function, unicode_literals)

import unittest

import numpy as np
from astropy.io import fits

from photutils.isophote import build_test_data
from photutils.isophote.sample import Sample
from photutils.isophote.fitter import Fitter
from photutils.isophote.isophote import Isophote, IsophoteList
from photutils.isophote.tests.test_data import TEST_DATA


class TestIsophote(unittest.TestCase):

    def test_fit(self):

        # low noise image, fitted perfectly by sample.
        test_data = build_test_data.build(noise=1.E-10)
        sample = Sample(test_data, 40)
        fitter = Fitter(sample)
        iso = fitter.fit(maxit=400)

        self.assertTrue(iso.valid)
        self.assertTrue(iso.stop_code == 0 or iso.stop_code == 2)

        # fitted values
        self.assertLessEqual(iso.intens,        201., 2)
        self.assertGreaterEqual(iso.intens,     199., 2)
        self.assertLessEqual(iso.int_err,       0.0010, 2)
        self.assertGreaterEqual(iso.int_err,    0.0009, 2)
        self.assertLessEqual(iso.pix_stddev,    0.03, 2)
        self.assertGreaterEqual(iso.pix_stddev, 0.02, 2)
        self.assertLessEqual(abs(iso.grad),     4.25, 2)
        self.assertGreaterEqual(abs(iso.grad),  4.20, 2)

        # integrals
        self.assertLessEqual(iso.tflux_e,    1.85E6, 2)
        self.assertGreaterEqual(iso.tflux_e, 1.82E6, 2)
        self.assertLessEqual(iso.tflux_c,    2.025E6, 2)
        self.assertGreaterEqual(iso.tflux_c, 2.022E6, 2)

        # deviations from perfect ellipticity
        self.assertLessEqual(abs(iso.a3), 0.01, 2)
        self.assertLessEqual(abs(iso.b3), 0.01, 2)
        self.assertLessEqual(abs(iso.a4), 0.01, 2)
        self.assertLessEqual(abs(iso.b4), 0.01, 2)

    def test_m51(self):

        image = fits.open(TEST_DATA + "M51.fits")
        test_data = image[0].data

        sample = Sample(test_data, 21.44)
        fitter = Fitter(sample)
        iso = fitter.fit()

        self.assertTrue(iso.valid)
        self.assertTrue(iso.stop_code == 0 or iso.stop_code == 2)

        # geometry
        g = iso.sample.geometry
        self.assertGreaterEqual(g.x0,  257 - 1.5)   # position within 1.5 pixel
        self.assertLessEqual(g.x0,     257 + 1.5)
        self.assertGreaterEqual(g.y0,  259 - 1.5)
        self.assertLessEqual(g.y0,     259 + 2.0)
        self.assertGreaterEqual(g.eps, 0.19 - 0.05) # eps within 0.05
        self.assertLessEqual(g.eps,    0.19 + 0.05)
        self.assertGreaterEqual(g.pa,  0.62 - 0.05) # pa within 5 deg
        self.assertLessEqual(g.pa,     0.62 + 0.05)

        # fitted values
        self.assertAlmostEqual(iso.intens,     682.9,  1)
        self.assertAlmostEqual(iso.rms,         83.27, 2)
        self.assertAlmostEqual(iso.int_err,      7.63, 2)
        self.assertAlmostEqual(iso.pix_stddev, 117.8,  1)
        self.assertAlmostEqual(iso.grad,       -36.08, 2)

        # integrals
        self.assertLessEqual(iso.tflux_e,    1.20E6, 2)
        self.assertGreaterEqual(iso.tflux_e, 1.19E6, 2)
        self.assertLessEqual(iso.tflux_c,    1.38E6, 2)
        self.assertGreaterEqual(iso.tflux_c, 1.36E6, 2)

        # deviations from perfect ellipticity
        self.assertLessEqual(abs(iso.a3), 0.05, 2)
        self.assertLessEqual(abs(iso.b3), 0.05, 2)
        self.assertLessEqual(abs(iso.a4), 0.05, 2)
        self.assertLessEqual(abs(iso.b4), 0.05, 2)

    def test_m51_niter(self):
        # compares with old STSDAS task. In this task, the
        # default for the starting value of SMA is 10; it
        # fits with 20 iterations.
        image = fits.open(TEST_DATA + "M51.fits")
        test_data = image[0].data

        sample = Sample(test_data, 10)
        fitter = Fitter(sample)
        iso = fitter.fit()

        self.assertTrue(iso.valid)
        self.assertEqual(iso.niter, 50)

class TestIsophoteList(unittest.TestCase):

    def test_isophote_list(self):
        test_data = build_test_data.build()
        iso_list = []
        for k in range(10):
            sample = Sample(test_data, float(k+10.))
            sample.update()
            iso_list.append(Isophote(sample, k, True, 0))

        result = IsophoteList(iso_list)

        # make sure it can be indexed as a list.
        self.assertIsInstance(result[0], Isophote)

        array = np.array([])
        # make sure the important arrays contain floats.
        # especially the sma array, which is derived
        # from a property in the Isophote class.
        self.assertEqual(type(result.sma), type(array))
        self.assertIsInstance(result.sma[0], float)

        self.assertEqual(type(result.intens), type(array))
        self.assertIsInstance(result.intens[0], float)

        self.assertEqual(type(result.rms), type(array))
        self.assertEqual(type(result.int_err), type(array))
        self.assertEqual(type(result.pix_stddev), type(array))
        self.assertEqual(type(result.grad), type(array))
        self.assertEqual(type(result.grad_error), type(array))
        self.assertEqual(type(result.grad_r_error), type(array))
        self.assertEqual(type(result.sarea), type(array))
        self.assertEqual(type(result.niter), type(array))
        self.assertEqual(type(result.ndata), type(array))
        self.assertEqual(type(result.nflag), type(array))
        self.assertEqual(type(result.valid), type(array))
        self.assertEqual(type(result.stop_code), type(array))
        self.assertEqual(type(result.tflux_c), type(array))
        self.assertEqual(type(result.tflux_e), type(array))
        self.assertEqual(type(result.npix_c), type(array))
        self.assertEqual(type(result.npix_e), type(array))
        self.assertEqual(type(result.a3), type(array))
        self.assertEqual(type(result.a4), type(array))
        self.assertEqual(type(result.b3), type(array))
        self.assertEqual(type(result.b4), type(array))

        samples = result.sample
        self.assertIsInstance(samples, list)
        self.assertIsInstance(samples[0], Sample)

        iso = result.get_closest(13.6)
        self.assertIsInstance(iso, Isophote)
        self.assertEqual(iso.sma, 14.)




