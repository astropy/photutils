from __future__ import (absolute_import, division, print_function, unicode_literals)

import unittest

import numpy as np
from astropy.io import fits

from ..sample import Sample
from ..integrator import NEAREST_NEIGHBOR, BI_LINEAR, MEAN, MEDIAN


class TestIntegrator(unittest.TestCase):

    def _init_test(self, integrmode=BI_LINEAR, sma=40.):

        image = fits.open("data/synth_highsnr.fits")
        test_data = image[0].data

        self.sample = Sample(test_data, sma, integrmode=integrmode)

        s = self.sample.extract()

        self.assertEqual(len(s), 3)
        self.assertEqual(len(s[0]), len(s[1]))
        self.assertEqual(len(s[0]), len(s[2]))

        return s

    def test_bilinear(self):

        s = self._init_test()

        self.assertEqual(len(s[0]), 225)

        # intensities
        self.assertAlmostEqual(np.mean(s[2]), 200.76, 2)
        self.assertAlmostEqual(np.std(s[2]),  21.55, 2)

        # radii
        self.assertAlmostEqual(np.max(s[1]), 40.0, 2)
        self.assertAlmostEqual(np.min(s[1]), 32.0, 2)

        self.assertEqual(self.sample.total_points, 225)
        self.assertEqual(self.sample.actual_points, 225)

    def test_bilinear_small(self):

        # small radius forces sub-pixel sampling
        s = self._init_test(sma=10.)

        # intensities
        self.assertAlmostEqual(np.mean(s[2]), 1045.4, 1)
        self.assertAlmostEqual(np.std(s[2]),  143.0, 1)

        # radii
        self.assertAlmostEqual(np.max(s[1]), 10.0, 1)
        self.assertAlmostEqual(np.min(s[1]), 8.0, 1)

        self.assertEqual(self.sample.total_points, 57)
        self.assertEqual(self.sample.actual_points, 57)

    def test_nearest_neighbor(self):

        s = self._init_test(integrmode=NEAREST_NEIGHBOR)

        self.assertEqual(len(s[0]), 225)
        # intensities
        self.assertAlmostEqual(np.mean(s[2]), 201.1, 1)
        self.assertAlmostEqual(np.std(s[2]),  21.8, 1)
        # radii
        self.assertAlmostEqual(np.max(s[1]), 40.0, 2)
        self.assertAlmostEqual(np.min(s[1]), 32.0, 2)

        self.assertEqual(self.sample.total_points, 225)
        self.assertEqual(self.sample.actual_points, 225)

    def test_mean(self):

        s = self._init_test(integrmode=MEAN)

        self.assertEqual(len(s[0]), 64)
        # intensities
        self.assertAlmostEqual(np.mean(s[2]), 199.9, 1)
        self.assertAlmostEqual(np.std(s[2]),  21.3, 1)
        # radii
        self.assertAlmostEqual(np.max(s[1]), 40.00, 2)
        self.assertAlmostEqual(np.min(s[1]), 32.01, 2)

        self.assertAlmostEqual(self.sample.sector_area, 12.4, 1)
        self.assertEqual(self.sample.total_points, 64)
        self.assertEqual(self.sample.actual_points, 64)

    def test_mean_small(self):

        s = self._init_test(sma=5., integrmode=MEAN)

        self.assertEqual(len(s[0]), 29)
        # intensities
        self.assertAlmostEqual(np.mean(s[2]), 2339.0, 1)
        self.assertAlmostEqual(np.std(s[2]),  284.7, 1)
        # radii
        self.assertAlmostEqual(np.max(s[1]), 5.00, 2)
        self.assertAlmostEqual(np.min(s[1]), 4.00, 2)

        self.assertAlmostEqual(self.sample.sector_area, 2.0, 1)
        self.assertEqual(self.sample.total_points, 29)
        self.assertEqual(self.sample.actual_points, 29)

    def test_median(self):

        s = self._init_test(integrmode=MEDIAN)

        self.assertEqual(len(s[0]), 64)
        # intensities
        self.assertAlmostEqual(np.mean(s[2]), 199.9, 1)
        self.assertAlmostEqual(np.std(s[2]),  21.3, 1)
        # radii
        self.assertAlmostEqual(np.max(s[1]), 40.00, 2)
        self.assertAlmostEqual(np.min(s[1]), 32.01, 2)

        self.assertAlmostEqual(self.sample.sector_area, 12.4, 1)
        self.assertEqual(self.sample.total_points, 64)
        self.assertEqual(self.sample.actual_points, 64)
