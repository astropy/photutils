from __future__ import (absolute_import, division, print_function, unicode_literals)

import unittest

import numpy as np

from photutils.isophote import build_test_data
from photutils.isophote.integrator import MEDIAN, MEAN, BI_LINEAR, NEAREST_NEIGHBOR
from photutils.isophote.sample import Sample
from photutils.isophote.isophote import Isophote

test_data = build_test_data.build(background=100., i0=0., noise=10.)


class TestSample(unittest.TestCase):

    def test_scatter(self):
        '''
        Checks that the pixel standard deviation can be reliably estimated
        from the rms scatter and the sector area.

        The test data is just a flat image with noise. No galaxy. We define 
        the noise rms and then compare how close the pixel std dev estimated 
        at extraction matches this input noise.
        '''
        self._doit(NEAREST_NEIGHBOR, 7., 12.)
        self._doit(BI_LINEAR,        7., 12.)
        self._doit(MEAN,             7., 12.)
        self._doit(MEDIAN,           6., 15.) # the median is not so good at estimating rms

    def _doit(self, integrmode, amin, amax):
        sample = Sample(test_data, 50., astep=0.2, integrmode=integrmode)
        sample.update()
        iso = Isophote(sample, 0, True, 0)

        self.assertLess(iso.pix_stddev, amax)
        self.assertGreater(iso.pix_stddev, amin)

    def test_coordinates(self):
        sample = Sample(test_data, 50.)
        sample.update()

        x, y = sample.coordinates()

        array = np.array([])
        self.assertEqual(type(x), type(array))
        self.assertEqual(type(y), type(array))

    def test_sclip(self):
        sample = Sample(test_data, 50., nclip=3)
        sample.update()

        x, y = sample.coordinates()

        array = np.array([])
        self.assertEqual(type(x), type(array))
        self.assertEqual(type(y), type(array))
