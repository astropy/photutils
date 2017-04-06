from __future__ import (absolute_import, division, print_function, unicode_literals)

from os import path
import unittest

import numpy as np
from astropy.io import fits

from photutils.isophote.geometry import Geometry
from photutils.isophote.ellipse import Ellipse
from photutils.isophote.model import build_model
from photutils.isophote.build_test_data import build

TEST_DATA = path.dirname(path.dirname(path.dirname(__file__))) + "/isophote/tests/data/"

verb = False


class TestModel(unittest.TestCase):

    def test_model(self):
        name = "M105-S001-RGB"
        test_data = fits.open(TEST_DATA + name + ".fits")
        image = test_data[0].data[0]

        g = Geometry(530., 511, 10., 0.1, 10./180.*np.pi)
        ellipse = Ellipse(image, geometry=g, verbose=verb, threshold=1.e5)
        isophote_list = ellipse.fit_image(verbose=verb)
        model = build_model(image, isophote_list, fill=np.mean(image[10:100,10:100]), verbose=verb)

        self.assertEqual(image.shape, model.shape)

        residual = image - model

        self.assertLessEqual(np.mean(residual),  5.0)
        self.assertGreaterEqual(np.mean(residual), -5.0)

    def test_2(self):
        image = build(eps=0.5, pa=np.pi/3., noise=1.e-2)

        g = Geometry(256., 256., 10., 0.5, np.pi/3.)
        ellipse = Ellipse(image, geometry=g, verbose=verb, threshold=1.e5)
        isophote_list = ellipse.fit_image(verbose=verb)
        model = build_model(image, isophote_list, fill=np.mean(image[0:50,0:50]), verbose=verb)

        self.assertEqual(image.shape, model.shape)

        residual = image - model

        self.assertLessEqual(np.mean(residual),  5.0)
        self.assertGreaterEqual(np.mean(residual), -5.0)


