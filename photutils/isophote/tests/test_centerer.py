from __future__ import (absolute_import, division, print_function, unicode_literals)

import unittest

import numpy as np
from astropy.io import fits

from ..geometry import Geometry
from ..centerer import Centerer

TEST_DATA = "data/"


class TestCenterer(unittest.TestCase):

    def test_centerer(self):
        image = fits.open(TEST_DATA + "M51.fits")
        test_data = image[0].data

        geometry = Geometry(252, 253, 10., 0.2, np.pi/2)

        centerer = Centerer(test_data, geometry, False)
        centerer.center()

        self.assertEqual(geometry.x0, 257.)
        self.assertEqual(geometry.y0, 258.)