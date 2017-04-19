from __future__ import (absolute_import, division, print_function, unicode_literals)

import numpy as np
from astropy.io import fits

from photutils.isophote.geometry import Geometry
from photutils.isophote.centerer import Centerer
from photutils.isophote.tests.test_data import TEST_DATA


def test_centerer():
    image = fits.open(TEST_DATA + "M51.fits")
    test_data = image[0].data

    geometry = Geometry(252, 253, 10., 0.2, np.pi/2)

    centerer = Centerer(test_data, geometry, False)
    centerer.center()

    assert geometry.x0 == 257.
    assert geometry.y0 == 258.