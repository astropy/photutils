from __future__ import (absolute_import, division, print_function, unicode_literals)

import numpy as np
from astropy.io import fits

from photutils.isophote.geometry import Geometry
from photutils.isophote.ellipse import Ellipse
from photutils.isophote.model import build_model
from photutils.isophote.build_test_data import build
from photutils.isophote.tests.test_data import TEST_DATA

verb = False


def test_model():
    name = "M105-S001-RGB"
    test_data = fits.open(TEST_DATA + name + ".fits")
    image = test_data[0].data[0]

    g = Geometry(530., 511, 10., 0.1, 10./180.*np.pi)
    ellipse = Ellipse(image, geometry=g, verbose=verb, threshold=1.e5)
    isophote_list = ellipse.fit_image(verbose=verb)
    model = build_model(image, isophote_list, fill=np.mean(image[10:100,10:100]), verbose=verb)

    assert image.shape == model.shape

    residual = image - model

    assert np.mean(residual) <= 5.0
    assert np.mean(residual) >= -5.0


def test_2():
    image = build(eps=0.5, pa=np.pi/3., noise=1.e-2)

    g = Geometry(256., 256., 10., 0.5, np.pi/3.)
    ellipse = Ellipse(image, geometry=g, verbose=verb, threshold=1.e5)
    isophote_list = ellipse.fit_image(verbose=verb)
    model = build_model(image, isophote_list, fill=np.mean(image[0:50,0:50]), verbose=verb)

    assert image.shape == model.shape

    residual = image - model

    assert np.mean(residual) <= 5.0
    assert np.mean(residual) >= -5.0


