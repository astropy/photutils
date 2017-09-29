from __future__ import (absolute_import, division, print_function, unicode_literals)

import pytest

import numpy as np
from astropy.io import fits
from astropy.tests.helper import remote_data

from photutils.isophote.geometry import Geometry
from photutils.isophote.ellipse import Ellipse
from photutils.isophote.model import build_model

from .make_test_data import make_test_image
from ...datasets import get_path

verb = False

try:
    import scipy
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


@remote_data
@pytest.mark.skipif('not HAS_SCIPY')
def test_model():
    path = get_path('isophote/M105-S001-RGB.fits',
                    location='photutils-datasets', cache=True)
    hdu = fits.open(path)
    data = hdu[0].data[0]
    hdu.close()

    g = Geometry(530., 511, 10., 0.1, 10./180.*np.pi)
    ellipse = Ellipse(data, geometry=g, verbose=verb, threshold=1.e5)
    isophote_list = ellipse.fit_image(verbose=verb)
    model = build_model(data, isophote_list, fill=np.mean(data[10:100,10:100]), verbose=verb)

    assert data.shape == model.shape

    residual = data - model

    assert np.mean(residual) <= 5.0
    assert np.mean(residual) >= -5.0


@pytest.mark.skipif('not HAS_SCIPY')
def test_2():
    data = make_test_image(eps=0.5, pa=np.pi/3., noise=1.e-2,
                           random_state=123)

    g = Geometry(256., 256., 10., 0.5, np.pi/3.)
    ellipse = Ellipse(data, geometry=g, verbose=verb, threshold=1.e5)
    isophote_list = ellipse.fit_image(verbose=verb)
    model = build_model(data, isophote_list, fill=np.mean(data[0:50,0:50]), verbose=verb)

    assert data.shape == model.shape

    residual = data - model

    assert np.mean(residual) <= 5.0
    assert np.mean(residual) >= -5.0


