# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
import pytest

from astropy.io import fits

from .make_test_data import make_test_image
from ..ellipse import Ellipse
from ..geometry import EllipseGeometry
from ..model import build_ellipse_model
from ...datasets import get_path

try:
    import scipy    # noqa
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


@pytest.mark.remote_data
@pytest.mark.skipif('not HAS_SCIPY')
def test_model():
    path = get_path('isophote/M105-S001-RGB.fits',
                    location='photutils-datasets', cache=True)
    hdu = fits.open(path)
    data = hdu[0].data[0]
    hdu.close()

    g = EllipseGeometry(530., 511, 10., 0.1, 10./180.*np.pi)
    ellipse = Ellipse(data, geometry=g, threshold=1.e5)
    isophote_list = ellipse.fit_image()
    model = build_ellipse_model(data.shape, isophote_list,
                                fill=np.mean(data[10:100, 10:100]))

    assert data.shape == model.shape

    residual = data - model
    assert np.mean(residual) <= 5.0
    assert np.mean(residual) >= -5.0


@pytest.mark.skipif('not HAS_SCIPY')
def test_model_simulated_data():
    data = make_test_image(eps=0.5, pa=np.pi/3., noise=1.e-2,
                           random_state=123)

    g = EllipseGeometry(256., 256., 10., 0.5, np.pi/3.)
    ellipse = Ellipse(data, geometry=g, threshold=1.e5)
    isophote_list = ellipse.fit_image()
    model = build_ellipse_model(data.shape, isophote_list,
                                fill=np.mean(data[0:50, 0:50]))

    assert data.shape == model.shape

    residual = data - model
    assert np.mean(residual) <= 5.0
    assert np.mean(residual) >= -5.0
