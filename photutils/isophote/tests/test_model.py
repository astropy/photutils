# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the model module.
"""

import warnings
from contextlib import nullcontext

import numpy as np
import pytest
from astropy.io import fits
from astropy.utils.data import get_pkg_data_filename

from photutils.datasets import get_path
from photutils.isophote.ellipse import Ellipse
from photutils.isophote.geometry import EllipseGeometry
from photutils.isophote.model import build_ellipse_model
from photutils.isophote.tests.make_test_data import make_test_image
from photutils.tests.helper import PYTEST_LT_80
from photutils.utils._optional_deps import HAS_SCIPY


@pytest.mark.remote_data
@pytest.mark.skipif(not HAS_SCIPY, reason='scipy is required')
def test_model():
    path = get_path('isophote/M105-S001-RGB.fits',
                    location='photutils-datasets', cache=True)
    hdu = fits.open(path)
    data = hdu[0].data[0]
    hdu.close()

    g = EllipseGeometry(530.0, 511, 10.0, 0.1, 10.0 / 180.0 * np.pi)
    ellipse = Ellipse(data, geometry=g, threshold=1.0e5)

    # NOTE: this sometimes emits warnings (e.g., py38, ubuntu), but
    # sometimes not. Here we simply ignore any RuntimeWarning, whether
    # there is one or not.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        isophote_list = ellipse.fit_image()
    model = build_ellipse_model(data.shape, isophote_list,
                                fill=np.mean(data[10:100, 10:100]))

    assert data.shape == model.shape

    residual = data - model
    assert np.mean(residual) <= 5.0
    assert np.mean(residual) >= -5.0


@pytest.mark.skipif(not HAS_SCIPY, reason='scipy is required')
def test_model_simulated_data():
    data = make_test_image(nx=200, ny=200, i0=10.0, sma=5.0, eps=0.5,
                           pa=np.pi / 3.0, noise=0.05, seed=0)

    g = EllipseGeometry(100.0, 100.0, 5.0, 0.5, np.pi / 3.0)
    ellipse = Ellipse(data, geometry=g, threshold=1.0e5)
    isophote_list = ellipse.fit_image()
    model = build_ellipse_model(data.shape, isophote_list,
                                fill=np.mean(data[0:50, 0:50]))

    assert data.shape == model.shape

    residual = data - model
    assert np.mean(residual) <= 5.0
    assert np.mean(residual) >= -5.0


@pytest.mark.skipif(not HAS_SCIPY, reason='scipy is required')
def test_model_minimum_radius():
    # This test requires a "defective" image that drives the
    # model building algorithm into a corner, where it fails.
    # With the algorithm fixed, it bypasses the failure and
    # succeeds in building the model image.
    filepath = get_pkg_data_filename('data/minimum_radius_test.fits')
    with fits.open(filepath) as hdu:
        data = hdu[0].data
        g = EllipseGeometry(50.0, 45, 530.0, 0.1, 10.0 / 180.0 * np.pi)
        g.find_center(data)
        ellipse = Ellipse(data, geometry=g)

        match1 = 'Degrees of freedom'
        ctx1 = pytest.warns(RuntimeWarning, match=match1)
        if PYTEST_LT_80:
            ctx2 = nullcontext()
            ctx3 = nullcontext()
        else:
            match2 = 'Mean of empty slice'
            match3 = 'invalid value encountered'
            ctx2 = pytest.warns(RuntimeWarning, match=match2)
            ctx3 = pytest.warns(RuntimeWarning, match=match3)
        with ctx1, ctx2, ctx3:
            isophote_list = ellipse.fit_image(sma0=40, minsma=0,
                                              maxsma=350.0, step=0.4, nclip=3)

        model = build_ellipse_model(data.shape, isophote_list,
                                    fill=np.mean(data[0:50, 0:50]))

        # It's enough that the algorithm reached this point. The
        # actual accuracy of the modelling is being tested elsewhere.
        assert data.shape == model.shape
