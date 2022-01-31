# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the model module.
"""
import warnings

from astropy.io import fits
from astropy.utils.data import get_pkg_data_filename
import numpy as np
import pytest

from .make_test_data import make_test_image
from ..ellipse import Ellipse
from ..geometry import EllipseGeometry
from ..model import build_ellipse_model
from ...datasets import get_path
from ...utils._optional_deps import HAS_SCIPY  # noqa


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
 
@pytest.mark.remote_data
@pytest.mark.skipif('not HAS_SCIPY')
def test_model_parallel():
    path = get_path('isophote/M105-S001-RGB.fits',
                    location='photutils-datasets', cache=True)
    hdu = fits.open(path)
    data = hdu[0].data[0]
    hdu.close()

    g = EllipseGeometry(530., 511, 10., 0.1, 10./180.*np.pi)
    ellipse = Ellipse(data, geometry=g, threshold=1.e5)
    isophote_list = ellipse.fit_image()
    # test parallel on a thread pool, w/ size equal to CPU thread count
    model = build_ellipse_model(data.shape, isophote_list, nthreads=mp.cpu_count(),
                                fill=np.mean(data[10:100, 10:100]))

    assert data.shape == model.shape

    residual = data - model
    assert np.mean(residual) <= 5.0
    assert np.mean(residual) >= -5.0


@pytest.mark.skipif('not HAS_SCIPY')
def test_model_simulated_data():
    data = make_test_image(nx=200, ny=200, i0=10., sma=5., eps=0.5,
                           pa=np.pi/3., noise=0.05, seed=0)

    g = EllipseGeometry(100., 100., 5., 0.5, np.pi/3.)
    ellipse = Ellipse(data, geometry=g, threshold=1.e5)
    isophote_list = ellipse.fit_image()
    model = build_ellipse_model(data.shape, isophote_list,
                                fill=np.mean(data[0:50, 0:50]))

    assert data.shape == model.shape

    residual = data - model
    assert np.mean(residual) <= 5.0
    assert np.mean(residual) >= -5.0

@pytest.mark.skipif('not HAS_SCIPY')
def test_model_large_array():
    data = make_test_image(nx=5000, ny=5000, i0=100., sma=500., eps=0.3,
                           pa=np.pi/4., noise=0.05, seed=0)

    g = EllipseGeometry(2500., 2500., 100., 0.3, np.pi/4.)
    ellipse = Ellipse(data, geometry=g, threshold=1.e5)
    isophote_list = ellipse.fit_image()
    # test parallel on a thread pool, w/ size equal to CPU thread count
    model = build_ellipse_model(data.shape, isophote_list, nthreads=mp.cpu_count(),
                                fill=np.mean(data[0:50, 0:50]))

    assert data.shape == model.shape

    residual = data - model
    
    assert np.mean(residual) <= 5.0
    assert np.mean(residual) >= -5.0

@pytest.mark.skipif('not HAS_SCIPY')
def test_model_minimum_radius():
    # This test requires a "defective" image that drives the
    # model building algorithm into a corner, where it fails.
    # With the algorithm fixed, it bypasses the failure and
    # succeeds in building the model image.
    filepath = get_pkg_data_filename('data/minimum_radius_test.fits')
    with fits.open(filepath) as hdu:
        data = hdu[0].data

        g = EllipseGeometry(50., 45, 530., 0.1, 10. / 180. * np.pi)
        g.find_center(data)
        ellipse = Ellipse(data, geometry=g)
        with pytest.warns(RuntimeWarning, match='Degrees of freedom'):
            isophote_list = ellipse.fit_image(sma0=40, minsma=0, maxsma=350.,
                                              step=0.4, nclip=3)

        model = build_ellipse_model(data.shape, isophote_list,
                                    fill=np.mean(data[0:50, 0:50]))

        # It's enough that the algorithm reached this point. The
        # actual accuracy of the modelling is being tested elsewhere.
        assert data.shape == model.shape
