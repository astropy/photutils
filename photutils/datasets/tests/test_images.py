# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the images module.
"""

import astropy.units as u
import numpy as np
import pytest
from astropy.modeling.models import Moffat2D
from astropy.table import QTable
from numpy.testing import assert_allclose

from photutils.datasets import make_model_image
from photutils.psf import (CircularGaussianPSF, CircularGaussianSigmaPRF,
                           ImagePSF)


def test_make_model_image():
    params = QTable()
    params['x_0'] = [50, 70, 90]
    params['y_0'] = [50, 50, 50]
    params['gamma'] = [1.7, 2.32, 5.8]
    params['alpha'] = [2.9, 5.7, 4.6]
    model = Moffat2D(amplitude=1)
    shape = (300, 500)
    model_shape = (11, 11)
    image = make_model_image(shape, model, params, model_shape=model_shape)
    assert image.shape == shape
    assert image.sum() > 1

    # test variable model shape
    params['model_shape'] = [9, 7, 11]
    image = make_model_image(shape, model, params, model_shape=model_shape)
    assert image.shape == shape
    assert image.sum() > 1

    # test local_bkg
    params['local_bkg'] = [1, 2, 3]
    image = make_model_image(shape, model, params, model_shape=model_shape)
    assert image.shape == shape
    assert image.sum() > 1


def test_make_model_image_units():
    unit = u.Jy
    params = QTable()
    params['x_0'] = [30, 50, 70.5]
    params['y_0'] = [50, 50, 50.5]
    params['flux'] = [1, 2, 3] * unit
    model = CircularGaussianSigmaPRF(sigma=1.5)
    shape = (300, 500)
    model_shape = (11, 11)
    image = make_model_image(shape, model, params, model_shape=model_shape)
    assert image.shape == shape
    assert isinstance(image, u.Quantity)
    assert image.unit == unit
    assert model.flux == 1.0  # default flux (unchanged)

    params['local_bkg'] = [0.1, 0.2, 0.3] * unit
    image = make_model_image(shape, model, params, model_shape=model_shape)
    assert image.shape == shape
    assert isinstance(image, u.Quantity)
    assert image.unit == unit

    match = 'The local_bkg column must have the same flux units'
    params['local_bkg'] = [0.1, 0.2, 0.3]
    with pytest.raises(ValueError, match=match):
        make_model_image(shape, model, params, model_shape=model_shape)


def test_make_model_image_units_no_overlap():
    """
    Test that the model image is created with the correct units when
    there is no overlap between the model and the image.
    """
    unit = u.Jy
    params = QTable()
    params['x_0'] = [50, 70.5]
    params['y_0'] = [50, 50.5]
    params['flux'] = [2, 3] * unit
    model = CircularGaussianSigmaPRF(sigma=1.5)
    shape = (10, 12)
    image = make_model_image(shape, model, params)
    assert image.shape == shape
    assert isinstance(image, u.Quantity)
    assert image.unit == unit
    assert model.flux == 1.0  # default flux (unchanged)

    params['flux'] = [2, 3]
    image = make_model_image(shape, model, params)
    assert image.shape == shape
    assert not isinstance(image, u.Quantity)
    assert model.flux == 1.0  # default flux (unchanged)


def test_make_model_image_discretize_method():
    params = QTable()
    params['x_0'] = [50, 70, 90]
    params['y_0'] = [50, 50, 50]
    params['gamma'] = [1.7, 2.32, 5.8]
    params['alpha'] = [2.9, 5.7, 4.6]
    model = Moffat2D(amplitude=1)
    shape = (300, 500)
    model_shape = (11, 11)
    for method in ('interp', 'oversample'):
        image = make_model_image(shape, model, params, model_shape=model_shape,
                                 discretize_method=method)
        assert image.shape == shape
        assert image.sum() > 1


def test_make_model_image_no_overlap():
    params = QTable()
    params['x_0'] = [50]
    params['y_0'] = [50]
    params['gamma'] = [1.7]
    params['alpha'] = [2.9]
    model = Moffat2D(amplitude=1)
    shape = (10, 10)
    model_shape = (3, 3)
    data = make_model_image(shape, model, params, model_shape=model_shape)
    assert data.shape == shape
    assert np.sum(data) == 0


def test_make_model_image_inputs():
    match = 'shape must be a 2-tuple'
    with pytest.raises(ValueError, match=match):
        make_model_image(100, Moffat2D(), QTable())

    match = 'model must be a Model instance'
    with pytest.raises(TypeError, match=match):
        make_model_image((100, 100), None, QTable())

    match = 'model must be a 2D model'
    model = Moffat2D()
    model.n_inputs = 1
    with pytest.raises(ValueError, match=match):
        make_model_image((100, 100), model, QTable())

    match = 'params_table must be an astropy Table'
    model = Moffat2D()
    with pytest.raises(TypeError, match=match):
        make_model_image((100, 100), model, None)

    match = 'not in model parameter names'
    model = Moffat2D()
    with pytest.raises(ValueError, match=match):
        make_model_image((100, 100), model, QTable(), x_name='invalid')

    match = 'not in params_table column names'
    model = Moffat2D()
    with pytest.raises(ValueError, match=match):
        make_model_image((100, 100), model, QTable(), y_name='invalid')

    model = Moffat2D()
    params = QTable()
    with pytest.raises(ValueError, match=match):
        make_model_image((100, 100), model, params)

    model = Moffat2D()
    params = QTable()
    params['x_0'] = [50, 70, 90]
    with pytest.raises(ValueError, match=match):
        make_model_image((100, 100), model, params)

    match = 'model_shape must be specified if the model does not have'
    params = QTable()
    params['x_0'] = [50]
    params['y_0'] = [50]
    params['gamma'] = [1.7]
    params['alpha'] = [2.9]
    model = Moffat2D(amplitude=1)
    shape = (100, 100)
    with pytest.raises(ValueError, match=match):
        make_model_image(shape, model, params)


def test_make_model_image_bbox():
    model1 = CircularGaussianPSF(x_0=50, y_0=50, fwhm=10)
    yy, xx = np.mgrid[:101, :101]
    model2 = ImagePSF(model1(xx, yy), x_0=50, y_0=50)

    params = QTable()
    params['x_0'] = [50, 70, 90]
    params['y_0'] = [50, 50, 50]
    shape = (100, 151)
    image1 = make_model_image(shape, model2, params, bbox_factor=10)
    image2 = make_model_image(shape, model2, params, bbox_factor=None)
    assert_allclose(image1, image2)

    image3 = make_model_image(shape, model1, params, bbox_factor=10)
    image4 = make_model_image(shape, model1, params, bbox_factor=None)
    assert_allclose(image3, image4)

    model1.bbox_factor = 10
    image5 = make_model_image(shape, model1, params)
    assert np.sum(image5) > np.sum(image4)
    assert_allclose(image3, image4)


def test_make_model_image_params_map():
    params = QTable()
    params['x_0'] = [50, 70, 90]
    params['y_0'] = [50, 50, 50]
    params['gamma'] = [1.7, 2.32, 5.8]
    params['alpha'] = [2.9, 5.7, 4.6]
    model = Moffat2D(amplitude=1)
    shape = (300, 500)
    model_shape = (11, 11)
    image = make_model_image(shape, model, params, model_shape=model_shape)

    params = QTable()
    params['x_0'] = [50, 70, 90]
    params['y_0'] = [50, 50, 50]
    params['gamma2'] = [1.7, 2.32, 5.8]
    params['alpha4'] = [2.9, 5.7, 4.6]
    params_map = {'gamma': 'gamma2', 'alpha': 'alpha4'}
    model = Moffat2D(amplitude=1)
    shape = (300, 500)
    model_shape = (11, 11)
    image2 = make_model_image(shape, model, params, model_shape=model_shape,
                              params_map=params_map)
    assert_allclose(image, image2)


def test_make_model_image_nonfinite():
    params = QTable()
    params['x_0'] = [50, np.nan, 90, 100]
    params['y_0'] = [50, 50, 50, 50]
    params['gamma'] = [1.7, 2.32, 5.8, np.inf]
    params['alpha'] = [2.9, 5.7, 4.6, 3.1]
    model = Moffat2D(amplitude=1)
    shape = (300, 500)
    model_shape = (11, 11)
    image = make_model_image(shape, model, params, model_shape=model_shape)
    assert image.shape == shape
    assert image.sum() < 33
    assert image[50, 100] == 0

    # all invalid sources
    params = QTable()
    params['x_0'] = [50, np.nan, 90, 100]
    params['y_0'] = [-np.inf, 50, 50, 50]
    params['gamma'] = [1.7, 2.32, 5.8, np.inf]
    params['alpha'] = [2.9, 5.7, np.nan, 3.1]
    model = Moffat2D(amplitude=1)
    shape = (300, 500)
    model_shape = (11, 11)
    image = make_model_image(shape, model, params, model_shape=model_shape)
    assert image.shape == shape
    assert image.sum() == 0
