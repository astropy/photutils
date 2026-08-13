# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the images module.
"""

import astropy.units as u
import numpy as np
import pytest
from astropy.modeling.models import Moffat2D, Polynomial2D
from astropy.table import QTable
from numpy.testing import assert_allclose

from photutils.datasets import make_model_image
from photutils.datasets.images import _model_shape_from_bbox
from photutils.psf import (CircularGaussianPSF, CircularGaussianSigmaPRF,
                           ImagePSF)


def test_make_model_image():
    """
    Test the basic functionality of make_model_image.
    """
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

    # Test variable model shape
    params['model_shape'] = [9, 7, 11]
    image = make_model_image(shape, model, params, model_shape=model_shape)
    assert image.shape == shape
    assert image.sum() > 1

    # Test local_bkg
    params['local_bkg'] = [1, 2, 3]
    image = make_model_image(shape, model, params, model_shape=model_shape)
    assert image.shape == shape
    assert image.sum() > 1

    # Test scalar shape
    del params['model_shape']
    del params['local_bkg']
    image = make_model_image(300, model, params, model_shape=model_shape)
    assert image.shape == (300, 300)


def test_make_model_image_variable_shape_pairs():
    """
    Test a model_shape column containing (ny, nx) pairs.
    """
    params = QTable()
    params['x_0'] = [50, 70, 90]
    params['y_0'] = [50, 50, 50]
    params['gamma'] = [1.7, 2.32, 5.8]
    params['alpha'] = [2.9, 5.7, 4.6]
    params['model_shape'] = np.array([[9, 7], [7, 9], [11, 11]])
    model = Moffat2D(amplitude=1)
    shape = (300, 500)
    image = make_model_image(shape, model, params)
    assert image.shape == shape
    assert image.sum() > 1


def test_make_model_image_variable_shape_invalid():
    """
    Test that invalid values in a model_shape column raise an error.
    """
    params = QTable()
    params['x_0'] = [50, 70]
    params['y_0'] = [50, 50]
    params['gamma'] = [1.7, 2.32]
    params['alpha'] = [2.9, 5.7]
    model = Moffat2D(amplitude=1)
    shape = (300, 500)

    match = 'model_shape must be > 0'
    for model_shapes in ([0, 11], [-5, 11]):
        params['model_shape'] = model_shapes
        with pytest.raises(ValueError, match=match):
            make_model_image(shape, model, params)

    match = 'model_shape must be a finite value'
    params['model_shape'] = [np.nan, 11.0]
    with pytest.raises(ValueError, match=match):
        make_model_image(shape, model, params)


def test_make_model_image_nonfinite_local_bkg():
    """
    Test that sources with a non-finite local_bkg value are skipped.
    """
    params = QTable()
    params['x_0'] = [50.0, 70.0, 90.0]
    params['y_0'] = [50.0, 50.0, 50.0]
    params['gamma'] = [1.7, 2.32, 5.8]
    params['alpha'] = [2.9, 5.7, 4.6]
    model = Moffat2D(amplitude=1)
    shape = (100, 150)
    model_shape = (11, 11)
    params['local_bkg'] = [0.0, 0.0, 0.0]
    image0 = make_model_image(shape, model, params, model_shape=model_shape)

    params['local_bkg'] = [np.nan, 0.0, np.inf]
    image = make_model_image(shape, model, params, model_shape=model_shape)
    assert not np.any(np.isnan(image))
    assert image[50, 50] == 0  # first source skipped
    assert image[50, 90] == 0  # third source skipped
    assert_allclose(image[50, 70], image0[50, 70])


def test_make_model_image_units():
    """
    Test that the model image is created with the correct units when the
    flux column has units.
    """
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
    assert model.flux == 1.0  # Default flux (unchanged)

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
    assert model.flux == 1.0  # Default flux (unchanged)

    params['flux'] = [2, 3]
    image = make_model_image(shape, model, params)
    assert image.shape == shape
    assert not isinstance(image, u.Quantity)
    assert model.flux == 1.0  # Default flux (unchanged)


def test_make_model_image_units_no_sources_rendered():
    """
    Test that the output units do not depend on whether any sources
    are actually rendered.
    """
    unit = u.Jy
    model = CircularGaussianSigmaPRF(sigma=1.5)
    shape = (10, 12)

    # Zero-row table
    params = QTable()
    params['x_0'] = np.array([], dtype=float)
    params['y_0'] = np.array([], dtype=float)
    params['flux'] = np.array([], dtype=float) * unit
    image = make_model_image(shape, model, params, model_shape=(5, 5))
    assert isinstance(image, u.Quantity)
    assert image.unit == unit
    assert np.all(image.value == 0)

    # All rows non-finite
    params = QTable()
    params['x_0'] = [np.nan]
    params['y_0'] = [5.0]
    params['flux'] = [1.0] * unit
    image = make_model_image(shape, model, params, model_shape=(5, 5))
    assert isinstance(image, u.Quantity)
    assert image.unit == unit
    assert np.all(image.value == 0)

    # Zero-row table without units
    params = QTable()
    params['x_0'] = np.array([], dtype=float)
    params['y_0'] = np.array([], dtype=float)
    image = make_model_image(shape, model, params, model_shape=(5, 5))
    assert not isinstance(image, u.Quantity)
    assert np.all(image == 0)


def test_make_model_image_discretize_method():
    """
    Test the model image when using different discretization methods.
    """
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
    """
    Test the model image when there is no overlap between the model and
    the image.
    """
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
    """
    Test that the appropriate exceptions are raised for invalid inputs.
    """
    match = 'shape must have 1 or 2 elements'
    with pytest.raises(ValueError, match=match):
        make_model_image((100, 100, 3), Moffat2D(), QTable())

    match = 'shape must be > 0'
    with pytest.raises(ValueError, match=match):
        make_model_image((-100, 100), Moffat2D(), QTable())

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

    match = 'Invalid discretize_method'
    with pytest.raises(ValueError, match=match):
        make_model_image(shape, model, params, model_shape=(11, 11),
                         discretize_method='invalid')


def test_make_model_image_bbox():
    """
    Test the model image when using a PSF model that has a bounding box
    and the bbox_factor keyword to control the size of the bounding box.
    """
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


def test_make_model_image_params_map():
    """
    Test the model image when using a parameter mapping to map the model
    parameter names to different column names in the input table.
    """
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
    """
    Test the model image when the input table contains non-finite
    values.
    """
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

    # All invalid sources
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


def test_make_model_image_progress_bar():
    """
    Test the model image with progress_bar=True.
    """
    pytest.importorskip('tqdm')
    params = QTable()
    params['x_0'] = [50, 70, 90]
    params['y_0'] = [50, 50, 50]
    params['gamma'] = [1.7, 2.32, 5.8]
    params['alpha'] = [2.9, 5.7, 4.6]
    model = Moffat2D(amplitude=1)
    shape = (300, 500)
    model_shape = (11, 11)
    image = make_model_image(shape, model, params, model_shape=model_shape,
                             progress_bar=True)
    assert image.shape == shape
    assert image.sum() > 1


def test_model_shape_from_bbox_no_bbox():
    """
    Test that _model_shape_from_bbox raises an error when the model does
    not have a bounding_box attribute.
    """
    model = Polynomial2D(degree=2)
    match = 'model does not have a bounding_box attribute'
    with pytest.raises(ValueError, match=match):
        _model_shape_from_bbox(model)
