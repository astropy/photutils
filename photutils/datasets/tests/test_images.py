# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the images module.
"""

import astropy.units as u
import numpy as np
import pytest
from astropy.modeling.models import Moffat2D
from astropy.table import QTable
from astropy.utils.exceptions import (AstropyDeprecationWarning,
                                      AstropyUserWarning)
from numpy.testing import assert_allclose

from photutils.datasets import (make_gaussian_prf_sources_image,
                                make_gaussian_sources_image, make_model_image,
                                make_model_sources_image, make_test_psf_data)
from photutils.psf import IntegratedGaussianPRF
from photutils.utils._optional_deps import HAS_SCIPY


@pytest.fixture(name='source_params')
def fixture_source_params():
    # this can be remove when the image deprecations are removed
    params = QTable()
    params['flux'] = [1, 2, 3]
    params['x_mean'] = [30, 50, 70.5]
    params['y_mean'] = [50, 50, 50.5]
    params['x_stddev'] = [1, 2, 3.5]
    params['y_stddev'] = [2, 1, 3.5]
    params['theta'] = np.array([0.0, 30, 50]) * np.pi / 180.0
    return params


@pytest.fixture(name='source_params_prf')
def fixture_source_params_prf():
    # this can be remove when the image deprecations are removed
    params = QTable()
    params['x_0'] = [30, 50, 70.5]
    params['y_0'] = [50, 50, 50.5]
    params['amplitude'] = np.array([1, 2, 3]) / (2 * np.pi)  # sigma = 1
    return params


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


@pytest.mark.skipif(not HAS_SCIPY, reason='scipy is required')
def test_make_model_image_units():
    unit = u.Jy
    params = QTable()
    params['x_0'] = [30, 50, 70.5]
    params['y_0'] = [50, 50, 50.5]
    params['flux'] = [1, 2, 3] * unit
    model = IntegratedGaussianPRF(sigma=1.5)
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
    with pytest.raises(ValueError, match=match):
        params['local_bkg'] = [0.1, 0.2, 0.3]
        make_model_image(shape, model, params, model_shape=model_shape)


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
    with pytest.raises(ValueError, match=match):
        make_model_image((100, 100), None, QTable())

    match = 'model must be a 2D model'
    with pytest.raises(ValueError, match=match):
        model = Moffat2D()
        model.n_inputs = 1
        make_model_image((100, 100), model, QTable())

    match = 'params_table must be an astropy Table'
    with pytest.raises(ValueError, match=match):
        model = Moffat2D()
        make_model_image((100, 100), model, None)

    match = 'not in model parameter names'
    with pytest.raises(ValueError, match=match):
        model = Moffat2D()
        make_model_image((100, 100), model, QTable(), x_name='invalid')
    with pytest.raises(ValueError, match=match):
        model = Moffat2D()
        make_model_image((100, 100), model, QTable(), y_name='invalid')

    match = '"x_0" not in psf_params column names'
    with pytest.raises(ValueError, match=match):
        model = Moffat2D()
        params = QTable()
        make_model_image((100, 100), model, params)

    match = '"y_0" not in psf_params column names'
    with pytest.raises(ValueError, match=match):
        model = Moffat2D()
        params = QTable()
        params['x_0'] = [50, 70, 90]
        make_model_image((100, 100), model, params)

    match = 'model_shape must be specified if the model does not have'
    with pytest.raises(ValueError, match=match):
        params = QTable()
        params['x_0'] = [50]
        params['y_0'] = [50]
        params['gamma'] = [1.7]
        params['alpha'] = [2.9]
        model = Moffat2D(amplitude=1)
        shape = (100, 100)
        make_model_image(shape, model, params)


def test_make_model_sources_image():
    params = QTable()
    params['x_0'] = [50, 70, 90]
    params['y_0'] = [50, 50, 50]
    params['gamma'] = [1.7, 2.32, 5.8]
    params['alpha'] = [2.9, 5.7, 4.6]
    model = Moffat2D(amplitude=1)
    with pytest.warns(AstropyDeprecationWarning):
        image = make_model_sources_image((300, 500), model, params)
        assert image.sum() > 1


def test_make_gaussian_sources_image(source_params):
    with pytest.warns(AstropyDeprecationWarning):
        shape = (100, 100)
        image = make_gaussian_sources_image(shape, source_params)
        assert image.shape == shape
        assert_allclose(image.sum(), source_params['flux'].sum())


def test_make_gaussian_sources_image_amplitude(source_params):
    with pytest.warns(AstropyDeprecationWarning):
        params = source_params.copy()
        params.remove_column('flux')
        params['amplitude'] = [1, 2, 3]
        shape = (100, 100)
        image = make_gaussian_sources_image(shape, params)
        assert image.shape == shape


def test_make_gaussian_sources_image_desc_oversample(source_params):
    with pytest.warns(AstropyDeprecationWarning):
        shape = (100, 100)
        image = make_gaussian_sources_image(shape, source_params,
                                            oversample=10)
        assert image.shape == shape
        assert_allclose(image.sum(), source_params['flux'].sum())


@pytest.mark.skipif(not HAS_SCIPY, reason='scipy is required')
def test_make_gaussian_prf_sources_image(source_params_prf):
    with pytest.warns(AstropyDeprecationWarning):
        shape = (100, 100)
        image = make_gaussian_prf_sources_image(shape, source_params_prf)
        assert image.shape == shape
        flux = source_params_prf['amplitude'] * (2 * np.pi)  # sigma = 1
        assert_allclose(image.sum(), flux.sum(), rtol=1.0e-6)


@pytest.mark.skipif(not HAS_SCIPY, reason='scipy is required')
def test_make_test_psf_data():
    with pytest.warns(AstropyDeprecationWarning):
        psf_model = IntegratedGaussianPRF(flux=100, sigma=1.5)
        psf_shape = (5, 5)
        nsources = 10
        shape = (100, 100)
        data, true_params = make_test_psf_data(shape, psf_model, psf_shape,
                                               nsources,
                                               flux_range=(500, 1000),
                                               min_separation=10, seed=0)

        assert isinstance(data, np.ndarray)
        assert data.shape == shape
        assert isinstance(true_params, QTable)
        assert len(true_params) == nsources
        assert true_params['x'].min() >= 0
        assert true_params['y'].min() >= 0

        match = 'Unable to produce'
        with pytest.warns(AstropyUserWarning, match=match):
            nsources = 100
            make_test_psf_data(shape, psf_model, psf_shape, nsources,
                               flux_range=(500, 1000), min_separation=100,
                               seed=0)
