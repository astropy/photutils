# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the simulation module.
"""

import numpy as np
import pytest
from astropy.modeling.models import Gaussian2D
from astropy.table import Table
from numpy.testing import assert_equal

from photutils.psf import (CircularGaussianPRF, make_psf_model,
                           make_psf_model_image)


def test_make_psf_model_image():
    shape = (401, 451)
    n_sources = 100
    model = CircularGaussianPRF(fwhm=2.7)
    data, params = make_psf_model_image(shape, model, n_sources)
    assert data.shape == shape
    assert isinstance(params, Table)
    assert len(params) == n_sources

    model_shape = (13, 13)
    data2, params2 = make_psf_model_image(shape, model, n_sources,
                                          model_shape=model_shape)
    assert_equal(data, data2)
    assert len(params2) == n_sources

    flux = (100, 200)
    fwhm = (2.5, 4.5)
    alpha = (0, 1)
    n_sources = 10
    data, params = make_psf_model_image(shape, model, n_sources,
                                        seed=0, flux=flux, fwhm=fwhm,
                                        alpha=alpha)
    assert len(params) == n_sources
    colnames = ('id', 'x_0', 'y_0', 'flux', 'fwhm')
    for colname in colnames:
        assert colname in params.colnames
    assert 'alpha' not in params.colnames
    assert np.min(params['flux']) >= flux[0]
    assert np.max(params['flux']) <= flux[1]
    assert np.min(params['fwhm']) >= fwhm[0]
    assert np.max(params['fwhm']) <= fwhm[1]


def test_make_psf_model_image_custom():
    shape = (401, 451)
    n_sources = 100
    model = Gaussian2D()
    psf_model = make_psf_model(model, x_name='x_mean', y_name='y_mean')
    data, params = make_psf_model_image(shape, psf_model, n_sources,
                                        model_shape=(11, 11))
    assert data.shape == shape
    assert isinstance(params, Table)
    assert len(params) == n_sources


def test_make_psf_model_image_inputs():
    shape = (50, 50)
    match = 'psf_model must be an Astropy Model subclass'
    with pytest.raises(TypeError, match=match):
        make_psf_model_image(shape, None, 2)

    match = 'psf_model must be two-dimensional'
    model = CircularGaussianPRF(fwhm=2.7)
    model.n_inputs = 3
    with pytest.raises(ValueError, match=match):
        make_psf_model_image(shape, model, 2)

    match = 'model_shape must be specified if the model does not have'
    model = CircularGaussianPRF(fwhm=2.7)
    model.bounding_box = None
    with pytest.raises(ValueError, match=match):
        make_psf_model_image(shape, model, 2)

    match = 'Invalid PSF model - could not find PSF parameter names'
    model = Gaussian2D()
    with pytest.raises(ValueError, match=match):
        make_psf_model_image(shape, model, 2)
