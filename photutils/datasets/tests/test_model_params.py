# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the model_params module.
"""

import numpy as np
import pytest
from astropy.table import QTable, Table
from astropy.utils.exceptions import AstropyUserWarning

from photutils.datasets import (make_model_params, make_random_models_table,
                                params_table_to_models)
from photutils.psf import CircularGaussianPSF


def test_make_model_params():
    shape = (100, 100)
    n_sources = 10
    flux = (100, 1000)
    params = make_model_params(shape, n_sources, flux=flux)
    assert isinstance(params, Table)
    assert len(params) == 10
    cols = ('id', 'x_0', 'y_0', 'flux')
    for col in cols:
        assert col in params.colnames
        assert np.min(params[col]) >= 0
    assert np.min(params['flux']) >= flux[0]
    assert np.max(params['flux']) <= flux[1]

    # test extra parameters
    sigma = (1, 2)
    alpha = (0, 1)
    params = make_model_params((120, 100), 5, flux=flux, sigma=sigma,
                               alpha=alpha, min_separation=3, border_size=10,
                               seed=0)
    cols = ('id', 'x_0', 'y_0', 'flux', 'sigma', 'alpha')
    for col in cols:
        assert col in params.colnames
        assert np.min(params[col]) >= 0
    assert np.min(params['flux']) >= flux[0]
    assert np.max(params['flux']) <= flux[1]
    assert np.min(params['sigma']) >= sigma[0]
    assert np.max(params['sigma']) <= sigma[1]
    assert np.min(params['alpha']) >= alpha[0]
    assert np.max(params['alpha']) <= alpha[1]

    match = 'flux must be a 2-tuple'
    with pytest.raises(ValueError, match=match):
        make_model_params(shape, n_sources, flux=(1, 2, 3))

    match = 'must be a 2-tuple'
    with pytest.raises(ValueError, match=match):
        make_model_params(shape, n_sources, flux=(1, 2), alpha=(1, 2, 3))


def test_make_model_params_nsources():
    """
    Test case when the number of the possible sources is less than
    ``n_sources``.
    """
    match = r'Unable to produce .* coordinates within the given shape'
    with pytest.warns(AstropyUserWarning, match=match):
        shape = (200, 500)
        n_sources = 100
        params = make_model_params(shape, n_sources, min_separation=50,
                                   amplitude=(100, 500), x_stddev=(1, 5),
                                   y_stddev=(1, 5), theta=(0, np.pi))
        assert len(params) < 100


def test_make_model_params_border_size():
    shape = (10, 10)
    n_sources = 10
    flux = (100, 1000)
    match = 'border_size is too large for the given shape'
    with pytest.raises(ValueError, match=match):
        make_model_params(shape, n_sources, flux=flux, border_size=20)


def test_make_random_models_table():
    param_ranges = {'x_0': (0, 300), 'y_0': (0, 500),
                    'gamma': (1, 3), 'alpha': (1.5, 3)}
    source_table = make_random_models_table(10, param_ranges)
    assert len(source_table) == 10
    assert 'id' in source_table.colnames
    cols = ('x_0', 'y_0', 'gamma', 'alpha')
    for col in cols:
        assert col in source_table.colnames
        assert np.min(source_table[col]) >= param_ranges[col][0]
        assert np.max(source_table[col]) <= param_ranges[col][1]


def test_params_table_to_models():
    tbl = QTable()
    tbl['x_0'] = [1, 2, 3]
    tbl['y_0'] = [4, 5, 6]
    tbl['flux'] = [100, 200, 300]
    tbl['name'] = ['a', 'b', 'c']
    model = CircularGaussianPSF()
    models = params_table_to_models(tbl, model)

    assert len(models) == 3
    for i, model in enumerate(models):
        assert model.x_0 == tbl['x_0'][i]
        assert model.y_0 == tbl['y_0'][i]
        assert model.flux == tbl['flux'][i]
        assert model.name == tbl['name'][i]

    tbl = QTable()
    tbl['invalid1'] = [1, 2, 3]
    tbl['invalid2'] = [4, 5, 6]
    match = 'No matching model parameter names found in params_table'
    with pytest.raises(ValueError, match=match):
        params_table_to_models(tbl, model)
