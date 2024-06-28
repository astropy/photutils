# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the sources module.
"""

import numpy as np
import pytest
from astropy.table import Table
from astropy.utils.exceptions import (AstropyDeprecationWarning,
                                      AstropyUserWarning)

from photutils.datasets import (make_model_params, make_random_gaussians_table,
                                make_random_models_table)
from photutils.utils._optional_deps import HAS_SCIPY


@pytest.mark.skipif(not HAS_SCIPY, reason='scipy is required')
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


@pytest.mark.skipif(not HAS_SCIPY, reason='scipy is required')
def test_make_model_params_nsources():
    """
    Test case when the number of the possible sources is less than
    ``n_sources``.
    """
    with pytest.warns(AstropyUserWarning):
        shape = (200, 500)
        n_sources = 100
        params = make_model_params(shape, n_sources, min_separation=50,
                                   amplitude=(100, 500), x_stddev=(1, 5),
                                   y_stddev=(1, 5), theta=(0, np.pi))
        assert len(params) < 100


@pytest.mark.skipif(not HAS_SCIPY, reason='scipy is required')
def test_make_model_params_border_size():
    shape = (10, 10)
    n_sources = 10
    flux = (100, 1000)
    with pytest.raises(ValueError):
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


def test_make_random_gaussians_table():
    with pytest.warns(AstropyDeprecationWarning):
        n_sources = 5
        param_ranges = dict([('amplitude', [500, 1000]), ('x_mean', [0, 500]),
                             ('y_mean', [0, 300]), ('x_stddev', [1, 5]),
                             ('y_stddev', [1, 5]), ('theta', [0, np.pi])])

        table = make_random_gaussians_table(n_sources, param_ranges, seed=0)
        assert 'flux' in table.colnames
        assert len(table) == n_sources


def test_make_random_gaussians_table_no_stddev():
    with pytest.warns(AstropyDeprecationWarning):
        n_sources = 5
        param_ranges = dict([('amplitude', [500, 1000]), ('x_mean', [0, 500]),
                             ('y_mean', [0, 300])])

        table = make_random_gaussians_table(n_sources, param_ranges, seed=0)
        assert 'flux' in table.colnames
        assert len(table) == n_sources
        assert 'x_stddev' not in table.colnames
        assert 'y_stddev' not in table.colnames


def test_make_random_gaussians_table_flux():
    with pytest.warns(AstropyDeprecationWarning):
        n_sources = 5
        param_ranges = dict([('flux', [500, 1000]), ('x_mean', [0, 500]),
                             ('y_mean', [0, 300]), ('x_stddev', [1, 5]),
                             ('y_stddev', [1, 5]), ('theta', [0, np.pi])])
        table = make_random_gaussians_table(n_sources, param_ranges, seed=0)
        assert 'amplitude' in table.colnames
        assert len(table) == n_sources
