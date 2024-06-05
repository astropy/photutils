# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the sources module.
"""

import numpy as np
import pytest
from astropy.table import Table
from astropy.utils.exceptions import AstropyDeprecationWarning

from photutils.datasets import (make_model_params, make_random_gaussians_table,
                                make_random_models_table)
from photutils.utils._optional_deps import HAS_SCIPY


@pytest.mark.skipif(not HAS_SCIPY, reason='scipy is required')
def test_make_model_params():
    shape = (100, 100)
    n_sources = 10
    flux_range = (100, 1000)
    params = make_model_params(shape, n_sources, flux_range)
    assert isinstance(params, Table)
    assert len(params) == 10
    cols = ('id', 'x_0', 'y_0', 'flux')
    for col in cols:
        assert col in params.colnames
        assert np.min(params[col]) >= 0
    assert np.min(params['flux']) >= flux_range[0]
    assert np.max(params['flux']) <= flux_range[1]


@pytest.mark.skipif(not HAS_SCIPY, reason='scipy is required')
def test_make_model_params_border_size():
    shape = (10, 10)
    n_sources = 10
    flux_range = (100, 1000)
    with pytest.raises(ValueError):
        make_model_params(shape, n_sources, flux_range, border_size=20)


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
