# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the sources module.
"""

import numpy as np
import pytest
from astropy.utils.exceptions import AstropyDeprecationWarning

from photutils.datasets import (make_random_gaussians_table,
                                make_random_models_table)


def test_make_random_models_table():
    param_ranges = {'x_0': (0, 300), 'y_0': (0, 500),
                    'gamma': (1, 3), 'alpha': (1.5, 3)}
    source_table = make_random_models_table(10, param_ranges)
    assert len(source_table) == 10
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
