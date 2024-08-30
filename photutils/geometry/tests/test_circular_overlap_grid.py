# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the circular_overlap_grid module.
"""

import pytest
from numpy.testing import assert_allclose

from photutils.geometry import circular_overlap_grid

grid_sizes = [50, 500, 1000]
circ_sizes = [0.2, 0.4, 0.8]
use_exacts = [0, 1]
subsamples = [1, 5, 10]


@pytest.mark.parametrize('grid_size', grid_sizes)
@pytest.mark.parametrize('circ_size', circ_sizes)
@pytest.mark.parametrize('use_exact', use_exacts)
@pytest.mark.parametrize('subsample', subsamples)
def test_circular_overlap_grid(grid_size, circ_size, use_exact, subsample):
    """
    Test normalization of the overlap grid to make sure that a fully
    enclosed pixel has a value of 1.0.
    """
    g = circular_overlap_grid(-1.0, 1.0, -1.0, 1.0, grid_size, grid_size,
                              circ_size, use_exact, subsample)
    assert_allclose(g.max(), 1.0)
