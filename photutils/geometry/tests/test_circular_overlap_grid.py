# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the circular_overlap_grid module.
"""

import itertools

from numpy.testing import assert_allclose
import pytest

from .. import circular_overlap_grid


grid_sizes = [50, 500, 1000]
circ_sizes = [0.2, 0.4, 0.8]
use_exact = [0, 1]
subsamples = [1, 5, 10]
arg_list = ['grid_size', 'circ_size', 'use_exact', 'subsample']


@pytest.mark.parametrize(('grid_size', 'circ_size', 'use_exact', 'subsample'),
                         list(itertools.product(grid_sizes, circ_sizes,
                                                use_exact, subsamples)))
def test_circular_overlap_grid(grid_size, circ_size, use_exact, subsample):
    """
    Test normalization of the overlap grid to make sure that a fully
    enclosed pixel has a value of 1.0.
    """

    g = circular_overlap_grid(-1.0, 1.0, -1.0, 1.0, grid_size, grid_size,
                              circ_size, use_exact, subsample)
    assert_allclose(g.max(), 1.0)
