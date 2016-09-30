# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import division

import itertools
from numpy.testing import assert_allclose
from astropy.tests.helper import pytest
from photutils.geometry import polygon_overlap_grid

grid_sizes = [50, 500]
polygons = [[(-1, -1),(-1, 1),(1, 1)],[(-1, -1),(-1, 1),(1, 1),(1, -1)]]
subsamples = [1]
arg_list = ['grid_size', 'polygon', 'subsample']

@pytest.mark.parametrize(('grid_size', 'polygon', 'subsample'),
                         list(itertools.product(grid_sizes, polygons, subsamples)))
def test_polygon_overlap_grid(grid_size, polygon, subsample):
    """
    Test normalization of the overlap grid to make sure that a fully enclosed pixel has a value of 1.0.
    """
    g = polygon_overlap_grid(-1.0, 1.0, -1.0, 1.0, grid_size, grid_size, polygon, 0, subsample)
    assert_allclose(g.max(), 1.0)
