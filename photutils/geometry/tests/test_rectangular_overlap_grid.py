# Licensed under a 3-clause BSD style license - see LICENSE.rst

import itertools

from numpy.testing import assert_allclose
import pytest

from .. import rectangular_overlap_grid


grid_sizes = [50, 500, 1000]
rect_sizes = [0.2, 0.4, 0.8]
angles = [0.0, 0.5, 1.0]
subsamples = [1, 5, 10]
arg_list = ['grid_size', 'rect_size', 'angle', 'subsample']


@pytest.mark.parametrize(('grid_size', 'rect_size', 'angle', 'subsample'),
                         list(itertools.product(grid_sizes, rect_sizes,
                                                angles, subsamples)))
def test_rectangular_overlap_grid(grid_size, rect_size, angle, subsample):
    """
    Test normalization of the overlap grid to make sure that a fully
    enclosed pixel has a value of 1.0.
    """

    g = rectangular_overlap_grid(-1.0, 1.0, -1.0, 1.0, grid_size, grid_size,
                                 rect_size, rect_size, angle, 0, subsample)
    assert_allclose(g.max(), 1.0)
