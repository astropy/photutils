# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the rectangular_overlap_grid module.
"""

import math

import numpy as np
import pytest
from numpy.testing import assert_allclose

from photutils.geometry import rectangular_overlap_grid

grid_sizes = [50, 500, 1000]
rect_sizes = [0.2, 0.4, 0.8]
angles = [0.0, 0.5, 1.0]
subsamples = [1, 5, 10]


@pytest.mark.parametrize('grid_size', grid_sizes)
@pytest.mark.parametrize('rect_size', rect_sizes)
@pytest.mark.parametrize('angle', angles)
@pytest.mark.parametrize('subsample', subsamples)
def test_rectangular_overlap_grid(grid_size, rect_size, angle, subsample):
    """
    Test normalization of the overlap grid to make sure that a fully
    enclosed pixel has a value of 1.0.
    """
    g = rectangular_overlap_grid(-1.0, 1.0, -1.0, 1.0, grid_size, grid_size,
                                 rect_size, rect_size, angle, 0, subsample)
    assert_allclose(g.max(), 1.0)


def test_axis_aligned_rectangle_exact():
    """
    Exact mode reproduces the analytic overlap for an axis-aligned
    rectangle that exactly fills an integer number of pixels.
    """
    grid = rectangular_overlap_grid(-2.5, 2.5, -2.5, 2.5, 5, 5,
                                    2.0, 2.0, 0.0, 1, 1)
    assert_allclose(grid.sum(), 4.0)
    expected = np.zeros((5, 5))
    # Center pixel fully inside.
    expected[2, 2] = 1.0
    # Edge pixels: half overlap.
    expected[1, 2] = expected[3, 2] = 0.5
    expected[2, 1] = expected[2, 3] = 0.5
    # Corner pixels: quarter overlap.
    expected[1, 1] = expected[1, 3] = 0.25
    expected[3, 1] = expected[3, 3] = 0.25
    assert_allclose(grid, expected)


def test_rectangle_exact_total_area_preserved():
    """
    For any rotation angle, the sum of the exact overlap fractions times
    the pixel area equals the analytic rectangle area, provided the
    rectangle fits inside the grid.
    """
    width, height = 3.7, 1.4
    pixel_area = (10.0 / 25) ** 2
    for theta in (0.0, 0.1, math.pi / 6, math.pi / 4, math.pi / 3, 1.7):
        grid = rectangular_overlap_grid(-5.0, 5.0, -5.0, 5.0, 25, 25,
                                        width, height, theta, 1, 1)
        assert_allclose(grid.sum() * pixel_area, width * height,
                        rtol=1e-12, atol=1e-12)


def test_rectangle_exact_matches_high_subpixel():
    """
    The exact result should agree with a high subpixel approximation to
    within the subpixel-sampling error.
    """
    args = (-3.5, 3.5, -3.5, 3.5, 35, 35, 2.0, 1.0, math.pi / 5)
    exact = rectangular_overlap_grid(*args, 1, 1)
    sub = rectangular_overlap_grid(*args, 0, 64)
    assert np.abs(exact - sub).max() < 0.02


def test_rectangle_exact_full_pixel_is_one():
    """
    A pixel completely inside the rectangle must have overlap = 1.
    """
    grid = rectangular_overlap_grid(-2.0, 2.0, -2.0, 2.0, 4, 4,
                                    10.0, 10.0, 0.3, 1, 1)
    assert_allclose(grid, 1.0)


def test_rectangle_exact_no_overlap():
    """
    A rectangle entirely outside the grid yields all zeros.
    """
    grid = rectangular_overlap_grid(10.0, 12.0, 10.0, 12.0, 4, 4,
                                    1.0, 1.0, 0.0, 1, 1)
    assert_allclose(grid, 0.0)


@pytest.mark.parametrize('use_exact', [0, 1])
def test_rectangular_overlap_grid_no_readonly_inputs(use_exact):
    """
    Test that the scalar-input overlap grid function takes no array
    inputs (so read-only input arrays cannot occur) and returns a
    freshly allocated, writeable output array of shape (ny, nx).
    """
    g = rectangular_overlap_grid(-5.0, 5.0, -5.0, 5.0, 10, 12, 4.0, 2.0,
                                 0.5, use_exact, 5)
    assert g.shape == (12, 10)
    assert g.flags.writeable
