# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the ellipse_overlap module.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from photutils.geometry import elliptical_overlap_grid

grid_sizes = [50, 500, 1000]
maj_sizes = [0.2, 0.4, 0.8]
min_sizes = [0.2, 0.4, 0.8]
angles = [0.0, 0.5, 1.0]
use_exacts = [0, 1]
subsamples = [1, 5, 10]


@pytest.mark.parametrize('grid_size', grid_sizes)
@pytest.mark.parametrize('maj_size', maj_sizes)
@pytest.mark.parametrize('min_size', min_sizes)
@pytest.mark.parametrize('angle', angles)
@pytest.mark.parametrize('use_exact', use_exacts)
@pytest.mark.parametrize('subsample', subsamples)
def test_elliptical_overlap_grid(grid_size, maj_size, min_size, angle,
                                 use_exact, subsample):
    """
    Test normalization of the overlap grid to make sure that a fully
    enclosed pixel has a value of 1.0.
    """
    g = elliptical_overlap_grid(-1.0, 1.0, -1.0, 1.0, grid_size, grid_size,
                                maj_size, min_size, angle, use_exact,
                                subsample)
    assert_allclose(g.max(), 1.0)


@pytest.mark.parametrize('theta', [0.0, 0.3])
def test_elliptical_overlap_smaller_than_pixel(theta):
    """
    An ellipse smaller than a pixel, centered exactly on a pixel
    center, must return the ellipse area, not 1.0.
    """
    area = np.pi * 0.4 * 0.3
    grid = elliptical_overlap_grid(-1.5, 1.5, -1.5, 1.5, 3, 3,
                                   0.4, 0.3, theta, 1, 1)
    assert_allclose(grid[1, 1], area, rtol=1e-10)
    assert_allclose(grid.sum(), area, rtol=1e-10)


@pytest.mark.parametrize(('rx', 'ry', 'theta'), [
    (3.0, 1.5, 0.7),
    (2.2, 0.4, -1.2),
    (4.0, 4.0, 0.0),
    (0.4, 0.3, 0.5),
])
def test_elliptical_overlap_exact_total_area(rx, ry, theta):
    """
    The exact-mode total overlap times the pixel area equals the
    analytic ellipse area when the ellipse fits inside the grid, for
    various axis ratios and rotation angles.
    """
    grid = elliptical_overlap_grid(-5.0, 5.0, -5.0, 5.0, 100, 100,
                                   rx, ry, theta, 1, 1)
    pixel_area = (10.0 / 100) ** 2
    assert_allclose(grid.sum() * pixel_area, np.pi * rx * ry, rtol=1e-10)


def test_elliptical_overlap_exact_matches_high_subpixel():
    """
    The exact result should agree with a high subpixel approximation to
    within the subpixel-sampling error.
    """
    args = (-2.0, 2.0, -2.0, 2.0, 20, 20, 1.5, 0.8, 0.4)
    exact = elliptical_overlap_grid(*args, 1, 1)
    sub = elliptical_overlap_grid(*args, 0, 64)
    assert np.abs(exact - sub).max() < 0.02


def test_elliptical_overlap_grid_validation():
    """
    Test that invalid use_exact and subpixels inputs raise errors.
    """
    match = 'use_exact must be 0 or 1'
    with pytest.raises(ValueError, match=match):
        elliptical_overlap_grid(-1.0, 1.0, -1.0, 1.0, 4, 4, 0.5, 0.3,
                                0.1, 2, 5)

    match = 'subpixels must be a strictly positive integer'
    for subpixels in (0, -1):
        with pytest.raises(ValueError, match=match):
            elliptical_overlap_grid(-1.0, 1.0, -1.0, 1.0, 4, 4, 0.5, 0.3,
                                    0.1, 0, subpixels)

    # subpixels is ignored (not validated) for the exact method
    g = elliptical_overlap_grid(-1.0, 1.0, -1.0, 1.0, 4, 4, 0.5, 0.3,
                                0.1, 1, 0)
    assert np.isfinite(g).all()


@pytest.mark.parametrize('use_exact', use_exacts)
def test_elliptical_overlap_grid_no_readonly_inputs(use_exact):
    """
    Test that the scalar-input overlap grid function takes no array
    inputs (so read-only input arrays cannot occur) and returns a
    freshly allocated, writeable output array of shape (ny, nx).
    """
    g = elliptical_overlap_grid(-5.0, 5.0, -5.0, 5.0, 10, 12, 4.0, 2.0,
                                0.5, use_exact, 5)
    assert g.shape == (12, 10)
    assert g.flags.writeable
