# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the polygon_overlap_grid module.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from photutils.geometry._polygon_overlap import polygon_overlap_grid


def test_polygon_overlap_clockwise_input():
    """
    Clockwise-ordered vertices should give the same result as
    counter-clockwise ones.
    """
    vx_ccw = np.array([-1.0, 1.0, 1.0, -1.0])
    vy_ccw = np.array([-1.0, -1.0, 1.0, 1.0])
    vx_cw = vx_ccw[::-1].copy()
    vy_cw = vy_ccw[::-1].copy()
    grid_ccw = polygon_overlap_grid(-2.5, 2.5, -2.5, 2.5, 5, 5,
                                    vx_ccw, vy_ccw, 1, 1)
    grid_cw = polygon_overlap_grid(-2.5, 2.5, -2.5, 2.5, 5, 5,
                                   vx_cw, vy_cw, 1, 1)
    assert_allclose(grid_cw, grid_ccw)


@pytest.mark.parametrize(('leg', 'dx', 'dy'), [
    (3.0, 0.0, 0.0),  # base case
    (1.0, 0.0, 0.0),  # small triangle
    (6.0, 0.0, 0.0),  # large triangle
    (3.0, 0.5, 0.0),  # shifted half-pixel in x
    (3.0, 0.0, 0.5),  # shifted half-pixel in y
    (3.0, 0.25, 0.25),  # shifted diagonally
    (2.0, 1.0, -1.0),  # different size with offset
])
def test_polygon_overlap_triangle_area(leg, dx, dy):
    """
    The total overlap of a right triangle with given leg length equals
    its analytic area, for various sizes and pixel offsets.
    """
    vx = np.array([0.0, leg, 0.0]) + dx
    vy = np.array([0.0, 0.0, leg]) + dy
    margin = 0.5
    xmin = vx.min() - margin
    xmax = vx.max() + margin
    ymin = vy.min() - margin
    ymax = vy.max() + margin
    nx = max(4, int(leg * 4))
    pixel_area = ((xmax - xmin) / nx) * ((ymax - ymin) / nx)
    grid = polygon_overlap_grid(
        xmin, xmax, ymin, ymax, nx, nx, vx, vy, 1, 1,
    )
    assert_allclose(grid.sum() * pixel_area, 0.5 * leg * leg)


@pytest.mark.parametrize(('half_side', 'subpix', 'atol'), [
    (1.0, 16, 0.05),  # base case, 2x2 square
    (2.0, 16, 0.5),  # larger square, 4x4 area
    (0.5, 4, 0.3),  # small square
    (1.0, 1, 2.0),  # subpixel=1 (center-of-pixel, coarse by design)
    (1.0, 32, 0.02),  # high subpixel sampling
])
def test_polygon_overlap_subpixel_mode(half_side, subpix, atol):
    """
    Subpixel mode returns values in [0, 1] and the total weighted area
    matches the analytic area for various square sizes and sampling
    sizes.
    """
    s = half_side
    vx = np.array([-s, s, s, -s])
    vy = np.array([-s, -s, s, s])
    lim = s + 1.0
    nx = 5
    pixel_area = (2.0 * lim / nx) ** 2
    grid = polygon_overlap_grid(-lim, lim, -lim, lim, nx, nx,
                                vx, vy, 0, subpix)
    assert grid.min() >= 0.0
    assert grid.max() <= 1.0
    assert_allclose(grid.sum() * pixel_area, (2.0 * s) ** 2, atol=atol)


def test_polygon_overlap_no_intersection():
    """
    A polygon that does not intersect the grid yields all zeros.
    """
    vx = np.array([10.0, 12.0, 11.0])
    vy = np.array([10.0, 10.0, 12.0])
    grid = polygon_overlap_grid(-2.0, 2.0, -2.0, 2.0, 4, 4, vx, vy, 1, 1)
    assert_allclose(grid, 0.0)


def test_polygon_overlap_nonconvex_l_shape():
    """
    The exact-mode kernel handles non-convex (L-shaped) polygons.
    The L-shape used here has area 12 (4x4 square minus 2x2 corner).
    """
    vx = np.array([0.0, 4.0, 4.0, 2.0, 2.0, 0.0])
    vy = np.array([0.0, 0.0, 2.0, 2.0, 4.0, 4.0])
    grid = polygon_overlap_grid(-1.0, 5.0, -1.0, 5.0, 60, 60, vx, vy, 1, 1)
    pixel_area = (6.0 / 60) ** 2
    assert_allclose(grid.sum() * pixel_area, 12.0, atol=1e-12)


def test_polygon_overlap_nonconvex_star():
    """
    The exact-mode kernel handles a 5-pointed star (non-convex,
    non-self-intersecting). The total area should match the shoelace
    area of the polygon.
    """
    n_points = 5
    r_outer, r_inner = 5.0, 2.0
    angles = (np.pi / 2
              + np.arange(2 * n_points) * np.pi / n_points)
    radii = np.where(np.arange(2 * n_points) % 2 == 0, r_outer, r_inner)
    vx = np.ascontiguousarray(radii * np.cos(angles))
    vy = np.ascontiguousarray(radii * np.sin(angles))
    shoelace = abs(0.5 * np.sum(vx * np.roll(vy, -1)
                                - np.roll(vx, -1) * vy))

    grid = polygon_overlap_grid(-6.0, 6.0, -6.0, 6.0, 120, 120,
                                vx, vy, 1, 1)
    pixel_area = (12.0 / 120) ** 2
    assert_allclose(grid.sum() * pixel_area, shoelace, atol=1e-10)


@pytest.mark.parametrize(('width', 'height'), [
    (2.0, 2.0),  # square
    (4.0, 1.0),  # wide rectangle
    (1.0, 4.0),  # tall rectangle
    (0.5, 0.5),  # sub-pixel rectangle
    (3.0, 2.5),  # non-integer dimensions
])
def test_polygon_overlap_rectangle_area(width, height):
    """
    The total overlap of an axis-aligned rectangle equals width * height.
    """
    hw, hh = width / 2.0, height / 2.0
    vx = np.array([-hw, hw, hw, -hw])
    vy = np.array([-hh, -hh, hh, hh])
    margin = 0.5
    xmin, xmax = -hw - margin, hw + margin
    ymin, ymax = -hh - margin, hh + margin
    nx = max(4, int((width + 2 * margin) * 10))
    ny = max(4, int((height + 2 * margin) * 10))
    pixel_area = ((xmax - xmin) / nx) * ((ymax - ymin) / ny)
    grid = polygon_overlap_grid(
        xmin, xmax, ymin, ymax, nx, ny, vx, vy, 1, 1,
    )
    assert_allclose(grid.sum() * pixel_area, width * height)


@pytest.mark.parametrize('radius', [1.0, 2.0, 3.5, 0.8])
def test_polygon_overlap_hexagon_area(radius):
    """
    The total overlap of a regular hexagon equals its analytic area
    (3*sqrt(3)/2 * radius^2).
    """
    angles = np.pi / 6.0 + np.arange(6) * np.pi / 3.0
    vx = np.ascontiguousarray(radius * np.cos(angles))
    vy = np.ascontiguousarray(radius * np.sin(angles))
    expected_area = 1.5 * np.sqrt(3.0) * radius ** 2
    margin = 0.5
    n = max(60, int(2.0 * radius * 30))
    grid = polygon_overlap_grid(
        -radius - margin, radius + margin,
        -radius - margin, radius + margin,
        n, n, vx, vy, 1, 1,
    )
    pixel_area = ((2.0 * (radius + margin)) / n) ** 2
    assert_allclose(grid.sum() * pixel_area, expected_area, atol=1e-10)


def test_polygon_overlap_validation():
    """
    Input validation errors are raised for malformed polygons.
    """
    vx = np.array([0.0, 1.0])
    vy = np.array([0.0, 0.0])
    match = 'at least 3 vertices'
    with pytest.raises(ValueError, match=match):
        polygon_overlap_grid(-1.0, 1.0, -1.0, 1.0, 4, 4, vx, vy, 1, 1)

    vx = np.array([0.0, 1.0, 0.0])
    vy = np.array([0.0, 0.0])
    match = 'same length'
    with pytest.raises(ValueError, match=match):
        polygon_overlap_grid(-1.0, 1.0, -1.0, 1.0, 4, 4, vx, vy, 1, 1)

    n_too_many = 600
    angles = np.linspace(0.0, 2 * np.pi, n_too_many, endpoint=False)
    vx = np.cos(angles)
    vy = np.sin(angles)
    match = 'too many vertices'
    with pytest.raises(ValueError, match=match):
        polygon_overlap_grid(-1.0, 1.0, -1.0, 1.0, 4, 4, vx, vy, 1, 1)
