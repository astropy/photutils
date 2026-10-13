# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the _polygon_overlap module.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from photutils.geometry._polygon_overlap import polygon_overlap_grid


def test_polygon_overlap_clockwise_input():
    """
    Clockwise-ordered vertices should give the same result as
    counterclockwise ones.
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


@pytest.mark.parametrize('use_exact', [1, 0])
def test_polygon_overlap_readonly_vertices(use_exact):
    """
    Regression test that read-only (nonwritable) vertex arrays are
    accepted and give results identical to writable arrays.

    The vertex arrays are buffered into ``const`` typed memoryviews so
    that read-only arrays do not raise a ``ValueError``.
    """
    vx = np.array([-4.0, 6.0, 3.0, -5.0])
    vy = np.array([-3.0, -2.0, 5.0, 4.0])
    expected = polygon_overlap_grid(-10.0, 10.0, -10.0, 10.0, 40, 40,
                                    vx, vy, use_exact, 5)

    vx_ro = vx.copy()
    vy_ro = vy.copy()
    vx_ro.setflags(write=False)
    vy_ro.setflags(write=False)
    result = polygon_overlap_grid(-10.0, 10.0, -10.0, 10.0, 40, 40,
                                  vx_ro, vy_ro, use_exact, 5)
    assert_allclose(result, expected, rtol=1e-12)


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


@pytest.mark.parametrize('subpix', [1, 3, 5])
def test_polygon_overlap_subpixel_convex_fast_path(subpix):
    """
    Test that the convex interior/exterior fast path of the subpixel
    mode returns results identical to sampling every pixel.

    The same hexagon with a duplicated vertex is geometrically
    identical but is reported as nonconvex (the zero-length edge fails
    ``convex_edge_normals``), which disables the fast path and forces
    the full scanline sampling for every pixel.
    """
    angles = np.pi / 6.0 + np.arange(6) * np.pi / 3.0
    vx = np.ascontiguousarray(4.0 * np.cos(angles))
    vy = np.ascontiguousarray(4.0 * np.sin(angles))
    vx_dup = np.append(vx, vx[0])
    vy_dup = np.append(vy, vy[0])
    fast = polygon_overlap_grid(-5.0, 5.0, -5.0, 5.0, 40, 40,
                                vx, vy, 0, subpix)
    slow = polygon_overlap_grid(-5.0, 5.0, -5.0, 5.0, 40, 40,
                                vx_dup, vy_dup, 0, subpix)
    assert_array_equal(fast, slow)
    # Sanity check that interior pixels exist and are exactly 1
    assert fast.max() == 1.0


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
    The exact-mode kernel handles nonconvex (L-shaped) polygons.
    The L-shape used here has area 12 (4x4 square minus 2x2 corner).
    """
    vx = np.array([0.0, 4.0, 4.0, 2.0, 2.0, 0.0])
    vy = np.array([0.0, 0.0, 2.0, 2.0, 4.0, 4.0])
    grid = polygon_overlap_grid(-1.0, 5.0, -1.0, 5.0, 60, 60, vx, vy, 1, 1)
    pixel_area = (6.0 / 60) ** 2
    assert_allclose(grid.sum() * pixel_area, 12.0, atol=1e-12)


def test_polygon_overlap_nonconvex_star():
    """
    The exact-mode kernel handles a 5-pointed star (nonconvex,
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
    (0.5, 0.5),  # subpixel rectangle
    (3.0, 2.5),  # noninteger dimensions
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

    vx = np.array([0.0, 1.0, 0.0])
    vy = np.array([0.0, 0.0, 1.0])
    match = 'use_exact must be 0 or 1'
    with pytest.raises(ValueError, match=match):
        polygon_overlap_grid(-1.0, 1.0, -1.0, 1.0, 4, 4, vx, vy, 2, 1)

    match = 'subpixels must be a strictly positive integer'
    for subpixels in (0, -1):
        with pytest.raises(ValueError, match=match):
            polygon_overlap_grid(-1.0, 1.0, -1.0, 1.0, 4, 4, vx, vy, 0,
                                 subpixels)


def test_polygon_overlap_vertices_input_types():
    """
    Test that integer arrays and Python lists are accepted for the
    vertex coordinates and give the same result as float64 arrays,
    and that non-1D vertex arrays raise an error.
    """
    vx = np.array([0.0, 4.0, 0.0])
    vy = np.array([0.0, 0.0, 4.0])
    expected = polygon_overlap_grid(-1.0, 5.0, -1.0, 5.0, 12, 12,
                                    vx, vy, 1, 1)

    result = polygon_overlap_grid(-1.0, 5.0, -1.0, 5.0, 12, 12,
                                  np.array([0, 4, 0]),
                                  np.array([0, 0, 4]), 1, 1)
    assert_allclose(result, expected)

    result = polygon_overlap_grid(-1.0, 5.0, -1.0, 5.0, 12, 12,
                                  [0.0, 4.0, 0.0], [0.0, 0.0, 4.0], 1, 1)
    assert_allclose(result, expected)

    match = 'must be 1D arrays'
    with pytest.raises(ValueError, match=match):
        polygon_overlap_grid(-1.0, 1.0, -1.0, 1.0, 4, 4,
                             np.zeros((3, 2)), np.zeros((3, 2)), 1, 1)


def test_polygon_overlap_many_vertices():
    """
    There is no limit on the number of polygon vertices.

    A 10000-vertex polygon approximating a circle should have the
    analytic circle area.
    """
    n_verts = 10000
    radius = 3.0
    angles = np.linspace(0.0, 2 * np.pi, n_verts, endpoint=False)
    vx = np.ascontiguousarray(radius * np.cos(angles))
    vy = np.ascontiguousarray(radius * np.sin(angles))
    n = 70
    grid = polygon_overlap_grid(-3.5, 3.5, -3.5, 3.5, n, n, vx, vy, 1, 1)
    pixel_area = (7.0 / n) ** 2
    assert_allclose(grid.sum() * pixel_area, np.pi * radius**2, rtol=1e-6)


def test_polygon_overlap_subpixel_on_horizontal_boundary():
    """
    Test that subpixel centers landing exactly on a horizontal polygon
    edge are excluded (classified as outside).

    A wide rectangle spanning y in [-1, 1] is sampled by a single pixel
    with ``subpixels=3``, so the subpixel-row centers fall exactly at y
    = -1, 0, +1. The top and bottom rows lie exactly on the rectangle's
    horizontal edges (with all column centers strictly inside in x), so
    only the middle row (3 of 9 subpixels) is counted, giving 1/3.

    Before horizontal-edge handling, the bottom edge (lower y, region
    above) was incorrectly counted, which would give 2/3 instead.
    """
    vx = np.array([-2.0, 2.0, 2.0, -2.0])
    vy = np.array([-1.0, -1.0, 1.0, 1.0])
    grid = polygon_overlap_grid(-1.5, 1.5, -1.5, 1.5, 1, 1, vx, vy, 0, 3)
    assert_allclose(grid[0, 0], 1.0 / 3.0)


def test_polygon_overlap_subpixel_on_vertical_boundary():
    """
    Test that subpixel centers landing exactly on a vertical polygon
    edge are excluded (classified as outside).

    A tall rectangle spanning x in [-1, 1] is sampled by a single pixel
    with ``subpixels=3``, so the subpixel-column centers fall exactly
    at x = -1, 0, +1. The left and right columns lie exactly on the
    rectangle's vertical edges (with all row centers strictly inside in
    y), so only the middle column (3 of 9 subpixels) is counted, giving
    1/3.
    """
    vx = np.array([-1.0, 1.0, 1.0, -1.0])
    vy = np.array([-2.0, -2.0, 2.0, 2.0])
    grid = polygon_overlap_grid(-1.5, 1.5, -1.5, 1.5, 1, 1, vx, vy, 0, 3)
    assert_allclose(grid[0, 0], 1.0 / 3.0)


def test_polygon_overlap_subpixel_on_all_boundaries():
    """
    Test that subpixel centers on horizontal edges, vertical edges, and
    at corners are all excluded.

    A unit square spanning [-1, 1] in both axes is sampled by a single
    pixel with ``subpixels=3``, so the 9 subpixel centers fall at every
    combination of (-1, 0, +1). Of these, only the central center
    (0, 0) is strictly interior. The four edge-midpoint centers (two
    horizontal, two vertical) and the four corner centers all lie on the
    boundary and are excluded, giving 1/9.
    """
    vx = np.array([-1.0, 1.0, 1.0, -1.0])
    vy = np.array([-1.0, -1.0, 1.0, 1.0])
    grid = polygon_overlap_grid(-1.5, 1.5, -1.5, 1.5, 1, 1, vx, vy, 0, 3)
    assert_allclose(grid[0, 0], 1.0 / 9.0)


def test_polygon_overlap_subpixel_on_reflex_vertex():
    """
    Test that a subpixel center landing exactly on a vertex that is a
    local maximum in y, with the polygon interior above it, is
    excluded.

    Neither edge at such a vertex straddles the sample row, so the
    vertex produces no crossing. The notch vertex at (1, 1) is sampled
    by a single pixel with ``subpixels=3``, so the 9 subpixel centers
    fall at every combination of (0.5, 1, 1.5). The three centers in
    the bottom row are below the notch edges and the vertex sample is on
    the boundary, leaving 5 of 9.
    """
    vx = np.array([0.0, 1.0, 2.0, 2.0, 0.0])
    vy = np.array([0.0, 1.0, 0.0, 3.0, 3.0])
    for sign in (1, -1):
        grid = polygon_overlap_grid(0.25, 1.75, 0.25, 1.75, 1, 1,
                                    vx[::sign], vy[::sign], 0, 3)
        assert_allclose(grid[0, 0], 5.0 / 9.0)


def test_polygon_overlap_subpixel_on_vertex_matches_center():
    """
    Test that a subpixel center landing exactly on a vertex whose
    neighbors lie on opposite sides of the sample row is excluded.

    The subpixel fraction of a single pixel must equal the mean of the
    ``center`` method evaluated on the same sample points.
    """
    rng = np.random.default_rng(0)
    for _ in range(500):
        # A triangle with a vertex on the middle sample at (1, 1), one
        # vertex below the sample row, and one vertex above it
        vx = np.array([1.0, rng.uniform(-3, 0), rng.uniform(-3, 0)])
        vy = np.array([1.0, rng.uniform(-3, 0.9), rng.uniform(1.1, 4)])
        for sign in (1, -1):
            args = (vx[::sign], vy[::sign], 0)
            subpix = polygon_overlap_grid(0.25, 1.75, 0.25, 1.75, 1, 1,
                                          *args, 3)
            center = polygon_overlap_grid(0.25, 1.75, 0.25, 1.75, 3, 3,
                                          *args, 1)
            assert center[1, 1] == 0.0
            assert_allclose(subpix[0, 0], center.mean())
