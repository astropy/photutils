#!/usr/bin/env python3
# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Benchmarks for the photutils.geometry subpackage.

The benchmarks cover the per-call cost of the overlap grid functions
(circular_overlap_grid, elliptical_overlap_grid,
rectangular_overlap_grid, and polygon_overlap_grid) for each overlap
method (exact, center, and subpixel), the scaling with grid size, and
the scaling of the polygon overlap with the number of vertices for
convex and non-convex polygons.

Run ``python benchmarks/bench_geometry.py --help`` to see the
available options.
"""

import argparse
from functools import partial

import numpy as np
from bench_helpers import print_environment, time_best

from photutils.geometry import (circular_overlap_grid, elliptical_overlap_grid,
                                rectangular_overlap_grid)
from photutils.geometry._polygon_overlap import polygon_overlap_grid

# The theta used for all rotated shapes
THETA = 0.5


def make_polygon_vertices(n_vertices, radius, *, star=False):
    """
    Return the vertex coordinates of a regular or star-shaped polygon
    centered on the origin.

    Parameters
    ----------
    n_vertices : int
        The number of polygon vertices.

    radius : float
        The circumradius of the polygon.

    star : bool, optional
        If `True`, alternate vertices are placed at half the radius,
        producing a non-convex (star-shaped) polygon. ``n_vertices``
        must be even in that case.

    Returns
    -------
    vertices_x, vertices_y : 1D `~numpy.ndarray`
        The x and y vertex coordinates.
    """
    angles = np.linspace(0.0, 2.0 * np.pi, n_vertices, endpoint=False)
    radii = np.full(n_vertices, radius)
    if star:
        radii[1::2] = 0.5 * radius
    vertices_x = np.ascontiguousarray(radii * np.cos(angles))
    vertices_y = np.ascontiguousarray(radii * np.sin(angles))
    return vertices_x, vertices_y


def make_grid_functions(grid_size, *, fill_frac=0.4, n_vertices=6):
    """
    Return the overlap grid functions with their shape arguments bound,
    leaving only ``use_exact`` and ``subpixels`` free.

    The grid spans ``grid_size`` unit pixels per side and the shapes
    are centered on the origin with a size of ``fill_frac`` times the
    grid half-extent, so the boundary band, the wholly-inside region,
    and the wholly-outside region are all exercised.

    Parameters
    ----------
    grid_size : int
        The number of grid pixels per side.

    fill_frac : float, optional
        The shape size as a fraction of the grid half-extent.

    n_vertices : int, optional
        The number of vertices of the (convex) benchmark polygon.

    Returns
    -------
    result : list of (str, callable) tuples
        The shape name and callable pairs; each callable takes
        ``(use_exact, subpixels)``.
    """
    extent = grid_size / 2.0
    grid_args = (-extent, extent, -extent, extent, grid_size, grid_size)
    radius = fill_frac * extent
    vertices_x, vertices_y = make_polygon_vertices(n_vertices, radius)
    return [
        ('circle', partial(circular_overlap_grid, *grid_args, radius)),
        ('ellipse', partial(elliptical_overlap_grid, *grid_args, radius,
                            0.6 * radius, THETA)),
        ('rectangle', partial(rectangular_overlap_grid, *grid_args,
                              1.6 * radius, 1.0 * radius, THETA)),
        (f'polygon ({n_vertices} vert)',
         partial(polygon_overlap_grid, *grid_args, vertices_x, vertices_y)),
    ]


def time_grid(func, use_exact, subpixels, *, n_iter, repeats):
    """
    Return the best per-call time of an overlap grid callable.

    Parameters
    ----------
    func : callable
        The overlap grid callable taking ``(use_exact, subpixels)``.

    use_exact, subpixels : int
        The overlap method arguments.

    n_iter : int
        The number of calls per timing; the per-call time is returned.

    repeats : int
        The number of repeats for each timing (best time is kept).

    Returns
    -------
    result : float
        The best per-call wall-clock time in seconds.
    """
    def run():
        """
        Call the overlap grid callable ``n_iter`` times.
        """
        for _ in range(n_iter):
            func(use_exact, subpixels)

    return time_best(run, repeats=repeats) / n_iter


def bench_shapes(*, grid_size=200, subpixels=5, n_iter=10, repeats=3):
    """
    Benchmark each overlap grid function for each overlap method.

    Parameters
    ----------
    grid_size : int, optional
        The number of grid pixels per side.

    subpixels : int, optional
        The number of subpixels (per dimension) for the subpixel
        method.

    n_iter : int, optional
        The number of calls per timing; the per-call time is reported.

    repeats : int, optional
        The number of repeats for each timing (best time is kept).
    """
    methods = [('exact', 1, 1), ('center', 0, 1),
               (f'subpixel({subpixels})', 0, subpixels)]
    print(f'\n== Overlap grid functions ({grid_size}x{grid_size} grid, '
          f'per-call time) ==')
    header = f'{"shape":>18}'
    header += ''.join(f'{name:>14}' for name, _, _ in methods)
    print(header)

    for name, func in make_grid_functions(grid_size):
        cells = ''
        for _, use_exact, subpix in methods:
            t_call = time_grid(func, use_exact, subpix, n_iter=n_iter,
                               repeats=repeats)
            cells += f'{t_call * 1e3:>12.3f}ms'
        print(f'{name:>18}{cells}')


def bench_grid_sizes(*, sizes=(50, 100, 200, 400, 800), n_iter=3,
                     repeats=3):
    """
    Benchmark the exact-mode overlap grid functions across grid sizes.

    The shape size scales with the grid so the boundary-band fraction
    stays constant.

    Parameters
    ----------
    sizes : tuple of int, optional
        The grid sizes (pixels per side).

    n_iter : int, optional
        The number of calls per timing; the per-call time is reported.

    repeats : int, optional
        The number of repeats for each timing (best time is kept).
    """
    print('\n== Exact overlap vs grid size (per-call time) ==')
    names = [name for name, _ in make_grid_functions(sizes[0])]
    header = f'{"grid":>10}'
    header += ''.join(f'{name:>18}' for name in names)
    print(header)

    for size in sizes:
        cells = ''
        for _, func in make_grid_functions(size):
            t_call = time_grid(func, 1, 1, n_iter=n_iter, repeats=repeats)
            cells += f'{t_call * 1e3:>16.3f}ms'
        print(f'{f"{size}x{size}":>10}{cells}')


def bench_polygon_vertices(*, n_vertices=(4, 8, 32, 128, 1024),
                           grid_size=200, subpixels=5, n_iter=3,
                           repeats=3):
    """
    Benchmark the polygon overlap grid across vertex counts for convex
    and non-convex (star-shaped) polygons.

    Convex polygons use the interior/exterior fast path, so only the
    boundary band pays the per-vertex clipping cost; non-convex
    polygons clip every pixel in the bounding box.

    Parameters
    ----------
    n_vertices : tuple of int, optional
        The vertex counts. Each must be even (required for the star
        shapes).

    grid_size : int, optional
        The number of grid pixels per side.

    subpixels : int, optional
        The number of subpixels (per dimension) for the subpixel
        method.

    n_iter : int, optional
        The number of calls per timing; the per-call time is reported.

    repeats : int, optional
        The number of repeats for each timing (best time is kept).
    """
    extent = grid_size / 2.0
    grid_args = (-extent, extent, -extent, extent, grid_size, grid_size)
    radius = 0.4 * extent
    columns = [('exact convex', False, 1, 1),
               ('exact star', True, 1, 1),
               (f'subpix({subpixels}) convex', False, 0, subpixels),
               (f'subpix({subpixels}) star', True, 0, subpixels)]

    print(f'\n== polygon_overlap_grid vs vertex count '
          f'({grid_size}x{grid_size} grid, per-call time) ==')
    header = f'{"vertices":>10}'
    header += ''.join(f'{name:>18}' for name, _, _, _ in columns)
    print(header)

    for n_verts in n_vertices:
        cells = ''
        for _, star, use_exact, subpix in columns:
            vertices_x, vertices_y = make_polygon_vertices(
                n_verts, radius, star=star)
            func = partial(polygon_overlap_grid, *grid_args,
                           vertices_x, vertices_y)
            t_call = time_grid(func, use_exact, subpix, n_iter=n_iter,
                               repeats=repeats)
            cells += f'{t_call * 1e3:>16.3f}ms'
        print(f'{n_verts:>10}{cells}')


def parse_int_list(text):
    """
    Parse a comma-separated list of positive integers.

    Parameters
    ----------
    text : str
        The comma-separated integers (e.g., ``'50,100,200'``).

    Returns
    -------
    result : list of int
        The parsed integers.
    """
    values = [int(item) for item in text.split(',')]
    if any(value < 1 for value in values):
        msg = 'values must be positive integers'
        raise ValueError(msg)
    return values


def main():
    """
    Run the photutils.geometry benchmarks.
    """
    parser = argparse.ArgumentParser(
        description='Benchmarks for the photutils.geometry subpackage.')
    parser.add_argument('--grid-size', type=int, default=200,
                        help='grid pixels per side for the shape and '
                             'polygon benchmarks (default: %(default)s)')
    parser.add_argument('--sizes', type=parse_int_list,
                        default=[50, 100, 200, 400, 800],
                        help='comma-separated grid sizes for the '
                             'grid-size scaling benchmark '
                             '(default: 50,100,200,400,800)')
    parser.add_argument('--n-vertices', type=parse_int_list,
                        default=[4, 8, 32, 128, 1024],
                        help='comma-separated (even) vertex counts for '
                             'the polygon benchmark '
                             '(default: 4,8,32,128,1024)')
    parser.add_argument('--subpixels', type=int, default=5,
                        help='subpixels per dimension for the subpixel '
                             'method (default: %(default)s)')
    parser.add_argument('--repeats', type=int, default=3,
                        help='number of repeats per timing; the best '
                             'time is reported (default: %(default)s)')
    parser.add_argument('--which', default='all',
                        choices=['all', 'shapes', 'sizes', 'polygon'],
                        help='which benchmark to run '
                             '(default: %(default)s)')
    args = parser.parse_args()

    if any(n % 2 for n in args.n_vertices):
        parser.error('--n-vertices values must be even')

    print_environment()

    if args.which in ('all', 'shapes'):
        bench_shapes(grid_size=args.grid_size, subpixels=args.subpixels,
                     repeats=args.repeats)
    if args.which in ('all', 'sizes'):
        bench_grid_sizes(sizes=args.sizes, repeats=args.repeats)
    if args.which in ('all', 'polygon'):
        bench_polygon_vertices(n_vertices=args.n_vertices,
                               grid_size=args.grid_size,
                               subpixels=args.subpixels,
                               repeats=args.repeats)


if __name__ == '__main__':
    main()
