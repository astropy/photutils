#!/usr/bin/env python3
# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Benchmarks for the photutils.utils subpackage.

The benchmarks cover the scaling of ``ImageDepth`` with the number
of apertures (including the non-overlapping mode and concurrent
calls from multiple threads), ``ShepardIDWInterpolator`` construction
and evaluation, cutout generation, ``calc_total_error``, the
NaN-ignoring statistics functions, random-coordinate generation with
a minimum separation, and the local WCS helper functions.

Run ``python benchmarks/bench_utils.py --help`` to see the available
options.
"""

import argparse
import warnings
from concurrent.futures import ThreadPoolExecutor
from functools import partial

import numpy as np
from astropy.utils.exceptions import AstropyUserWarning
from bench_helpers import (format_sweep_cells, make_image, parse_thread_counts,
                           print_environment, time_best)

from photutils.datasets import make_wcs
from photutils.utils import (CutoutImage, ImageDepth, ShepardIDWInterpolator,
                             calc_total_error)
from photutils.utils._coords import make_random_xycoords
from photutils.utils._stats import (nanmax, nanmean, nanmedian, nanmin, nanstd,
                                    nansum, nanvar)
from photutils.utils._wcs_helpers import (compute_local_wcs_jacobian,
                                          wcs_pixel_scale_angle)
from photutils.utils.cutouts import _make_cutouts

APER_RADIUS = 4.0


def make_depth_inputs(size, *, seed=0):
    """
    Return a noise image and a source mask for the ImageDepth
    benchmarks.

    The mask contains compact square "sources" covering ~5% of the
    image, so that plenty of unmasked area remains after the mask is
    dilated by the aperture radius.

    Parameters
    ----------
    size : int
        The image size; the image is ``(size, size)``.

    seed : int, optional
        The random number generator seed.

    Returns
    -------
    data : 2D `~numpy.ndarray`
        The noise image.

    mask : 2D bool `~numpy.ndarray`
        The source mask.
    """
    data = make_image((size, size), seed=seed)
    rng = np.random.default_rng(seed)
    mask = np.zeros(data.shape, dtype=bool)
    blob_size = 20
    n_blobs = max(1, (size * size) // (20 * blob_size**2))
    for _ in range(n_blobs):
        y = rng.integers(0, size - blob_size)
        x = rng.integers(0, size - blob_size)
        mask[y:y + blob_size, x:x + blob_size] = True
    return data, mask


def make_depth(n_apertures, *, overlap=True):
    """
    Return a seeded ImageDepth instance for the benchmarks.

    Parameters
    ----------
    n_apertures : int
        The number of circular apertures.

    overlap : bool, optional
        Whether to allow the apertures to overlap.

    Returns
    -------
    depth : `~photutils.utils.ImageDepth`
        The ImageDepth instance.
    """
    return ImageDepth(APER_RADIUS, n_sigma=5.0, n_apertures=n_apertures,
                      n_iters=2, mask_pad=2, overlap=overlap, seed=0,
                      progress_bar=False)


def run_image_depth(depth, data, mask, n_iter):
    """
    Run an ImageDepth calculation repeatedly.

    Parameters
    ----------
    depth : `~photutils.utils.ImageDepth`
        The ImageDepth instance.

    data : 2D `~numpy.ndarray`
        The image.

    mask : 2D bool `~numpy.ndarray`
        The source mask.

    n_iter : int
        The number of calls.
    """
    with warnings.catch_warnings():
        # The non-overlapping mode may warn when fewer apertures than
        # requested can be placed; that is irrelevant for timing
        warnings.simplefilter('ignore', AstropyUserWarning)
        for _ in range(n_iter):
            depth(data, mask)


def bench_image_depth(*, size=1000, n_apertures_list=(100, 500, 2000),
                      n_iter=1, repeats=3):
    """
    Benchmark ImageDepth versus the number of apertures.

    Parameters
    ----------
    size : int, optional
        The image size; the image is ``(size, size)``.

    n_apertures_list : tuple of int, optional
        The numbers of apertures to place.

    n_iter : int, optional
        The number of calls per timing; the per-call time is reported.

    repeats : int, optional
        The number of repeats for each timing (best time is kept).
    """
    data, mask = make_depth_inputs(size)
    run_image_depth(make_depth(50), data, mask, 1)  # warm up

    print(f'\n== ImageDepth ({size}x{size} image, radius '
          f'{APER_RADIUS}, 2 iterations, per-call time) ==')
    print(f'{"n_apertures":>12}{"overlap":>16}{"no overlap":>16}')
    for n_apertures in n_apertures_list:
        times = []
        for overlap in (True, False):
            depth = make_depth(n_apertures, overlap=overlap)
            bench = partial(run_image_depth, depth, data, mask, n_iter)
            times.append(time_best(bench, repeats=repeats) / n_iter)
        print(f'{n_apertures:>12}'
              f'{f"{times[0] * 1e3:.2f}ms":>16}'
              f'{f"{times[1] * 1e3:.2f}ms":>16}')


def run_image_depth_concurrent(depth, data, mask, n_calls, n_threads):
    """
    Run concurrent ImageDepth calls on a shared instance.

    Parameters
    ----------
    depth : `~photutils.utils.ImageDepth`
        The shared ImageDepth instance.

    data : 2D `~numpy.ndarray`
        The image.

    mask : 2D bool `~numpy.ndarray`
        The source mask.

    n_calls : int
        The total number of calls.

    n_threads : int
        The number of worker threads.
    """
    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        futures = [executor.submit(depth, data, mask)
                   for _ in range(n_calls)]
        for future in futures:
            future.result()


def bench_depth_threads(*, size=500, n_apertures=500, n_calls=8,
                        thread_counts=(1, 2, 4, 8), repeats=3):
    """
    Benchmark concurrent ImageDepth calls on a shared instance.

    Parameters
    ----------
    size : int, optional
        The image size; the image is ``(size, size)``.

    n_apertures : int, optional
        The number of circular apertures.

    n_calls : int, optional
        The total number of ImageDepth calls per timing.

    thread_counts : tuple of int, optional
        The numbers of worker threads.

    repeats : int, optional
        The number of repeats for each timing (best time is kept).
    """
    data, mask = make_depth_inputs(size)
    depth = make_depth(n_apertures)
    depth(data, mask)  # warm up

    times = []
    for n_threads in thread_counts:
        bench = partial(run_image_depth_concurrent, depth, data, mask,
                        n_calls, n_threads)
        times.append(time_best(bench, repeats=repeats))

    print(f'\n== ImageDepth thread scaling ({size}x{size} image, '
          f'{n_apertures} apertures, {n_calls} concurrent calls on a '
          'shared instance) ==')
    header = ''.join(f'{f"{n} thr":>18}' for n in thread_counts)
    print(header)
    print(''.join(f'{cell:>18}' for cell in format_sweep_cells(times)))


def bench_idw(*, n_coords_list=(1_000, 10_000, 100_000),
              n_positions_list=(100, 1_000, 10_000),
              n_neighbors_list=(1, 8, 64), repeats=3):
    """
    Benchmark ShepardIDWInterpolator construction and evaluation.

    Parameters
    ----------
    n_coords_list : tuple of int, optional
        The numbers of known data points for the construction
        benchmark.

    n_positions_list : tuple of int, optional
        The numbers of query positions for the evaluation benchmark.

    n_neighbors_list : tuple of int, optional
        The numbers of neighbors for the evaluation benchmark.

    repeats : int, optional
        The number of repeats for each timing (best time is kept).
    """
    rng = np.random.default_rng(0)

    print('\n== ShepardIDWInterpolator construction (2D coords) ==')
    print(f'{"n_coords":>12}{"time":>12}')
    for n_coords in n_coords_list:
        coords = rng.random((n_coords, 2))
        values = np.sin(coords[:, 0] + coords[:, 1])
        bench = partial(ShepardIDWInterpolator, coords, values)
        t_call = time_best(bench, repeats=repeats)
        print(f'{n_coords:>12}{f"{t_call * 1e3:.2f}ms":>12}')

    n_coords = 10_000
    coords = rng.random((n_coords, 2))
    values = np.sin(coords[:, 0] + coords[:, 1])
    interp = ShepardIDWInterpolator(coords, values)

    print(f'\n== ShepardIDWInterpolator evaluation ({n_coords} coords, '
          '8 neighbors) ==')
    print(f'{"n_positions":>12}{"time":>12}{"per-pos":>12}')
    for n_positions in n_positions_list:
        positions = rng.random((n_positions, 2))
        bench = partial(interp, positions)
        t_call = time_best(bench, repeats=repeats)
        print(f'{n_positions:>12}{f"{t_call * 1e3:.2f}ms":>12}'
              f'{f"{t_call / n_positions * 1e6:.1f}us":>12}')

    n_positions = 1000
    positions = rng.random((n_positions, 2))
    print(f'\n== ShepardIDWInterpolator evaluation ({n_coords} coords, '
          f'{n_positions} positions) ==')
    print(f'{"n_neighbors":>12}{"time":>12}')
    for n_neighbors in n_neighbors_list:
        bench = partial(interp, positions, n_neighbors=n_neighbors)
        t_call = time_best(bench, repeats=repeats)
        print(f'{n_neighbors:>12}{f"{t_call * 1e3:.2f}ms":>12}')


def run_cutout_image(data, positions, shape, n_iter):
    """
    Create single cutouts repeatedly with CutoutImage.

    Parameters
    ----------
    data : 2D `~numpy.ndarray`
        The image.

    positions : 2D `~numpy.ndarray`
        The (y, x) cutout positions.

    shape : 2-tuple of int
        The cutout shape.

    n_iter : int
        The number of passes over the positions.
    """
    for _ in range(n_iter):
        for position in positions:
            CutoutImage(data, position, shape, mode='partial')


def bench_cutouts(*, size=1000, n_sources_list=(100, 1_000, 10_000),
                  cutout_shape=(11, 11), repeats=3):
    """
    Benchmark cutout generation.

    Parameters
    ----------
    size : int, optional
        The image size; the image is ``(size, size)``.

    n_sources_list : tuple of int, optional
        The numbers of cutout positions.

    cutout_shape : 2-tuple of int, optional
        The cutout shape.

    repeats : int, optional
        The number of repeats for each timing (best time is kept).
    """
    data = make_image((size, size))
    rng = np.random.default_rng(0)

    ny, nx = cutout_shape
    print(f'\n== _make_cutouts ({size}x{size} image, {ny}x{nx} '
          'cutouts) ==')
    print(f'{"n_sources":>12}{"time":>12}{"per-source":>14}')
    for n_sources in n_sources_list:
        xpos = rng.uniform(0, size - 1, n_sources)
        ypos = rng.uniform(0, size - 1, n_sources)
        bench = partial(_make_cutouts, data, xpos, ypos, cutout_shape)
        t_call = time_best(bench, repeats=repeats)
        print(f'{n_sources:>12}{f"{t_call * 1e3:.2f}ms":>12}'
              f'{f"{t_call / n_sources * 1e6:.1f}us":>14}')

    n_sources = 1000
    positions = np.column_stack(
        [rng.uniform(0, size - 1, n_sources),
         rng.uniform(0, size - 1, n_sources)])
    n_iter = 3
    bench = partial(run_cutout_image, data, positions, (25, 25), n_iter)
    t_call = time_best(bench, repeats=repeats) / (n_iter * n_sources)
    print('\n== CutoutImage (25x25 cutouts, per-cutout time) ==')
    print(f'{f"{t_call * 1e6:.1f}us":>12}')


def bench_total_error(*, sizes=(500, 1000, 2000), n_iter=3, repeats=3):
    """
    Benchmark calc_total_error versus image size.

    Parameters
    ----------
    sizes : tuple of int, optional
        The image sizes.

    n_iter : int, optional
        The number of calls per timing; the per-call time is reported.

    repeats : int, optional
        The number of repeats for each timing (best time is kept).
    """
    print('\n== calc_total_error (per-call time) ==')
    print(f'{"size":>12}{"scalar gain":>14}{"2D gain":>14}')
    for size in sizes:
        data = make_image((size, size))
        bkg_error = np.sqrt(np.abs(data))
        gain_image = np.full(data.shape, 2.0)
        times = []
        for gain in (2.0, gain_image):
            def run(data=data, bkg_error=bkg_error, gain=gain):
                for _ in range(n_iter):
                    calc_total_error(data, bkg_error, gain)

            times.append(time_best(run, repeats=repeats) / n_iter)
        print(f'{size:>12}'
              f'{f"{times[0] * 1e3:.2f}ms":>14}'
              f'{f"{times[1] * 1e3:.2f}ms":>14}')


def bench_nan_stats(*, size=2000, n_iter=3, repeats=3):
    """
    Benchmark the NaN-ignoring statistics functions.

    Float64 arrays dispatch to bottleneck (if installed); other
    dtypes use NumPy.

    Parameters
    ----------
    size : int, optional
        The array size; the array is ``(size, size)``.

    n_iter : int, optional
        The number of calls per timing; the per-call time is reported.

    repeats : int, optional
        The number of repeats for each timing (best time is kept).
    """
    data = make_image((size, size))
    rng = np.random.default_rng(0)
    nan_mask = rng.random(data.shape) < 0.01
    data[nan_mask] = np.nan
    data32 = data.astype(np.float32)

    funcs = [('nansum', nansum), ('nanmin', nanmin), ('nanmax', nanmax),
             ('nanmean', nanmean), ('nanmedian', nanmedian),
             ('nanstd', nanstd), ('nanvar', nanvar)]

    print(f'\n== NaN-ignoring statistics ({size}x{size} array, 1% NaN, '
          'per-call time) ==')
    print(f'{"function":>12}{"float64":>12}{"float32":>12}')
    for name, func in funcs:
        times = []
        for arr in (data, data32):
            def run(func=func, arr=arr):
                for _ in range(n_iter):
                    func(arr)

            times.append(time_best(run, repeats=repeats) / n_iter)
        print(f'{name:>12}'
              f'{f"{times[0] * 1e3:.2f}ms":>12}'
              f'{f"{times[1] * 1e3:.2f}ms":>12}')


def bench_random_coords(*, size=1000, n_coords_list=(1_000, 10_000),
                        min_separations=(0.0, 5.0), repeats=3):
    """
    Benchmark make_random_xycoords with and without a minimum
    separation.

    Parameters
    ----------
    size : int, optional
        The coordinate range; coordinates span ``(0, size)`` in x
        and y.

    n_coords_list : tuple of int, optional
        The numbers of coordinates to generate.

    min_separations : tuple of float, optional
        The minimum separations to apply.

    repeats : int, optional
        The number of repeats for each timing (best time is kept).
    """
    print('\n== make_random_xycoords (per-call time) ==')
    header = f'{"n_coords":>12}'
    header += ''.join(f'{f"min_sep={ms:g}":>16}' for ms in min_separations)
    print(header)
    for n_coords in n_coords_list:
        cells = [f'{n_coords:>12}']
        for min_separation in min_separations:
            bench = partial(make_random_xycoords, n_coords, (0, size),
                            (0, size), min_separation=min_separation,
                            seed=0)
            t_call = time_best(bench, repeats=repeats)
            cells.append(f'{f"{t_call * 1e3:.2f}ms":>16}')
        print(''.join(cells))


def bench_wcs_helpers(*, n_iter=100, repeats=3):
    """
    Benchmark the per-call cost of the local WCS helper functions.

    Parameters
    ----------
    n_iter : int, optional
        The number of calls per timing; the per-call time is reported.

    repeats : int, optional
        The number of repeats for each timing (best time is kept).
    """
    wcs = make_wcs((1000, 1000))
    skycoord = wcs.pixel_to_world(500.0, 500.0)

    funcs = [('local Jacobian', compute_local_wcs_jacobian),
             ('scale/angle', wcs_pixel_scale_angle)]

    print('\n== WCS helpers (TAN WCS, per-call time) ==')
    print(f'{"function":>16}{"time":>12}')
    for name, func in funcs:
        def run(func=func):
            for _ in range(n_iter):
                func(skycoord, wcs)

        t_call = time_best(run, repeats=repeats) / n_iter
        print(f'{name:>16}{f"{t_call * 1e6:.1f}us":>12}')


def main():
    """
    Run the photutils.utils benchmarks.
    """
    parser = argparse.ArgumentParser(
        description='Benchmarks for the photutils.utils subpackage.')
    parser.add_argument('--size', type=int, default=1000,
                        help='image size for the ImageDepth and cutout '
                             'benchmarks (default: %(default)s)')
    parser.add_argument('--threads', type=parse_thread_counts,
                        default=[1, 2, 4, 8],
                        help='comma-separated thread counts for the '
                             'ImageDepth thread-scaling benchmark '
                             '(default: 1,2,4,8)')
    parser.add_argument('--repeats', type=int, default=3,
                        help='number of repeats per timing; the best '
                             'time is reported (default: %(default)s)')
    parser.add_argument('--which', default='all',
                        choices=['all', 'image-depth', 'depth-threads',
                                 'idw', 'cutouts', 'total-error',
                                 'nan-stats', 'random-coords', 'wcs'],
                        help='which benchmark to run '
                             '(default: %(default)s)')
    args = parser.parse_args()

    print_environment()

    if args.which in ('all', 'image-depth'):
        bench_image_depth(size=args.size, repeats=args.repeats)
    if args.which in ('all', 'depth-threads'):
        bench_depth_threads(thread_counts=args.threads,
                            repeats=args.repeats)
    if args.which in ('all', 'idw'):
        bench_idw(repeats=args.repeats)
    if args.which in ('all', 'cutouts'):
        bench_cutouts(size=args.size, repeats=args.repeats)
    if args.which in ('all', 'total-error'):
        bench_total_error(repeats=args.repeats)
    if args.which in ('all', 'nan-stats'):
        bench_nan_stats(repeats=args.repeats)
    if args.which in ('all', 'random-coords'):
        bench_random_coords(repeats=args.repeats)
    if args.which in ('all', 'wcs'):
        bench_wcs_helpers(repeats=args.repeats)


if __name__ == '__main__':
    main()
