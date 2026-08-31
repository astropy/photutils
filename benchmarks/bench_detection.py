#!/usr/bin/env python3
# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Benchmarks for the photutils.detection subpackage.

The benchmarks cover the star finder classes (DAOStarFinder,
IRAFStarFinder, and StarFinder) for both source detection and
catalog-only (xycoords) runs, the find_peaks detection modes, the
fast circular (min_separation) peak detection versus the equivalent
circular-footprint maximum filter, and concurrent find_stars calls on
a shared finder instance across thread counts.

Run ``python benchmarks/bench_detection.py --help`` to see the
available options.
"""

import argparse
import warnings
from concurrent.futures import ThreadPoolExecutor
from functools import partial

import numpy as np
from astropy.modeling.models import Gaussian2D
from astropy.stats import gaussian_fwhm_to_sigma
from bench_helpers import (format_sweep_cells, parse_thread_counts,
                           print_environment, time_best)

from photutils.centroids import centroid_com
from photutils.detection import (DAOStarFinder, IRAFStarFinder, StarFinder,
                                 find_peaks)
from photutils.utils.exceptions import NoDetectionsWarning

FWHM = 4.0
THRESHOLD = 10.0


def make_star_image(n_sources, *, spacing=25, fwhm=FWHM, amplitude=100.0,
                    noise_std=1.0, seed=0):
    """
    Return an image with a grid of Gaussian stars and their positions.

    Parameters
    ----------
    n_sources : int
        The number of stars. The image is a square grid of
        ``spacing``-sized cells with one star per cell, so the image
        size is ``ceil(sqrt(n_sources)) * spacing`` per side.

    spacing : int, optional
        The grid cell size in pixels.

    fwhm : float, optional
        The FWHM of the Gaussian stars in pixels.

    amplitude : float, optional
        The amplitude of the Gaussian stars.

    noise_std : float, optional
        The standard deviation of the Gaussian noise.

    seed : int, optional
        The random number generator seed.

    Returns
    -------
    data : 2D `~numpy.ndarray`
        The image.

    xypos : Nx2 `~numpy.ndarray`
        The (x, y) star positions.
    """
    n_grid = int(np.ceil(np.sqrt(n_sources)))
    size = n_grid * spacing
    rng = np.random.default_rng(seed)
    data = rng.normal(0.0, noise_std, (size, size))

    sigma = fwhm * gaussian_fwhm_to_sigma
    yy, xx = np.mgrid[0:spacing, 0:spacing]
    xypos = np.empty((n_sources, 2))
    for i in range(n_sources):
        gy, gx = divmod(i, n_grid)
        xc = (spacing - 1) / 2.0 + rng.uniform(-0.5, 0.5)
        yc = (spacing - 1) / 2.0 + rng.uniform(-0.5, 0.5)
        model = Gaussian2D(amplitude, xc, yc, sigma, sigma)
        x0 = gx * spacing
        y0 = gy * spacing
        data[y0:y0 + spacing, x0:x0 + spacing] += model(xx, yy)
        xypos[i] = (x0 + xc, y0 + yc)

    return data, xypos


def make_gaussian_kernel(size, *, fwhm=FWHM):
    """
    Return a 2D Gaussian kernel array for StarFinder.

    Parameters
    ----------
    size : int
        The kernel size; the kernel is ``(size, size)``.

    fwhm : float, optional
        The FWHM of the Gaussian kernel in pixels.

    Returns
    -------
    result : 2D `~numpy.ndarray`
        The kernel array.
    """
    cen = (size - 1) / 2.0
    sigma = fwhm * gaussian_fwhm_to_sigma
    yy, xx = np.mgrid[0:size, 0:size]
    model = Gaussian2D(1.0, cen, cen, sigma, sigma)
    return model(xx, yy)


def make_finders(xypos=None):
    """
    Return the star finder name and instance pairs.

    Parameters
    ----------
    xypos : Nx2 `~numpy.ndarray` or `None`, optional
        If not `None`, the DAOStarFinder and IRAFStarFinder instances
        are configured with these source positions (skipping the
        source-finding step).

    Returns
    -------
    result : list of (str, finder) tuples
        The finder name and instance pairs.
    """
    kwargs = {}
    if xypos is not None:
        kwargs['xycoords'] = np.round(xypos).astype(int)
    finders = [
        ('DAOStarFinder', DAOStarFinder(THRESHOLD, FWHM, **kwargs)),
        ('IRAFStarFinder', IRAFStarFinder(THRESHOLD, FWHM, **kwargs)),
    ]
    if xypos is None:  # StarFinder does not support xycoords
        kernel = make_gaussian_kernel(11)
        finders.append(('StarFinder', StarFinder(THRESHOLD, kernel)))
    return finders


def bench_star_finders(*, n_sources=1000, repeats=3, seed=0):
    """
    Benchmark the star finder classes.

    Each finder is timed for a full detection run and, for the
    DAOFIND-style finders, a catalog-only run with the source
    positions given via ``xycoords`` (skipping the source-finding
    step).

    Parameters
    ----------
    n_sources : int, optional
        The number of stars in the image.

    repeats : int, optional
        The number of repeats for each timing (best time is kept).

    seed : int, optional
        The random number generator seed.
    """
    data, xypos = make_star_image(n_sources, seed=seed)
    print(f'\n== Star finders ({n_sources} stars, '
          f'{data.shape[0]}x{data.shape[1]} image) ==')
    print(f'{"finder":>16}{"detect":>12}{"catalog-only":>14}{"n_found":>9}')

    catalog_times = {name: 'n/a' for name, _ in make_finders()}
    for name, finder in make_finders(xypos=xypos):
        t_best = time_best(partial(finder, data), repeats=repeats)
        catalog_times[name] = f'{t_best:.4f}s'

    for name, finder in make_finders():
        tbl = finder(data)
        t_best = time_best(partial(finder, data), repeats=repeats)
        print(f'{name:>16}{f"{t_best:.4f}s":>12}'
              f'{catalog_times[name]:>14}{len(tbl):>9}')


def bench_find_peaks(*, n_sources=1000, repeats=3, seed=0):
    """
    Benchmark the find_peaks detection modes.

    Parameters
    ----------
    n_sources : int, optional
        The number of stars in the image.

    repeats : int, optional
        The number of repeats for each timing (best time is kept).

    seed : int, optional
        The random number generator seed.
    """
    data, _ = make_star_image(n_sources, seed=seed)
    print(f'\n== find_peaks ({n_sources} stars, '
          f'{data.shape[0]}x{data.shape[1]} image) ==')
    print(f'{"mode":>34}{"time":>12}{"n_peaks":>9}')

    modes = [
        ('box_size=3', {'box_size': 3}),
        ('box_size=11', {'box_size': 11}),
        ('footprint=ones((11, 11))', {'footprint': np.ones((11, 11))}),
        ('min_separation=10', {'min_separation': 10}),
        ('box_size=3 + centroid_com', {'box_size': 3,
                                       'centroid_func': centroid_com}),
    ]
    for name, kwargs in modes:
        tbl = find_peaks(data, THRESHOLD, **kwargs)
        bench = partial(find_peaks, data, THRESHOLD, **kwargs)
        t_best = time_best(bench, repeats=repeats)
        print(f'{name:>34}{f"{t_best:.4f}s":>12}{len(tbl):>9}')


def bench_min_separation(*, n_sources=1000, radii=(5, 10, 25, 50),
                         repeats=3, seed=0):
    """
    Benchmark the fast circular (min_separation) peak detection
    against the equivalent circular-footprint maximum filter.

    Parameters
    ----------
    n_sources : int, optional
        The number of stars in the image.

    radii : tuple of int, optional
        The circular-region radii in pixels.

    repeats : int, optional
        The number of repeats for each timing (best time is kept).

    seed : int, optional
        The random number generator seed.
    """
    data, _ = make_star_image(n_sources, seed=seed)
    print(f'\n== find_peaks min_separation vs circular footprint '
          f'({n_sources} stars, {data.shape[0]}x{data.shape[1]} '
          'image) ==')
    print(f'{"radius":>8}{"footprint":>12}{"min_separation":>16}'
          f'{"speedup":>9}')

    for radius in radii:
        idx = np.arange(-radius, radius + 1)
        xx, yy = np.meshgrid(idx, idx)
        footprint = np.array((xx**2 + yy**2) <= radius**2, dtype=int)

        with warnings.catch_warnings():
            warnings.simplefilter('ignore', NoDetectionsWarning)
            bench_ref = partial(find_peaks, data, THRESHOLD,
                                footprint=footprint)
            t_ref = time_best(bench_ref, repeats=repeats)
            bench_fast = partial(find_peaks, data, THRESHOLD,
                                 min_separation=radius)
            t_fast = time_best(bench_fast, repeats=repeats)

        print(f'{radius:>8}{f"{t_ref:.4f}s":>12}{f"{t_fast:.4f}s":>16}'
              f'{f"{t_ref / t_fast:.1f}x":>9}')


def run_finder_concurrent(finder, data, n_calls, n_threads):
    """
    Run concurrent find_stars calls on a shared finder instance.

    Parameters
    ----------
    finder : `~photutils.detection.StarFinderBase`
        The shared star finder instance.

    data : 2D `~numpy.ndarray`
        The image.

    n_calls : int
        The total number of calls.

    n_threads : int
        The number of worker threads.
    """
    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        futures = [executor.submit(finder, data) for _ in range(n_calls)]
        for future in futures:
            future.result()


def bench_finder_threads(*, n_sources=1000, n_calls=8,
                         thread_counts=(1, 2, 4, 8), repeats=3, seed=0):
    """
    Benchmark concurrent find_stars calls on a shared DAOStarFinder.

    Parameters
    ----------
    n_sources : int, optional
        The number of stars in the image.

    n_calls : int, optional
        The total number of find_stars calls per timing.

    thread_counts : tuple of int, optional
        The numbers of worker threads.

    repeats : int, optional
        The number of repeats for each timing (best time is kept).

    seed : int, optional
        The random number generator seed.
    """
    data, _ = make_star_image(n_sources, seed=seed)
    finder = DAOStarFinder(THRESHOLD, FWHM)
    finder(data)  # warm up

    times = []
    for n_threads in thread_counts:
        bench = partial(run_finder_concurrent, finder, data, n_calls,
                        n_threads)
        times.append(time_best(bench, repeats=repeats))

    print(f'\n== DAOStarFinder thread scaling ({n_sources} stars, '
          f'{data.shape[0]}x{data.shape[1]} image, {n_calls} concurrent '
          'calls on a shared instance) ==')
    header = ''.join(f'{f"{n} thr":>18}' for n in thread_counts)
    print(header)
    print(''.join(f'{cell:>18}' for cell in format_sweep_cells(times)))


def parse_int_list(text):
    """
    Parse a comma-separated list of positive integers.

    Parameters
    ----------
    text : str
        The comma-separated integers (e.g., ``'5,10,25'``).

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
    Run the photutils.detection benchmarks.
    """
    parser = argparse.ArgumentParser(
        description='Benchmarks for the photutils.detection subpackage.')
    parser.add_argument('--n-sources', type=int, default=1000,
                        help='number of stars in the image '
                             '(default: %(default)s)')
    parser.add_argument('--radii', type=parse_int_list,
                        default=[5, 10, 25, 50],
                        help='comma-separated radii for the '
                             'min-separation benchmark '
                             '(default: 5,10,25,50)')
    parser.add_argument('--n-calls', type=int, default=8,
                        help='number of concurrent find_stars calls for '
                             'the thread-scaling benchmark '
                             '(default: %(default)s)')
    parser.add_argument('--n-threads', type=parse_thread_counts,
                        default=[1, 2, 4, 8],
                        help='comma-separated thread counts for the '
                             'thread-scaling benchmark (default: 1,2,4,8)')
    parser.add_argument('--repeats', type=int, default=3,
                        help='number of repeats per timing; the best '
                             'time is reported (default: %(default)s)')
    parser.add_argument('--seed', type=int, default=0,
                        help='random number generator seed '
                             '(default: %(default)s)')
    parser.add_argument('--which', default='all',
                        choices=['all', 'finders', 'find-peaks',
                                 'min-separation', 'threads'],
                        help='which benchmark to run '
                             '(default: %(default)s)')
    args = parser.parse_args()

    print_environment()

    if args.which in ('all', 'finders'):
        bench_star_finders(n_sources=args.n_sources, repeats=args.repeats,
                           seed=args.seed)
    if args.which in ('all', 'find-peaks'):
        bench_find_peaks(n_sources=args.n_sources, repeats=args.repeats,
                         seed=args.seed)
    if args.which in ('all', 'min-separation'):
        bench_min_separation(n_sources=args.n_sources, radii=args.radii,
                             repeats=args.repeats, seed=args.seed)
    if args.which in ('all', 'threads'):
        bench_finder_threads(n_sources=args.n_sources, n_calls=args.n_calls,
                             thread_counts=args.n_threads,
                             repeats=args.repeats, seed=args.seed)


if __name__ == '__main__':
    main()
