#!/usr/bin/env python3
# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Benchmarks for ApertureStats in the photutils.aperture subpackage.

The benchmarks cover computing all ApertureStats properties for all
of the pixel-based aperture types, with and without sigma clipping,
the cold cost of each individual ApertureStats property (including
its lazy dependencies) for a circular aperture, and the thread
scaling of the ``n_threads`` keyword.

Run ``python benchmarks/bench_aperture_stats.py --help`` to see the
available options.
"""

import argparse
from functools import partial

from astropy.stats import SigmaClip
from bench_helpers import (format_sweep_cells, make_aperture_inputs,
                           make_apertures, parse_thread_counts,
                           print_environment, time_best)

from photutils.aperture import ApertureStats, CircularAperture


def _compute_all_properties(data, aperture, *, error=None, wcs=None,
                            sigma_clip=None, n_threads=1):
    """
    Construct an ApertureStats instance and compute every property.
    """
    apstats = ApertureStats(data, aperture, error=error, wcs=wcs,
                            sigma_clip=sigma_clip, n_threads=n_threads)
    for prop in apstats.properties:
        getattr(apstats, prop)


def _compute_median(data, aperture, *, error=None, wcs=None,
                    sigma_clip=None, n_threads=1):
    """
    Construct an ApertureStats instance and compute the median.
    """
    return ApertureStats(data, aperture, error=error, wcs=wcs,
                         sigma_clip=sigma_clip, n_threads=n_threads).median


def _compute_property(data, aperture, prop, *, error=None, wcs=None,
                      sigma_clip=None, n_threads=1):
    """
    Construct an ApertureStats instance and compute one property.
    """
    apstats = ApertureStats(data, aperture, error=error, wcs=wcs,
                            sigma_clip=sigma_clip, n_threads=n_threads)
    return getattr(apstats, prop)


def bench_aperture_types(size, n_sources, n_threads_list, *, repeats=3,
                         seed=0):
    """
    Benchmark ApertureStats for all pixel-based aperture types.

    For each aperture type, the time to compute the median and the
    time to compute all ApertureStats properties are measured, with
    and without sigma clipping. When more than one thread count is
    requested, additional tables sweep the all-properties timings
    over the thread counts, with speedups relative to the first
    thread count.

    Parameters
    ----------
    size : int
        The image size; the image is ``(size, size)``.

    n_sources : int
        The number of aperture positions.

    n_threads_list : list of int
        The thread counts to sweep.

    repeats : int, optional
        The number of repeats for each timing (best time is kept).

    seed : int, optional
        The random number generator seed.
    """
    print(f'\n== ApertureStats aperture types '
          f'({n_sources} sources, {size}x{size} image) ==')
    print(f'{"aperture":>22}{"median":>10}{"all props":>12}'
          f'{"all props+clip":>16}')
    (data, error, wcs,
     positions) = make_aperture_inputs(size, n_sources, seed=seed)
    sigma_clip = SigmaClip(sigma=3.0, maxiters=10)

    for name, aperture in make_apertures(positions):
        t_median = time_best(
            partial(_compute_median, data, aperture, error=error,
                    wcs=wcs), repeats=repeats)
        t_all = time_best(
            partial(_compute_all_properties, data, aperture, error=error,
                    wcs=wcs), repeats=repeats)
        t_all_clip = time_best(
            partial(_compute_all_properties, data, aperture, error=error,
                    wcs=wcs, sigma_clip=sigma_clip), repeats=repeats)
        print(f'{name:>22}{f"{t_median:.3f}s":>10}'
              f'{f"{t_all:.3f}s":>12}{f"{t_all_clip:.3f}s":>16}')

    if len(n_threads_list) == 1:
        return

    for label, sigclip in (('all properties', None),
                           ('all properties + sigma clip', sigma_clip)):
        print(f'\n== ApertureStats aperture types n_threads sweep: '
              f'{label} ({n_sources} sources, {size}x{size} image) ==')
        header = f'{"aperture":>22}'
        header += ''.join(f'{f"n={n}":>18}' for n in n_threads_list)
        print(header)

        for name, aperture in make_apertures(positions):
            times = [time_best(
                partial(_compute_all_properties, data, aperture,
                        error=error, wcs=wcs, sigma_clip=sigclip,
                        n_threads=n_threads),
                repeats=repeats) for n_threads in n_threads_list]
            row = f'{name:>22}'
            row += ''.join(f'{cell:>18}'
                           for cell in format_sweep_cells(times))
            print(row)


def bench_properties(size, n_sources, n_threads_list, *, repeats=3,
                     seed=0):
    """
    Benchmark each ApertureStats property for a circular aperture.

    Each property is timed on a fresh ApertureStats instance for
    each thread count, so a timing includes the cost of the
    property's uncomputed lazy dependencies (e.g., the pixel gather
    and sigma clipping). Two tables are printed (without and with
    sigma clipping), with the properties listed by decreasing
    single-threaded time. Speedups are relative to the first thread
    count.

    Parameters
    ----------
    size : int
        The image size; the image is ``(size, size)``.

    n_sources : int
        The number of aperture positions.

    n_threads_list : list of int
        The thread counts to sweep.

    repeats : int, optional
        The number of repeats for each timing (best time is kept).

    seed : int, optional
        The random number generator seed.
    """
    (data, error, wcs,
     positions) = make_aperture_inputs(size, n_sources, seed=seed)
    aperture = CircularAperture(positions, r=8.0)
    sigma_clip = SigmaClip(sigma=3.0, maxiters=10)
    prop_names = ApertureStats(data, aperture).properties

    for label, sigclip in (('no sigma clipping', None),
                           ('sigma clipping', sigma_clip)):
        print(f'\n== ApertureStats properties, {label} (cold, incl. '
              f'dependencies; CircularAperture r=8, {n_sources} '
              f'sources) ==')
        header = f'{"property":>24}'
        header += ''.join(f'{f"n={n}":>18}' for n in n_threads_list)
        print(header)

        results = []
        for prop in prop_names:
            times = [time_best(
                partial(_compute_property, data, aperture, prop,
                        error=error, wcs=wcs, sigma_clip=sigclip,
                        n_threads=n_threads),
                repeats=repeats) for n_threads in n_threads_list]
            results.append((prop, times))

        results.sort(key=lambda result: result[1][0], reverse=True)
        for prop, times in results:
            row = f'{prop:>24}'
            row += ''.join(f'{cell:>18}'
                           for cell in format_sweep_cells(times))
            print(row)


def bench_thread_scaling(size, n_sources, n_threads_list, *, repeats=3,
                         seed=0):
    """
    Benchmark ApertureStats thread scaling for all pixel-based
    aperture types.

    Two sweeps are measured for each aperture type: the aperture
    ``sum`` (with an error array), which runs entirely in the
    threaded ``sum_method`` gather, and the sigma-clipped ``median``,
    whose pixel gather and sigma clipping are threaded (the clipped
    order statistics reuse the clip kernel's sorted values).
    Properties dominated by the serial per-source reductions (e.g.,
    the image moments) benefit less. Speedups are relative to the
    first thread count.

    Parameters
    ----------
    size : int
        The image size; the image is ``(size, size)``.

    n_sources : int
        The number of aperture positions.

    n_threads_list : list of int
        The thread counts to sweep.

    repeats : int, optional
        The number of repeats for each timing (best time is kept).

    seed : int, optional
        The random number generator seed.
    """
    (data, error, wcs,
     positions) = make_aperture_inputs(size, n_sources, seed=seed)
    sigma_clip = SigmaClip(sigma=3.0, maxiters=10)
    sweeps = (('sum', 'sum', None),
              ('sigma-clipped median', 'median', sigma_clip))

    for label, prop, sigclip in sweeps:
        print(f'\n== ApertureStats n_threads sweep: {label} '
              f'({n_sources} sources, {size}x{size} image) ==')
        header = f'{"aperture":>22}'
        header += ''.join(f'{f"n={n}":>16}' for n in n_threads_list)
        print(header)

        for name, aperture in make_apertures(positions):
            times = [time_best(
                partial(_compute_property, data, aperture, prop,
                        error=error, wcs=wcs, sigma_clip=sigclip,
                        n_threads=n_threads),
                repeats=repeats) for n_threads in n_threads_list]
            row = f'{name:>22}'
            row += ''.join(f'{cell:>16}'
                           for cell in format_sweep_cells(times))
            print(row)


def main():
    """
    Run the aperture benchmarks.
    """
    parser = argparse.ArgumentParser(
        description='Benchmarks for ApertureStats.')
    parser.add_argument('--size', type=int, default=2048,
                        help='image size (default: %(default)s)')
    parser.add_argument('--n-sources', type=int, default=1000,
                        help='number of sources (default: %(default)s)')
    parser.add_argument('--repeats', type=int, default=3,
                        help='number of repeats per timing; the best '
                             'time is reported (default: %(default)s)')
    parser.add_argument('--seed', type=int, default=0,
                        help='random number generator seed '
                             '(default: %(default)s)')
    parser.add_argument('--which', default='all',
                        choices=['all', 'types', 'properties', 'threads'],
                        help='which benchmark to run '
                             '(default: %(default)s)')
    parser.add_argument('--n-threads', type=parse_thread_counts,
                        default='1,2,4,8',
                        help='comma-separated thread counts for the '
                             'threads benchmark (default: %(default)s)')
    args = parser.parse_args()

    print_environment()

    if args.which in ('all', 'types'):
        bench_aperture_types(args.size, args.n_sources, args.n_threads,
                             repeats=args.repeats, seed=args.seed)
    if args.which in ('all', 'properties'):
        bench_properties(args.size, args.n_sources, args.n_threads,
                         repeats=args.repeats, seed=args.seed)
    if args.which in ('all', 'threads'):
        bench_thread_scaling(args.size, args.n_sources, args.n_threads,
                             repeats=args.repeats, seed=args.seed)


if __name__ == '__main__':
    main()
