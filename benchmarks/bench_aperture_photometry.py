#!/usr/bin/env python3
# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Benchmarks for AperturePhotometry in the photutils.aperture
subpackage.

The benchmarks cover computing the flux (with an error array) for all
of the pixel-based aperture types and each aperture overlap method
(exact, center, and subpixel), and the thread scaling of the
``n_threads`` keyword.

Run ``python benchmarks/bench_aperture_photometry.py --help`` to see
the available options.
"""

import argparse
from functools import partial

from bench_utils import (format_sweep_cells, make_aperture_inputs,
                         make_apertures, parse_thread_counts,
                         print_environment, time_best)

from photutils.aperture import AperturePhotometry


def _compute_flux(data, aperture, *, error=None, method='exact',
                  n_threads=1):
    """
    Construct an AperturePhotometry instance and compute the flux.
    """
    return AperturePhotometry(data, aperture, error=error,
                              method=method, n_threads=n_threads).flux


def bench_photometry(size, n_sources, *, repeats=3, seed=0):
    """
    Benchmark AperturePhotometry for all pixel-based aperture types.

    For each aperture type, the time to compute the flux (with an
    error array) is measured for each aperture overlap method.

    Parameters
    ----------
    size : int
        The image size; the image is ``(size, size)``.

    n_sources : int
        The number of aperture positions.

    repeats : int, optional
        The number of repeats for each timing (best time is kept).

    seed : int, optional
        The random number generator seed.
    """
    print(f'\n== AperturePhotometry '
          f'({n_sources} sources, {size}x{size} image) ==')
    print(f'{"aperture":>22}{"exact":>10}{"center":>10}{"subpixel":>10}')
    data, error, _, positions = make_aperture_inputs(size, n_sources,
                                                     seed=seed)

    for name, aperture in make_apertures(positions):
        row = f'{name:>22}'
        for method in ('exact', 'center', 'subpixel'):
            t_flux = time_best(
                partial(_compute_flux, data, aperture, error=error,
                        method=method), repeats=repeats)
            row += f'{f"{t_flux:.3f}s":>10}'
        print(row)


def bench_thread_scaling(size, n_sources, n_threads_list, *, repeats=3,
                         seed=0):
    """
    Benchmark AperturePhotometry thread scaling for all pixel-based
    aperture types.

    For each aperture type, the time to compute the flux (with an
    error array and the 'exact' method) is measured for each thread
    count. Speedups are relative to the first thread count.

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
    print(f'\n== AperturePhotometry n_threads sweep '
          f'({n_sources} sources, {size}x{size} image, exact method) ==')
    header = f'{"aperture":>22}'
    header += ''.join(f'{f"n={n}":>16}' for n in n_threads_list)
    print(header)
    data, error, _, positions = make_aperture_inputs(size, n_sources,
                                                     seed=seed)

    for name, aperture in make_apertures(positions):
        times = [time_best(partial(_compute_flux, data, aperture,
                                   error=error, n_threads=n_threads),
                           repeats=repeats)
                 for n_threads in n_threads_list]
        row = f'{name:>22}'
        row += ''.join(f'{cell:>16}' for cell in format_sweep_cells(times))
        print(row)


def main():
    """
    Run the AperturePhotometry benchmarks.
    """
    parser = argparse.ArgumentParser(
        description='Benchmarks for AperturePhotometry.')
    parser.add_argument('--size', type=int, default=2048,
                        help='image size (default: %(default)s)')
    parser.add_argument('--n-sources', type=int, default=10000,
                        help='number of sources (default: %(default)s)')
    parser.add_argument('--repeats', type=int, default=3,
                        help='number of repeats per timing; the best '
                             'time is reported (default: %(default)s)')
    parser.add_argument('--seed', type=int, default=0,
                        help='random number generator seed '
                             '(default: %(default)s)')
    parser.add_argument('--which', default='all',
                        choices=['all', 'types', 'threads'],
                        help='which benchmark to run '
                             '(default: %(default)s)')
    parser.add_argument('--n-threads', type=parse_thread_counts,
                        default='1,2,4,8',
                        help='comma-separated thread counts for the '
                             'threads benchmark (default: %(default)s)')
    args = parser.parse_args()

    print_environment()

    if args.which in ('all', 'types'):
        bench_photometry(args.size, args.n_sources, repeats=args.repeats,
                         seed=args.seed)
    if args.which in ('all', 'threads'):
        bench_thread_scaling(args.size, args.n_sources, args.n_threads,
                             repeats=args.repeats, seed=args.seed)


if __name__ == '__main__':
    main()
