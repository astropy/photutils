#!/usr/bin/env python3
# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Benchmarks for the photutils.centroids subpackage.

The benchmarks cover the per-call cost of the single-source centroid
functions (centroid_com, centroid_quadratic, centroid_1dg, and
centroid_2dg) and centroid_sources at many positions.

Run ``python benchmarks/bench_centroids.py --help`` to see the
available options.
"""

import argparse
from functools import partial

import numpy as np
from astropy.modeling.models import Gaussian2D
from astropy.stats import gaussian_fwhm_to_sigma
from bench_helpers import print_environment, time_best

from photutils.centroids import (centroid_1dg, centroid_2dg, centroid_com,
                                 centroid_quadratic, centroid_sources)

CENTROID_FUNCS = [('centroid_com', centroid_com),
                  ('centroid_quadratic', centroid_quadratic),
                  ('centroid_1dg', centroid_1dg),
                  ('centroid_2dg', centroid_2dg)]

# Centroid functions that accept an error keyword
ERROR_FUNCS = (centroid_1dg, centroid_2dg)


def make_source_cutout(size, *, fwhm=4.0, amplitude=100.0, noise_std=1.0,
                       seed=0):
    """
    Return a cutout image containing a single Gaussian source.

    Parameters
    ----------
    size : int
        The cutout size; the cutout is ``(size, size)``.

    fwhm : float, optional
        The FWHM of the Gaussian source in pixels.

    amplitude : float, optional
        The amplitude of the Gaussian source.

    noise_std : float, optional
        The standard deviation of the Gaussian noise.

    seed : int, optional
        The random number generator seed.

    Returns
    -------
    data : 2D `~numpy.ndarray`
        The cutout image.

    error : 2D `~numpy.ndarray`
        The error array.
    """
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size]
    xc = yc = (size - 1) / 2.0
    sigma = fwhm * gaussian_fwhm_to_sigma
    model = Gaussian2D(amplitude, xc, yc, sigma, sigma)
    data = model(xx, yy) + rng.normal(0.0, noise_std, (size, size))
    error = np.full(data.shape, noise_std)
    return data, error


def make_sources_image(n_sources, *, spacing=25, fwhm=4.0, amplitude=100.0,
                       noise_std=1.0, seed=0):
    """
    Return an image with a grid of Gaussian sources and their
    positions.

    Parameters
    ----------
    n_sources : int
        The number of sources. The image is a square grid of
        ``spacing``-sized cells with one source per cell, so the image
        size is ``ceil(sqrt(n_sources)) * spacing`` per side.

    spacing : int, optional
        The grid cell size in pixels.

    fwhm : float, optional
        The FWHM of the Gaussian sources in pixels.

    amplitude : float, optional
        The amplitude of the Gaussian sources.

    noise_std : float, optional
        The standard deviation of the Gaussian noise.

    seed : int, optional
        The random number generator seed.

    Returns
    -------
    data : 2D `~numpy.ndarray`
        The image.

    xpos, ypos : 1D `~numpy.ndarray`
        The source positions.
    """
    n_grid = int(np.ceil(np.sqrt(n_sources)))
    size = n_grid * spacing
    rng = np.random.default_rng(seed)
    data = rng.normal(0.0, noise_std, (size, size))

    sigma = fwhm * gaussian_fwhm_to_sigma
    yy, xx = np.mgrid[0:spacing, 0:spacing]
    xpos = np.empty(n_sources)
    ypos = np.empty(n_sources)
    for i in range(n_sources):
        gy, gx = divmod(i, n_grid)
        xc = (spacing - 1) / 2.0 + rng.uniform(-0.5, 0.5)
        yc = (spacing - 1) / 2.0 + rng.uniform(-0.5, 0.5)
        model = Gaussian2D(amplitude, xc, yc, sigma, sigma)
        x0 = gx * spacing
        y0 = gy * spacing
        data[y0:y0 + spacing, x0:x0 + spacing] += model(xx, yy)
        xpos[i] = x0 + xc
        ypos[i] = y0 + yc

    return data, xpos, ypos


def bench_functions(*, cutout_size=21, n_iter=200, repeats=3, seed=0):
    """
    Benchmark the per-call cost of the single-source centroid
    functions.

    Each function is called on the same single-source cutout; the
    Gaussian fitting functions are also timed with an error array.

    Parameters
    ----------
    cutout_size : int, optional
        The cutout size; the cutout is ``(cutout_size, cutout_size)``.

    n_iter : int, optional
        The number of calls per timing; the per-call time is reported.

    repeats : int, optional
        The number of repeats for each timing (best time is kept).

    seed : int, optional
        The random number generator seed.
    """
    print(f'\n== Single-source centroid functions '
          f'({cutout_size}x{cutout_size} cutout, per-call time) ==')
    print(f'{"function":>22}{"no error":>12}{"with error":>13}')
    data, error = make_source_cutout(cutout_size, seed=seed)

    def run(func, n_iter, **kwargs):
        """
        Call the centroid function repeatedly on the cutout.

        Parameters
        ----------
        func : callable
            The centroid function.

        n_iter : int
            The number of calls.

        **kwargs : dict, optional
            Keyword arguments passed to ``func``.
        """
        for _ in range(n_iter):
            func(data, **kwargs)

    for name, func in CENTROID_FUNCS:
        t_call = time_best(partial(run, func, n_iter),
                           repeats=repeats) / n_iter
        cells = f'{t_call * 1e6:>10.1f}us'
        if func in ERROR_FUNCS:
            t_err = time_best(partial(run, func, n_iter, error=error),
                              repeats=repeats) / n_iter
            cells += f'{t_err * 1e6:>11.1f}us'
        else:
            cells += f'{"n/a":>13}'
        print(f'{name:>22}{cells}')


def bench_centroid_sources(*, n_sources=1000, box_size=11, repeats=3,
                           seed=0):
    """
    Benchmark centroid_sources at many positions for each centroid
    function.

    Parameters
    ----------
    n_sources : int, optional
        The number of sources.

    box_size : int, optional
        The cutout box size used for each source.

    repeats : int, optional
        The number of repeats for each timing (best time is kept).

    seed : int, optional
        The random number generator seed.
    """
    data, xpos, ypos = make_sources_image(n_sources, seed=seed)
    print(f'\n== centroid_sources ({n_sources} sources, '
          f'{data.shape[0]}x{data.shape[1]} image, box_size={box_size}) ==')
    print(f'{"function":>22}{"time":>12}')

    for name, func in CENTROID_FUNCS:
        bench = partial(centroid_sources, data, xpos, ypos,
                        box_size=box_size, centroid_func=func)
        t_best = time_best(bench, repeats=repeats)
        print(f'{name:>22}{f"{t_best:.4f}s":>12}')


def main():
    """
    Run the photutils.centroids benchmarks.
    """
    parser = argparse.ArgumentParser(
        description='Benchmarks for the photutils.centroids subpackage.')
    parser.add_argument('--cutout-size', type=int, default=21,
                        help='single-source cutout size '
                             '(default: %(default)s)')
    parser.add_argument('--n-sources', type=int, default=1000,
                        help='number of sources for centroid_sources '
                             '(default: %(default)s)')
    parser.add_argument('--box-size', type=int, default=11,
                        help='centroid_sources cutout box size '
                             '(default: %(default)s)')
    parser.add_argument('--repeats', type=int, default=3,
                        help='number of repeats per timing; the best '
                             'time is reported (default: %(default)s)')
    parser.add_argument('--seed', type=int, default=0,
                        help='random number generator seed '
                             '(default: %(default)s)')
    parser.add_argument('--which', default='all',
                        choices=['all', 'functions', 'sources'],
                        help='which benchmark to run '
                             '(default: %(default)s)')
    args = parser.parse_args()

    print_environment()

    if args.which in ('all', 'functions'):
        bench_functions(cutout_size=args.cutout_size, repeats=args.repeats,
                        seed=args.seed)
    if args.which in ('all', 'sources'):
        bench_centroid_sources(n_sources=args.n_sources,
                               box_size=args.box_size, repeats=args.repeats,
                               seed=args.seed)


if __name__ == '__main__':
    main()
