#!/usr/bin/env python3
# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Benchmarks for the photutils.morphology subpackage.

The benchmarks cover the per-call cost of data_properties (catalog
construction alone and with the morphological properties computed,
with and without a mask, background, and WCS) and the scaling of gini
with array size, with and without a mask.

Run ``python benchmarks/bench_morphology.py --help`` to see the
available options.
"""

import argparse
from functools import partial

import numpy as np
from astropy.modeling.models import Gaussian2D
from astropy.stats import gaussian_fwhm_to_sigma
from bench_utils import print_environment, time_best

from photutils.datasets import make_wcs
from photutils.morphology import data_properties, gini

# The morphological properties accessed by the data_properties
# benchmark (computing them is lazy, so construction is timed
# separately from property access)
PROPERTY_NAMES = ['x_centroid', 'y_centroid', 'semimajor_axis',
                  'semiminor_axis', 'orientation', 'eccentricity',
                  'area']

BACKGROUND_PROPERTY_NAMES = ['background_mean', 'background_sum']


def make_source_cutout(size, *, amplitude=100.0, noise_std=1.0, seed=0):
    """
    Return a cutout image containing a single Gaussian source.

    The source FWHM scales with the cutout size so that the source
    fills a constant fraction of the cutout.

    Parameters
    ----------
    size : int
        The cutout size; the cutout is ``(size, size)``.

    amplitude : float, optional
        The amplitude of the Gaussian source.

    noise_std : float, optional
        The standard deviation of the Gaussian noise.

    seed : int, optional
        The random number generator seed.

    Returns
    -------
    result : 2D `~numpy.ndarray`
        The cutout image.
    """
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size]
    xc = yc = (size - 1) / 2.0
    sigma = (size / 8.0) * gaussian_fwhm_to_sigma
    model = Gaussian2D(amplitude, xc, yc, sigma, sigma)
    return model(xx, yy) + rng.normal(0.0, noise_std, (size, size))


def run_data_properties(data, n_iter, property_names, **kwargs):
    """
    Call data_properties repeatedly and access the given properties.

    Parameters
    ----------
    data : 2D `~numpy.ndarray`
        The cutout image.

    n_iter : int
        The number of calls.

    property_names : list of str
        The names of the properties to access on each returned
        catalog.

    **kwargs : dict, optional
        Keyword arguments passed to ``data_properties``.
    """
    for _ in range(n_iter):
        props = data_properties(data, **kwargs)
        for name in property_names:
            getattr(props, name)


def bench_data_properties(*, cutout_size=51, n_iter=10, repeats=3,
                          seed=0):
    """
    Benchmark the per-call cost of data_properties.

    Catalog construction alone is timed separately from construction
    plus computing the morphological properties (the properties are
    computed lazily), and the mask, background, and WCS variants are
    timed with their relevant properties.

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
    data = make_source_cutout(cutout_size, seed=seed)
    mask = data < 5.0  # isolate source pixels
    background = 1.0
    wcs = make_wcs(data.shape)

    variants = [
        ('construct only', {}, []),
        ('+ properties', {}, PROPERTY_NAMES),
        ('+ mask', {'mask': mask}, PROPERTY_NAMES),
        ('+ background', {'background': background},
         PROPERTY_NAMES + BACKGROUND_PROPERTY_NAMES),
        ('+ wcs (sky_centroid)', {'wcs': wcs}, ['sky_centroid']),
    ]

    print(f'\n== data_properties ({cutout_size}x{cutout_size} cutout, '
          f'per-call time) ==')
    print(f'{"variant":>22}{"time":>12}')
    for name, kwargs, property_names in variants:
        bench = partial(run_data_properties, data, n_iter,
                        property_names, **kwargs)
        t_call = time_best(bench, repeats=repeats) / n_iter
        print(f'{name:>22}{f"{t_call * 1e3:.3f}ms":>12}')


def run_gini(data, n_iter, **kwargs):
    """
    Call gini repeatedly on the data.

    Parameters
    ----------
    data : 2D `~numpy.ndarray`
        The cutout image.

    n_iter : int
        The number of calls.

    **kwargs : dict, optional
        Keyword arguments passed to ``gini``.
    """
    for _ in range(n_iter):
        gini(data, **kwargs)


def bench_gini(*, sizes=(64, 256, 1024), n_iter=10, repeats=3, seed=0):
    """
    Benchmark gini across array sizes, with and without a mask.

    Parameters
    ----------
    sizes : tuple of int, optional
        The array sizes (pixels per side).

    n_iter : int, optional
        The number of calls per timing; the per-call time is reported.

    repeats : int, optional
        The number of repeats for each timing (best time is kept).

    seed : int, optional
        The random number generator seed.
    """
    print('\n== gini vs array size (per-call time) ==')
    print(f'{"size":>12}{"no mask":>12}{"with mask":>12}')
    rng = np.random.default_rng(seed)
    for size in sizes:
        data = make_source_cutout(size, seed=seed)
        # A sparse mask keeps the number of unmasked values comparable
        # to the no-mask case, so the columns show the masking overhead
        mask = rng.random(data.shape) < 0.01
        cells = ''
        for kwargs in ({}, {'mask': mask}):
            bench = partial(run_gini, data, n_iter, **kwargs)
            t_call = time_best(bench, repeats=repeats) / n_iter
            cells += f'{f"{t_call * 1e3:.3f}ms":>12}'
        print(f'{f"{size}x{size}":>12}{cells}')


def parse_int_list(text):
    """
    Parse a comma-separated list of positive integers.

    Parameters
    ----------
    text : str
        The comma-separated integers (e.g., ``'64,256,1024'``).

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
    Run the photutils.morphology benchmarks.
    """
    parser = argparse.ArgumentParser(
        description='Benchmarks for the photutils.morphology '
                    'subpackage.')
    parser.add_argument('--cutout-size', type=int, default=51,
                        help='data_properties cutout size '
                             '(default: %(default)s)')
    parser.add_argument('--sizes', type=parse_int_list,
                        default=[64, 256, 1024],
                        help='comma-separated array sizes for the gini '
                             'benchmark (default: 64,256,1024)')
    parser.add_argument('--n-iter', type=int, default=10,
                        help='number of calls per timing '
                             '(default: %(default)s)')
    parser.add_argument('--repeats', type=int, default=3,
                        help='number of repeats per timing; the best '
                             'time is reported (default: %(default)s)')
    parser.add_argument('--seed', type=int, default=0,
                        help='random number generator seed '
                             '(default: %(default)s)')
    parser.add_argument('--which', default='all',
                        choices=['all', 'data-properties', 'gini'],
                        help='which benchmark to run '
                             '(default: %(default)s)')
    args = parser.parse_args()

    print_environment()

    if args.which in ('all', 'data-properties'):
        bench_data_properties(cutout_size=args.cutout_size,
                              n_iter=args.n_iter,
                              repeats=args.repeats, seed=args.seed)
    if args.which in ('all', 'gini'):
        bench_gini(sizes=args.sizes, n_iter=args.n_iter,
                   repeats=args.repeats, seed=args.seed)


if __name__ == '__main__':
    main()
