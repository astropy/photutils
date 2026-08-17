#!/usr/bin/env python3
# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Benchmarks for the photutils.datasets subpackage.

The benchmarks cover the scaling of ``make_model_image`` with the
number of sources, the model discretization methods, the scaling of
``make_model_params`` with the number of sources, the noise
functions, the WCS factories, and the example-image functions.

Run ``python benchmarks/bench_datasets.py --help`` to see the
available options.
"""

import argparse
from functools import partial

import numpy as np
from astropy.modeling.models import Gaussian2D
from bench_helpers import print_environment, time_best

from photutils.datasets import (apply_poisson_noise, make_4gaussians_image,
                                make_100gaussians_image, make_gwcs,
                                make_model_image, make_model_params,
                                make_noise_image, make_wcs)
from photutils.utils._optional_deps import HAS_GWCS

MODEL_SHAPE = (25, 25)


def make_source_params(shape, n_sources):
    """
    Return a table of random Gaussian2D source parameters.

    Parameters
    ----------
    shape : 2-tuple of int
        The shape of the target image.

    n_sources : int
        The number of sources to generate.

    Returns
    -------
    params : `~astropy.table.QTable`
        The table of model parameters.
    """
    return make_model_params(shape, n_sources, x_name='x_mean',
                             y_name='y_mean', min_separation=1,
                             border_size=10, amplitude=(50, 200),
                             x_stddev=(1, 3), y_stddev=(1, 3),
                             theta=(0, np.pi), seed=0)


def run_model_image(shape, params, n_iter, **kwargs):
    """
    Render a model image repeatedly.

    Parameters
    ----------
    shape : 2-tuple of int
        The shape of the output image.

    params : `~astropy.table.QTable`
        The table of model parameters.

    n_iter : int
        The number of calls.

    **kwargs : dict, optional
        Additional keyword arguments passed to ``make_model_image``
        (e.g., ``discretize_method``).
    """
    model = Gaussian2D()
    for _ in range(n_iter):
        make_model_image(shape, model, params, model_shape=MODEL_SHAPE,
                         x_name='x_mean', y_name='y_mean', **kwargs)


def bench_model_image(*, size=500, n_sources_list=(100, 400, 1600),
                      n_iter=3, repeats=3):
    """
    Benchmark make_model_image versus the number of sources.

    Parameters
    ----------
    size : int, optional
        The image size; the image is ``(size, size)``.

    n_sources_list : tuple of int, optional
        The numbers of sources to render.

    n_iter : int, optional
        The number of calls per timing; the per-call time is reported.

    repeats : int, optional
        The number of repeats for each timing (best time is kept).
    """
    shape = (size, size)
    print(f'\n== make_model_image ({size}x{size} image, '
          f'{MODEL_SHAPE[0]}x{MODEL_SHAPE[1]} model shape, '
          'per-call time) ==')
    print(f'{"n_sources":>12}{"time":>12}{"per-source":>14}')
    for n_sources in n_sources_list:
        params = make_source_params(shape, n_sources)
        bench = partial(run_model_image, shape, params, n_iter)
        t_call = time_best(bench, repeats=repeats) / n_iter
        t_src = t_call / len(params)
        print(f'{n_sources:>12}{f"{t_call * 1e3:.2f}ms":>12}'
              f'{f"{t_src * 1e6:.1f}us":>14}')


def bench_discretize(*, size=300, n_sources=200, n_iter=3, repeats=3):
    """
    Benchmark make_model_image for each discretization method.

    The 'integrate' method is omitted because it is extremely slow.

    Parameters
    ----------
    size : int, optional
        The image size; the image is ``(size, size)``.

    n_sources : int, optional
        The number of sources to render.

    n_iter : int, optional
        The number of calls per timing; the per-call time is reported.

    repeats : int, optional
        The number of repeats for each timing (best time is kept).
    """
    shape = (size, size)
    params = make_source_params(shape, n_sources)

    print(f'\n== discretization methods ({size}x{size} image, '
          f'{len(params)} sources, per-call time) ==')
    print(f'{"method":>14}{"time":>12}')
    for method in ('center', 'interp', 'oversample'):
        bench = partial(run_model_image, shape, params, n_iter,
                        discretize_method=method)
        t_call = time_best(bench, repeats=repeats) / n_iter
        print(f'{method:>14}{f"{t_call * 1e3:.2f}ms":>12}')


def run_model_params(shape, n_sources, min_separation, n_iter):
    """
    Generate a table of source parameters repeatedly.

    Parameters
    ----------
    shape : 2-tuple of int
        The shape of the target image.

    n_sources : int
        The number of sources to generate.

    min_separation : float
        The minimum separation between source centers.

    n_iter : int
        The number of calls.
    """
    for _ in range(n_iter):
        make_model_params(shape, n_sources, min_separation=min_separation,
                          flux=(100, 500), seed=0)


def bench_model_params(*, size=500, n_sources_list=(100, 400, 1600),
                       n_iter=10, repeats=3):
    """
    Benchmark make_model_params versus the number of sources.

    The timing includes the minimum-separation (KDTree) filtering of
    the random coordinates.

    Parameters
    ----------
    size : int, optional
        The image size; the image is ``(size, size)``.

    n_sources_list : tuple of int, optional
        The numbers of sources to generate.

    n_iter : int, optional
        The number of calls per timing; the per-call time is reported.

    repeats : int, optional
        The number of repeats for each timing (best time is kept).
    """
    shape = (size, size)
    print(f'\n== make_model_params ({size}x{size} image, per-call '
          'time) ==')
    print(f'{"n_sources":>12}{"min_sep=1":>14}{"min_sep=5":>14}')
    for n_sources in n_sources_list:
        cells = ''
        for min_separation in (1, 5):
            bench = partial(run_model_params, shape, n_sources,
                            min_separation, n_iter)
            t_call = time_best(bench, repeats=repeats) / n_iter
            cells += f'{f"{t_call * 1e3:.2f}ms":>14}'
        print(f'{n_sources:>12}{cells}')


def run_noise(func, n_iter, **kwargs):
    """
    Call a noise function repeatedly.

    Parameters
    ----------
    func : callable
        The zero-argument noise function variant.

    n_iter : int
        The number of calls.

    **kwargs : dict, optional
        Keyword arguments passed to ``func``.
    """
    for _ in range(n_iter):
        func(**kwargs)


def bench_noise(*, size=1000, n_iter=10, repeats=3):
    """
    Benchmark the noise functions.

    Parameters
    ----------
    size : int, optional
        The image size; the image is ``(size, size)``.

    n_iter : int, optional
        The number of calls per timing; the per-call time is reported.

    repeats : int, optional
        The number of repeats for each timing (best time is kept).
    """
    shape = (size, size)
    data = np.full(shape, 100.0)
    variants = (
        ('make_noise_image (gaussian)',
         partial(make_noise_image, shape, distribution='gaussian',
                 mean=0.0, stddev=2.0, seed=0)),
        ('make_noise_image (poisson)',
         partial(make_noise_image, shape, distribution='poisson',
                 mean=5.0, seed=0)),
        ('apply_poisson_noise',
         partial(apply_poisson_noise, data, seed=0)),
    )

    print(f'\n== noise functions ({size}x{size} image, per-call '
          'time) ==')
    print(f'{"variant":>30}{"time":>12}')
    for label, func in variants:
        bench = partial(run_noise, func, n_iter)
        t_call = time_best(bench, repeats=repeats) / n_iter
        print(f'{label:>30}{f"{t_call * 1e3:.2f}ms":>12}')


def bench_wcs(*, size=1000, n_iter=100, repeats=3):
    """
    Benchmark the WCS factory functions.

    The gWCS benchmark is skipped if the optional gwcs package is not
    installed.

    Parameters
    ----------
    size : int, optional
        The image size; the image is ``(size, size)``.

    n_iter : int, optional
        The number of calls per timing; the per-call time is reported.

    repeats : int, optional
        The number of repeats for each timing (best time is kept).
    """
    shape = (size, size)
    variants = [('make_wcs', partial(make_wcs, shape))]
    if HAS_GWCS:
        variants.append(('make_gwcs', partial(make_gwcs, shape)))

    print('\n== WCS factories (per-call time) ==')
    print(f'{"function":>12}{"time":>12}')
    for label, func in variants:
        bench = partial(run_noise, func, n_iter)
        t_call = time_best(bench, repeats=repeats) / n_iter
        print(f'{label:>12}{f"{t_call * 1e3:.3f}ms":>12}')
    if not HAS_GWCS:
        print('make_gwcs skipped (gwcs is not installed)')


def bench_examples(*, n_iter=3, repeats=3):
    """
    Benchmark the example-image functions.

    Parameters
    ----------
    n_iter : int, optional
        The number of calls per timing; the per-call time is reported.

    repeats : int, optional
        The number of repeats for each timing (best time is kept).
    """
    variants = (
        ('make_4gaussians_image', make_4gaussians_image),
        ('make_100gaussians_image', make_100gaussians_image),
    )

    print('\n== example images (per-call time) ==')
    print(f'{"function":>26}{"time":>12}')
    for label, func in variants:
        bench = partial(run_noise, func, n_iter)
        t_call = time_best(bench, repeats=repeats) / n_iter
        print(f'{label:>26}{f"{t_call * 1e3:.2f}ms":>12}')


def parse_int_list(text):
    """
    Parse a comma-separated list of positive integers.

    Parameters
    ----------
    text : str
        The comma-separated integers (e.g., ``'100,400,1600'``).

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
    Run the photutils.datasets benchmarks.
    """
    parser = argparse.ArgumentParser(
        description='Benchmarks for the photutils.datasets subpackage.')
    parser.add_argument('--size', type=int, default=500,
                        help='image size for the model-image and '
                             'model-params benchmarks '
                             '(default: %(default)s)')
    parser.add_argument('--n-sources-list', type=parse_int_list,
                        default=[100, 400, 1600],
                        help='comma-separated numbers of sources for '
                             'the scaling benchmarks '
                             '(default: 100,400,1600)')
    parser.add_argument('--n-iter', type=int, default=3,
                        help='number of calls per timing '
                             '(default: %(default)s)')
    parser.add_argument('--repeats', type=int, default=3,
                        help='number of repeats per timing; the best '
                             'time is reported (default: %(default)s)')
    parser.add_argument('--which', default='all',
                        choices=['all', 'model-image', 'discretize',
                                 'model-params', 'noise', 'wcs',
                                 'examples'],
                        help='which benchmark to run '
                             '(default: %(default)s)')
    args = parser.parse_args()

    print_environment()

    if args.which in ('all', 'model-image'):
        bench_model_image(size=args.size,
                          n_sources_list=args.n_sources_list,
                          n_iter=args.n_iter, repeats=args.repeats)
    if args.which in ('all', 'discretize'):
        bench_discretize(n_iter=args.n_iter, repeats=args.repeats)
    if args.which in ('all', 'model-params'):
        bench_model_params(size=args.size,
                           n_sources_list=args.n_sources_list,
                           n_iter=args.n_iter * 3,
                           repeats=args.repeats)
    if args.which in ('all', 'noise'):
        bench_noise(n_iter=args.n_iter * 3, repeats=args.repeats)
    if args.which in ('all', 'wcs'):
        bench_wcs(n_iter=args.n_iter * 30, repeats=args.repeats)
    if args.which in ('all', 'examples'):
        bench_examples(n_iter=args.n_iter, repeats=args.repeats)


if __name__ == '__main__':
    main()
