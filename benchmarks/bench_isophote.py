#!/usr/bin/env python3
# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Benchmarks for the photutils.isophote subpackage.

The benchmarks cover full isophote fitting with Ellipse.fit_image,
elliptical sample extraction with EllipseSample.extract for each
integration mode, model-image reconstruction with
build_ellipse_model, and the scaling of the private
build_ellipse_model_c Cython kernel with image size.

Run ``python benchmarks/bench_isophote.py --help`` to see the
available options.
"""

import argparse
import warnings
from functools import partial

import numpy as np
from bench_helpers import print_environment, time_best

from photutils.isophote import (Ellipse, EllipseGeometry, EllipseSample,
                                build_ellipse_model)
from photutils.isophote._ellipse_model import build_ellipse_model_c
from photutils.isophote.tests.make_test_data import make_test_image

FIT_MODES = ('bilinear', 'nearest_neighbor', 'mean', 'median')

EXTRACT_MODES = ('bilinear', 'nearest_neighbor', 'mean', 'median')

# Geometry of the reference isophote in the simulated galaxy image
X0 = Y0 = 256.0
SMA0 = 20.0
EPS = 0.2
PA = 0.0


def make_galaxy_image(*, seed=0):
    """
    Return the simulated galaxy image used by the benchmarks.

    Parameters
    ----------
    seed : int, optional
        The random number generator seed.

    Returns
    -------
    result : 2D `~numpy.ndarray`
        The simulated galaxy image.
    """
    return make_test_image(seed=seed)


def run_fit_image(data, *, integrmode='bilinear', maxsma=200.0):
    """
    Run a full isophote fit on the data.

    RuntimeWarnings from degenerate inner and outer isophotes (e.g.,
    zero gradients) are routine during fitting and are suppressed.

    Parameters
    ----------
    data : 2D `~numpy.ndarray`
        The galaxy image.

    integrmode : str, optional
        The integration mode.

    maxsma : float, optional
        The maximum semimajor axis length.

    Returns
    -------
    result : `~photutils.isophote.IsophoteList`
        The fitted isophotes.
    """
    geometry = EllipseGeometry(X0, Y0, SMA0, EPS, PA)
    ellipse = Ellipse(data, geometry=geometry)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        return ellipse.fit_image(integrmode=integrmode, maxsma=maxsma)


def bench_fit_image(*, maxsma=200.0, seed=0):
    """
    Benchmark Ellipse.fit_image for each integration mode.

    Each fit is timed once (a full fit takes seconds and its runtime
    is stable).

    Parameters
    ----------
    maxsma : float, optional
        The maximum semimajor axis length.

    seed : int, optional
        The random number generator seed.
    """
    data = make_galaxy_image(seed=seed)
    print(f'\n== Ellipse.fit_image ({data.shape[1]}x{data.shape[0]} '
          f'image, maxsma={maxsma:g}) ==')
    print(f'{"mode":>18}{"time":>12}{"isophotes":>12}')
    for mode in FIT_MODES:
        bench = partial(run_fit_image, data, integrmode=mode,
                        maxsma=maxsma)
        try:
            t_run = time_best(bench, repeats=1)
            isolist = bench()
        except ValueError as exc:
            print(f'{mode:>18}  failed ({exc})')
            continue
        print(f'{mode:>18}{f"{t_run:.3f}s":>12}{len(isolist):>12}')


def run_extract(data, sma, n_iter, *, integrmode='bilinear'):
    """
    Extract an elliptical sample repeatedly from fresh samples.

    A fresh EllipseSample is created for every extraction because
    extract caches its result on the instance.

    Parameters
    ----------
    data : 2D `~numpy.ndarray`
        The galaxy image.

    sma : float
        The semimajor axis length of the sample.

    n_iter : int
        The number of extractions.

    integrmode : str, optional
        The integration mode.
    """
    for _ in range(n_iter):
        sample = EllipseSample(data, sma, x0=X0, y0=Y0, eps=EPS,
                               position_angle=PA, integrmode=integrmode)
        sample.extract()


def bench_sample(*, sma_list=(20.0, 80.0), n_iter=20, repeats=3,
                 seed=0):
    """
    Benchmark EllipseSample.extract for each integration mode.

    Parameters
    ----------
    sma_list : tuple of float, optional
        The semimajor axis lengths to sample at.

    n_iter : int, optional
        The number of calls per timing; the per-call time is reported.

    repeats : int, optional
        The number of repeats for each timing (best time is kept).

    seed : int, optional
        The random number generator seed.
    """
    data = make_galaxy_image(seed=seed)
    print('\n== EllipseSample.extract (per-call time) ==')
    header = f'{"mode":>18}'
    for sma in sma_list:
        header += f'{f"sma={sma:g}":>12}'
    print(header)
    for mode in EXTRACT_MODES:
        cells = ''
        for sma in sma_list:
            bench = partial(run_extract, data, sma, n_iter,
                            integrmode=mode)
            t_call = time_best(bench, repeats=repeats) / n_iter
            cells += f'{f"{t_call * 1e3:.3f}ms":>12}'
        print(f'{mode:>18}{cells}')


def bench_model(*, maxsma=200.0, repeats=3, seed=0):
    """
    Benchmark build_ellipse_model, with and without high harmonics.

    The isophote fit that provides the input IsophoteList is run once
    and is not included in the timings.

    Parameters
    ----------
    maxsma : float, optional
        The maximum semimajor axis length of the isophote fit.

    repeats : int, optional
        The number of repeats for each timing (best time is kept).

    seed : int, optional
        The random number generator seed.
    """
    data = make_galaxy_image(seed=seed)
    isolist = run_fit_image(data, maxsma=maxsma)
    print(f'\n== build_ellipse_model ({data.shape[1]}x{data.shape[0]} '
          f'image, {len(isolist)} isophotes) ==')
    print(f'{"variant":>18}{"time":>12}')
    for name, high_harmonics in (('no harmonics', False),
                                 ('high harmonics', True)):
        bench = partial(build_ellipse_model, data.shape, isolist,
                        high_harmonics=high_harmonics)
        t_run = time_best(bench, repeats=repeats)
        print(f'{name:>18}{f"{t_run * 1e3:.1f}ms":>12}')


def run_kernel(size, *, harmonics=False):
    """
    Call the build_ellipse_model_c kernel for a size x size image.

    The isophote parameter arrays mimic the finely spaced (0.01 pixel)
    interpolated arrays that build_ellipse_model passes to the kernel,
    with the outermost isophote at 0.35 * size.

    Parameters
    ----------
    size : int
        The image size; the image is ``(size, size)``.

    harmonics : bool, optional
        Whether to include the harmonic arrays.
    """
    max_sma = 0.35 * size
    n_sma = int(max_sma / 0.01)
    sma = np.linspace(0.5, max_sma, n_sma)
    intens = np.full(n_sma, 100.0)
    eps = np.full(n_sma, EPS)
    pa = np.full(n_sma, 0.4)
    x0 = np.full(n_sma, size / 2.0)
    y0 = np.full(n_sma, size / 2.0)
    harmonic_arrays = []
    if harmonics:
        harmonic_arrays = [np.full(n_sma, 0.01) for _ in range(4)]
    build_ellipse_model_c(size, size, sma, intens, eps, pa, x0, y0,
                          *harmonic_arrays)


def bench_kernel(*, sizes=(256, 512), repeats=3):
    """
    Benchmark the build_ellipse_model_c kernel across image sizes.

    Parameters
    ----------
    sizes : tuple of int, optional
        The image sizes (pixels per side).

    repeats : int, optional
        The number of repeats for each timing (best time is kept).
    """
    print('\n== build_ellipse_model_c kernel vs image size ==')
    print(f'{"size":>12}{"no harmonics":>14}{"harmonics":>14}')
    for size in sizes:
        cells = ''
        for harmonics in (False, True):
            bench = partial(run_kernel, size, harmonics=harmonics)
            t_run = time_best(bench, repeats=repeats)
            cells += f'{f"{t_run * 1e3:.1f}ms":>14}'
        print(f'{f"{size}x{size}":>12}{cells}')


def parse_int_list(text):
    """
    Parse a comma-separated list of positive integers.

    Parameters
    ----------
    text : str
        The comma-separated integers (e.g., ``'256,512'``).

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


def parse_float_list(text):
    """
    Parse a comma-separated list of positive floats.

    Parameters
    ----------
    text : str
        The comma-separated floats (e.g., ``'20,80'``).

    Returns
    -------
    result : list of float
        The parsed floats.
    """
    values = [float(item) for item in text.split(',')]
    if any(value <= 0 for value in values):
        msg = 'values must be positive'
        raise ValueError(msg)
    return values


def main():
    """
    Run the photutils.isophote benchmarks.
    """
    parser = argparse.ArgumentParser(
        description='Benchmarks for the photutils.isophote subpackage.')
    parser.add_argument('--maxsma', type=float, default=200.0,
                        help='maximum semimajor axis length for the '
                             'isophote fits (default: %(default)s)')
    parser.add_argument('--sma-list', type=parse_float_list,
                        default=[20.0, 80.0],
                        help='comma-separated semimajor axis lengths '
                             'for the sample benchmark (default: '
                             '20,80)')
    parser.add_argument('--sizes', type=parse_int_list,
                        default=[256, 512],
                        help='comma-separated image sizes for the '
                             'kernel benchmark (default: 256,512)')
    parser.add_argument('--n-iter', type=int, default=20,
                        help='number of calls per timing for the '
                             'sample benchmark (default: %(default)s)')
    parser.add_argument('--repeats', type=int, default=3,
                        help='number of repeats per timing; the best '
                             'time is reported (default: %(default)s)')
    parser.add_argument('--seed', type=int, default=0,
                        help='random number generator seed '
                             '(default: %(default)s)')
    parser.add_argument('--which', default='all',
                        choices=['all', 'fit-image', 'sample', 'model',
                                 'kernel'],
                        help='which benchmark to run '
                             '(default: %(default)s)')
    args = parser.parse_args()

    print_environment()

    if args.which in ('all', 'fit-image'):
        bench_fit_image(maxsma=args.maxsma, seed=args.seed)
    if args.which in ('all', 'sample'):
        bench_sample(sma_list=args.sma_list, n_iter=args.n_iter,
                     repeats=args.repeats, seed=args.seed)
    if args.which in ('all', 'model'):
        bench_model(maxsma=args.maxsma, repeats=args.repeats,
                    seed=args.seed)
    if args.which in ('all', 'kernel'):
        bench_kernel(sizes=args.sizes, repeats=args.repeats)


if __name__ == '__main__':
    main()
