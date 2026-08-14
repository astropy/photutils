#!/usr/bin/env python3
# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Benchmarks for the photutils.background subpackage.

The benchmarks cover Background2D construction across image sizes,
box sizes, and n_threads values; full-size background and background
RMS map generation; the scalar background and background RMS
estimator classes; and LocalBackground.

Run ``python benchmarks/bench_background.py --help`` to see the
available options.
"""

import argparse
import os
import platform
import sys
import time
from functools import partial

import numpy as np
from astropy.stats import SigmaClip

from photutils.background import (Background2D, BiweightLocationBackground,
                                  BiweightScaleBackgroundRMS, LocalBackground,
                                  MADStdBackgroundRMS, MeanBackground,
                                  MedianBackground, MMMBackground,
                                  ModeEstimatorBackground,
                                  SExtractorBackground, StdBackgroundRMS)

ESTIMATOR_CLASSES = [MeanBackground, MedianBackground,
                     ModeEstimatorBackground, MMMBackground,
                     SExtractorBackground, BiweightLocationBackground,
                     StdBackgroundRMS, MADStdBackgroundRMS,
                     BiweightScaleBackgroundRMS]


def time_best(func, *, repeats=3):
    """
    Return the best wall-clock time of ``repeats`` calls to ``func``.

    Parameters
    ----------
    func : callable
        The zero-argument callable to time.

    repeats : int, optional
        The number of times to call ``func``.

    Returns
    -------
    result : float
        The best (minimum) wall-clock time in seconds.
    """
    best = np.inf
    for _ in range(repeats):
        t0 = time.perf_counter()
        func()
        best = min(best, time.perf_counter() - t0)
    return best


def make_image(shape, *, seed=0):
    """
    Return a Gaussian-noise image of the given shape.

    Parameters
    ----------
    shape : tuple of int
        The shape of the output image.

    seed : int, optional
        The random number generator seed.

    Returns
    -------
    result : 2D `~numpy.ndarray`
        The noise image.
    """
    rng = np.random.default_rng(seed)
    return rng.normal(100.0, 5.0, shape)


def print_environment():
    """
    Print information about the runtime environment.
    """
    import astropy
    import scipy

    import photutils
    from photutils.utils._optional_deps import HAS_BOTTLENECK

    gil = getattr(sys, '_is_gil_enabled', lambda: True)()
    print(f'python {platform.python_version()} (GIL enabled: {gil}), '
          f'{os.cpu_count()} CPUs')
    print(f'photutils {photutils.__version__}, numpy {np.__version__}, '
          f'scipy {scipy.__version__}, astropy {astropy.__version__}, '
          f'bottleneck: {HAS_BOTTLENECK}')


def bench_background2d(sizes, box_sizes, n_threads_list, *, repeats=3,
                       seed=0):
    """
    Benchmark Background2D construction.

    The construction time is measured for each combination of image
    size, box size, and number of threads. Speedups are reported
    relative to the first value in ``n_threads_list``.

    Parameters
    ----------
    sizes : list of int
        The image sizes; each image is ``(size, size)``.

    box_sizes : list of int
        The box sizes; each box is ``(box_size, box_size)``.

    n_threads_list : list of int
        The ``n_threads`` values to benchmark.

    repeats : int, optional
        The number of repeats for each timing (best time is kept).

    seed : int, optional
        The random number generator seed.
    """
    print('\n== Background2D construction ==')
    header = f'{"image":>12} {"box":>6}'
    for n_threads in n_threads_list:
        header += f'{f"n_threads={n_threads}":>19}'
    print(header)

    for size in sizes:
        data = make_image((size, size), seed=seed)
        for box_size in box_sizes:
            row = f'{f"{size}x{size}":>12} {box_size:>6}'
            t_ref = None
            for n_threads in n_threads_list:
                func = partial(Background2D, data, box_size,
                               n_threads=n_threads)
                t_best = time_best(func, repeats=repeats)
                if t_ref is None:
                    t_ref = t_best
                    cell = f'{t_best:.3f}s'
                else:
                    cell = f'{t_best:.3f}s ({t_ref / t_best:.2f}x)'
                row += f'{cell:>19}'
            print(row)


def bench_maps(sizes, *, box_size=64, repeats=3, seed=0):
    """
    Benchmark full-size background and background RMS map generation.

    The ``background`` and ``background_rms`` properties are
    recalculated on each access, so their cost is paid every time
    they are used.

    Parameters
    ----------
    sizes : list of int
        The image sizes; each image is ``(size, size)``.

    box_size : int, optional
        The box size used for the Background2D instance.

    repeats : int, optional
        The number of repeats for each timing (best time is kept).

    seed : int, optional
        The random number generator seed.
    """
    print(f'\n== Full-size map generation (box={box_size}) ==')
    print(f'{"image":>12}{"background":>14}{"background_rms":>17}')
    for size in sizes:
        data = make_image((size, size), seed=seed)
        bkg = Background2D(data, box_size)
        t_bkg = time_best(partial(getattr, bkg, 'background'),
                          repeats=repeats)
        t_rms = time_best(partial(getattr, bkg, 'background_rms'),
                          repeats=repeats)
        print(f'{f"{size}x{size}":>12}{f"{t_bkg:.3f}s":>14}'
              f'{f"{t_rms:.3f}s":>17}')


def bench_estimators(*, size=2048, repeats=3, seed=0):
    """
    Benchmark the scalar background and background RMS estimators.

    Each estimator is timed on the full image, with and without
    sigma clipping.

    Parameters
    ----------
    size : int, optional
        The image size; the image is ``(size, size)``.

    repeats : int, optional
        The number of repeats for each timing (best time is kept).

    seed : int, optional
        The random number generator seed.
    """
    print(f'\n== Scalar estimators ({size}x{size} image) ==')
    print(f'{"class":>28}{"no clipping":>14}{"sigma clipping":>17}')
    data = make_image((size, size), seed=seed)
    for cls in ESTIMATOR_CLASSES:
        est_noclip = cls(sigma_clip=None)
        est_clip = cls(sigma_clip=SigmaClip(sigma=3.0, maxiters=10))
        t_noclip = time_best(partial(est_noclip, data), repeats=repeats)
        t_clip = time_best(partial(est_clip, data), repeats=repeats)
        print(f'{cls.__name__:>28}{f"{t_noclip:.3f}s":>14}'
              f'{f"{t_clip:.3f}s":>17}')


def bench_local_background(*, size=2048, n_positions=1000, repeats=3,
                           seed=0):
    """
    Benchmark LocalBackground at many positions.

    Parameters
    ----------
    size : int, optional
        The image size; the image is ``(size, size)``.

    n_positions : int, optional
        The number of aperture positions.

    repeats : int, optional
        The number of repeats for each timing (best time is kept).

    seed : int, optional
        The random number generator seed.
    """
    print(f'\n== LocalBackground ({n_positions} positions, '
          f'{size}x{size} image) ==')
    data = make_image((size, size), seed=seed)
    rng = np.random.default_rng(seed)
    x = rng.uniform(50, size - 50, n_positions)
    y = rng.uniform(50, size - 50, n_positions)
    local_bkg = LocalBackground(5, 10)
    t_local = time_best(partial(local_bkg, data, x, y), repeats=repeats)
    print(f'{t_local:.3f}s')


def main():
    """
    Run the photutils.background benchmarks.
    """
    parser = argparse.ArgumentParser(
        description='Benchmarks for the photutils.background subpackage.')
    parser.add_argument('--sizes', default='1024,2048,4096',
                        help='comma-separated image sizes '
                             '(default: %(default)s)')
    parser.add_argument('--box-sizes', default='32,64,128',
                        help='comma-separated box sizes '
                             '(default: %(default)s)')
    parser.add_argument('--n-threads', default='1,2,4,8',
                        help='comma-separated n_threads values '
                             '(default: %(default)s)')
    parser.add_argument('--repeats', type=int, default=3,
                        help='number of repeats per timing; the best '
                             'time is reported (default: %(default)s)')
    parser.add_argument('--seed', type=int, default=0,
                        help='random number generator seed '
                             '(default: %(default)s)')
    parser.add_argument('--which', default='all',
                        choices=['all', 'background2d', 'maps',
                                 'estimators', 'local'],
                        help='which benchmark to run '
                             '(default: %(default)s)')
    args = parser.parse_args()

    sizes = [int(size) for size in args.sizes.split(',')]
    box_sizes = [int(box) for box in args.box_sizes.split(',')]
    n_threads_list = [int(nthr) for nthr in args.n_threads.split(',')]

    print_environment()

    if args.which in ('all', 'background2d'):
        bench_background2d(sizes, box_sizes, n_threads_list,
                           repeats=args.repeats, seed=args.seed)
    if args.which in ('all', 'maps'):
        bench_maps(sizes, repeats=args.repeats, seed=args.seed)
    if args.which in ('all', 'estimators'):
        bench_estimators(repeats=args.repeats, seed=args.seed)
    if args.which in ('all', 'local'):
        bench_local_background(repeats=args.repeats, seed=args.seed)


if __name__ == '__main__':
    main()
