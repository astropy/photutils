#!/usr/bin/env python3
# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Benchmarks for the photutils.psf_matching subpackage.

The benchmarks cover the per-call cost of the kernel-making functions
(``make_kernel`` and ``make_wiener_kernel`` for each penalty), the
scaling of the kernel computation with the PSF size, the per-call
cost of the window classes, and ``resize_psf`` for each spline
order.

Run ``python benchmarks/bench_psf_matching.py --help`` to see the
available options.
"""

import argparse
from functools import partial

import numpy as np
from astropy.modeling.models import Gaussian2D
from bench_utils import print_environment, time_best

from photutils.psf_matching import (CosineBellWindow, HanningWindow,
                                    SplitCosineBellWindow, TopHatWindow,
                                    TukeyWindow, make_kernel,
                                    make_wiener_kernel, resize_psf)

# The (label, function, kwargs) variants for the kernel benchmarks
KERNEL_VARIANTS = (
    ('make_kernel', make_kernel, {}),
    ('make_kernel (window)', make_kernel,
     {'window': SplitCosineBellWindow(alpha=0.15, beta=0.3)}),
    ('make_wiener_kernel', make_wiener_kernel, {}),
    ('make_wiener_kernel (laplacian)', make_wiener_kernel,
     {'penalty': 'laplacian'}),
    ('make_wiener_kernel (biharmonic)', make_wiener_kernel,
     {'penalty': 'biharmonic'}),
)


def make_psf_pair(size, *, source_fraction=0.06, target_fraction=0.10):
    """
    Return centered, normalized source and target Gaussian PSFs.

    The PSF widths scale with the array size so that the PSFs fill a
    constant fraction of the array.

    Parameters
    ----------
    size : int
        The PSF array size; the arrays are ``(size, size)``. Must be
        odd.

    source_fraction : float, optional
        The source Gaussian sigma as a fraction of the array size.

    target_fraction : float, optional
        The target Gaussian sigma as a fraction of the array size.

    Returns
    -------
    source_psf : 2D `~numpy.ndarray`
        The source (narrower) PSF.

    target_psf : 2D `~numpy.ndarray`
        The target (broader) PSF.
    """
    cen = (size - 1) / 2.0
    yy, xx = np.mgrid[0:size, 0:size]
    psfs = []
    for fraction in (source_fraction, target_fraction):
        sigma = fraction * size
        psf = Gaussian2D(1.0, cen, cen, sigma, sigma)(xx, yy)
        psfs.append(psf / psf.sum())
    return tuple(psfs)


def run_kernel(func, source_psf, target_psf, n_iter, **kwargs):
    """
    Compute a matching kernel repeatedly.

    Parameters
    ----------
    func : callable
        The kernel-making function.

    source_psf : 2D `~numpy.ndarray`
        The source PSF.

    target_psf : 2D `~numpy.ndarray`
        The target PSF.

    n_iter : int
        The number of calls.

    **kwargs : dict, optional
        Additional keyword arguments passed to ``func`` (e.g.,
        ``window`` or ``penalty``).
    """
    for _ in range(n_iter):
        func(source_psf, target_psf, **kwargs)


def bench_kernels(*, size=101, n_iter=20, repeats=3):
    """
    Benchmark each kernel-making function variant.

    Parameters
    ----------
    size : int, optional
        The PSF array size; the arrays are ``(size, size)``. Must be
        odd.

    n_iter : int, optional
        The number of calls per timing; the per-call time is reported.

    repeats : int, optional
        The number of repeats for each timing (best time is kept).
    """
    source_psf, target_psf = make_psf_pair(size)

    print(f'\n== kernel functions ({size}x{size} PSFs, per-call time) ==')
    print(f'{"variant":>34}{"time":>12}')
    for label, func, kwargs in KERNEL_VARIANTS:
        bench = partial(run_kernel, func, source_psf, target_psf,
                        n_iter, **kwargs)
        t_call = time_best(bench, repeats=repeats) / n_iter
        print(f'{label:>34}{f"{t_call * 1e3:.3f}ms":>12}')


def bench_size_scaling(*, sizes=(25, 51, 101, 201), n_iter=20,
                       repeats=3):
    """
    Benchmark the kernel computation versus the PSF size.

    Parameters
    ----------
    sizes : tuple of int, optional
        The PSF array sizes. Each must be odd.

    n_iter : int, optional
        The number of calls per timing; the per-call time is reported.

    repeats : int, optional
        The number of repeats for each timing (best time is kept).
    """
    variants = (
        ('make_kernel', make_kernel, {}),
        ('make_wiener_kernel', make_wiener_kernel,
         {'penalty': 'laplacian'}),
    )

    print('\n== scaling with the PSF size (per-call time) ==')
    header = f'{"size":>8}'
    for label, _, _ in variants:
        header += f'{label:>22}'
    print(header)
    for size in sizes:
        source_psf, target_psf = make_psf_pair(size)
        cells = ''
        for _, func, kwargs in variants:
            bench = partial(run_kernel, func, source_psf, target_psf,
                            n_iter, **kwargs)
            t_call = time_best(bench, repeats=repeats) / n_iter
            cells += f'{f"{t_call * 1e3:.3f}ms":>22}'
        print(f'{size:>8}{cells}')


def run_window(window, shape, n_iter):
    """
    Evaluate a window function repeatedly.

    Parameters
    ----------
    window : callable
        The window class instance.

    shape : tuple of int
        The array shape passed to the window.

    n_iter : int
        The number of calls.
    """
    for _ in range(n_iter):
        window(shape)


def bench_windows(*, size=101, n_iter=100, repeats=3):
    """
    Benchmark each window class.

    Parameters
    ----------
    size : int, optional
        The window array size; the arrays are ``(size, size)``.

    n_iter : int, optional
        The number of calls per timing; the per-call time is reported.

    repeats : int, optional
        The number of repeats for each timing (best time is kept).
    """
    windows = (
        SplitCosineBellWindow(alpha=0.4, beta=0.3),
        TukeyWindow(alpha=0.5),
        HanningWindow(),
        CosineBellWindow(alpha=0.5),
        TopHatWindow(beta=0.4),
    )

    shape = (size, size)
    print(f'\n== window classes ({size}x{size} array, per-call time) ==')
    print(f'{"class":>24}{"time":>12}')
    for window in windows:
        bench = partial(run_window, window, shape, n_iter)
        t_call = time_best(bench, repeats=repeats) / n_iter
        print(f'{window.__class__.__name__:>24}'
              f'{f"{t_call * 1e3:.3f}ms":>12}')


def run_resize(psf, ratio, order, n_iter):
    """
    Resize a PSF repeatedly.

    Parameters
    ----------
    psf : 2D `~numpy.ndarray`
        The PSF array.

    ratio : float
        The ratio of the input to output pixel scale.

    order : int
        The spline interpolation order.

    n_iter : int
        The number of calls.
    """
    for _ in range(n_iter):
        resize_psf(psf, ratio, 1.0, order=order)


def bench_resize(*, size=101, ratios=(0.5, 2.0), n_iter=20, repeats=3):
    """
    Benchmark resize_psf for each spline order.

    Parameters
    ----------
    size : int, optional
        The PSF array size; the array is ``(size, size)``. Must be
        odd.

    ratios : tuple of float, optional
        The ratios of the input to output pixel scale (a ratio > 1
        upsamples the PSF).

    n_iter : int, optional
        The number of calls per timing; the per-call time is reported.

    repeats : int, optional
        The number of repeats for each timing (best time is kept).
    """
    psf, _ = make_psf_pair(size)
    orders = (1, 3, 5)

    print(f'\n== resize_psf ({size}x{size} PSF, per-call time) ==')
    header = f'{"ratio":>8}'
    for order in orders:
        header += f'{f"order={order}":>12}'
    print(header)
    for ratio in ratios:
        cells = ''
        for order in orders:
            bench = partial(run_resize, psf, ratio, order, n_iter)
            t_call = time_best(bench, repeats=repeats) / n_iter
            cells += f'{f"{t_call * 1e3:.3f}ms":>12}'
        print(f'{ratio:>8}{cells}')


def parse_int_list(text):
    """
    Parse a comma-separated list of positive odd integers.

    Parameters
    ----------
    text : str
        The comma-separated integers (e.g., ``'25,101,201'``).

    Returns
    -------
    result : list of int
        The parsed integers.
    """
    values = [int(item) for item in text.split(',')]
    if any(value < 1 or value % 2 == 0 for value in values):
        msg = 'values must be positive odd integers'
        raise ValueError(msg)
    return values


def main():
    """
    Run the photutils.psf_matching benchmarks.
    """
    parser = argparse.ArgumentParser(
        description='Benchmarks for the photutils.psf_matching '
                    'subpackage.')
    parser.add_argument('--size', type=int, default=101,
                        help='PSF array size; must be odd '
                             '(default: %(default)s)')
    parser.add_argument('--sizes', type=parse_int_list,
                        default=[25, 51, 101, 201],
                        help='comma-separated PSF sizes for the '
                             'scaling benchmark; each must be odd '
                             '(default: 25,51,101,201)')
    parser.add_argument('--n-iter', type=int, default=20,
                        help='number of calls per timing '
                             '(default: %(default)s)')
    parser.add_argument('--repeats', type=int, default=3,
                        help='number of repeats per timing; the best '
                             'time is reported (default: %(default)s)')
    parser.add_argument('--which', default='all',
                        choices=['all', 'kernels', 'size-scaling',
                                 'windows', 'resize'],
                        help='which benchmark to run '
                             '(default: %(default)s)')
    args = parser.parse_args()

    print_environment()

    if args.which in ('all', 'kernels'):
        bench_kernels(size=args.size, n_iter=args.n_iter,
                      repeats=args.repeats)
    if args.which in ('all', 'size-scaling'):
        bench_size_scaling(sizes=args.sizes, n_iter=args.n_iter,
                           repeats=args.repeats)
    if args.which in ('all', 'windows'):
        bench_windows(size=args.size, n_iter=args.n_iter * 5,
                      repeats=args.repeats)
    if args.which in ('all', 'resize'):
        bench_resize(size=args.size, n_iter=args.n_iter,
                     repeats=args.repeats)


if __name__ == '__main__':
    main()
