#!/usr/bin/env python3
# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Benchmarks for the photutils.profiles subpackage.

The benchmarks cover the per-call cost of constructing each profile
class and computing its profile for each aperture overlap method, the
scaling of the profile computation with the number of radial bins,
and the cost of the RadialProfile extras (the raw data profile and
the Gaussian and Moffat model fits).

Run ``python benchmarks/bench_profiles.py --help`` to see the
available options.
"""

import argparse
from functools import partial

import numpy as np
from astropy.modeling.models import Gaussian2D
from astropy.stats import gaussian_fwhm_to_sigma
from bench_helpers import print_environment, time_best

from photutils.profiles import (CurveOfGrowth, EllipticalCurveOfGrowth,
                                EnsquaredCurveOfGrowth, RadialProfile)

# The overlap methods used by the profile-classes benchmark
METHODS = ('exact', 'center', 'subpixel')


def make_source_image(size, *, amplitude=100.0, noise_std=1.0, seed=0):
    """
    Return an image with a single Gaussian source and an error array.

    The source FWHM scales with the image size so that the source
    fills a constant fraction of the image.

    Parameters
    ----------
    size : int
        The image size; the image is ``(size, size)``.

    amplitude : float, optional
        The amplitude of the Gaussian source.

    noise_std : float, optional
        The standard deviation of the Gaussian noise.

    seed : int, optional
        The random number generator seed.

    Returns
    -------
    data : 2D `~numpy.ndarray`
        The image.

    error : 2D `~numpy.ndarray`
        The error array.

    xycen : tuple of float
        The ``(x, y)`` source center.
    """
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size]
    xc = yc = (size - 1) / 2.0
    sigma = (size / 8.0) * gaussian_fwhm_to_sigma
    model = Gaussian2D(amplitude, xc, yc, sigma, sigma)
    data = model(xx, yy) + rng.normal(0.0, noise_std, (size, size))
    error = np.full((size, size), noise_std)
    return data, error, (xc, yc)


def make_profile(profile_class, data, error, xycen, n_radii, max_radius,
                 **kwargs):
    """
    Construct a profile instance covering the given radial range.

    Parameters
    ----------
    profile_class : class
        The profile class to construct.

    data : 2D `~numpy.ndarray`
        The image.

    error : 2D `~numpy.ndarray`
        The error array.

    xycen : tuple of float
        The ``(x, y)`` source center.

    n_radii : int
        The number of radial bins.

    max_radius : float
        The maximum radius (or half-size) in pixels.

    **kwargs : dict, optional
        Additional keyword arguments passed to the profile class
        (e.g., ``method`` or ``axis_ratio``).

    Returns
    -------
    result : `~photutils.profiles.ProfileBase`
        The profile instance.
    """
    if profile_class is RadialProfile:
        radii = np.linspace(0.0, max_radius, n_radii + 1)
    else:
        radii = np.linspace(1.0, max_radius, n_radii)
    if profile_class is EllipticalCurveOfGrowth:
        kwargs.setdefault('axis_ratio', 0.5)
        kwargs.setdefault('theta', 0.5)
    return profile_class(data, xycen, radii, error=error, **kwargs)


def run_profile(profile_class, data, error, xycen, n_radii, max_radius,
                n_iter, **kwargs):
    """
    Construct a profile and compute its outputs repeatedly.

    Each call constructs a new instance and accesses the ``profile``,
    ``profile_error``, and ``area`` attributes (the computation is
    lazy and cached per instance).

    Parameters
    ----------
    profile_class : class
        The profile class to construct.

    data : 2D `~numpy.ndarray`
        The image.

    error : 2D `~numpy.ndarray`
        The error array.

    xycen : tuple of float
        The ``(x, y)`` source center.

    n_radii : int
        The number of radial bins.

    max_radius : float
        The maximum radius (or half-size) in pixels.

    n_iter : int
        The number of calls.

    **kwargs : dict, optional
        Additional keyword arguments passed to the profile class.
    """
    for _ in range(n_iter):
        profile = make_profile(profile_class, data, error, xycen,
                               n_radii, max_radius, **kwargs)
        for name in ('profile', 'profile_error', 'area'):
            getattr(profile, name)


def bench_classes(*, size=501, n_radii=100, n_iter=5, repeats=3, seed=0):
    """
    Benchmark each profile class for each overlap method.

    Each timing includes profile construction and the computation of
    the ``profile``, ``profile_error``, and ``area`` attributes.

    Parameters
    ----------
    size : int, optional
        The image size; the image is ``(size, size)``.

    n_radii : int, optional
        The number of radial bins.

    n_iter : int, optional
        The number of calls per timing; the per-call time is reported.

    repeats : int, optional
        The number of repeats for each timing (best time is kept).

    seed : int, optional
        The random number generator seed.
    """
    data, error, xycen = make_source_image(size, seed=seed)
    max_radius = size / 4.0

    classes = (RadialProfile, CurveOfGrowth, EnsquaredCurveOfGrowth,
               EllipticalCurveOfGrowth)

    print(f'\n== profile classes ({size}x{size} image, {n_radii} radii, '
          'per-call time) ==')
    header = f'{"class":>24}'
    for method in METHODS:
        header += f'{method:>12}'
    print(header)
    for profile_class in classes:
        cells = ''
        for method in METHODS:
            bench = partial(run_profile, profile_class, data, error,
                            xycen, n_radii, max_radius, n_iter,
                            method=method)
            t_call = time_best(bench, repeats=repeats) / n_iter
            cells += f'{f"{t_call * 1e3:.2f}ms":>12}'
        print(f'{profile_class.__name__:>24}{cells}')


def bench_radii_scaling(*, size=501, n_radii_list=(25, 100, 400),
                        n_iter=5, repeats=3, seed=0):
    """
    Benchmark the profile computation versus the number of radii.

    Parameters
    ----------
    size : int, optional
        The image size; the image is ``(size, size)``.

    n_radii_list : tuple of int, optional
        The numbers of radial bins.

    n_iter : int, optional
        The number of calls per timing; the per-call time is reported.

    repeats : int, optional
        The number of repeats for each timing (best time is kept).

    seed : int, optional
        The random number generator seed.
    """
    data, error, xycen = make_source_image(size, seed=seed)
    max_radius = size / 4.0

    classes = (RadialProfile, CurveOfGrowth)

    print(f'\n== scaling with the number of radii ({size}x{size} image, '
          'per-call time) ==')
    header = f'{"n_radii":>12}'
    for profile_class in classes:
        header += f'{profile_class.__name__:>16}'
    print(header)
    for n_radii in n_radii_list:
        cells = ''
        for profile_class in classes:
            bench = partial(run_profile, profile_class, data, error,
                            xycen, n_radii, max_radius, n_iter)
            t_call = time_best(bench, repeats=repeats) / n_iter
            cells += f'{f"{t_call * 1e3:.2f}ms":>16}'
        print(f'{n_radii:>12}{cells}')


def run_radial_extra(data, error, xycen, n_radii, max_radius, n_iter,
                     attributes):
    """
    Construct a RadialProfile and access the given attributes.

    Parameters
    ----------
    data : 2D `~numpy.ndarray`
        The image.

    error : 2D `~numpy.ndarray`
        The error array.

    xycen : tuple of float
        The ``(x, y)`` source center.

    n_radii : int
        The number of radial bins.

    max_radius : float
        The maximum radius in pixels.

    n_iter : int
        The number of calls.

    attributes : list of str
        The names of the attributes to access on each instance.
    """
    for _ in range(n_iter):
        profile = make_profile(RadialProfile, data, error, xycen,
                               n_radii, max_radius)
        for name in ['profile', *attributes]:
            getattr(profile, name)


def bench_radial_extras(*, size=501, n_radii=100, n_iter=5, repeats=3,
                        seed=0):
    """
    Benchmark the extra lazily-computed RadialProfile attributes.

    The baseline row times construction plus the ``profile``
    computation alone; the remaining rows add one extra attribute
    each.

    Parameters
    ----------
    size : int, optional
        The image size; the image is ``(size, size)``.

    n_radii : int, optional
        The number of radial bins.

    n_iter : int, optional
        The number of calls per timing; the per-call time is reported.

    repeats : int, optional
        The number of repeats for each timing (best time is kept).

    seed : int, optional
        The random number generator seed.
    """
    data, error, xycen = make_source_image(size, seed=seed)
    max_radius = size / 4.0

    variants = [
        ('profile only', []),
        ('+ data_profile', ['data_profile']),
        ('+ gaussian_fit', ['gaussian_fit']),
        ('+ moffat_fit', ['moffat_fit']),
    ]

    print(f'\n== RadialProfile extras ({size}x{size} image, {n_radii} '
          'radii, per-call time) ==')
    print(f'{"variant":>18}{"time":>12}')
    for name, attributes in variants:
        bench = partial(run_radial_extra, data, error, xycen, n_radii,
                        max_radius, n_iter, attributes)
        t_call = time_best(bench, repeats=repeats) / n_iter
        print(f'{name:>18}{f"{t_call * 1e3:.2f}ms":>12}')


def parse_int_list(text):
    """
    Parse a comma-separated list of positive integers.

    Parameters
    ----------
    text : str
        The comma-separated integers (e.g., ``'25,100,400'``).

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
    Run the photutils.profiles benchmarks.
    """
    parser = argparse.ArgumentParser(
        description='Benchmarks for the photutils.profiles subpackage.')
    parser.add_argument('--size', type=int, default=501,
                        help='image size (default: %(default)s)')
    parser.add_argument('--n-radii', type=int, default=100,
                        help='number of radial bins '
                             '(default: %(default)s)')
    parser.add_argument('--n-radii-list', type=parse_int_list,
                        default=[25, 100, 400],
                        help='comma-separated numbers of radial bins '
                             'for the scaling benchmark '
                             '(default: 25,100,400)')
    parser.add_argument('--n-iter', type=int, default=5,
                        help='number of calls per timing '
                             '(default: %(default)s)')
    parser.add_argument('--repeats', type=int, default=3,
                        help='number of repeats per timing; the best '
                             'time is reported (default: %(default)s)')
    parser.add_argument('--seed', type=int, default=0,
                        help='random number generator seed '
                             '(default: %(default)s)')
    parser.add_argument('--which', default='all',
                        choices=['all', 'classes', 'radii-scaling',
                                 'radial-extras'],
                        help='which benchmark to run '
                             '(default: %(default)s)')
    args = parser.parse_args()

    print_environment()

    if args.which in ('all', 'classes'):
        bench_classes(size=args.size, n_radii=args.n_radii,
                      n_iter=args.n_iter, repeats=args.repeats,
                      seed=args.seed)
    if args.which in ('all', 'radii-scaling'):
        bench_radii_scaling(size=args.size,
                            n_radii_list=args.n_radii_list,
                            n_iter=args.n_iter, repeats=args.repeats,
                            seed=args.seed)
    if args.which in ('all', 'radial-extras'):
        bench_radial_extras(size=args.size, n_radii=args.n_radii,
                            n_iter=args.n_iter, repeats=args.repeats,
                            seed=args.seed)


if __name__ == '__main__':
    main()
