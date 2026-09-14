#!/usr/bin/env python3
# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Benchmarks for the local WCS helper functions.

The benchmarks cover the per-call cost of the scalar helpers on
astropy TAN, TAN-SIP, and gwcs transforms, the vectorized helpers
versus the number of positions (with the speedup over a per-source
loop), and the aperture ``to_pixel`` and ``to_sky`` conversions that
use the helpers.

Run ``python benchmarks/bench_wcs_helpers.py --help`` to see the
available options.
"""

import argparse
from functools import partial

import astropy.units as u
import numpy as np
from astropy.io.fits import Header
from astropy.wcs import WCS
from bench_helpers import parse_int_list, print_environment, time_best

from photutils.aperture import (CircularAperture, EllipticalAperture,
                                SkyCircularAperture, SkyEllipticalAperture)
from photutils.datasets import make_gwcs, make_wcs
from photutils.utils._wcs_helpers import (compute_local_wcs_jacobian,
                                          compute_pixel_scale_angles,
                                          compute_pixel_to_sky_jacobians,
                                          compute_pixel_to_sky_mean_scales,
                                          pixel_shape_to_sky_svd,
                                          pixel_to_sky_mean_scale,
                                          pixel_to_sky_svd_scales,
                                          sky_shape_to_pixel_svd,
                                          sky_to_pixel_mean_scale,
                                          sky_to_pixel_svd_scales,
                                          wcs_pixel_scale_angle)

# The per-source loops are capped at this many calls and the time is
# scaled to the requested number of positions
MAX_LOOP_CALLS = 500


def make_sip_wcs(shape, *, coeff=1e-6):
    """
    Return a TAN-SIP WCS with a small quadratic distortion.

    Parameters
    ----------
    shape : tuple of int
        The ``(ny, nx)`` image shape.

    coeff : float, optional
        The quadratic SIP coefficient.

    Returns
    -------
    wcs : `~astropy.wcs.WCS`
        The distorted WCS.
    """
    header = Header()
    header['NAXIS'] = 2
    header['NAXIS1'] = shape[1]
    header['NAXIS2'] = shape[0]
    header['CRPIX1'] = shape[1] / 2
    header['CRPIX2'] = shape[0] / 2
    header['CRVAL1'] = 197.8925
    header['CRVAL2'] = -1.36555556
    header['CTYPE1'] = 'RA---TAN-SIP'
    header['CTYPE2'] = 'DEC--TAN-SIP'
    cdelt = 0.1 / 3600
    header['CD1_1'] = -cdelt
    header['CD1_2'] = 0.0
    header['CD2_1'] = 0.0
    header['CD2_2'] = cdelt
    header['A_ORDER'] = 2
    header['A_2_0'] = coeff
    header['B_ORDER'] = 2
    header['B_0_2'] = coeff
    return WCS(header)


def make_wcs_cases(shape):
    """
    Return the (name, wcs) pairs benchmarked.

    Parameters
    ----------
    shape : tuple of int
        The ``(ny, nx)`` image shape.

    Returns
    -------
    result : list of (str, wcs) tuples
        The WCS name and instance pairs.
    """
    return [('TAN', make_wcs(shape)),
            ('TAN-SIP', make_sip_wcs(shape)),
            ('gwcs', make_gwcs(shape))]


def make_positions(shape, n_positions, *, seed=0):
    """
    Return random pixel positions inside the image.

    Parameters
    ----------
    shape : tuple of int
        The ``(ny, nx)`` image shape.

    n_positions : int
        The number of positions.

    seed : int, optional
        The random number generator seed.

    Returns
    -------
    x, y : 1D `~numpy.ndarray`
        The pixel coordinates.
    """
    rng = np.random.default_rng(seed)
    margin = 20.0
    x = rng.uniform(margin, shape[1] - margin, n_positions)
    y = rng.uniform(margin, shape[0] - margin, n_positions)
    return x, y


def make_scalar_cases(skycoord, pixcoord, wcs):
    """
    Return the (name, callable) pairs for the scalar helpers.

    Parameters
    ----------
    skycoord : `~astropy.coordinates.SkyCoord`
        The sky position for the sky-input helpers.

    pixcoord : tuple of float
        The ``(x, y)`` pixel position for the pixel-input helpers.

    wcs : WCS object
        The WCS transformation.

    Returns
    -------
    result : list of (str, callable) tuples
        The helper name and zero-argument callable pairs.
    """
    width, height, angle = 2.0, 1.0, 0.5
    return [
        ('local Jacobian', partial(compute_local_wcs_jacobian, skycoord, wcs)),
        ('scale/angle', partial(wcs_pixel_scale_angle, skycoord, wcs)),
        ('sky->pix mean scale',
         partial(sky_to_pixel_mean_scale, skycoord, wcs)),
        ('sky->pix SVD scales',
         partial(sky_to_pixel_svd_scales, skycoord, wcs)),
        ('sky->pix shape SVD',
         partial(sky_shape_to_pixel_svd, skycoord, wcs, width, height,
                 angle)),
        ('pix->sky mean scale',
         partial(pixel_to_sky_mean_scale, pixcoord, wcs)),
        ('pix->sky SVD scales',
         partial(pixel_to_sky_svd_scales, pixcoord, wcs)),
        ('pix->sky shape SVD',
         partial(pixel_shape_to_sky_svd, pixcoord, wcs, width, height,
                 angle)),
    ]


def bench_scalar_helpers(*, shape=(2000, 2000), n_iter=50, repeats=3):
    """
    Benchmark the per-call cost of the scalar helper functions.

    Parameters
    ----------
    shape : tuple of int, optional
        The ``(ny, nx)`` image shape.

    n_iter : int, optional
        The number of calls per timing; the per-call time is reported.

    repeats : int, optional
        The number of repeats for each timing (best time is kept).
    """
    pixcoord = (shape[1] / 2, shape[0] / 2)
    cases = make_wcs_cases(shape)
    names = [name for name, _ in cases]

    print('\n== Scalar WCS helpers (per-call time) ==')
    print(f'{"function":>20}' + ''.join(f'{name:>12}' for name in names))
    rows = {}
    for _, wcs in cases:
        skycoord = wcs.pixel_to_world(*pixcoord)
        for name, func in make_scalar_cases(skycoord, pixcoord, wcs):
            def run(func=func):
                for _ in range(n_iter):
                    func()

            t_call = time_best(run, repeats=repeats) / n_iter
            rows.setdefault(name, []).append(f'{t_call * 1e6:.0f}us')
    for name, cells in rows.items():
        print(f'{name:>20}' + ''.join(f'{cell:>12}' for cell in cells))


def bench_vectorized_helpers(*, shape=(2000, 2000),
                             n_positions_list=(100, 1_000, 10_000, 100_000),
                             repeats=3):
    """
    Benchmark the vectorized helpers versus the number of positions.

    The speedup over a per-source loop of the scalar equivalents is
    also reported. The loops are capped at ``MAX_LOOP_CALLS`` calls and
    scaled to the number of positions.

    Parameters
    ----------
    shape : tuple of int, optional
        The ``(ny, nx)`` image shape.

    n_positions_list : tuple of int, optional
        The numbers of positions.

    repeats : int, optional
        The number of repeats for each timing (best time is kept).
    """
    vectorized = [
        ('Jacobians', compute_pixel_to_sky_jacobians),
        ('mean scales', compute_pixel_to_sky_mean_scales),
        ('scale/angles', compute_pixel_scale_angles),
    ]

    for wcs_name, wcs in make_wcs_cases(shape):
        print(f'\n== Vectorized WCS helpers ({wcs_name}) ==')
        print(f'{"n_positions":>12}{"Jacobians":>12}{"mean scales":>14}'
              f'{"scale/angles":>14}{"loop speedup":>14}')
        for n_positions in n_positions_list:
            x, y = make_positions(shape, n_positions)
            times = [time_best(partial(func, x, y, wcs), repeats=repeats)
                     for _, func in vectorized]
            # Per-source loop of the scalar mean-scale helper, scaled
            # to n_positions
            n_calls = min(n_positions, MAX_LOOP_CALLS)

            def loop(x=x, y=y, wcs=wcs, n_calls=n_calls):
                for i in range(n_calls):
                    pixel_to_sky_mean_scale((x[i], y[i]), wcs)

            t_loop = time_best(loop, repeats=1) * n_positions / n_calls
            cells = [f'{t * 1e3:.2f}ms' for t in times]
            cells.append(f'{t_loop / times[1]:.0f}x')
            print(f'{n_positions:>12}{cells[0]:>12}{cells[1]:>14}'
                  f'{cells[2]:>14}{cells[3]:>14}')


def bench_aperture_conversions(*, shape=(2000, 2000), n_iter=20, repeats=3):
    """
    Benchmark the aperture ``to_pixel`` and ``to_sky`` conversions
    that use the WCS helpers.

    Parameters
    ----------
    shape : tuple of int, optional
        The ``(ny, nx)`` image shape.

    n_iter : int, optional
        The number of calls per timing; the per-call time is reported.

    repeats : int, optional
        The number of repeats for each timing (best time is kept).
    """
    x0, y0 = shape[1] / 2, shape[0] / 2
    cases = make_wcs_cases(shape)
    names = [name for name, _ in cases]

    print('\n== Aperture conversions (per-call time) ==')
    print(f'{"conversion":>32}' + ''.join(f'{name:>12}' for name in names))
    rows = {}
    for _, wcs in cases:
        skycoord = wcs.pixel_to_world(x0, y0)
        circle = CircularAperture((x0, y0), r=5.0)
        ellipse = EllipticalAperture((x0, y0), a=5.0, b=3.0, theta=0.5)
        sky_circle = SkyCircularAperture(skycoord, r=0.5 * u.arcsec)
        sky_ellipse = SkyEllipticalAperture(skycoord, a=0.5 * u.arcsec,
                                            b=0.3 * u.arcsec,
                                            theta=30 * u.deg)
        funcs = [
            ('CircularAperture.to_sky', partial(circle.to_sky, wcs)),
            ('SkyCircularAperture.to_pixel',
             partial(sky_circle.to_pixel, wcs)),
            ('EllipticalAperture.to_sky', partial(ellipse.to_sky, wcs)),
            ('SkyEllipticalAperture.to_pixel',
             partial(sky_ellipse.to_pixel, wcs)),
        ]
        for name, func in funcs:
            def run(func=func):
                for _ in range(n_iter):
                    func()

            t_call = time_best(run, repeats=repeats) / n_iter
            rows.setdefault(name, []).append(f'{t_call * 1e6:.0f}us')
    for name, cells in rows.items():
        print(f'{name:>32}' + ''.join(f'{cell:>12}' for cell in cells))


def main():
    """
    Run the WCS helper benchmarks.
    """
    parser = argparse.ArgumentParser(
        description='Benchmarks for the local WCS helper functions.')
    parser.add_argument('--n-positions', type=parse_int_list,
                        default=[100, 1_000, 10_000, 100_000],
                        help='comma-separated numbers of positions for '
                             'the vectorized-helper benchmark '
                             '(default: 100,1000,10000,100000)')
    parser.add_argument('--repeats', type=int, default=3,
                        help='number of repeats per timing; the best '
                             'time is reported (default: %(default)s)')
    parser.add_argument('--which', default='all',
                        choices=['all', 'scalar', 'vectorized',
                                 'apertures'],
                        help='which benchmark to run '
                             '(default: %(default)s)')
    args = parser.parse_args()

    print_environment()

    if args.which in ('all', 'scalar'):
        bench_scalar_helpers(repeats=args.repeats)
    if args.which in ('all', 'vectorized'):
        bench_vectorized_helpers(n_positions_list=args.n_positions,
                                 repeats=args.repeats)
    if args.which in ('all', 'apertures'):
        bench_aperture_conversions(repeats=args.repeats)


if __name__ == '__main__':
    main()
