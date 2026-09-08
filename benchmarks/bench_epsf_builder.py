#!/usr/bin/env python3
# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Benchmarks for ePSF building with EPSFBuilder.

The benchmarks cover the wall-clock time of a full build and its
number of iterations versus the number of stars and versus the
oversampling factor, and the cost of the stages of a single build
iteration (residual stacking, sigma clipping and median, smoothing
and recentering, and star fitting).

Run ``python benchmarks/bench_epsf_builder.py --help`` to see the
available options.
"""

import argparse
import time
import warnings
from functools import partial

import numpy as np
from astropy.nddata import NDData
from astropy.table import Table
from bench_helpers import parse_int_list, print_environment, time_best

from photutils.datasets import make_model_image
from photutils.psf import CircularGaussianPRF, EPSFBuilder, extract_stars


def make_stars(n_stars, *, fwhm=2.8, cutout_size=25, seed=0):
    """
    Return extracted star cutouts from a noisy scene of Gaussian stars.

    The stars are placed on a regular grid with random subpixel
    offsets so that exactly ``n_stars`` well-separated stars are
    generated. They have fluxes spanning a factor of ten and
    Poisson-like noise plus a constant background noise.

    Parameters
    ----------
    n_stars : int
        The number of stars.

    fwhm : float, optional
        The FWHM of the stars in pixels.

    cutout_size : int, optional
        The size of the star cutouts in pixels.

    seed : int, optional
        The random number generator seed.

    Returns
    -------
    result : `~photutils.psf.EPSFStars`
        The extracted stars.
    """
    rng = np.random.default_rng(seed)
    separation = cutout_size + 10
    n_side = int(np.ceil(np.sqrt(n_stars)))
    size = (n_side + 1) * separation

    # Grid cell centers, leaving a border of one cell around the image
    grid = (np.arange(n_side) + 1) * separation
    yy, xx = np.meshgrid(grid, grid, indexing='ij')
    xpos = xx.ravel()[:n_stars] + rng.uniform(-0.5, 0.5, n_stars)
    ypos = yy.ravel()[:n_stars] + rng.uniform(-0.5, 0.5, n_stars)
    flux = rng.uniform(1e4, 1e5, n_stars)
    params = Table({'x_0': xpos, 'y_0': ypos, 'flux': flux})

    model = CircularGaussianPRF(fwhm=fwhm)
    data = make_model_image((size, size), model, params,
                            model_shape=(cutout_size + 4, cutout_size + 4))
    data += rng.normal(0.0, np.sqrt(np.abs(data) + 100.0))
    catalog = Table({'x': xpos, 'y': ypos})
    return extract_stars(NDData(data), catalog, size=cutout_size)


def make_builder(oversampling, *, maxiters=10, **kwargs):
    """
    Return an `~photutils.psf.EPSFBuilder` with the benchmark settings.

    Parameters
    ----------
    oversampling : int
        The oversampling factor.

    maxiters : int, optional
        The maximum number of build iterations.

    **kwargs : dict, optional
        Additional keyword arguments passed to `EPSFBuilder`.

    Returns
    -------
    result : `~photutils.psf.EPSFBuilder`
        The builder.
    """
    return EPSFBuilder(oversampling=oversampling, maxiters=maxiters,
                       progress_bar=False, **kwargs)


def build(builder, stars):
    """
    Build an ePSF, suppressing the builder warnings.

    Parameters
    ----------
    builder : `~photutils.psf.EPSFBuilder`
        The builder.

    stars : `~photutils.psf.EPSFStars`
        The stars.

    Returns
    -------
    result : `~photutils.psf.EPSFBuildResults`
        The build results.
    """
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return builder(stars)


def bench_stars(*, n_stars_list=(100, 400, 1600), oversampling=2, fwhm=2.8,
                cutout_size=25, maxiters=10, repeats=3, seed=0):
    """
    Benchmark a full build versus the number of stars.

    Parameters
    ----------
    n_stars_list : tuple of int, optional
        The numbers of stars.

    oversampling : int, optional
        The oversampling factor.

    fwhm : float, optional
        The FWHM of the stars in pixels.

    cutout_size : int, optional
        The size of the star cutouts in pixels.

    maxiters : int, optional
        The maximum number of build iterations.

    repeats : int, optional
        The number of repeats for each timing (best time is kept).

    seed : int, optional
        The random number generator seed.
    """
    print(f'\n== EPSFBuilder versus the number of stars '
          f'(oversampling={oversampling}, fwhm={fwhm}, '
          f'cutout_size={cutout_size}, maxiters={maxiters}) ==')
    print(f'{"n_stars":>10}{"iterations":>12}{"total":>12}'
          f'{"per iter":>12}{"per star-iter":>16}')
    for n_stars in n_stars_list:
        stars = make_stars(n_stars, fwhm=fwhm, cutout_size=cutout_size,
                           seed=seed)
        builder = make_builder(oversampling, maxiters=maxiters)
        result = build(builder, stars)
        t_build = time_best(partial(build, builder, stars),
                            repeats=repeats)
        per_iter = t_build / result.iterations
        print(f'{n_stars:>10}{result.iterations:>12}'
              f'{f"{t_build:.2f}s":>12}{f"{per_iter:.3f}s":>12}'
              f'{f"{per_iter / n_stars * 1e3:.3f}ms":>16}')


def bench_oversampling(*, oversampling_list=(1, 2, 4), n_stars=400,
                       fwhm=2.8, cutout_size=25, maxiters=10, repeats=3,
                       seed=0):
    """
    Benchmark a full build versus the oversampling factor.

    Parameters
    ----------
    oversampling_list : tuple of int, optional
        The oversampling factors.

    n_stars : int, optional
        The number of stars.

    fwhm : float, optional
        The FWHM of the stars in pixels.

    cutout_size : int, optional
        The size of the star cutouts in pixels.

    maxiters : int, optional
        The maximum number of build iterations.

    repeats : int, optional
        The number of repeats for each timing (best time is kept).

    seed : int, optional
        The random number generator seed.
    """
    stars = make_stars(n_stars, fwhm=fwhm, cutout_size=cutout_size,
                       seed=seed)
    print(f'\n== EPSFBuilder versus the oversampling factor '
          f'(n_stars={n_stars}, fwhm={fwhm}, cutout_size={cutout_size}, '
          f'maxiters={maxiters}) ==')
    print(f'{"oversampling":>14}{"ePSF shape":>14}{"iterations":>12}'
          f'{"total":>12}{"per iter":>12}')
    for oversampling in oversampling_list:
        builder = make_builder(oversampling, maxiters=maxiters)
        result = build(builder, stars)
        t_build = time_best(partial(build, builder, stars),
                            repeats=repeats)
        shape = 'x'.join(str(s) for s in result.epsf.data.shape)
        print(f'{oversampling:>14}{shape:>14}{result.iterations:>12}'
              f'{f"{t_build:.2f}s":>12}'
              f'{f"{t_build / result.iterations:.3f}s":>12}')


def bench_stages(*, n_stars=400, oversampling=2, fwhm=2.8, cutout_size=25,
                 repeats=3, seed=0):
    """
    Benchmark the stages of a single build iteration.

    The stages are timed on the ePSF and star centers of a converged
    build, so that the star fits start close to their solutions as
    they do in the later build iterations.

    Parameters
    ----------
    n_stars : int, optional
        The number of stars.

    oversampling : int, optional
        The oversampling factor.

    fwhm : float, optional
        The FWHM of the stars in pixels.

    cutout_size : int, optional
        The size of the star cutouts in pixels.

    repeats : int, optional
        The number of repeats for each timing (best time is kept).

    seed : int, optional
        The random number generator seed.
    """
    from photutils.psf.epsf_builder import _suppress_alias_modes
    from photutils.psf.image_models import ImagePSF
    from photutils.utils._stats import nanmedian

    stars = make_stars(n_stars, fwhm=fwhm, cutout_size=cutout_size,
                       seed=seed)
    builder = make_builder(oversampling, maxiters=10)
    result = build(builder, stars)
    epsf = result.epsf
    stars = result.fitted_stars

    def stack():
        """
        Stack the resampled star residuals.
        """
        return builder._resample_residuals(stars, epsf)

    residuals = stack()

    def clip_and_median():
        """
        Sigma clip the residual stack and take its median.
        """
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            clipped = builder._sigma_clip(residuals, axis=0, masked=False,
                                          return_bounds=False)
            return nanmedian(clipped, axis=0)

    median = clip_and_median()
    new_epsf = epsf.data + median

    def smooth_and_recenter():
        """
        Smooth, low-pass filter, recenter, and normalize the ePSF.
        """
        builder._update_auto_parameters(new_epsf)
        smoothed = builder._smooth_epsf(new_epsf)
        smoothed = _suppress_alias_modes(smoothed, builder.oversampling)
        temp = ImagePSF(data=smoothed, origin=epsf.origin,
                        oversampling=builder.oversampling, fill_value=0.0)
        return builder._normalize_epsf(builder._recenter_epsf(temp))

    def fit():
        """
        Fit the ePSF to the stars.
        """
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            return builder._fit_stars(epsf, stars)

    def iteration():
        """
        Run a full build iteration.
        """
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            return builder._process_iteration(stars, epsf, 2)

    timings = [('residual stacking', time_best(stack, repeats=repeats)),
               ('sigma clip and median',
                time_best(clip_and_median, repeats=repeats)),
               ('smooth and recenter',
                time_best(smooth_and_recenter, repeats=repeats)),
               ('star fitting', time_best(fit, repeats=repeats)),
               ('full iteration', time_best(iteration, repeats=repeats))]

    kernel_shape = (None if result.smoothing_kernel is None
                    else result.smoothing_kernel.shape)
    print(f'\n== Build iteration stages (n_stars={n_stars}, '
          f'oversampling={oversampling}, fwhm={fwhm}, '
          f'cutout_size={cutout_size}, fit_shape={result.fit_shape}, '
          f'kernel={kernel_shape}) ==')
    print(f'{"stage":>26}{"time":>12}{"fraction":>12}')
    total = timings[-1][1]
    for name, t_stage in timings:
        print(f'{name:>26}{f"{t_stage:.3f}s":>12}'
              f'{f"{t_stage / total * 100:.0f}%":>12}')


def main():
    """
    Run the ePSF building benchmarks.
    """
    parser = argparse.ArgumentParser(
        description='Benchmarks for ePSF building with EPSFBuilder.')
    parser.add_argument('--n-stars-list', type=parse_int_list,
                        default=[100, 400, 1600],
                        help='comma-separated numbers of stars for the '
                             'stars benchmark (default: 100,400,1600)')
    parser.add_argument('--n-stars', type=int, default=400,
                        help='number of stars for the oversampling and '
                             'stages benchmarks (default: %(default)s)')
    parser.add_argument('--oversampling-list', type=parse_int_list,
                        default=[1, 2, 4],
                        help='comma-separated oversampling factors for '
                             'the oversampling benchmark (default: 1,2,4)')
    parser.add_argument('--oversampling', type=int, default=2,
                        help='oversampling factor for the stars and '
                             'stages benchmarks (default: %(default)s)')
    parser.add_argument('--fwhm', type=float, default=2.8,
                        help='FWHM of the stars in pixels '
                             '(default: %(default)s)')
    parser.add_argument('--cutout-size', type=int, default=25,
                        help='size of the star cutouts in pixels '
                             '(default: %(default)s)')
    parser.add_argument('--maxiters', type=int, default=10,
                        help='maximum number of build iterations '
                             '(default: %(default)s)')
    parser.add_argument('--repeats', type=int, default=3,
                        help='number of repeats per timing; the best '
                             'time is reported (default: %(default)s)')
    parser.add_argument('--seed', type=int, default=0,
                        help='random number generator seed '
                             '(default: %(default)s)')
    parser.add_argument('--which', default='all',
                        choices=['all', 'stars', 'oversampling', 'stages'],
                        help='which benchmark to run '
                             '(default: %(default)s)')
    args = parser.parse_args()

    print_environment()
    t0 = time.perf_counter()

    if args.which in ('all', 'stars'):
        bench_stars(n_stars_list=args.n_stars_list,
                    oversampling=args.oversampling, fwhm=args.fwhm,
                    cutout_size=args.cutout_size, maxiters=args.maxiters,
                    repeats=args.repeats, seed=args.seed)
    if args.which in ('all', 'oversampling'):
        bench_oversampling(oversampling_list=args.oversampling_list,
                           n_stars=args.n_stars, fwhm=args.fwhm,
                           cutout_size=args.cutout_size,
                           maxiters=args.maxiters, repeats=args.repeats,
                           seed=args.seed)
    if args.which in ('all', 'stages'):
        bench_stages(n_stars=args.n_stars, oversampling=args.oversampling,
                     fwhm=args.fwhm, cutout_size=args.cutout_size,
                     repeats=args.repeats, seed=args.seed)

    print(f'\ntotal benchmark time: {time.perf_counter() - t0:.1f}s')


if __name__ == '__main__':
    main()
