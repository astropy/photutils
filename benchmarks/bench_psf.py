#!/usr/bin/env python3
# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Benchmarks for the photutils.psf subpackage.

The benchmarks cover the per-call evaluation cost of the PSF models
(the functional models, ImagePSF, and GriddedPSFModel), PSFPhotometry
versus the number of sources and versus the PSF model type,
IterativePSFPhotometry, SourceGrouper scaling, ePSF building
(extract_stars and EPSFBuilder), the Gaussian fitting helpers
(fit_2dgaussian and fit_fwhm), and make_psf_model_image.

Run ``python benchmarks/bench_psf.py --help`` to see the available
options.
"""

import argparse
from functools import partial

import numpy as np
from astropy.nddata import NDData
from astropy.stats import gaussian_fwhm_to_sigma
from astropy.table import Table
from bench_helpers import print_environment, time_best

from photutils.detection import DAOStarFinder
from photutils.psf import (AiryDiskPSF, CircularGaussianPRF,
                           CircularGaussianPSF, CircularGaussianSigmaPRF,
                           EPSFBuilder, GaussianPRF, GaussianPSF,
                           GriddedPSFModel, ImagePSF, IterativePSFPhotometry,
                           MoffatPSF, PSFPhotometry, SourceGrouper,
                           extract_stars, fit_2dgaussian, fit_fwhm,
                           make_psf_model_image)

# The FWHM (pixels) used for all of the PSF models
FWHM = 2.8

# The shape of the model cutouts used to render the scene images
MODEL_SHAPE = (25, 25)


def make_epsf_data(*, fwhm=FWHM, oversampling=4, half_size=12):
    """
    Return an oversampled circular-Gaussian ePSF image.

    Parameters
    ----------
    fwhm : float, optional
        The FWHM of the Gaussian in detector pixels.

    oversampling : int, optional
        The oversampling factor of the output image.

    half_size : int, optional
        The half size of the ePSF in detector pixels; the output
        image has ``2 * half_size * oversampling + 1`` pixels along
        each axis.

    Returns
    -------
    result : 2D `~numpy.ndarray`
        The oversampled ePSF image, normalized to sum to
        ``oversampling**2``.
    """
    size = 2 * half_size * oversampling + 1
    yy, xx = np.mgrid[0:size, 0:size] / oversampling
    cen = (size - 1) / 2.0 / oversampling
    model = CircularGaussianPSF(flux=1.0, x_0=cen, y_0=cen, fwhm=fwhm)
    data = model(xx, yy)
    return data / data.sum() * oversampling**2


def make_image_psf(*, fwhm=FWHM, oversampling=4):
    """
    Return an `~photutils.psf.ImagePSF` model of a circular Gaussian.

    Parameters
    ----------
    fwhm : float, optional
        The FWHM of the Gaussian in detector pixels.

    oversampling : int, optional
        The oversampling factor of the ePSF image.

    Returns
    -------
    result : `~photutils.psf.ImagePSF`
        The image PSF model.
    """
    data = make_epsf_data(fwhm=fwhm, oversampling=oversampling)
    return ImagePSF(data, oversampling=oversampling)


def make_gridded_psf_model(*, fwhm=FWHM, oversampling=4, n_psfs_side=3,
                           image_size=2048):
    """
    Return a synthetic `~photutils.psf.GriddedPSFModel`.

    All of the grid planes hold the same circular-Gaussian ePSF
    image, on an ``n_psfs_side`` x ``n_psfs_side`` grid of fiducial
    positions spanning the image.

    Parameters
    ----------
    fwhm : float, optional
        The FWHM of the Gaussian in detector pixels.

    oversampling : int, optional
        The oversampling factor of the ePSF images.

    n_psfs_side : int, optional
        The number of fiducial positions along each axis.

    image_size : int, optional
        The size of the detector image spanned by the grid.

    Returns
    -------
    result : `~photutils.psf.GriddedPSFModel`
        The gridded PSF model.
    """
    epsf = make_epsf_data(fwhm=fwhm, oversampling=oversampling)
    coords = np.linspace(0, image_size - 1, n_psfs_side)
    grid_xypos = [(x, y) for y in coords for x in coords]
    planes = np.repeat(epsf[None], len(grid_xypos), axis=0)
    meta = {'grid_xypos': grid_xypos, 'oversampling': oversampling}
    return GriddedPSFModel(NDData(planes, meta=meta))


def make_psf_models():
    """
    Return the PSF models used by the model benchmarks.

    All of the models have roughly the same FWHM.

    Returns
    -------
    result : list of (str, model) tuples
        The model name and instance pairs.
    """
    sigma = FWHM * gaussian_fwhm_to_sigma
    return [
        ('CircularGaussianPRF', CircularGaussianPRF(fwhm=FWHM)),
        ('CircularGaussianPSF', CircularGaussianPSF(fwhm=FWHM)),
        ('CircularGaussianSigmaPRF', CircularGaussianSigmaPRF(sigma=sigma)),
        ('GaussianPRF', GaussianPRF(x_fwhm=FWHM, y_fwhm=FWHM)),
        ('GaussianPSF', GaussianPSF(x_fwhm=FWHM, y_fwhm=FWHM)),
        ('MoffatPSF', MoffatPSF(alpha=FWHM, beta=2.5)),
        ('AiryDiskPSF', AiryDiskPSF(radius=FWHM)),
        ('ImagePSF', make_image_psf()),
        ('GriddedPSFModel', make_gridded_psf_model()),
    ]


def make_scene(psf_model, shape, n_sources, *, flux_range=(100, 1000),
               min_separation=10, noise_std=0.1, seed=0):
    """
    Return a noisy image of PSF sources and their true parameters.

    Parameters
    ----------
    psf_model : 2D `astropy.modeling.Model`
        The PSF model used to render the sources.

    shape : tuple of int
        The shape of the output image.

    n_sources : int
        The number of sources.

    flux_range : tuple of float, optional
        The lower and upper bounds of the source fluxes.

    min_separation : float, optional
        The minimum separation between sources in pixels.

    noise_std : float, optional
        The standard deviation of the Gaussian noise.

    seed : int, optional
        The random number generator seed.

    Returns
    -------
    data : 2D `~numpy.ndarray`
        The image.

    params : `~astropy.table.Table`
        The true source parameters (``x_0``, ``y_0``, and ``flux``).
    """
    data, params = make_psf_model_image(shape, psf_model, n_sources,
                                        model_shape=MODEL_SHAPE,
                                        min_separation=min_separation,
                                        border_size=15, seed=seed,
                                        flux=flux_range)
    rng = np.random.default_rng(seed)
    data = data + rng.normal(0.0, noise_std, shape)
    return data, params


def run_model_eval(model, xx, yy, n_iter):
    """
    Evaluate a PSF model on the given grid repeatedly.

    Parameters
    ----------
    model : 2D `astropy.modeling.Model`
        The PSF model to evaluate.

    xx, yy : 2D `~numpy.ndarray`
        The x and y coordinate grids.

    n_iter : int
        The number of calls.
    """
    for _ in range(n_iter):
        model(xx, yy)


def bench_model_eval(*, sizes=(25, 361), n_iter=20, repeats=3):
    """
    Benchmark the per-call evaluation cost of each PSF model.

    Each model is evaluated on square coordinate grids with the model
    centered on the grid.

    Parameters
    ----------
    sizes : tuple of int, optional
        The grid sizes; each grid is ``(size, size)``.

    n_iter : int, optional
        The number of calls per timing; the per-call time is reported.

    repeats : int, optional
        The number of repeats for each timing (best time is kept).
    """
    models = make_psf_models()

    print('\n== PSF model evaluation (per-call time) ==')
    header = f'{"model":>26}'
    for size in sizes:
        header += f'{f"{size}x{size}":>12}'
    print(header)
    for name, model in models:
        cells = ''
        for size in sizes:
            yy, xx = np.mgrid[0:size, 0:size].astype(float)
            cen = (size - 1) / 2.0
            model.x_0 = cen
            model.y_0 = cen
            bench = partial(run_model_eval, model, xx, yy, n_iter)
            t_call = time_best(bench, repeats=repeats) / n_iter
            cells += f'{f"{t_call * 1e3:.3f}ms":>12}'
        print(f'{name:>26}{cells}')


def run_photometry(phot, data, init_params, error=None):
    """
    Run a PSF photometry instance once.

    Parameters
    ----------
    phot : `~photutils.psf.PSFPhotometry`
        The photometry instance.

    data : 2D `~numpy.ndarray`
        The image.

    init_params : `~astropy.table.Table`
        The initial source parameters.

    error : 2D `~numpy.ndarray`, optional
        The error array.
    """
    phot(data, error=error, init_params=init_params)


def bench_photometry(*, size=2048, n_sources_list=(100, 400, 1600),
                     repeats=3, seed=0):
    """
    Benchmark PSFPhotometry versus the number of sources.

    The source positions and fluxes are given via ``init_params``, so
    the timings exclude source detection. The variants are a basic
    run, a run with an error array, and a run with a SourceGrouper.

    Parameters
    ----------
    size : int, optional
        The image size; the image is ``(size, size)``.

    n_sources_list : tuple of int, optional
        The numbers of sources.

    repeats : int, optional
        The number of repeats for each timing (best time is kept).

    seed : int, optional
        The random number generator seed.
    """
    psf_model = CircularGaussianPRF(fwhm=FWHM)
    grouper = SourceGrouper(min_separation=15)

    variants = [
        ('basic', {}, None),
        ('+ error', {}, 'error'),
        ('+ grouper', {'grouper': grouper}, None),
    ]

    print(f'\n== PSFPhotometry ({size}x{size} image, init_params given, '
          'per-call time) ==')
    header = f'{"n_sources":>12}'
    for name, _, _ in variants:
        header += f'{name:>22}'
    print(header)
    for n_sources in n_sources_list:
        data, params = make_scene(psf_model, (size, size), n_sources,
                                  seed=seed)
        error = np.full(data.shape, 0.1)
        init_params = params['x_0', 'y_0', 'flux']
        cells = ''
        for _, kwargs, use_error in variants:
            phot = PSFPhotometry(psf_model, fit_shape=(5, 5),
                                 group_warning_threshold=10**9, **kwargs)
            bench = partial(run_photometry, phot, data, init_params,
                            error if use_error else None)
            t_call = time_best(bench, repeats=repeats)
            per_source = t_call / n_sources * 1e3
            cells += f'{f"{t_call:.3f}s ({per_source:.2f}ms/src)":>22}'
        print(f'{n_sources:>12}{cells}')


def bench_iterative(*, size=512, n_sources=100, repeats=3, seed=0):
    """
    Benchmark finder-based PSFPhotometry and IterativePSFPhotometry.

    The baseline row is a single PSFPhotometry pass using a
    DAOStarFinder; the other rows run IterativePSFPhotometry in the
    ``new`` and ``all`` modes.

    Parameters
    ----------
    size : int, optional
        The image size; the image is ``(size, size)``.

    n_sources : int, optional
        The number of sources.

    repeats : int, optional
        The number of repeats for each timing (best time is kept).

    seed : int, optional
        The random number generator seed.
    """
    psf_model = CircularGaussianPRF(fwhm=FWHM)
    data, _ = make_scene(psf_model, (size, size), n_sources, seed=seed)
    finder = DAOStarFinder(threshold=1.0, fwhm=FWHM)
    grouper = SourceGrouper(min_separation=15)
    kwargs = {'grouper': grouper, 'aperture_radius': 4.0,
              'group_warning_threshold': 10**9}

    variants = [
        ('PSFPhotometry (finder)',
         PSFPhotometry(psf_model, (5, 5), finder=finder, **kwargs)),
        ("IterativePSFPhotometry (mode='new')",
         IterativePSFPhotometry(psf_model, (5, 5), finder, maxiters=2,
                                mode='new', **kwargs)),
        ("IterativePSFPhotometry (mode='all')",
         IterativePSFPhotometry(psf_model, (5, 5), finder, maxiters=2,
                                mode='all', **kwargs)),
    ]

    print(f'\n== iterative photometry ({size}x{size} image, {n_sources} '
          'sources, source detection included) ==')
    print(f'{"variant":>38}{"time":>12}')
    for name, phot in variants:
        bench = partial(phot, data)
        t_call = time_best(bench, repeats=repeats)
        print(f'{name:>38}{f"{t_call:.3f}s":>12}')


def bench_model_types(*, size=512, n_sources=100, repeats=3, seed=0):
    """
    Benchmark PSFPhotometry versus the PSF model type.

    For each PSF model, the scene is rendered with that model and then
    fit with the same model, with the source positions and fluxes
    given via ``init_params``.

    Parameters
    ----------
    size : int, optional
        The image size; the image is ``(size, size)``.

    n_sources : int, optional
        The number of sources.

    repeats : int, optional
        The number of repeats for each timing (best time is kept).

    seed : int, optional
        The random number generator seed.
    """
    print(f'\n== PSFPhotometry versus PSF model ({size}x{size} image, '
          f'{n_sources} sources, per-call time) ==')
    print(f'{"model":>26}{"time":>22}')
    for name, psf_model in make_psf_models():
        data, params = make_scene(psf_model, (size, size), n_sources,
                                  seed=seed)
        init_params = params['x_0', 'y_0', 'flux']
        phot = PSFPhotometry(psf_model, fit_shape=(5, 5),
                             group_warning_threshold=10**9)
        bench = partial(run_photometry, phot, data, init_params)
        t_call = time_best(bench, repeats=repeats)
        per_source = t_call / n_sources * 1e3
        cell = f'{t_call:.3f}s ({per_source:.2f}ms/src)'
        print(f'{name:>26}{cell:>22}')


def run_grouper(grouper, x, y, n_iter):
    """
    Run a SourceGrouper repeatedly.

    Parameters
    ----------
    grouper : `~photutils.psf.SourceGrouper`
        The grouper instance.

    x, y : 1D `~numpy.ndarray`
        The source positions.

    n_iter : int
        The number of calls.
    """
    for _ in range(n_iter):
        grouper(x, y)


def bench_grouper(*, size=4096, n_sources_list=(4000, 16000, 64000),
                  separations=(5, 25), n_iter=5, repeats=3, seed=0):
    """
    Benchmark SourceGrouper versus the number of sources.

    Parameters
    ----------
    size : int, optional
        The side length of the square field containing the positions.

    n_sources_list : tuple of int, optional
        The numbers of sources.

    separations : tuple of float, optional
        The ``min_separation`` values.

    n_iter : int, optional
        The number of calls per timing; the per-call time is reported.

    repeats : int, optional
        The number of repeats for each timing (best time is kept).

    seed : int, optional
        The random number generator seed.
    """
    rng = np.random.default_rng(seed)

    print(f'\n== SourceGrouper ({size}x{size} field, per-call time) ==')
    header = f'{"n_sources":>12}'
    for separation in separations:
        header += f'{f"min_sep={separation}":>16}'
    print(header)
    for n_sources in n_sources_list:
        x = rng.uniform(0, size, n_sources)
        y = rng.uniform(0, size, n_sources)
        cells = ''
        for separation in separations:
            grouper = SourceGrouper(min_separation=separation)
            bench = partial(run_grouper, grouper, x, y, n_iter)
            t_call = time_best(bench, repeats=repeats) / n_iter
            cells += f'{f"{t_call * 1e3:.2f}ms":>16}'
        print(f'{n_sources:>12}{cells}')


def bench_epsf(*, size=2048, n_stars_list=(25, 100), oversampling=4,
               maxiters=5, repeats=3, seed=0):
    """
    Benchmark ePSF building (extract_stars and EPSFBuilder).

    Parameters
    ----------
    size : int, optional
        The image size; the image is ``(size, size)``.

    n_stars_list : tuple of int, optional
        The numbers of stars.

    oversampling : int, optional
        The EPSFBuilder oversampling factor.

    maxiters : int, optional
        The maximum number of EPSFBuilder iterations.

    repeats : int, optional
        The number of repeats for each timing (best time is kept).

    seed : int, optional
        The random number generator seed.
    """
    psf_model = CircularGaussianPRF(fwhm=FWHM)

    print(f'\n== ePSF building ({size}x{size} image, '
          f'oversampling={oversampling}, maxiters={maxiters}) ==')
    print(f'{"n_stars":>12}{"extract_stars":>16}{"EPSFBuilder":>16}')
    for n_stars in n_stars_list:
        data, params = make_scene(psf_model, (size, size), n_stars,
                                  flux_range=(500, 1000),
                                  min_separation=25, noise_std=0.01,
                                  seed=seed)
        nddata = NDData(data)
        catalog = Table({'x': params['x_0'], 'y': params['y_0']})

        bench = partial(extract_stars, nddata, catalog, size=25)
        t_extract = time_best(bench, repeats=repeats)

        stars = extract_stars(nddata, catalog, size=25)
        builder = EPSFBuilder(oversampling=oversampling, maxiters=maxiters,
                              progress_bar=False)
        bench = partial(builder, stars)
        t_build = time_best(bench, repeats=repeats)

        print(f'{n_stars:>12}{f"{t_extract * 1e3:.2f}ms":>16}'
              f'{f"{t_build:.3f}s":>16}')


def bench_fit_helpers(*, size=1024, n_sources=100, n_iter=5, repeats=3,
                      seed=0):
    """
    Benchmark the Gaussian fitting helpers.

    The helpers are timed on a single-source cutout image and at many
    positions (via ``xypos``) on a full image.

    Parameters
    ----------
    size : int, optional
        The full-image size; the image is ``(size, size)``.

    n_sources : int, optional
        The number of sources in the full image.

    n_iter : int, optional
        The number of calls per timing (single-source case only); the
        per-call time is reported.

    repeats : int, optional
        The number of repeats for each timing (best time is kept).

    seed : int, optional
        The random number generator seed.
    """
    rng = np.random.default_rng(seed)
    cutout_size = MODEL_SHAPE[0]
    cen = (cutout_size - 1) / 2.0
    yy, xx = np.mgrid[0:cutout_size, 0:cutout_size]
    cutout_model = CircularGaussianPRF(flux=500.0, x_0=cen, y_0=cen,
                                       fwhm=FWHM)
    cutout = cutout_model(xx, yy) + rng.normal(0.0, 0.01, MODEL_SHAPE)

    psf_model = CircularGaussianPRF(fwhm=FWHM)
    data, params = make_scene(psf_model, (size, size), n_sources,
                              flux_range=(500, 1000), min_separation=25,
                              noise_std=0.01, seed=seed)
    xypos = np.column_stack([params['x_0'], params['y_0']])

    variants = [
        ('fit_fwhm', partial(fit_fwhm)),
        ('fit_2dgaussian (fix_fwhm=True)',
         partial(fit_2dgaussian, fix_fwhm=True)),
        ('fit_2dgaussian (fix_fwhm=False)',
         partial(fit_2dgaussian, fix_fwhm=False)),
    ]

    print(f'\n== Gaussian fitting helpers (per-call time; '
          f'{n_sources} sources in the many-source column) ==')
    print(f'{"function":>34}{"1 source":>14}{f"{n_sources} sources":>14}')
    for name, func in variants:

        def run_single(func=func):
            for _ in range(n_iter):
                func(cutout, fit_shape=7)

        t_single = time_best(run_single, repeats=repeats) / n_iter
        bench = partial(func, data, xypos=xypos, fit_shape=7)
        t_many = time_best(bench, repeats=repeats)
        print(f'{name:>34}{f"{t_single * 1e3:.2f}ms":>14}'
              f'{f"{t_many:.3f}s":>14}')


def bench_simulation(*, size=2048, n_sources_list=(100, 400, 1600),
                     model_sizes=(15, 25, 45), repeats=3, seed=0):
    """
    Benchmark make_psf_model_image versus the number of sources.

    Parameters
    ----------
    size : int, optional
        The image size; the image is ``(size, size)``.

    n_sources_list : tuple of int, optional
        The numbers of sources.

    model_sizes : tuple of int, optional
        The model cutout sizes; each cutout is ``(msize, msize)``.

    repeats : int, optional
        The number of repeats for each timing (best time is kept).

    seed : int, optional
        The random number generator seed.
    """
    psf_model = CircularGaussianPRF(fwhm=FWHM)

    print(f'\n== make_psf_model_image ({size}x{size} image, per-call '
          'time) ==')
    header = f'{"n_sources":>12}'
    for model_size in model_sizes:
        header += f'{f"shape={model_size}":>14}'
    print(header)
    for n_sources in n_sources_list:
        cells = ''
        for model_size in model_sizes:
            bench = partial(make_psf_model_image, (size, size), psf_model,
                            n_sources, model_shape=(model_size, model_size),
                            min_separation=10, seed=seed,
                            flux=(100, 1000))
            t_call = time_best(bench, repeats=repeats)
            cells += f'{f"{t_call:.3f}s":>14}'
        print(f'{n_sources:>12}{cells}')


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
    Run the photutils.psf benchmarks.
    """
    parser = argparse.ArgumentParser(
        description='Benchmarks for the photutils.psf subpackage.')
    parser.add_argument('--n-sources-list', type=parse_int_list,
                        default=[100, 400, 1600],
                        help='comma-separated numbers of sources for the '
                             'photometry and simulation benchmarks '
                             '(default: 100,400,1600)')
    parser.add_argument('--n-grouper-list', type=parse_int_list,
                        default=[4000, 16000, 64000],
                        help='comma-separated numbers of sources for the '
                             'grouper benchmark (default: 4000,16000,64000)')
    parser.add_argument('--n-stars-list', type=parse_int_list,
                        default=[25, 100],
                        help='comma-separated numbers of stars for the '
                             'ePSF benchmark (default: 25,100)')
    parser.add_argument('--sizes', type=parse_int_list, default=[25, 361],
                        help='comma-separated grid sizes for the '
                             'model-evaluation benchmark (default: 25,361)')
    parser.add_argument('--repeats', type=int, default=3,
                        help='number of repeats per timing; the best '
                             'time is reported (default: %(default)s)')
    parser.add_argument('--seed', type=int, default=0,
                        help='random number generator seed '
                             '(default: %(default)s)')
    parser.add_argument('--which', default='all',
                        choices=['all', 'model-eval', 'photometry',
                                 'iterative', 'model-types', 'grouper',
                                 'epsf', 'fit-helpers', 'simulation'],
                        help='which benchmark to run '
                             '(default: %(default)s)')
    args = parser.parse_args()

    print_environment()

    if args.which in ('all', 'model-eval'):
        bench_model_eval(sizes=args.sizes, repeats=args.repeats)
    if args.which in ('all', 'photometry'):
        bench_photometry(n_sources_list=args.n_sources_list,
                         repeats=args.repeats, seed=args.seed)
    if args.which in ('all', 'iterative'):
        bench_iterative(repeats=args.repeats, seed=args.seed)
    if args.which in ('all', 'model-types'):
        bench_model_types(repeats=args.repeats, seed=args.seed)
    if args.which in ('all', 'grouper'):
        bench_grouper(n_sources_list=args.n_grouper_list,
                      repeats=args.repeats, seed=args.seed)
    if args.which in ('all', 'epsf'):
        bench_epsf(n_stars_list=args.n_stars_list, repeats=args.repeats,
                   seed=args.seed)
    if args.which in ('all', 'fit-helpers'):
        bench_fit_helpers(repeats=args.repeats, seed=args.seed)
    if args.which in ('all', 'simulation'):
        bench_simulation(n_sources_list=args.n_sources_list,
                         repeats=args.repeats, seed=args.seed)


if __name__ == '__main__':
    main()
