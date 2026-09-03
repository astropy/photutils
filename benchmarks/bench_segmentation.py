#!/usr/bin/env python3
# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Benchmarks for the photutils.segmentation subpackage.

The benchmarks cover source detection (detect_threshold and
detect_sources), source deblending (deblend_sources) across the
threshold modes and process counts, the combined SourceFinder class,
SegmentationImage operations (relabeling, border-label removal,
source masks, and polygons), SourceCatalog property calculations, the
SourceCatalog n_threads keyword, and concurrent SourceCatalog runs
across thread counts.

Run ``python benchmarks/bench_segmentation.py --help`` to see the
available options.
"""

import argparse
import time
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from operator import attrgetter

import numpy as np
from astropy.convolution import convolve
from astropy.modeling.models import Gaussian2D
from astropy.stats import gaussian_fwhm_to_sigma
from bench_helpers import (format_sweep_cells, parse_thread_counts,
                           print_environment, time_best)

from photutils.datasets import make_wcs
from photutils.segmentation import (SegmentationImage, SourceCatalog,
                                    SourceFinder, deblend_sources,
                                    detect_sources, detect_threshold,
                                    make_2dgaussian_kernel)
from photutils.utils import circular_footprint
from photutils.utils._optional_deps import HAS_RASTERIO, HAS_SHAPELY

FWHM = 4.0
THRESHOLD = 5.0
N_PIXELS = 10


def make_source_image(n_sources, *, spacing=25, fwhm=FWHM, offset=6.0,
                      noise_std=1.0, seed=0):
    """
    Return an image with a grid of blended Gaussian-source pairs.

    Each grid cell contains a pair of overlapping Gaussian sources
    separated by ``offset`` pixels, so that each detected segment can
    be deblended into two sources.

    Parameters
    ----------
    n_sources : int
        The total number of Gaussian sources. The number of grid
        cells is ``ceil(n_sources / 2)``.

    spacing : int, optional
        The grid cell size in pixels.

    fwhm : float, optional
        The FWHM of the Gaussian sources in pixels.

    offset : float, optional
        The separation of the two sources within a cell in pixels.

    noise_std : float, optional
        The standard deviation of the Gaussian noise.

    seed : int, optional
        The random number generator seed.

    Returns
    -------
    data : 2D `~numpy.ndarray`
        The image.
    """
    n_cells = (n_sources + 1) // 2
    n_grid = int(np.ceil(np.sqrt(n_cells)))
    size = n_grid * spacing
    rng = np.random.default_rng(seed)
    data = rng.normal(0.0, noise_std, (size, size))

    sigma = fwhm * gaussian_fwhm_to_sigma
    yy, xx = np.mgrid[0:spacing, 0:spacing]
    cen = (spacing - 1) / 2.0
    half = offset / 2.0
    for i in range(n_sources):
        cell, pair = divmod(i, 2)
        gy, gx = divmod(cell, n_grid)
        sign = 1.0 if pair == 0 else -1.0
        amplitude = 100.0 if pair == 0 else 60.0
        xc = cen + sign * half + rng.uniform(-0.5, 0.5)
        yc = cen + sign * half + rng.uniform(-0.5, 0.5)
        model = Gaussian2D(amplitude, xc, yc, sigma, sigma)
        x0 = gx * spacing
        y0 = gy * spacing
        data[y0:y0 + spacing, x0:x0 + spacing] += model(xx, yy)

    return data


def make_inputs(n_sources, *, seed=0):
    """
    Return the image, convolved image, and segmentation image.

    Parameters
    ----------
    n_sources : int
        The total number of Gaussian sources.

    seed : int, optional
        The random number generator seed.

    Returns
    -------
    data : 2D `~numpy.ndarray`
        The image.

    convolved_data : 2D `~numpy.ndarray`
        The convolved image.

    segm : `~photutils.segmentation.SegmentationImage`
        The segmentation image from detect_sources.
    """
    data = make_source_image(n_sources, seed=seed)
    kernel = make_2dgaussian_kernel(3.0, size=5)
    convolved_data = convolve(data, kernel)
    segm = detect_sources(convolved_data, THRESHOLD, N_PIXELS)
    return data, convolved_data, segm


def bench_detect(*, n_sources=1000, repeats=3, seed=0):
    """
    Benchmark detect_threshold and detect_sources.

    Parameters
    ----------
    n_sources : int, optional
        The total number of Gaussian sources in the image.

    repeats : int, optional
        The number of repeats for each timing (best time is kept).

    seed : int, optional
        The random number generator seed.
    """
    data = make_source_image(n_sources, seed=seed)
    kernel = make_2dgaussian_kernel(3.0, size=5)
    convolved_data = convolve(data, kernel)

    print(f'\n== detect_threshold / detect_sources ({n_sources} sources, '
          f'{data.shape[0]}x{data.shape[1]} image) ==')
    print(f'{"benchmark":>34}{"time":>12}{"n_labels":>10}')

    bench = partial(detect_threshold, data, 2.0)
    t_best = time_best(bench, repeats=repeats)
    print(f'{"detect_threshold(n_sigma=2)":>34}{f"{t_best:.4f}s":>12}'
          f'{"n/a":>10}')

    for connectivity in (4, 8):
        segm = detect_sources(convolved_data, THRESHOLD, N_PIXELS,
                              connectivity=connectivity)
        bench = partial(detect_sources, convolved_data, THRESHOLD,
                        N_PIXELS, connectivity=connectivity)
        t_best = time_best(bench, repeats=repeats)
        name = f'detect_sources(connectivity={connectivity})'
        print(f'{name:>34}{f"{t_best:.4f}s":>12}{segm.n_labels:>10}')


def bench_deblend(*, n_sources=1000, process_counts=(1, 4), repeats=3,
                  seed=0):
    """
    Benchmark deblend_sources across modes and process counts.

    Parameters
    ----------
    n_sources : int, optional
        The total number of Gaussian sources in the image.

    process_counts : tuple of int, optional
        The n_processes values (for mode='exponential').

    repeats : int, optional
        The number of repeats for each timing (best time is kept).

    seed : int, optional
        The random number generator seed.
    """
    data, convolved_data, segm = make_inputs(n_sources, seed=seed)

    print(f'\n== deblend_sources ({n_sources} sources, '
          f'{segm.n_labels} segments, {data.shape[0]}x{data.shape[1]} '
          'image) ==')
    print(f'{"benchmark":>36}{"time":>12}{"n_labels":>10}')

    for mode in ('linear', 'exponential', 'sinh'):
        segm_deblended = deblend_sources(convolved_data, segm, N_PIXELS,
                                         mode=mode)
        bench = partial(deblend_sources, convolved_data, segm, N_PIXELS,
                        mode=mode)
        t_best = time_best(bench, repeats=repeats)
        name = f'mode={mode}'
        print(f'{name:>36}{f"{t_best:.4f}s":>12}'
              f'{segm_deblended.n_labels:>10}')

    for n_processes in process_counts:
        if n_processes == 1:
            continue
        bench = partial(deblend_sources, convolved_data, segm, N_PIXELS,
                        mode='exponential', n_processes=n_processes)
        t_best = time_best(bench, repeats=repeats)
        name = f'mode=exponential, n_processes={n_processes}'
        print(f'{name:>36}{f"{t_best:.4f}s":>12}{"":>10}')


def bench_finder(*, n_sources=1000, repeats=3, seed=0):
    """
    Benchmark the SourceFinder class (detection plus deblending).

    Parameters
    ----------
    n_sources : int, optional
        The total number of Gaussian sources in the image.

    repeats : int, optional
        The number of repeats for each timing (best time is kept).

    seed : int, optional
        The random number generator seed.
    """
    data = make_source_image(n_sources, seed=seed)
    kernel = make_2dgaussian_kernel(3.0, size=5)
    convolved_data = convolve(data, kernel)

    print(f'\n== SourceFinder ({n_sources} sources, '
          f'{data.shape[0]}x{data.shape[1]} image) ==')
    print(f'{"benchmark":>24}{"time":>12}{"n_labels":>10}')

    for deblend in (False, True):
        finder = SourceFinder(n_pixels=N_PIXELS, deblend=deblend)
        segm = finder(convolved_data, THRESHOLD)
        bench = partial(finder, convolved_data, THRESHOLD)
        t_best = time_best(bench, repeats=repeats)
        name = f'deblend={deblend}'
        print(f'{name:>24}{f"{t_best:.4f}s":>12}{segm.n_labels:>10}')


def bench_segmentation_image(*, n_sources=1000, repeats=3, seed=0):
    """
    Benchmark SegmentationImage operations.

    Each timing constructs a fresh SegmentationImage so that cached
    properties are recomputed and mutating methods start from the same
    state.

    Parameters
    ----------
    n_sources : int, optional
        The total number of Gaussian sources in the image.

    repeats : int, optional
        The number of repeats for each timing (best time is kept).

    seed : int, optional
        The random number generator seed.
    """
    _, _, segm = make_inputs(n_sources, seed=seed)
    segm_data = segm.data
    # Non-consecutive labels for the relabel benchmark
    segm_data_gaps = segm_data * 2

    print(f'\n== SegmentationImage operations ({segm.n_labels} segments, '
          f'{segm_data.shape[0]}x{segm_data.shape[1]} image) ==')
    print(f'{"benchmark":>36}{"time":>12}')

    footprint = circular_footprint(radius=5)

    def _run_properties():
        segm = SegmentationImage(segm_data.copy())
        _ = segm.slices, segm.areas, segm.bbox

    def _run_relabel():
        SegmentationImage(segm_data_gaps.copy()).relabel_consecutive()

    def _run_remove_border():
        segm = SegmentationImage(segm_data.copy())
        segm.remove_border_labels(border_width=10)

    def _run_source_mask_square():
        SegmentationImage(segm_data).make_source_mask(size=11)

    def _run_source_mask_circular():
        SegmentationImage(segm_data).make_source_mask(footprint=footprint)

    benchmarks = [
        ('labels + slices + areas + bbox', _run_properties),
        ('relabel_consecutive', _run_relabel),
        ('remove_border_labels(10)', _run_remove_border),
        ('make_source_mask(size=11)', _run_source_mask_square),
        ('make_source_mask(circular r=5)', _run_source_mask_circular),
    ]

    if HAS_RASTERIO and HAS_SHAPELY:
        def _run_polygons():
            _ = SegmentationImage(segm_data).polygons

        benchmarks.append(('polygons', _run_polygons))

    for name, func in benchmarks:
        t_best = time_best(func, repeats=repeats)
        print(f'{name:>36}{f"{t_best:.4f}s":>12}')


def bench_catalog(*, n_sources=1000, repeats=3, seed=0):
    """
    Benchmark all SourceCatalog property calculations.

    Every property listed by the SourceCatalog ``properties``
    attribute is timed, followed by the method-based measurements
    (flux_radius, circular_photometry, kron_photometry, and
    to_table). The catalog is constructed with convolved data, error,
    background, and WCS inputs so that all properties compute real
    values.

    Each timing constructs a fresh SourceCatalog so that cached
    properties are recomputed; the reported times therefore include
    the computation of any dependent properties (e.g., centroid_win
    includes the Kron flux and flux_radius it depends on).

    Parameters
    ----------
    n_sources : int, optional
        The total number of Gaussian sources in the image.

    repeats : int, optional
        The number of repeats for each timing (best time is kept).

    seed : int, optional
        The random number generator seed.
    """
    data, convolved_data, segm = make_inputs(n_sources, seed=seed)
    error = np.ones_like(data)
    background = np.full(data.shape, 0.1)
    wcs = make_wcs(data.shape)

    def _make_catalog():
        return SourceCatalog(data, segm, convolved_data=convolved_data,
                             error=error, background=background, wcs=wcs)

    catalog = _make_catalog()  # warm up shared segmentation-image caches

    print(f'\n== SourceCatalog ({segm.n_labels} segments, '
          f'{data.shape[0]}x{data.shape[1]} image; times include '
          'dependent properties) ==')
    print(f'{"benchmark":>32}{"time":>12}')

    benchmarks = [('constructor', lambda _cat: None)]
    benchmarks.extend((name, attrgetter(name))
                      for name in catalog.properties)
    benchmarks.extend([
        ('flux_radius(0.5)', lambda cat: cat.flux_radius(0.5)),
        ('circular_photometry(5)',
         lambda cat: cat.circular_photometry(5.0)),
        ('kron_photometry((2.5, 1.4))',
         lambda cat: cat.kron_photometry((2.5, 1.4))),
        ('to_table (default columns)', lambda cat: cat.to_table()),
    ])

    for name, func in benchmarks:
        def _bench(func=func):
            func(_make_catalog())

        t_best = time_best(_bench, repeats=repeats)
        print(f'{name:>32}{f"{t_best:.4f}s":>12}')


def run_catalog_concurrent(make_catalog, n_calls, n_threads):
    """
    Run concurrent SourceCatalog measurement jobs.

    Parameters
    ----------
    make_catalog : callable
        The zero-argument callable returning a fresh SourceCatalog.

    n_calls : int
        The total number of jobs.

    n_threads : int
        The number of worker threads.
    """
    def _job():
        make_catalog().to_table()

    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        futures = [executor.submit(_job) for _ in range(n_calls)]
        for future in futures:
            future.result()


def bench_catalog_threads(*, n_sources=1000, n_calls=8,
                          thread_counts=(1, 2, 4, 8), repeats=3, seed=0):
    """
    Benchmark concurrent SourceCatalog measurement runs.

    Each job constructs a fresh SourceCatalog over shared input arrays
    and computes the default to_table columns.

    Parameters
    ----------
    n_sources : int, optional
        The total number of Gaussian sources in the image.

    n_calls : int, optional
        The total number of catalog jobs per timing.

    thread_counts : tuple of int, optional
        The numbers of worker threads.

    repeats : int, optional
        The number of repeats for each timing (best time is kept).

    seed : int, optional
        The random number generator seed.
    """
    data, convolved_data, segm = make_inputs(n_sources, seed=seed)
    error = np.ones_like(data)

    def _make_catalog():
        return SourceCatalog(data, segm, convolved_data=convolved_data,
                             error=error)

    _make_catalog().to_table()  # warm up

    times = []
    for n_threads in thread_counts:
        bench = partial(run_catalog_concurrent, _make_catalog, n_calls,
                        n_threads)
        times.append(time_best(bench, repeats=repeats))

    print(f'\n== SourceCatalog thread scaling ({segm.n_labels} segments, '
          f'{data.shape[0]}x{data.shape[1]} image, {n_calls} concurrent '
          'to_table jobs) ==')
    header = ''.join(f'{f"{n} thr":>18}' for n in thread_counts)
    print(header)
    print(''.join(f'{cell:>18}' for cell in format_sweep_cells(times)))


def bench_catalog_n_threads(*, n_sources=1000, thread_counts=(1, 2, 4, 8),
                            repeats=3, seed=0):
    """
    Benchmark the SourceCatalog n_threads keyword.

    Each timing constructs a fresh SourceCatalog with the given
    ``n_threads`` and, outside the timed region, computes the
    prerequisites of the measurement (e.g., the isophotal centroid
    and shape parameters for the Kron radius, or the Kron flux for
    the flux radius), so that only the measurement itself is timed.
    The default ``to_table`` columns are timed from a cold catalog,
    so that row shows the end-to-end effect.

    Parameters
    ----------
    n_sources : int, optional
        The total number of Gaussian sources in the image.

    thread_counts : tuple of int, optional
        The ``n_threads`` values.

    repeats : int, optional
        The number of repeats for each timing (best time is kept).

    seed : int, optional
        The random number generator seed.
    """
    data, convolved_data, segm = make_inputs(n_sources, seed=seed)
    error = np.ones_like(data)

    def _make_catalog(n_threads):
        return SourceCatalog(data, segm, convolved_data=convolved_data,
                             error=error, n_threads=n_threads)

    def _time_measurement(func, n_threads, *, warm=()):
        best = np.inf
        for _ in range(repeats):
            cat = _make_catalog(n_threads)
            for item in warm:
                if isinstance(item, str):
                    getattr(cat, item)
                else:
                    item(cat)
            t0 = time.perf_counter()
            func(cat)
            best = min(best, time.perf_counter() - t0)
        return best

    _make_catalog(1).to_table()  # warm up

    shape = ('centroid', 'semimajor_axis', 'semiminor_axis',
             'orientation', 'ellipse_cxx', 'ellipse_cxy', 'ellipse_cyy')
    half_light = [lambda cat: cat.flux_radius(0.5)]
    benchmarks = [
        ('moments', attrgetter('moments'), ('bbox_xmin',)),
        ('moments_central', attrgetter('moments_central'),
         ('cutout_centroid',)),
        ('centroid_err', attrgetter('centroid_err'), ('covariance',)),
        ('segment_flux (+ error)', attrgetter('segment_flux'),
         ('bbox_xmin',)),
        ('min_value_xindex', attrgetter('min_value_xindex'),
         ('bbox_xmin',)),
        ('perimeter', attrgetter('perimeter'), ('bbox_xmin',)),
        ('centroid_quad', attrgetter('centroid_quad'), ('bbox_xmin',)),
        ('circular_photometry(5)',
         lambda cat: cat.circular_photometry(5.0), ('centroid',)),
        ('kron_radius', attrgetter('kron_radius'), shape),
        ('kron_flux (incl. kron_radius)', attrgetter('kron_flux'), shape),
        ('kron_photometry((2.5, 1.4))',
         lambda cat: cat.kron_photometry((2.5, 1.4)), ('kron_radius',)),
        ('flux_radius(0.5)', lambda cat: cat.flux_radius(0.5),
         ('kron_flux',)),
        ('centroid_win', attrgetter('centroid_win'), half_light),
        ('to_table (default columns)', lambda cat: cat.to_table(), ()),
    ]

    print(f'\n== SourceCatalog n_threads scaling ({segm.n_labels} '
          f'segments, {data.shape[0]}x{data.shape[1]} image; '
          'prerequisites precomputed except for to_table) ==')
    header = f'{"measurement":>32}'
    header += ''.join(f'{f"n={n}":>18}' for n in thread_counts)
    print(header)
    for name, func, warm in benchmarks:
        times = [_time_measurement(func, n_threads, warm=warm)
                 for n_threads in thread_counts]
        cells = ''.join(f'{cell:>18}' for cell in format_sweep_cells(times))
        print(f'{name:>32}{cells}')


def parse_int_list(text):
    """
    Parse a comma-separated list of positive integers.

    Parameters
    ----------
    text : str
        The comma-separated integers (e.g., ``'1,4'``).

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
    Run the photutils.segmentation benchmarks.
    """
    parser = argparse.ArgumentParser(
        description='Benchmarks for the photutils.segmentation subpackage.')
    parser.add_argument('--n-sources', type=int, default=1000,
                        help='total number of Gaussian sources in the '
                             'image (default: %(default)s)')
    parser.add_argument('--n-processes', type=parse_int_list, default=[1, 4],
                        help='comma-separated n_processes values for the '
                             'deblending benchmark (default: 1,4)')
    parser.add_argument('--n-calls', type=int, default=8,
                        help='number of concurrent catalog jobs for the '
                             'thread-scaling benchmark '
                             '(default: %(default)s)')
    parser.add_argument('--threads', type=parse_thread_counts,
                        default=[1, 2, 4, 8],
                        help='comma-separated thread counts for the '
                             'thread-scaling and n_threads benchmarks '
                             '(default: 1,2,4,8)')
    parser.add_argument('--repeats', type=int, default=3,
                        help='number of repeats per timing; the best '
                             'time is reported (default: %(default)s)')
    parser.add_argument('--seed', type=int, default=0,
                        help='random number generator seed '
                             '(default: %(default)s)')
    parser.add_argument('--which', default='all',
                        choices=['all', 'detect', 'deblend', 'finder',
                                 'segmimage', 'catalog', 'n-threads',
                                 'threads'],
                        help='which benchmark to run '
                             '(default: %(default)s)')
    args = parser.parse_args()

    print_environment()

    if args.which in ('all', 'detect'):
        bench_detect(n_sources=args.n_sources, repeats=args.repeats,
                     seed=args.seed)
    if args.which in ('all', 'deblend'):
        bench_deblend(n_sources=args.n_sources,
                      process_counts=args.n_processes,
                      repeats=args.repeats, seed=args.seed)
    if args.which in ('all', 'finder'):
        bench_finder(n_sources=args.n_sources, repeats=args.repeats,
                     seed=args.seed)
    if args.which in ('all', 'segmimage'):
        bench_segmentation_image(n_sources=args.n_sources,
                                 repeats=args.repeats, seed=args.seed)
    if args.which in ('all', 'catalog'):
        bench_catalog(n_sources=args.n_sources, repeats=args.repeats,
                      seed=args.seed)
    if args.which in ('all', 'n-threads'):
        bench_catalog_n_threads(n_sources=args.n_sources,
                                thread_counts=args.threads,
                                repeats=args.repeats, seed=args.seed)
    if args.which in ('all', 'threads'):
        bench_catalog_threads(n_sources=args.n_sources,
                              n_calls=args.n_calls,
                              thread_counts=args.threads,
                              repeats=args.repeats, seed=args.seed)


if __name__ == '__main__':
    main()
