#!/usr/bin/env python3
# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Benchmarks for source deblending (deblend_sources).

The benchmarks cover the two axes along which deblending is known to
be slow:

* many sources: an image with a grid of blended Gaussian-source
  pairs, sweeping the number of sources (per-source Python and numpy
  call overhead dominates)

* large sources: a single connected segment made of a broad Gaussian
  envelope with superposed peaks, sweeping the segment size and the
  number of peaks (per-level full-cutout work and the iterative
  watershed contrast loop dominate)

A per-stage breakdown of the single-source deblending pipeline
(multithresholding, marker building, watershed, contrast loop) and a
cProfile mode are also provided for bottleneck analysis.

Run ``python benchmarks/bench_deblend.py --help`` to see the
available options.
"""

import argparse
import cProfile
import pstats
import warnings
from functools import partial

import numpy as np
from astropy.modeling.models import Gaussian2D
from astropy.stats import gaussian_fwhm_to_sigma
from bench_helpers import print_environment, time_best
from bench_segmentation import N_PIXELS, THRESHOLD, make_inputs

from photutils.segmentation import detect_sources
from photutils.segmentation._deblend_reference import _SingleSourceDeblender
from photutils.segmentation.deblend import _DeblendParams, deblend_sources
from photutils.segmentation.utils import _make_binary_structure
from photutils.utils.exceptions import DeblendWarning

BLEND_THRESHOLD = 0.5
ENVELOPE_AMPLITUDE = 5.0
PEAK_FWHM = 8.0


def make_blended_image(size, n_peaks, *, amp_range=(3.0, 100.0), seed=0):
    """
    Return an image containing a single large blended source.

    The source is a broad Gaussian envelope with ``n_peaks`` compact
    Gaussian peaks superposed on it, so that detection at
    ``BLEND_THRESHOLD`` yields one large connected segment that
    deblending must split.

    Parameters
    ----------
    size : int
        The image size; the image is ``(size, size)``. The envelope
        sigma is ``size / 6``, so the segment area scales with the
        image area.

    n_peaks : int
        The number of compact peaks placed within one envelope sigma
        of the center.

    amp_range : tuple of float, optional
        The (min, max) peak amplitudes. The amplitudes are
        logarithmically spaced over this range.

    seed : int, optional
        The random number generator seed.

    Returns
    -------
    data : 2D `~numpy.ndarray`
        The image.
    """
    rng = np.random.default_rng(seed)
    data = rng.normal(0.0, 0.01, (size, size))

    cen = (size - 1) / 2.0
    sigma_env = size / 6.0
    yy, xx = np.mgrid[0:size, 0:size]
    envelope = Gaussian2D(ENVELOPE_AMPLITUDE, cen, cen, sigma_env,
                          sigma_env)
    data += envelope(xx, yy)

    sigma = PEAK_FWHM * gaussian_fwhm_to_sigma
    half = int(np.ceil(4.0 * sigma))
    yy_cut, xx_cut = np.mgrid[0:2 * half + 1, 0:2 * half + 1]
    amplitudes = np.geomspace(amp_range[1], amp_range[0], n_peaks)
    radii = sigma_env * np.sqrt(rng.uniform(0.0, 1.0, n_peaks))
    angles = rng.uniform(0.0, 2.0 * np.pi, n_peaks)
    for amplitude, radius, angle in zip(amplitudes, radii, angles,
                                        strict=True):
        xc = cen + radius * np.cos(angle)
        yc = cen + radius * np.sin(angle)
        x0 = int(xc) - half
        y0 = int(yc) - half
        model = Gaussian2D(amplitude, xc - x0, yc - y0, sigma, sigma)
        data[y0:y0 + 2 * half + 1,
             x0:x0 + 2 * half + 1] += model(xx_cut, yy_cut)

    return data


def make_blended_inputs(size, n_peaks, *, amp_range=(3.0, 100.0), seed=0):
    """
    Return the image and segmentation image for a single large
    blended source.

    Parameters
    ----------
    size : int
        The image size; the image is ``(size, size)``.

    n_peaks : int
        The number of compact peaks.

    amp_range : tuple of float, optional
        The (min, max) peak amplitudes.

    seed : int, optional
        The random number generator seed.

    Returns
    -------
    data : 2D `~numpy.ndarray`
        The image.

    segm : `~photutils.segmentation.SegmentationImage`
        The segmentation image containing a single label.
    """
    data = make_blended_image(size, n_peaks, amp_range=amp_range,
                              seed=seed)
    segm = detect_sources(data, BLEND_THRESHOLD, N_PIXELS)
    if segm.n_labels != 1:
        msg = (f'expected a single blended segment, got '
               f'{segm.n_labels}')
        raise ValueError(msg)
    return data, segm


def n_fallbacks(segm):
    """
    Return the number of deblending mode fallbacks recorded in a
    deblended segmentation image.

    Parameters
    ----------
    segm : `~photutils.segmentation.SegmentationImage`
        The deblended segmentation image.

    Returns
    -------
    result : int
        The total number of input labels whose deblending mode fell
        back to linear.
    """
    return (len(segm.info.get('nonposmin_labels', ()))
            + len(segm.info.get('n_markers_labels', ())))


def bench_many_sources(*, n_sources_sweep=(500, 1000, 2000, 4000),
                       repeats=3, seed=0):
    """
    Benchmark deblend_sources versus the number of sources.

    Each image contains a grid of blended Gaussian-source pairs, so
    the number of segments is half the number of sources and every
    segment deblends into two sources. The detect_sources time is
    included as a reference point.

    Parameters
    ----------
    n_sources_sweep : tuple of int, optional
        The numbers of Gaussian sources.

    repeats : int, optional
        The number of repeats for each timing (best time is kept).

    seed : int, optional
        The random number generator seed.
    """
    print('\n== deblend_sources: many small sources ==')
    header = (f'{"benchmark":>24}{"segments":>10}{"time":>12}'
              f'{"ms/segment":>12}{"n_labels":>10}')

    for n_sources in n_sources_sweep:
        _, data, segm = make_inputs(n_sources, seed=seed)

        print(f'\n-- {n_sources} sources, {segm.n_labels} segments, '
              f'{data.shape[0]}x{data.shape[1]} image --')
        print(header)

        bench = partial(detect_sources, data, THRESHOLD, N_PIXELS)
        t_best = time_best(bench, repeats=repeats)
        per_segment = 1000.0 * t_best / segm.n_labels
        print(f'{"detect_sources (ref)":>24}{segm.n_labels:>10}'
              f'{f"{t_best:.4f}s":>12}{per_segment:>12.3f}'
              f'{segm.n_labels:>10}')

        for mode in ('linear', 'exponential', 'sinh'):
            bench = partial(deblend_sources, data, segm, N_PIXELS,
                            mode=mode)
            segm_deblended = bench()
            t_best = time_best(bench, repeats=repeats)
            per_segment = 1000.0 * t_best / segm.n_labels
            name = f'mode={mode}'
            print(f'{name:>24}{segm.n_labels:>10}{f"{t_best:.4f}s":>12}'
                  f'{per_segment:>12.3f}{segm_deblended.n_labels:>10}')


def bench_large_source(*, size_sweep=(250, 500, 1000, 2000), n_peaks=8,
                       repeats=3, seed=0):
    """
    Benchmark deblend_sources versus the size of a single segment.

    Each image contains one connected segment (a Gaussian envelope
    with bright peaks) whose area scales with the image area. The
    contrast is set to 0 so that no markers are removed and the sweep
    measures the pure multithreshold plus watershed scaling.

    Parameters
    ----------
    size_sweep : tuple of int, optional
        The image sizes.

    n_peaks : int, optional
        The number of compact peaks in the segment.

    repeats : int, optional
        The number of repeats for each timing (best time is kept).

    seed : int, optional
        The random number generator seed.
    """
    print(f'\n== deblend_sources: single large segment ({n_peaks} peaks, '
          'contrast=0) ==')
    print(f'{"benchmark":>28}{"seg area":>10}{"time":>12}{"n_labels":>10}')

    for size in size_sweep:
        data, segm = make_blended_inputs(size, n_peaks,
                                         amp_range=(50.0, 100.0),
                                         seed=seed)
        area = int(segm.areas[0])
        for mode in ('linear', 'exponential'):
            bench = partial(deblend_sources, data, segm, N_PIXELS,
                            mode=mode, contrast=0.0)
            segm_deblended = bench()
            t_best = time_best(bench, repeats=repeats)
            name = f'size={size}, mode={mode}'
            print(f'{name:>28}{area:>10}{f"{t_best:.4f}s":>12}'
                  f'{segm_deblended.n_labels:>10}')


def bench_many_peaks(*, size=1000, n_peaks_sweep=(10, 25, 50, 100),
                     contrast_sweep=(0.0, 0.001, 0.01, 0.03),
                     repeats=3, seed=0):
    """
    Benchmark deblend_sources versus the number of peaks in one
    segment.

    The contrast criterion applies to the flux in each watershed
    basin (which includes a share of the envelope flux), so larger
    contrast values remove more markers. Each removal iteration
    re-runs the watershed over the full segment, so comparing the
    contrast=0 row (a single watershed call) to the larger-contrast
    rows isolates the cost of the iterative marker-removal loop. The
    number of watershed calls is at most the difference between the
    contrast=0 n_labels and the row's n_labels, plus one (batched
    removal may use fewer).

    Parameters
    ----------
    size : int, optional
        The image size.

    n_peaks_sweep : tuple of int, optional
        The numbers of peaks.

    contrast_sweep : tuple of float, optional
        The contrast values.

    repeats : int, optional
        The number of repeats for each timing (best time is kept).

    seed : int, optional
        The random number generator seed.
    """
    print(f'\n== deblend_sources: many peaks in one segment '
          f'({size}x{size} image, mode=exponential) ==')
    print(f'{"benchmark":>32}{"time":>12}{"n_labels":>10}'
          f'{"fallbacks":>10}')

    for n_peaks in n_peaks_sweep:
        data, segm = make_blended_inputs(size, n_peaks, seed=seed)
        for contrast in contrast_sweep:
            bench = partial(deblend_sources, data, segm, N_PIXELS,
                            contrast=contrast)
            segm_deblended = bench()
            t_best = time_best(bench, repeats=repeats)
            name = f'n_peaks={n_peaks}, contrast={contrast}'
            print(f'{name:>32}{f"{t_best:.4f}s":>12}'
                  f'{segm_deblended.n_labels:>10}'
                  f'{n_fallbacks(segm_deblended):>10}')


def bench_stages(*, size=1000, n_peaks=25, mode='exponential',
                 contrast=0.001, n_levels=32, repeats=3, seed=0):
    """
    Benchmark the stages of the single-source deblending pipeline.

    The stages are timed on the segment cutout of a single large
    blended source, using the private _SingleSourceDeblender class:

    * constructor: the segment mask and min/max/sum reductions
    * multithreshold: the ``n_levels`` per-level detection passes of
      the reference marker construction
    * make_markers: the multithreshold levels, the level quantization,
      and the compiled component-tree kernel
    * watershed: a single watershed call over the cutout
    * apply_watershed: the watershed contrast loop (one watershed
      call per removed marker)
    * deblend_source: the full pipeline

    Parameters
    ----------
    size : int, optional
        The image size.

    n_peaks : int, optional
        The number of compact peaks in the segment.

    mode : str, optional
        The mode used for spacing the multithreshold levels.

    contrast : float, optional
        The deblending contrast criterion.

    n_levels : int, optional
        The number of multithreshold levels.

    repeats : int, optional
        The number of repeats for each timing (best time is kept).

    seed : int, optional
        The random number generator seed.
    """
    data, segm = make_blended_inputs(size, n_peaks, seed=seed)
    label = segm.labels[0]
    slc = segm.slices[0]
    cutout = data[slc]
    segment_cutout = segm.data[slc]
    footprint = _make_binary_structure(2, 8)
    params = _DeblendParams(N_PIXELS, footprint, n_levels, contrast, mode)

    def _make_deblender():
        return _SingleSourceDeblender(cutout, segment_cutout, label,
                                      params)

    deblender = _make_deblender()
    markers = deblender.make_markers()
    n_markers = len(np.unique(markers[markers > 0]))
    final = deblender.apply_watershed(markers)
    n_final = len(np.unique(final[final > 0]))
    n_watershed = n_markers - n_final + 1

    from photutils.segmentation._deblend_watershed import deblend_watershed

    data_neg = np.ascontiguousarray(-cutout, dtype=np.float64)
    connectivity = 8 if footprint[0, 0] else 4

    def _run_constructor():
        _make_deblender()

    def _run_multithreshold():
        _make_deblender().multithreshold()

    def _run_make_markers():
        _make_deblender().make_markers()

    def _run_watershed():
        deblend_watershed(data_neg, markers, deblender.segment_mask,
                          connectivity)

    def _run_apply_watershed():
        deblender.apply_watershed(markers)

    def _run_deblend_source():
        _make_deblender().deblend_source()

    print(f'\n== single-source pipeline stages ({size}x{size} image, '
          f'cutout {cutout.shape[0]}x{cutout.shape[1]}, '
          f'{n_peaks} peaks, mode={mode}, contrast={contrast}, '
          f'n_levels={n_levels}) ==')
    print(f'{n_markers} markers, {n_final} final labels, '
          f'<={n_watershed} watershed calls')
    print(f'{"stage":>36}{"time":>12}')

    benchmarks = [
        ('constructor (mask + min/max/sum)', _run_constructor),
        (f'multithreshold ({n_levels} per-level detects)',
         _run_multithreshold),
        ('make_markers (component tree)', _run_make_markers),
        ('watershed (single call)', _run_watershed),
        ('apply_watershed (contrast loop)', _run_apply_watershed),
        ('deblend_source (full)', _run_deblend_source),
    ]
    for name, func in benchmarks:
        t_best = time_best(func, repeats=repeats)
        print(f'{name:>36}{f"{t_best:.4f}s":>12}')


def profile_case(name, func, *, limit=20):
    """
    Profile a callable with cProfile and print the top functions.

    Parameters
    ----------
    name : str
        The name of the profiled case.

    func : callable
        The zero-argument callable to profile.

    limit : int, optional
        The number of functions to print.
    """
    profiler = cProfile.Profile()
    profiler.enable()
    func()
    profiler.disable()

    print(f'\n== cProfile: {name} ==')
    stats = pstats.Stats(profiler)
    stats.sort_stats('tottime')
    stats.print_stats(limit)


def bench_profile(*, n_sources=2000, size=1000, n_peaks=25, seed=0):
    """
    Profile deblend_sources for the many-source and large-source
    scenarios.

    Parameters
    ----------
    n_sources : int, optional
        The number of sources for the many-source case.

    size : int, optional
        The image size for the large-source case.

    n_peaks : int, optional
        The number of peaks for the large-source case.

    seed : int, optional
        The random number generator seed.
    """
    _, data, segm = make_inputs(n_sources, seed=seed)
    profile_case(
        f'many small sources ({segm.n_labels} segments)',
        partial(deblend_sources, data, segm, N_PIXELS))

    data, segm = make_blended_inputs(size, n_peaks, seed=seed)
    profile_case(
        f'single large segment ({size}x{size}, {n_peaks} peaks)',
        partial(deblend_sources, data, segm, N_PIXELS))


def bench_threads(*, n_sources=4000, size=1000, n_peaks=25,
                  thread_counts=(1, 2, 4, 8), repeats=3, seed=0):
    """
    Benchmark deblend_sources thread scaling.

    Two regimes are timed: a field of many small blended pairs and a
    2x2 grid of large blended sources (whose per-source work is
    dominated by the GIL-releasing watershed and marker kernels).

    Parameters
    ----------
    n_sources : int, optional
        The total number of Gaussian sources for the many-source
        scene.

    size : int, optional
        The tile size for the large-source grid scene.

    n_peaks : int, optional
        The number of compact peaks per large source.

    thread_counts : tuple of int, optional
        The n_threads values.

    repeats : int, optional
        The number of repeats for each timing (best time is kept).

    seed : int, optional
        The random number generator seed.
    """
    from bench_helpers import format_sweep_cells

    _, data, segm = make_inputs(n_sources, seed=seed)
    tile = make_blended_image(size, n_peaks, seed=seed)
    grid = np.empty((2 * size, 2 * size))
    for iy in range(2):
        for ix in range(2):
            grid[iy * size:(iy + 1) * size,
                 ix * size:(ix + 1) * size] = tile
    segm_grid = detect_sources(grid, BLEND_THRESHOLD, N_PIXELS)

    scenes = [
        (f'{segm.n_labels} small segments', data, segm),
        (f'{segm_grid.n_labels} large segments', grid, segm_grid),
    ]
    print('\n== deblend_sources: thread scaling ==')
    header = ''.join(f'{f"{n} thr":>18}' for n in thread_counts)
    print(f'{"scene":>24}{header}')
    for name, scene_data, scene_segm in scenes:
        times = []
        for n_threads in thread_counts:
            bench = partial(deblend_sources, scene_data, scene_segm,
                            N_PIXELS, n_threads=n_threads)
            times.append(time_best(bench, repeats=repeats))
        cells = ''.join(f'{cell:>18}'
                        for cell in format_sweep_cells(times))
        print(f'{name:>24}{cells}')


def parse_int_list(text):
    """
    Parse a comma-separated list of positive integers.

    Parameters
    ----------
    text : str
        The comma-separated integers (e.g., ``'250,500,1000'``).

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
    Run the source deblending benchmarks.
    """
    parser = argparse.ArgumentParser(
        description='Benchmarks for source deblending.')
    parser.add_argument('--n-sources', type=parse_int_list,
                        default=[500, 1000, 2000, 4000],
                        help='comma-separated source counts for the '
                             'many-source benchmark '
                             '(default: 500,1000,2000,4000)')
    parser.add_argument('--sizes', type=parse_int_list,
                        default=[250, 500, 1000, 2000],
                        help='comma-separated image sizes for the '
                             'large-source benchmark '
                             '(default: 250,500,1000,2000)')
    parser.add_argument('--n-peaks', type=parse_int_list,
                        default=[10, 25, 50, 100],
                        help='comma-separated peak counts for the '
                             'many-peak benchmark '
                             '(default: 10,25,50,100)')
    parser.add_argument('--repeats', type=int, default=3,
                        help='number of repeats per timing; the best '
                             'time is reported (default: %(default)s)')
    parser.add_argument('--seed', type=int, default=0,
                        help='random number generator seed '
                             '(default: %(default)s)')
    parser.add_argument('--which', default='all',
                        choices=['all', 'many', 'large', 'peaks',
                                 'stages', 'threads', 'profile'],
                        help='which benchmark to run '
                             '(default: %(default)s)')
    args = parser.parse_args()

    warnings.filterwarnings('ignore', category=DeblendWarning)
    print_environment()

    if args.which in ('all', 'many'):
        bench_many_sources(n_sources_sweep=args.n_sources,
                           repeats=args.repeats, seed=args.seed)
    if args.which in ('all', 'large'):
        bench_large_source(size_sweep=args.sizes, repeats=args.repeats,
                           seed=args.seed)
    if args.which in ('all', 'peaks'):
        bench_many_peaks(n_peaks_sweep=args.n_peaks,
                         repeats=args.repeats, seed=args.seed)
    if args.which in ('all', 'stages'):
        bench_stages(repeats=args.repeats, seed=args.seed)
        bench_stages(contrast=0.03, repeats=args.repeats,
                     seed=args.seed)
    if args.which in ('all', 'threads'):
        bench_threads(repeats=args.repeats, seed=args.seed)
    if args.which == 'profile':
        bench_profile(seed=args.seed)


if __name__ == '__main__':
    main()
