#!/usr/bin/env python3
# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Benchmark and validation of photutils aperture photometry against SEP.

The script builds a synthetic scene of 2D Gaussian sources (with a
matching segmentation map) and, for a range of aperture shapes:

1. Validates that the photutils aperture-photometry entry points
   (``aperture_photometry``, ``AperturePhotometry``, and
   ``ApertureStats``) agree with each other for every ``mask_method``
   value ('none', 'mask', 'source_only', and 'correct'). The legacy
   ``aperture_photometry`` function does not support segmentation
   masking, so it is validated and benchmarked only for the
   scenarios without it.

2. Validates the photutils results against SEP for the shapes that
   SEP supports (circle, circular annulus, ellipse, and elliptical
   annulus). SEP exact mode (``subpix=0``) is compared against the
   photutils ``method='exact'`` code path.

3. Benchmarks the runtime of each photutils entry point and SEP
   across four masking scenarios: no masking, the ``mask`` keyword
   only, segmentation masking only, and ``mask`` plus segmentation
   masking.

4. Sweeps ``AperturePhotometry`` over the ``--n-threads`` counts
   against the single-threaded SEP baseline (no-masking scenario).

SEP segmentation masking maps to photutils as follows:
``mask_method='mask'`` corresponds to a positive SEP ``seg_id`` and
``mask_method='source_only'`` to a negative SEP ``seg_id``. The
photutils ``'correct'`` mask method has no SEP analogue and is only
cross-checked internally. SEP ``sum_ellipann`` has no ``segmap``
support, so segmentation scenarios are skipped for the elliptical
annulus when comparing to SEP.

Two SEP semantic differences are handled by the validation:

* SEP's ``mask`` array does not simply drop masked pixels: it
  rescales the sum by ``total_area / unmasked_area`` (it assumes
  masked pixels hold the aperture's average flux), while the
  photutils ``mask`` keyword drops masked pixels entirely. SEP value
  validation is therefore skipped for the ``mask``-keyword scenarios
  (their runtime is still benchmarked). SEP's segmentation masking
  does drop pixels and matches photutils.

* ``source_only`` on an annulus selects only the central
  target-source pixels, which generally lie outside the annulus,
  giving an empty aperture. ``ApertureStats`` reports an empty
  aperture as NaN while the other entry points report 0.0; the
  validation treats these as equivalent.

Run ``python benchmarks/bench_aperture_sep.py --help`` to see the
available options.
"""

import argparse
import sys

import numpy as np
from bench_helpers import parse_thread_counts, print_environment, time_best
from numpy.testing import assert_allclose

from photutils.aperture import (AperturePhotometry, ApertureStats,
                                CircularAnnulus, CircularAperture,
                                EllipticalAnnulus, EllipticalAperture,
                                PolygonAperture, RectangularAnnulus,
                                RectangularAperture, aperture_photometry)

try:
    import sep
    HAS_SEP = True
    SEP_IMPORT_ERROR = None
except ImportError as exc:
    HAS_SEP = False
    SEP_IMPORT_ERROR = str(exc)


def make_gaussian_scene(n_sources, shape, *, seed=0):
    """
    Build a synthetic scene of 2D Gaussian sources on a noisy
    background.

    Parameters
    ----------
    n_sources : int
        The number of Gaussian sources.

    shape : tuple of int
        The ``(ny, nx)`` shape of the image.

    seed : int, optional
        The random number generator seed.

    Returns
    -------
    data : 2D `~numpy.ndarray`
        The float64 image of summed Gaussians plus background noise.

    segm : 2D `~numpy.ndarray`
        The segmentation map, where each pixel is owned by the source
        (positive integer label) that contributes the most flux
        there, or 0 for the background.

    positions : 2D `~numpy.ndarray`
        The ``(x, y)`` source center positions, shape
        ``(n_sources, 2)``.

    labels : 1D `~numpy.ndarray`
        The per-source integer labels (``1 .. n_sources``).
    """
    rng = np.random.default_rng(seed)
    ny, nx = shape
    noise = 1.0
    data = rng.normal(0.0, noise, shape)
    segm = np.zeros(shape, dtype=np.intp)
    best = np.zeros(shape)  # owning (max) contribution per pixel

    margin = 20
    xc = rng.uniform(margin, nx - margin, n_sources)
    yc = rng.uniform(margin, ny - margin, n_sources)
    positions = np.column_stack([xc, yc])
    amps = rng.uniform(50.0, 500.0, n_sources)
    sigmas = rng.uniform(1.5, 3.0, n_sources)
    labels = np.arange(1, n_sources + 1)

    own_thresh = 5.0 * noise  # background stays unlabeled (label 0)
    for label, (x, y, amp, sig) in enumerate(
            zip(xc, yc, amps, sigmas, strict=True), start=1):
        half = int(np.ceil(4.0 * sig))
        x0, x1 = max(int(x) - half, 0), min(int(x) + half + 1, nx)
        y0, y1 = max(int(y) - half, 0), min(int(y) + half + 1, ny)
        yy, xx = np.mgrid[y0:y1, x0:x1]
        g = amp * np.exp(-((xx - x)**2 + (yy - y)**2) / (2.0 * sig**2))

        data[y0:y1, x0:x1] += g

        sub_best = best[y0:y1, x0:x1]
        sel = (g > sub_best) & (g > own_thresh)
        sub_best[sel] = g[sel]
        seg_sub = segm[y0:y1, x0:x1]
        seg_sub[sel] = label

    return np.ascontiguousarray(data), segm, positions, labels


def build_shapes():
    """
    Build the registry of aperture shapes to benchmark.

    Each entry provides a photutils aperture factory and, where SEP
    supports the shape, a SEP wrapper. SEP exact mode (``subpix=0``)
    is used so that the SEP results are directly comparable to the
    photutils ``method='exact'`` results.
    """
    r_circ = 6.0  # circle
    r_in, r_out = 6.0, 10.0  # circular annulus
    a_ell, b_ell, theta = 8.0, 4.0, 0.5  # ellipse
    # Elliptical annulus (SEP scales (a, b) by r_in/r_out)
    e_rin, e_rout = 0.6, 1.0
    a_in, a_out = a_ell * e_rin, a_ell * e_rout
    b_out = b_ell * e_rout
    # Rectangle and rectangular annulus (no SEP analogue)
    w, h = 12.0, 8.0
    w_in, w_out, h_out = 8.0, 14.0, 10.0
    # Polygon: regular hexagon of "radius" 7 (no SEP analogue)
    ang = np.linspace(0.0, 2.0 * np.pi, 6, endpoint=False)
    hexagon = np.column_stack([7.0 * np.cos(ang), 7.0 * np.sin(ang)])

    def sep_circle(data, x, y, **kwargs):
        """
        Run SEP exact circular-aperture photometry.

        Parameters
        ----------
        data : 2D `~numpy.ndarray`
            The image.

        x, y : 1D `~numpy.ndarray`
            The source center positions.

        **kwargs : dict
            Keyword arguments passed to the SEP function.

        Returns
        -------
        result : tuple of `~numpy.ndarray`
            The SEP ``(flux, flux_err, flag)`` arrays.
        """
        return sep.sum_circle(data, x, y, r_circ, subpix=0, **kwargs)

    def sep_circann(data, x, y, **kwargs):
        """
        Run SEP exact circular-annulus photometry.

        Parameters
        ----------
        data : 2D `~numpy.ndarray`
            The image.

        x, y : 1D `~numpy.ndarray`
            The source center positions.

        **kwargs : dict
            Keyword arguments passed to the SEP function.

        Returns
        -------
        result : tuple of `~numpy.ndarray`
            The SEP ``(flux, flux_err, flag)`` arrays.
        """
        return sep.sum_circann(data, x, y, r_in, r_out, subpix=0,
                               **kwargs)

    def sep_ellipse(data, x, y, **kwargs):
        """
        Run SEP exact elliptical-aperture photometry.

        Parameters
        ----------
        data : 2D `~numpy.ndarray`
            The image.

        x, y : 1D `~numpy.ndarray`
            The source center positions.

        **kwargs : dict
            Keyword arguments passed to the SEP function.

        Returns
        -------
        result : tuple of `~numpy.ndarray`
            The SEP ``(flux, flux_err, flag)`` arrays.
        """
        return sep.sum_ellipse(data, x, y, a_ell, b_ell, theta, 1.0,
                               subpix=0, **kwargs)

    def sep_ellipann(data, x, y, **kwargs):
        """
        Run SEP exact elliptical-annulus photometry.

        Parameters
        ----------
        data : 2D `~numpy.ndarray`
            The image.

        x, y : 1D `~numpy.ndarray`
            The source center positions.

        **kwargs : dict
            Keyword arguments passed to the SEP function.

        Returns
        -------
        result : tuple of `~numpy.ndarray`
            The SEP ``(flux, flux_err, flag)`` arrays.
        """
        return sep.sum_ellipann(data, x, y, a_ell, b_ell, theta, e_rin,
                                e_rout, subpix=0, **kwargs)

    return [
        {'name': 'circle',
         'aperture': lambda pos: CircularAperture(pos, r=r_circ),
         'sep': sep_circle, 'sep_segm': True},
        {'name': 'circ_annulus',
         'aperture': lambda pos: CircularAnnulus(pos, r_in=r_in,
                                                 r_out=r_out),
         'sep': sep_circann, 'sep_segm': True},
        {'name': 'ellipse',
         'aperture': lambda pos: EllipticalAperture(pos, a_ell, b_ell,
                                                    theta=theta),
         'sep': sep_ellipse, 'sep_segm': True},
        {'name': 'ellipse_annulus',
         'aperture': lambda pos: EllipticalAnnulus(pos, a_in, a_out,
                                                   b_out, theta=theta),
         'sep': sep_ellipann, 'sep_segm': False},
        {'name': 'rectangle',
         'aperture': lambda pos: RectangularAperture(pos, w, h,
                                                     theta=theta),
         'sep': None, 'sep_segm': False},
        {'name': 'rect_annulus',
         'aperture': lambda pos: RectangularAnnulus(pos, w_in, w_out,
                                                    h_out, theta=theta),
         'sep': None, 'sep_segm': False},
        {'name': 'polygon',
         'aperture': lambda pos: PolygonAperture(pos, hexagon),
         'sep': None, 'sep_segm': False},
    ]


def build_scenarios():
    """
    Build the four masking scenarios, each expanded over the
    applicable ``mask_method`` values.
    """
    return [
        {'name': 'no masking', 'use_mask': False, 'use_segm': False,
         'methods': ['none']},
        {'name': 'mask keyword only', 'use_mask': True,
         'use_segm': False, 'methods': ['none']},
        {'name': 'segmentation only', 'use_mask': False,
         'use_segm': True, 'methods': ['mask', 'source_only', 'correct']},
        {'name': 'mask + segmentation', 'use_mask': True,
         'use_segm': True, 'methods': ['mask', 'source_only', 'correct']},
    ]


def _phot_kwargs(scenario, method, segm, labels, maskarr):
    """
    Build the photutils keyword arguments for a masking scenario.

    Parameters
    ----------
    scenario : dict
        The masking scenario (an entry from ``build_scenarios``).

    method : str
        The segmentation ``mask_method``.

    segm : 2D `~numpy.ndarray`
        The segmentation map.

    labels : 1D `~numpy.ndarray`
        The per-source segmentation labels.

    maskarr : 2D `~numpy.ndarray` (bool)
        The pixel mask.

    Returns
    -------
    kwargs : dict
        The keyword arguments for the photutils entry points.
    """
    kwargs = {}
    if scenario['use_mask']:
        kwargs['mask'] = maskarr
    if scenario['use_segm']:
        kwargs['segmentation_image'] = segm
        kwargs['labels'] = labels
        kwargs['mask_method'] = method
    return kwargs


def run_legacy_photometry(data, aper, error, scenario, method, segm,
                          labels, maskarr):
    """
    Run the legacy ``aperture_photometry`` function.

    Parameters
    ----------
    data : 2D `~numpy.ndarray`
        The image.

    aper : `~photutils.aperture.Aperture`
        The aperture containing all source positions.

    error : 2D `~numpy.ndarray`
        The total error array.

    scenario : dict
        The masking scenario (an entry from ``build_scenarios``).

    method : str
        The segmentation ``mask_method``.

    segm : 2D `~numpy.ndarray`
        The segmentation map.

    labels : 1D `~numpy.ndarray`
        The per-source segmentation labels.

    maskarr : 2D `~numpy.ndarray` (bool)
        The pixel mask.

    Returns
    -------
    flux, flux_err : 1D `~numpy.ndarray`
        The per-source aperture sums and their errors.
    """
    kwargs = _phot_kwargs(scenario, method, segm, labels, maskarr)
    tbl = aperture_photometry(data, aper, error=error, method='exact',
                              **kwargs)
    return (np.asarray(tbl['aperture_sum']),
            np.asarray(tbl['aperture_sum_err']))


def run_aperture_photometry(data, aper, error, scenario, method, segm,
                            labels, maskarr, *, n_threads=1):
    """
    Run the ``AperturePhotometry`` class.

    Parameters
    ----------
    data : 2D `~numpy.ndarray`
        The image.

    aper : `~photutils.aperture.Aperture`
        The aperture containing all source positions.

    error : 2D `~numpy.ndarray`
        The total error array.

    scenario : dict
        The masking scenario (an entry from ``build_scenarios``).

    method : str
        The segmentation ``mask_method``.

    segm : 2D `~numpy.ndarray`
        The segmentation map.

    labels : 1D `~numpy.ndarray`
        The per-source segmentation labels.

    maskarr : 2D `~numpy.ndarray` (bool)
        The pixel mask.

    n_threads : int, optional
        The number of threads to use.

    Returns
    -------
    flux, flux_err : 1D `~numpy.ndarray`
        The per-source aperture fluxes and their errors.
    """
    kwargs = _phot_kwargs(scenario, method, segm, labels, maskarr)
    phot = AperturePhotometry(data, aper, error=error, method='exact',
                              n_threads=n_threads, **kwargs)
    return phot.flux, phot.flux_err


def run_aperture_stats(data, aper, error, scenario, method, segm,
                       labels, maskarr, *, n_threads=1):
    """
    Run the ``ApertureStats`` class (sum and sum error only).

    Parameters
    ----------
    data : 2D `~numpy.ndarray`
        The image.

    aper : `~photutils.aperture.Aperture`
        The aperture containing all source positions.

    error : 2D `~numpy.ndarray`
        The total error array.

    scenario : dict
        The masking scenario (an entry from ``build_scenarios``).

    method : str
        The segmentation ``mask_method``.

    segm : 2D `~numpy.ndarray`
        The segmentation map.

    labels : 1D `~numpy.ndarray`
        The per-source segmentation labels.

    maskarr : 2D `~numpy.ndarray` (bool)
        The pixel mask.

    n_threads : int, optional
        The number of threads to use.

    Returns
    -------
    flux, flux_err : 1D `~numpy.ndarray`
        The per-source aperture sums and their errors.
    """
    kwargs = _phot_kwargs(scenario, method, segm, labels, maskarr)
    stats = ApertureStats(data, aper, error=error, sum_method='exact',
                          n_threads=n_threads, **kwargs)
    return np.asarray(stats.sum), np.asarray(stats.sum_err)


def run_sep(shape, data, positions, error, scenario, method, segm32):
    """
    Run the SEP function for ``shape`` under the given scenario.

    Parameters
    ----------
    shape : dict
        The aperture shape (an entry from ``build_shapes``).

    data : 2D `~numpy.ndarray`
        The image.

    positions : 2D `~numpy.ndarray`
        The ``(x, y)`` source center positions.

    error : 2D `~numpy.ndarray`
        The total error array.

    scenario : dict
        The masking scenario (an entry from ``build_scenarios``).

    method : str
        The segmentation ``mask_method``.

    segm32 : 2D `~numpy.ndarray`
        The segmentation map as int32 (required by SEP).

    Returns
    -------
    flux, flux_err : 1D `~numpy.ndarray`
        The per-source aperture sums and their errors.
    """
    x = np.ascontiguousarray(positions[:, 0])
    y = np.ascontiguousarray(positions[:, 1])
    kwargs = {'err': error}
    if scenario['use_mask']:
        # SEP masks a pixel when mask > maskthresh (default 0.0)
        kwargs['mask'] = scenario['_maskarr']
    if scenario['use_segm']:
        kwargs['segmap'] = segm32
        seg_id = scenario['_labels'].astype(np.int32)
        if method == 'source_only':
            seg_id = -seg_id
        kwargs['seg_id'] = seg_id
    flux, flux_err, _ = shape['sep'](data, x, y, **kwargs)
    return np.asarray(flux), np.asarray(flux_err)


def _sanitize(arr):
    """
    Map empty-aperture sentinels to 0 so that the ``ApertureStats``
    NaN convention agrees with the 0.0 returned by the other entry
    points (and by SEP).

    Parameters
    ----------
    arr : array_like
        The values to sanitize.

    Returns
    -------
    result : `~numpy.ndarray`
        The float array with non-finite values replaced by 0.
    """
    return np.nan_to_num(np.asarray(arr, dtype=float), nan=0.0,
                         posinf=0.0, neginf=0.0)


def _sep_supported(shape, scenario, method, *, for_validation=False):
    """
    Determine whether SEP supports (and is comparable for) a cell.

    Parameters
    ----------
    shape : dict
        The aperture shape (an entry from ``build_shapes``).

    scenario : dict
        The masking scenario (an entry from ``build_scenarios``).

    method : str
        The segmentation ``mask_method``.

    for_validation : bool, optional
        Whether the cell is a value validation rather than a timing.

    Returns
    -------
    supported : bool
        Whether the SEP comparison is supported.
    """
    supported = (HAS_SEP and shape['sep'] is not None
                 and method in ('none', 'mask', 'source_only')
                 and (not scenario['use_segm'] or shape['sep_segm']))
    if for_validation:
        # SEP area-corrects mask-keyword pixels instead of dropping
        # them, so the values are not comparable (see the module
        # docstring)
        supported = supported and not scenario['use_mask']
    return supported


def validate(data, positions, labels, segm, maskarr, error, shapes,
             scenarios, *, internal_rtol=1e-9, sep_rtol=1e-5,
             atol=1e-5):
    """
    Validate photutils self-consistency and agreement with SEP.

    A small ``atol`` absorbs the negligible floating-point residue
    (~1e-8) that an empty aperture can leave in the photometry
    sum/error versus the NaN (mapped to 0) reported by
    ``ApertureStats``.

    Parameters
    ----------
    data : 2D `~numpy.ndarray`
        The image.

    positions : 2D `~numpy.ndarray`
        The ``(x, y)`` source center positions.

    labels : 1D `~numpy.ndarray`
        The per-source segmentation labels.

    segm : 2D `~numpy.ndarray`
        The segmentation map.

    maskarr : 2D `~numpy.ndarray` (bool)
        The pixel mask.

    error : 2D `~numpy.ndarray`
        The total error array.

    shapes : list of dict
        The aperture shapes from ``build_shapes``.

    scenarios : list of dict
        The masking scenarios from ``build_scenarios``.

    internal_rtol : float, optional
        The relative tolerance for the photutils internal agreement.

    sep_rtol : float, optional
        The relative tolerance for the SEP agreement.

    atol : float, optional
        The absolute tolerance for both comparisons.

    Returns
    -------
    n_fail : int
        The number of failed checks.
    """
    segm32 = np.ascontiguousarray(segm, dtype=np.int32)
    n_fail = 0
    print('\n== Validation ==')

    for scenario in scenarios:
        scenario['_maskarr'] = maskarr
        scenario['_labels'] = labels
        print(f'\nScenario: {scenario["name"]}')
        for shape in shapes:
            aper = shape['aperture'](positions)
            for method in scenario['methods']:
                ap_s, ap_e = run_aperture_photometry(
                    data, aper, error, scenario, method, segm, labels,
                    maskarr)
                st_s, st_e = run_aperture_stats(
                    data, aper, error, scenario, method, segm, labels,
                    maskarr)

                internal_ok = True
                try:
                    assert_allclose(_sanitize(st_s), _sanitize(ap_s),
                                    rtol=internal_rtol, atol=atol)
                    assert_allclose(_sanitize(st_e), _sanitize(ap_e),
                                    rtol=internal_rtol, atol=atol)
                    # The legacy function does not support
                    # segmentation masking
                    if not scenario['use_segm']:
                        lg_s, lg_e = run_legacy_photometry(
                            data, aper, error, scenario, method, segm,
                            labels, maskarr)
                        assert_allclose(_sanitize(lg_s),
                                        _sanitize(ap_s),
                                        rtol=internal_rtol, atol=atol)
                        assert_allclose(_sanitize(lg_e),
                                        _sanitize(ap_e),
                                        rtol=internal_rtol, atol=atol)
                except AssertionError:
                    internal_ok = False
                    n_fail += 1

                sep_str = ''
                if _sep_supported(shape, scenario, method,
                                  for_validation=True):
                    sep_s, _ = run_sep(shape, data, positions, error,
                                       scenario, method, segm32)
                    sep_s = _sanitize(sep_s)
                    ref = _sanitize(ap_s)
                    reldiff = np.max(np.abs(sep_s - ref)
                                     / np.maximum(np.abs(ref), 1e-12))
                    sep_ok = True
                    try:
                        assert_allclose(sep_s, ref, rtol=sep_rtol,
                                        atol=atol)
                    except AssertionError:
                        sep_ok = False
                        n_fail += 1
                    sep_str = (f'  SEP {"ok " if sep_ok else "FAIL"}'
                               f'(max reldiff {reldiff:.2e})')
                elif HAS_SEP and shape['sep'] is not None:
                    if scenario['use_mask']:
                        sep_str = '  SEP n/a (mask area-corrected)'
                    else:
                        sep_str = '  SEP n/a (no analogue)'

                status = 'PASS' if internal_ok else 'FAIL'
                print(f'  {shape["name"]:16s} {method:12s} '
                      f'internal {status}{sep_str}')

    result = 'ALL PASS' if n_fail == 0 else f'{n_fail} FAILURE(S)'
    print(f'\nValidation result: {result}')
    return n_fail


def benchmark(data, positions, labels, segm, maskarr, error, shapes,
              scenarios, *, repeats=3, n_threads=1):
    """
    Benchmark each photutils entry point and SEP across scenarios.

    The ``n_threads`` keyword applies to the ``AperturePhotometry``
    and ``ApertureStats`` classes only; the legacy
    ``aperture_photometry`` function and SEP are single threaded.

    Parameters
    ----------
    data : 2D `~numpy.ndarray`
        The image.

    positions : 2D `~numpy.ndarray`
        The ``(x, y)`` source center positions.

    labels : 1D `~numpy.ndarray`
        The per-source segmentation labels.

    segm : 2D `~numpy.ndarray`
        The segmentation map.

    maskarr : 2D `~numpy.ndarray` (bool)
        The pixel mask.

    error : 2D `~numpy.ndarray`
        The total error array.

    shapes : list of dict
        The aperture shapes from ``build_shapes``.

    scenarios : list of dict
        The masking scenarios from ``build_scenarios``.

    repeats : int, optional
        The number of repeats for each timing (best time is kept).

    n_threads : int, optional
        The number of threads for the class-based entry points.
    """
    segm32 = np.ascontiguousarray(segm, dtype=np.int32)
    n_src = positions.shape[0]
    print(f'\n== Benchmark (best of {repeats}, {n_src} sources, '
          f'{data.shape[0]}x{data.shape[1]} image, '
          f'n_threads={n_threads}) ==')

    header = (f'{"shape":16s} {"scenario":20s} {"method":12s} '
              f'{"ApPhot":>9s} {"legacy":>9s} {"ApStats":>9s} '
              f'{"SEP":>9s} {"ApPhot/SEP":>11s}')
    for scenario in scenarios:
        scenario['_maskarr'] = maskarr
        scenario['_labels'] = labels

    for shape in shapes:
        aper = shape['aperture'](positions)
        print('\n' + header)
        print('-' * len(header))
        for scenario in scenarios:
            for method in scenario['methods']:
                t_ap = time_best(
                    lambda a=aper, sc=scenario, m=method:
                    run_aperture_photometry(
                        data, a, error, sc, m, segm, labels,
                        maskarr, n_threads=n_threads), repeats=repeats)
                if scenario['use_segm']:
                    # The legacy function does not support
                    # segmentation masking
                    lg_ms = f'{"--":>9s}'
                else:
                    t_lg = time_best(
                        lambda a=aper, sc=scenario, m=method:
                        run_legacy_photometry(
                            data, a, error, sc, m, segm, labels,
                            maskarr), repeats=repeats)
                    lg_ms = f'{t_lg * 1e3:9.2f}'
                t_st = time_best(
                    lambda a=aper, sc=scenario, m=method:
                    run_aperture_stats(
                        data, a, error, sc, m, segm, labels,
                        maskarr, n_threads=n_threads), repeats=repeats)

                if _sep_supported(shape, scenario, method):
                    t_sep = time_best(
                        lambda s=shape, sc=scenario, m=method:
                        run_sep(s, data, positions, error, sc, m,
                                segm32), repeats=repeats)
                    sep_ms = f'{t_sep * 1e3:9.2f}'
                    ratio = f'{t_ap / t_sep:11.2f}'
                else:
                    sep_ms = f'{"--":>9s}'
                    ratio = f'{"--":>11s}'

                print(f'{shape["name"]:16s} {scenario["name"]:20s} '
                      f'{method:12s} {t_ap * 1e3:9.2f} '
                      f'{lg_ms} {t_st * 1e3:9.2f} '
                      f'{sep_ms} {ratio}')

    print('\n(times in ms; ApPhot/SEP = AperturePhotometry / SEP '
          'runtime ratio, lower is faster)')


def bench_thread_sweep(data, positions, error, shapes, n_threads_list,
                       *, repeats=3):
    """
    Sweep ``AperturePhotometry`` over thread counts against SEP.

    Only the no-masking scenario is timed. SEP is single threaded,
    so its baseline is one column; the final column is the
    ``AperturePhotometry`` to SEP runtime ratio at the last thread
    count.

    Parameters
    ----------
    data : 2D `~numpy.ndarray`
        The image.

    positions : 2D `~numpy.ndarray`
        The ``(x, y)`` source center positions.

    error : 2D `~numpy.ndarray`
        The total error array.

    shapes : list of dict
        The aperture shapes from ``build_shapes``.

    n_threads_list : list of int
        The thread counts to sweep.

    repeats : int, optional
        The number of repeats for each timing (best time is kept).
    """
    scenario = build_scenarios()[0]  # no masking
    method = 'none'
    n_src = positions.shape[0]
    n_last = n_threads_list[-1]
    print(f'\n== AperturePhotometry vs SEP n_threads sweep '
          f'(no masking, best of {repeats}, {n_src} sources, '
          f'{data.shape[0]}x{data.shape[1]} image) ==')
    header = f'{"shape":16s} {"SEP":>9s}'
    header += ''.join(f'{f"n={n}":>16s}' for n in n_threads_list)
    header += f'{f"ApPhot(n={n_last})/SEP":>20s}'
    print(header)
    print('-' * len(header))

    for shape in shapes:
        aper = shape['aperture'](positions)
        times = [time_best(
            lambda a=aper, n=n_threads:
            run_aperture_photometry(
                data, a, error, scenario, method, None, None, None,
                n_threads=n), repeats=repeats)
            for n_threads in n_threads_list]

        if HAS_SEP and shape['sep'] is not None:
            t_sep = time_best(
                lambda s=shape:
                run_sep(s, data, positions, error, scenario, method,
                        None), repeats=repeats)
            sep_ms = f'{t_sep * 1e3:9.2f}'
            ratio = f'{times[-1] / t_sep:20.2f}'
        else:
            sep_ms = f'{"--":>9s}'
            ratio = f'{"--":>20s}'

        row = f'{shape["name"]:16s} {sep_ms}'
        row += f'{times[0] * 1e3:16.2f}'
        row += ''.join(
            f'{f"{t * 1e3:.2f} ({times[0] / t:.1f}x)":>16s}'
            for t in times[1:])
        row += ratio
        print(row)

    print('\n(times in ms; speedups relative to the first thread '
          'count; ApPhot/SEP lower is faster)')


def main():
    """
    Run the SEP comparison benchmarks.
    """
    parser = argparse.ArgumentParser(
        description='Benchmark photutils aperture photometry against '
                    'SEP.')
    parser.add_argument('--size', type=int, default=1024,
                        help='image size (default: %(default)s)')
    parser.add_argument('--n-sources', type=int, default=1000,
                        help='number of sources (default: %(default)s)')
    parser.add_argument('--repeats', type=int, default=3,
                        help='number of repeats per timing; the best '
                             'time is reported (default: %(default)s)')
    parser.add_argument('--seed', type=int, default=0,
                        help='random number generator seed '
                             '(default: %(default)s)')
    parser.add_argument('--n-threads', type=parse_thread_counts,
                        default='1,8',
                        help='comma-separated thread counts for the '
                             'AperturePhotometry and ApertureStats '
                             'classes; the benchmark grid uses the '
                             'first count and the threads sweep uses '
                             'all of them; the legacy function and SEP '
                             'are single threaded '
                             '(default: %(default)s)')
    parser.add_argument('--which', default='all',
                        choices=['all', 'validate', 'benchmark',
                                 'threads'],
                        help='which suite to run (default: %(default)s)')
    args = parser.parse_args()

    print_environment()
    if HAS_SEP:
        print(f'sep {sep.__version__}')
    else:
        print(f'WARNING: the "sep" package is not importable in this '
              f'interpreter ({sys.executable}), so all SEP comparisons '
              f'and timings are skipped.\n  import error: '
              f'{SEP_IMPORT_ERROR}')

    data, segm, positions, labels = make_gaussian_scene(
        args.n_sources, (args.size, args.size), seed=args.seed)
    error = np.full(data.shape, 1.0)
    rng = np.random.default_rng(args.seed + 1)
    maskarr = rng.random(data.shape) < 0.005  # ~0.5% pixels masked

    shapes = build_shapes()
    scenarios = build_scenarios()

    n_fail = 0
    if args.which in ('all', 'validate'):
        n_fail = validate(data, positions, labels, segm, maskarr,
                          error, shapes, scenarios)
    if args.which in ('all', 'benchmark'):
        benchmark(data, positions, labels, segm, maskarr, error,
                  shapes, scenarios, repeats=args.repeats,
                  n_threads=args.n_threads[0])
    if args.which in ('all', 'threads'):
        bench_thread_sweep(data, positions, error, shapes,
                           args.n_threads, repeats=args.repeats)

    if n_fail:
        sys.exit(1)


if __name__ == '__main__':
    main()
