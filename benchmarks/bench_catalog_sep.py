#!/usr/bin/env python3
# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Benchmark and validation of SourceCatalog against SEP.

The script uses the same image of blended Gaussian-source pairs, the
same convolved image, and the same segmentation image as the
``SourceCatalog`` benchmarks in ``bench_segmentation.py`` (see its
``make_inputs`` function) and, for each ``aperture_mask_method`` that
has a SEP analogue ('none' and 'mask'):

1. Validates the ``SourceCatalog`` properties against the equivalent
   SEP measurements on the same inputs:

   * the isophotal properties (centroid and its error, covariance,
     semimajor and semiminor axes, orientation, ellipse coefficients,
     segment flux and area, peak value and position, and bounding
     box) against ``sep.extract`` run on the existing segmentation
     map (``segmentation_map=<array>``) with the same convolution
     kernel (``filter_type='conv'``)
   * ``kron_radius`` against ``sep.kron_radius``
   * ``kron_flux`` against ``sep.sum_ellipse`` in the Kron aperture
   * ``flux_radius`` against ``sep.flux_radius``
   * ``centroid_win`` against ``sep.winpos``
   * ``circular_photometry`` against ``sep.sum_circle``

2. Benchmarks each of these measurements, timing the ``SourceCatalog``
   property on a fresh catalog whose prerequisite properties are
   already computed (so each row is the cost of that step alone) and
   the SEP function with its inputs precomputed, and then the full
   chain of measurements from a cold catalog against the same chain
   of SEP calls.

SEP's ``segmap`` and positive ``seg_id`` inputs mask the pixels of
neighboring sources, which corresponds to the photutils
``aperture_mask_method='mask'``; without them, neighboring sources
are included (``aperture_mask_method='none'``). The photutils
``'correct'`` method has no SEP analogue. ``sep.winpos`` has no
segmentation-map support, so ``centroid_win`` is only compared and
benchmarked against SEP in the 'none' scenario.

The following conventions make the two packages comparable:

* SEP computes the isophotal properties from the internally filtered
  image and the segment flux from the unfiltered image, exactly as
  ``SourceCatalog`` uses ``convolved_data`` and ``data``. The plain
  SEP convolution (``filter_type='conv'``) matches the zero-padded
  ``astropy.convolution.convolve`` used to make ``convolved_data``.
  SEP stores pixel values as float32, so the isophotal properties
  agree to roughly 1e-7 relative precision.
* SEP objects are returned in segment-label order, matching the
  ``SourceCatalog`` label order.
* ``SourceCatalog`` applies its ``kron_params`` to the measured Kron
  radius (the minimum unscaled radius ``kron_params[1]`` and a NaN
  above the 6.0 measurement scale), while ``sep.kron_radius`` returns
  the raw measurement. The same limits are applied to the SEP value
  before it is compared and before it defines the SEP Kron aperture.
* ``sep.flux_radius`` measures the flux in 256 annuli out to the
  maximum radius and linearly interpolates the cumulative profile,
  whereas ``SourceCatalog.flux_radius`` solves for the exact-overlap
  circular aperture enclosing the flux fraction. The two agree only to
  roughly 1% (``FLUX_RADIUS_RTOL``). For the ``centroid_win``
  comparison, SEP is therefore given the window sigma derived from the
  photutils half-light radius so that the windowed algorithms are
  compared on the same window; its benchmark uses SEP's own chain.
* SEP weights each pixel's variance by its aperture overlap fraction,
  whereas photutils weights it by the square of the overlap fraction,
  so the aperture flux errors are not compared.
* The catalog has no ``background`` or ``wcs`` inputs, which have no
  SEP analogue.

Requires the optional ``sep`` package. Run ``python
benchmarks/bench_catalog_sep.py --help`` to see the available options.
"""

import argparse
import sys
import time

import astropy.units as u
import numpy as np
from astropy.stats import gaussian_fwhm_to_sigma
from bench_helpers import print_environment, time_best
from bench_segmentation import THRESHOLD, make_inputs
from numpy.testing import assert_allclose, assert_array_equal

from photutils.segmentation import SourceCatalog, make_2dgaussian_kernel

try:
    import sep
    HAS_SEP = True
    SEP_IMPORT_ERROR = None
except ImportError as exc:
    HAS_SEP = False
    SEP_IMPORT_ERROR = str(exc)

KRON_PARAMS = (2.5, 1.4, 0.0)  # the SourceCatalog default
MAX_KRON_RADIUS = 6.0  # the SourceCatalog Kron measurement scale
FRACTION = 0.5  # the flux_radius fraction (half-light radius)
RADIUS = 5.0  # the circular photometry radius
MIN_HALF_LIGHT_RADIUS = 0.5  # the centroid_win minimum (SourceExtractor)

SEP_RTOL = 1e-5
SEP_ATOL = 1e-5
FLUX_RADIUS_RTOL = 1e-2
CENTROID_WIN_ATOL = 1e-5

# The SourceCatalog properties that sep.extract also measures, in
# the (name, SEP field) pairs used by the validation
ISOPHOTAL_PROPERTIES = [
    ('x_centroid', 'x'),
    ('y_centroid', 'y'),
    ('x_centroid_err', 'errx2'),
    ('y_centroid_err', 'erry2'),
    ('covariance_xx', 'x2'),
    ('covariance_yy', 'y2'),
    ('covariance_xy', 'xy'),
    ('semimajor_axis', 'a'),
    ('semiminor_axis', 'b'),
    ('orientation', 'theta'),
    ('ellipse_cxx', 'cxx'),
    ('ellipse_cyy', 'cyy'),
    ('ellipse_cxy', 'cxy'),
    ('segment_flux', 'flux'),
    ('moments', 'cflux'),
    ('max_value', 'peak'),
    ('max_value_xindex', 'xpeak'),
    ('max_value_yindex', 'ypeak'),
    ('bbox_xmin', 'xmin'),
    ('bbox_xmax', 'xmax'),
    ('bbox_ymin', 'ymin'),
    ('bbox_ymax', 'ymax'),
    ('segment_area', 'npix'),
    ('area', 'npix'),
]
INTEGER_PROPERTIES = ('max_value_xindex', 'max_value_yindex', 'bbox_xmin',
                      'bbox_xmax', 'bbox_ymin', 'bbox_ymax', 'segment_area',
                      'area')


def make_scene(n_sources, *, seed=0):
    """
    Build the shared benchmark scene.

    Parameters
    ----------
    n_sources : int
        The total number of Gaussian sources in the image.

    seed : int, optional
        The random number generator seed.

    Returns
    -------
    scene : dict
        The ``data``, ``convolved_data``, ``kernel``, ``error``,
        ``segm`` (a `~photutils.segmentation.SegmentationImage`),
        ``segm32`` (the int32 segmentation array for SEP), and
        ``labels`` (int32, for SEP ``seg_id``) values.
    """
    data, convolved_data, segm = make_inputs(n_sources, seed=seed)
    kernel = make_2dgaussian_kernel(3.0, size=5)  # as in make_inputs
    return {'data': data,
            'convolved_data': convolved_data,
            'kernel': np.ascontiguousarray(kernel.array),
            'error': np.ones_like(data),
            'segm': segm,
            'segm32': np.ascontiguousarray(segm.data, dtype=np.int32),
            'labels': np.ascontiguousarray(segm.labels, dtype=np.int32)}


def build_scenarios():
    """
    Build the ``aperture_mask_method`` scenarios that have a SEP
    analogue.

    Returns
    -------
    result : list of dict
        The scenarios, each with the photutils ``method`` and a
        ``use_segm`` flag for the SEP ``segmap``/``seg_id`` inputs.
    """
    return [
        {'name': "aperture_mask_method='none' (SEP: no segmap)",
         'method': 'none', 'use_segm': False},
        {'name': "aperture_mask_method='mask' (SEP: segmap + seg_id)",
         'method': 'mask', 'use_segm': True},
    ]


def make_catalog(scene, method):
    """
    Make a fresh `~photutils.segmentation.SourceCatalog` for a
    scenario.

    Parameters
    ----------
    scene : dict
        The scene from ``make_scene``.

    method : str
        The ``aperture_mask_method``.

    Returns
    -------
    result : `~photutils.segmentation.SourceCatalog`
        The catalog.
    """
    return SourceCatalog(scene['data'], scene['segm'],
                         convolved_data=scene['convolved_data'],
                         error=scene['error'], aperture_mask_method=method,
                         kron_params=KRON_PARAMS)


def compute_isophotal(catalog):
    """
    Compute the isophotal properties of a catalog.

    Parameters
    ----------
    catalog : `~photutils.segmentation.SourceCatalog`
        The catalog.
    """
    for name, _ in ISOPHOTAL_PROPERTIES:
        getattr(catalog, name)


def property_values(catalog, name):
    """
    Return a catalog property as a plain float array in the SEP
    convention.

    Parameters
    ----------
    catalog : `~photutils.segmentation.SourceCatalog`
        The catalog.

    name : str
        The property name.

    Returns
    -------
    result : `~numpy.ndarray`
        The values (the orientation in radians, the zeroth-order
        moment for ``'moments'``).
    """
    values = getattr(catalog, name)
    if name == 'orientation':
        return values.to_value(u.rad)
    if name == 'moments':
        return np.asarray(values[:, 0, 0], dtype=float)
    return np.asarray(getattr(values, 'value', values), dtype=float)


def sep_values(objects, field):
    """
    Return a ``sep.extract`` field in the photutils convention.

    Parameters
    ----------
    objects : `~numpy.ndarray`
        The ``sep.extract`` structured array.

    field : str
        The field name.

    Returns
    -------
    result : `~numpy.ndarray`
        The values (the centroid errors as standard deviations).
    """
    values = np.asarray(objects[field], dtype=float)
    if field in ('errx2', 'erry2'):
        return np.sqrt(values)
    return values


def sep_seg_kwargs(scene, scenario):
    """
    Return the SEP ``segmap``/``seg_id`` keyword arguments for a
    scenario.

    Parameters
    ----------
    scene : dict
        The scene from ``make_scene``.

    scenario : dict
        The scenario (an entry from ``build_scenarios``).

    Returns
    -------
    result : dict
        The keyword arguments (empty for the 'none' scenario).
    """
    if not scenario['use_segm']:
        return {}
    return {'segmap': scene['segm32'], 'seg_id': scene['labels']}


def run_sep_extract(scene):
    """
    Run ``sep.extract`` on the existing segmentation map.

    The detection threshold only sets the bookkeeping fields of the
    SEP objects (the segmentation map defines the members of every
    source); it matches the ``detect_sources`` threshold used to make
    the scene.

    Parameters
    ----------
    scene : dict
        The scene from ``make_scene``.

    Returns
    -------
    objects : `~numpy.ndarray`
        The ``sep.extract`` structured array, in segment-label order.
    """
    objects, _ = sep.extract(scene['data'], THRESHOLD, err=scene['error'],
                             filter_kernel=scene['kernel'],
                             filter_type='conv',
                             segmentation_map=scene['segm32'])
    return objects


def sep_shape_inputs(objects):
    """
    Return the per-source SEP aperture inputs from the extracted
    objects.

    Parameters
    ----------
    objects : `~numpy.ndarray`
        The ``sep.extract`` structured array.

    Returns
    -------
    x, y, a, b, theta : `~numpy.ndarray`
        The contiguous float64 centroid and ellipse parameter arrays.
    """
    return tuple(np.ascontiguousarray(objects[field], dtype=np.float64)
                 for field in ('x', 'y', 'a', 'b', 'theta'))


def run_sep_kron_radius(scene, shape, seg_kwargs):
    """
    Run ``sep.kron_radius`` and apply the ``SourceCatalog``
    ``kron_params`` limits.

    Parameters
    ----------
    scene : dict
        The scene from ``make_scene``.

    shape : tuple of `~numpy.ndarray`
        The ``(x, y, a, b, theta)`` arrays from ``sep_shape_inputs``.

    seg_kwargs : dict
        The SEP ``segmap``/``seg_id`` keyword arguments.

    Returns
    -------
    kron_radius : `~numpy.ndarray`
        The unscaled Kron radius with the minimum radius applied and
        NaN above the measurement scale.

    raw_kron_radius : `~numpy.ndarray`
        The raw SEP measurement.
    """
    x, y, a, b, theta = shape
    raw, _ = sep.kron_radius(scene['data'], x, y, a, b, theta,
                             MAX_KRON_RADIUS, **seg_kwargs)
    raw = np.asarray(raw, dtype=float)
    kron_radius = np.where(raw > MAX_KRON_RADIUS, np.nan,
                           np.maximum(raw, KRON_PARAMS[1]))
    return kron_radius, raw


def run_sep_kron_flux(scene, shape, kron_radius, seg_kwargs):
    """
    Run ``sep.sum_ellipse`` in the Kron aperture (exact overlap).

    Parameters
    ----------
    scene : dict
        The scene from ``make_scene``.

    shape : tuple of `~numpy.ndarray`
        The ``(x, y, a, b, theta)`` arrays from ``sep_shape_inputs``.

    kron_radius : `~numpy.ndarray`
        The unscaled Kron radius (NaN where undefined).

    seg_kwargs : dict
        The SEP ``segmap``/``seg_id`` keyword arguments.

    Returns
    -------
    flux : `~numpy.ndarray`
        The Kron flux (NaN where the Kron radius is undefined).
    """
    x, y, a, b, theta = shape
    valid = np.isfinite(kron_radius)
    scale = np.where(valid, KRON_PARAMS[0] * kron_radius, 1.0)
    flux, _, _ = sep.sum_ellipse(scene['data'], x, y, a, b, theta, scale,
                                 err=scene['error'], subpix=0,
                                 **seg_kwargs)
    return np.where(valid, flux, np.nan)


def run_sep_flux_radius(scene, shape, kron_radius, kron_flux, seg_kwargs):
    """
    Run ``sep.flux_radius`` normalized by the Kron flux.

    The maximum radius is the circular Kron radius
    ``kron_params[0] * kron_radius * a``, as in
    ``SourceCatalog.flux_radius``.

    Parameters
    ----------
    scene : dict
        The scene from ``make_scene``.

    shape : tuple of `~numpy.ndarray`
        The ``(x, y, a, b, theta)`` arrays from ``sep_shape_inputs``.

    kron_radius : `~numpy.ndarray`
        The unscaled Kron radius (NaN where undefined).

    kron_flux : `~numpy.ndarray`
        The Kron flux.

    seg_kwargs : dict
        The SEP ``segmap``/``seg_id`` keyword arguments.

    Returns
    -------
    radius : `~numpy.ndarray`
        The radius enclosing ``FRACTION`` of the Kron flux (NaN where
        the Kron radius or flux is undefined or the flux is zero).
    """
    x, y, a, _, _ = shape
    rmax = KRON_PARAMS[0] * kron_radius * a
    valid = (np.isfinite(rmax) & np.isfinite(kron_flux)
             & (kron_flux != 0))
    rmax = np.where(valid, rmax, 1.0)
    normflux = np.where(valid, kron_flux, 1.0)
    radius, _ = sep.flux_radius(scene['data'], x, y, rmax, FRACTION,
                                normflux=normflux, subpix=5, **seg_kwargs)
    return np.where(valid, radius, np.nan)


def window_sigma(half_light_radius):
    """
    Return the windowed-centroid Gaussian sigma for a half-light
    radius, as in ``SourceCatalog.centroid_win``.

    Parameters
    ----------
    half_light_radius : `~numpy.ndarray`
        The half-light radius.

    Returns
    -------
    result : `~numpy.ndarray`
        The window sigma (non-finite radii use the minimum radius).
    """
    radius = np.where(np.isfinite(half_light_radius), half_light_radius,
                      MIN_HALF_LIGHT_RADIUS)
    radius = np.maximum(radius, MIN_HALF_LIGHT_RADIUS)
    return 2.0 * gaussian_fwhm_to_sigma * radius


def run_sep_winpos(scene, shape, sigma):
    """
    Run ``sep.winpos`` with the 'center' pixel overlap (``subpix=1``),
    matching the ``SourceCatalog`` default.

    Parameters
    ----------
    scene : dict
        The scene from ``make_scene``.

    shape : tuple of `~numpy.ndarray`
        The ``(x, y, a, b, theta)`` arrays from ``sep_shape_inputs``.

    sigma : `~numpy.ndarray`
        The per-source window sigma.

    Returns
    -------
    x, y : `~numpy.ndarray`
        The windowed centroid.
    """
    x0, y0 = shape[0], shape[1]
    x, y, _ = sep.winpos(scene['data'], x0, y0, sigma, subpix=1)
    return np.asarray(x), np.asarray(y)


def run_sep_sum_circle(scene, shape, seg_kwargs):
    """
    Run ``sep.sum_circle`` with exact overlap.

    Parameters
    ----------
    scene : dict
        The scene from ``make_scene``.

    shape : tuple of `~numpy.ndarray`
        The ``(x, y, a, b, theta)`` arrays from ``sep_shape_inputs``.

    seg_kwargs : dict
        The SEP ``segmap``/``seg_id`` keyword arguments.

    Returns
    -------
    flux : `~numpy.ndarray`
        The circular aperture flux.
    """
    x, y = shape[0], shape[1]
    flux, _, _ = sep.sum_circle(scene['data'], x, y, RADIUS,
                                err=scene['error'], subpix=0, **seg_kwargs)
    return np.asarray(flux)


def run_sep_chain(scene, scenario):
    """
    Run the full SEP measurement chain.

    Parameters
    ----------
    scene : dict
        The scene from ``make_scene``.

    scenario : dict
        The scenario (an entry from ``build_scenarios``).
    """
    seg_kwargs = sep_seg_kwargs(scene, scenario)
    objects = run_sep_extract(scene)
    shape = sep_shape_inputs(objects)
    kron_radius, _ = run_sep_kron_radius(scene, shape, seg_kwargs)
    kron_flux = run_sep_kron_flux(scene, shape, kron_radius, seg_kwargs)
    half_light = run_sep_flux_radius(scene, shape, kron_radius, kron_flux,
                                     seg_kwargs)
    if not scenario['use_segm']:
        run_sep_winpos(scene, shape, window_sigma(half_light))
    run_sep_sum_circle(scene, shape, seg_kwargs)


def run_catalog_chain(catalog):
    """
    Run the full ``SourceCatalog`` measurement chain.

    Parameters
    ----------
    catalog : `~photutils.segmentation.SourceCatalog`
        A fresh catalog.
    """
    compute_isophotal(catalog)
    _ = catalog.kron_radius
    _ = catalog.kron_flux
    catalog.flux_radius(FRACTION)
    _ = catalog.centroid_win
    catalog.circular_photometry(RADIUS)


def _check(name, phot, ref, *, rtol, atol, exact=False):
    """
    Compare photutils and SEP values and print one validation row.

    Parameters
    ----------
    name : str
        The row label.

    phot, ref : `~numpy.ndarray`
        The photutils and SEP values.

    rtol, atol : float
        The tolerances for `~numpy.testing.assert_allclose`.

    exact : bool, optional
        Whether to require exact equality (integer properties).

    Returns
    -------
    ok : bool
        Whether the comparison passed.
    """
    phot = np.asarray(phot, dtype=float)
    ref = np.asarray(ref, dtype=float)
    finite = np.isfinite(phot) & np.isfinite(ref)
    n_nan_mismatch = np.count_nonzero(np.isfinite(phot) != np.isfinite(ref))
    diff = np.abs(phot[finite] - ref[finite])
    max_abs = diff.max() if diff.size else 0.0
    scale = np.maximum(np.abs(ref[finite]), 1e-12)
    max_rel = (diff / scale).max() if diff.size else 0.0

    ok = n_nan_mismatch == 0
    try:
        if exact:
            assert_array_equal(phot[finite], ref[finite])
        else:
            assert_allclose(phot[finite], ref[finite], rtol=rtol, atol=atol)
    except AssertionError:
        ok = False

    status = 'ok  ' if ok else 'FAIL'
    print(f'  {name:26s} {status}  n={finite.sum():5d}  '
          f'max abs diff {max_abs:.2e}  max rel diff {max_rel:.2e}'
          + (f'  ({n_nan_mismatch} NaN mismatches)' if n_nan_mismatch
             else ''))
    return ok


def validate(scene, scenarios):
    """
    Validate the ``SourceCatalog`` properties against SEP.

    Parameters
    ----------
    scene : dict
        The scene from ``make_scene``.

    scenarios : list of dict
        The scenarios from ``build_scenarios``.

    Returns
    -------
    n_fail : int
        The number of failed checks.
    """
    n_fail = 0
    print('\n== Validation ==')
    objects = run_sep_extract(scene)
    shape = sep_shape_inputs(objects)

    for scenario in scenarios:
        print(f'\nScenario: {scenario["name"]}')
        seg_kwargs = sep_seg_kwargs(scene, scenario)
        catalog = make_catalog(scene, scenario['method'])

        if len(objects) != catalog.n_labels:
            print(f'  FAIL: SEP extracted {len(objects)} objects for '
                  f'{catalog.n_labels} labels')
            n_fail += 1
            continue

        if scenario is scenarios[0]:
            # The isophotal properties do not depend on the
            # aperture_mask_method
            for name, field in ISOPHOTAL_PROPERTIES:
                ok = _check(name, property_values(catalog, name),
                            sep_values(objects, field), rtol=SEP_RTOL,
                            atol=SEP_ATOL,
                            exact=name in INTEGER_PROPERTIES)
                n_fail += not ok
        else:
            print('  (isophotal properties are independent of the '
                  'aperture_mask_method; see the first scenario)')

        kron_radius, raw = run_sep_kron_radius(scene, shape, seg_kwargs)
        n_unclipped = np.count_nonzero(np.isfinite(kron_radius)
                                       & (raw == kron_radius))
        ok = _check(f'kron_radius ({n_unclipped} unclipped)',
                    property_values(catalog, 'kron_radius'), kron_radius,
                    rtol=SEP_RTOL, atol=SEP_ATOL)
        n_fail += not ok

        kron_flux = run_sep_kron_flux(scene, shape, kron_radius, seg_kwargs)
        ok = _check('kron_flux', catalog.kron_flux, kron_flux,
                    rtol=SEP_RTOL, atol=SEP_ATOL)
        n_fail += not ok

        half_light = catalog.flux_radius(FRACTION).value
        sep_half_light = run_sep_flux_radius(scene, shape, kron_radius,
                                             kron_flux, seg_kwargs)
        ok = _check(f'flux_radius({FRACTION}) [binned]', half_light,
                    sep_half_light, rtol=FLUX_RADIUS_RTOL, atol=SEP_ATOL)
        n_fail += not ok

        if scenario['use_segm']:
            print(f'  {"centroid_win":26s} n/a   (sep.winpos has no '
                  'segmap support)')
        else:
            xwin, ywin = run_sep_winpos(scene, shape,
                                        window_sigma(half_light))
            finite = np.isfinite(half_light)
            ok = _check('x_centroid_win [phot sigma]',
                        np.asarray(catalog.x_centroid_win)[finite],
                        xwin[finite], rtol=0.0, atol=CENTROID_WIN_ATOL)
            n_fail += not ok
            ok = _check('y_centroid_win [phot sigma]',
                        np.asarray(catalog.y_centroid_win)[finite],
                        ywin[finite], rtol=0.0, atol=CENTROID_WIN_ATOL)
            n_fail += not ok

        flux, _ = catalog.circular_photometry(RADIUS)
        ok = _check(f'circular_photometry({RADIUS:g})', flux,
                    run_sep_sum_circle(scene, shape, seg_kwargs),
                    rtol=SEP_RTOL, atol=SEP_ATOL)
        n_fail += not ok

    result = 'ALL PASS' if n_fail == 0 else f'{n_fail} FAILURE(S)'
    print(f'\nValidation result: {result}')
    return n_fail


def time_step(make_fresh_catalog, step, *, prepare=None, repeats=3):
    """
    Return the best time of ``step`` on fresh catalogs whose
    prerequisite properties are computed by ``prepare``.

    Parameters
    ----------
    make_fresh_catalog : callable
        The zero-argument callable returning a fresh catalog.

    step : callable
        The callable taking the catalog and computing the timed step.

    prepare : callable or `None`, optional
        The callable taking the catalog and computing the
        prerequisites outside of the timed region.

    repeats : int, optional
        The number of repeats (the best time is kept).

    Returns
    -------
    result : float
        The best wall-clock time in seconds.
    """
    best = np.inf
    for _ in range(repeats):
        catalog = make_fresh_catalog()
        if prepare is not None:
            prepare(catalog)
        t0 = time.perf_counter()
        step(catalog)
        best = min(best, time.perf_counter() - t0)
    return best


def benchmark(scene, scenarios, *, repeats=3):
    """
    Benchmark the ``SourceCatalog`` measurements against SEP.

    Parameters
    ----------
    scene : dict
        The scene from ``make_scene``.

    scenarios : list of dict
        The scenarios from ``build_scenarios``.

    repeats : int, optional
        The number of repeats for each timing (best time is kept).
    """
    n_src = scene['segm'].n_labels
    shape_ = scene['data'].shape
    print(f'\n== Benchmark (best of {repeats}, {n_src} segments, '
          f'{shape_[0]}x{shape_[1]} image; step times on a catalog with '
          'the prerequisites computed) ==')

    # The SEP inputs for the per-step timings
    objects = run_sep_extract(scene)
    shape = sep_shape_inputs(objects)

    header = (f'{"benchmark":34s} {"photutils":>11s} {"SEP":>11s} '
              f'{"phot/SEP":>10s}')

    for scenario in scenarios:
        seg_kwargs = sep_seg_kwargs(scene, scenario)
        kron_radius, _ = run_sep_kron_radius(scene, shape, seg_kwargs)
        kron_flux = run_sep_kron_flux(scene, shape, kron_radius, seg_kwargs)
        half_light = run_sep_flux_radius(scene, shape, kron_radius,
                                         kron_flux, seg_kwargs)
        sigma = window_sigma(half_light)

        def make_fresh_catalog(method=scenario['method']):
            return make_catalog(scene, method)

        def warm_kron_radius(catalog):
            compute_isophotal(catalog)
            _ = catalog.kron_radius

        def warm_kron_flux(catalog):
            warm_kron_radius(catalog)
            _ = catalog.kron_flux

        def warm_flux_radius(catalog):
            warm_kron_flux(catalog)
            catalog.flux_radius(FRACTION)

        rows = [
            ('isophotal properties (extract)', None, compute_isophotal,
             lambda: run_sep_extract(scene)),
            ('kron_radius', compute_isophotal,
             lambda cat: cat.kron_radius,
             lambda sk=seg_kwargs: run_sep_kron_radius(scene, shape, sk)),
            ('kron_flux', warm_kron_radius,
             lambda cat: cat.kron_flux,
             lambda kr=kron_radius, sk=seg_kwargs:
             run_sep_kron_flux(scene, shape, kr, sk)),
            (f'flux_radius({FRACTION})', warm_kron_flux,
             lambda cat: cat.flux_radius(FRACTION),
             lambda kr=kron_radius, kf=kron_flux, sk=seg_kwargs:
             run_sep_flux_radius(scene, shape, kr, kf, sk)),
            ('centroid_win', warm_flux_radius,
             lambda cat: cat.centroid_win,
             None if scenario['use_segm']
             else lambda sig=sigma: run_sep_winpos(scene, shape, sig)),
            (f'circular_photometry({RADIUS:g})',
             lambda cat: cat.x_centroid,
             lambda cat: cat.circular_photometry(RADIUS),
             lambda sk=seg_kwargs: run_sep_sum_circle(scene, shape, sk)),
            ('full chain (cold catalog)', None, run_catalog_chain,
             lambda sc=scenario: run_sep_chain(scene, sc)),
        ]

        print(f'\nScenario: {scenario["name"]}')
        print(header)
        print('-' * len(header))
        for name, prepare, step, sep_step in rows:
            t_phot = time_step(make_fresh_catalog, step, prepare=prepare,
                               repeats=repeats)
            if sep_step is None:
                sep_ms = f'{"--":>11s}'
                ratio = f'{"--":>10s}'
            else:
                t_sep = time_best(sep_step, repeats=repeats)
                sep_ms = f'{t_sep * 1e3:11.2f}'
                ratio = f'{t_phot / t_sep:10.2f}'
            print(f'{name:34s} {t_phot * 1e3:11.2f} {sep_ms} {ratio}')

    print('\n(times in ms; phot/SEP = SourceCatalog / SEP runtime ratio, '
          'lower is faster; -- = no SEP analogue)')


def main():
    """
    Run the SourceCatalog versus SEP comparison.
    """
    parser = argparse.ArgumentParser(
        description='Benchmark and validate SourceCatalog against SEP.')
    parser.add_argument('--n-sources', type=int, default=1000,
                        help='total number of Gaussian sources in the '
                             'image (default: %(default)s)')
    parser.add_argument('--repeats', type=int, default=3,
                        help='number of repeats per timing; the best '
                             'time is reported (default: %(default)s)')
    parser.add_argument('--seed', type=int, default=0,
                        help='random number generator seed '
                             '(default: %(default)s)')
    parser.add_argument('--which', default='all',
                        choices=['all', 'validate', 'benchmark'],
                        help='which part to run (default: %(default)s)')
    args = parser.parse_args()

    print_environment()
    if not HAS_SEP:
        print(f'sep is not available ({SEP_IMPORT_ERROR}); nothing to '
              'compare')
        sys.exit(1)
    print(f'sep {sep.__version__}')

    scene = make_scene(args.n_sources, seed=args.seed)
    # SEP needs room for every segment pixel on its pixel stack
    sep.set_extract_pixstack(max(sep.get_extract_pixstack(),
                                 scene['data'].size))
    scenarios = build_scenarios()

    n_fail = 0
    if args.which in ('all', 'validate'):
        n_fail = validate(scene, scenarios)
    if args.which in ('all', 'benchmark'):
        benchmark(scene, scenarios, repeats=args.repeats)

    if n_fail:
        sys.exit(1)


if __name__ == '__main__':
    main()
