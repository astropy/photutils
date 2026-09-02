# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the batch flux_radius solve driver.
"""

import math
from concurrent.futures import ThreadPoolExecutor

import astropy.units as u
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy.optimize import root_scalar

from photutils.geometry import circular_overlap_grid
from photutils.segmentation import SourceCatalog
from photutils.segmentation._batch_catalog import (batch_flux_radius_prepare,
                                                   batch_flux_radius_solve)
from photutils.segmentation._batch_results import BatchFluxRadiusArgs
from photutils.segmentation.tests._batch_scene import (make_batch_scene,
                                                       make_catalog)
from photutils.segmentation.utils import _mask_to_mirrored_value

# The batch solver accumulates the flux in a different order from the
# reference, which perturbs the Brent iteration path, so the roots can
# differ by up to the solver's absolute tolerance (xtol = 2e-12 pixels)
RADIUS_ATOL = 4e-12


def _reference_optimizer_args(cat):
    """
    Prepare the per-source flux-radius root-find arguments.

    This is a verbatim port of the per-source Python implementation
    of ``_flux_radius_optimizer_args`` that the batch Cython
    preparation replaces. It is the numerical reference for the
    preparation.
    """
    kron_flux = cat._kron_photometry[:, 0]  # unitless
    max_radius = cat._max_circular_kron_radius
    kwargs = cat._aperture_mask_kwargs['flux_radius']

    method = kwargs.get('method', 'exact')
    if method == 'exact':
        use_exact = 1
        subpixels = 1
    elif method == 'center':
        use_exact = 0
        subpixels = 1
    else:  # 'subpixel'
        use_exact = 0
        subpixels = kwargs.get('subpixels', 5)

    data_arr = cat._data
    mask_arr = cat._mask
    segm_data = cat._segmentation_image.data
    data_shape = data_arr.shape
    aperture_mask_method = cat.aperture_mask_method
    max_aper_size = max(data_arr.size, 1_000_000)

    x_centroid = np.atleast_1d(cat.x_centroid)
    y_centroid = np.atleast_1d(cat.y_centroid)

    args = []
    for label, xcen, ycen, kronflux, bkg, max_radius_ in zip(
            cat.labels, x_centroid, y_centroid,
            kron_flux, cat._local_background, max_radius, strict=True):

        if (np.any(~np.isfinite((xcen, ycen, kronflux, max_radius_)))
                or kronflux == 0):
            args.append(None)
            continue

        ixmin = math.floor(xcen - max_radius_ + 0.5)
        ixmax = math.ceil(xcen + max_radius_ + 0.5)
        iymin = math.floor(ycen - max_radius_ + 0.5)
        iymax = math.ceil(ycen + max_radius_ + 0.5)

        bbox_ny = iymax - iymin
        bbox_nx = ixmax - ixmin
        if bbox_ny * bbox_nx > max_aper_size:
            args.append(None)
            continue

        data_ymin = max(0, iymin)
        data_ymax = min(data_shape[0], iymax)
        data_xmin = max(0, ixmin)
        data_xmax = min(data_shape[1], ixmax)
        if data_ymin >= data_ymax or data_xmin >= data_xmax:
            args.append(None)
            continue

        slc_lg = (slice(data_ymin, data_ymax), slice(data_xmin, data_xmax))
        cutout_data = data_arr[slc_lg].astype(float) - bkg

        data_mask = ~np.isfinite(cutout_data)
        if mask_arr is not None:
            data_mask |= mask_arr[slc_lg]

        cutout_xcen = xcen - data_xmin
        cutout_ycen = ycen - data_ymin

        if aperture_mask_method != 'none':
            seg_cut = segm_data[slc_lg]
            segm_mask = (seg_cut != label) & (seg_cut != 0)
            if aperture_mask_method == 'mask':
                data_mask = data_mask | segm_mask
            elif aperture_mask_method == 'correct':
                cutout_data = _mask_to_mirrored_value(
                    cutout_data, segm_mask,
                    (cutout_xcen, cutout_ycen), mask=data_mask)

        clean_data = cutout_data.copy()
        clean_data[data_mask] = 0.0

        ny, nx = clean_data.shape
        xmin_edge = -0.5 - cutout_xcen
        xmax_edge = nx - 0.5 - cutout_xcen
        ymin_edge = -0.5 - cutout_ycen
        ymax_edge = ny - 0.5 - cutout_ycen
        grid_params = (xmin_edge, xmax_edge, ymin_edge, ymax_edge,
                       nx, ny, use_exact, subpixels)

        args.append([clean_data, grid_params, kronflux, max_radius_])

    return args


def _unpack_args(args):
    """
    Convert packed ``BatchFluxRadiusArgs`` to the per-source list of
    ``[clean_data, grid_params, kronflux, max_radius]`` entries (or
    `None`) of the reference implementation.
    """
    entries = []
    for i, count in enumerate(args.counts):
        if count == 0:
            entries.append(None)
            continue
        nx = int(args.nx[i])
        ny = int(args.ny[i])
        start = args.starts[i]
        clean_data = args.values[start:start + count].reshape(ny, nx)
        grid_params = (*(float(edge) for edge in args.grid_edges[i]),
                       nx, ny, args.use_exact, args.subpixels)
        entries.append([clean_data, grid_params, float(args.kronflux[i]),
                        float(args.max_radius[i])])
    return entries


def _pack_entries(entries, *, use_exact=1, subpixels=1):
    """
    Pack per-source reference entries (or `None`) into
    ``BatchFluxRadiusArgs``.
    """
    n_src = len(entries)
    counts = np.zeros(n_src, dtype=np.intp)
    nx = np.zeros(n_src, dtype=np.intp)
    ny = np.zeros(n_src, dtype=np.intp)
    grid_edges = np.zeros((n_src, 4))
    kronflux = np.zeros(n_src)
    max_radius = np.zeros(n_src)
    values = []
    for i, entry in enumerate(entries):
        if entry is None:
            continue
        clean_data, grid_params, kronflux[i], max_radius[i] = entry
        counts[i] = clean_data.size
        ny[i], nx[i] = clean_data.shape
        grid_edges[i] = grid_params[:4]
        values.append(clean_data.ravel())
    starts = np.concatenate(([0], np.cumsum(counts)[:-1])).astype(np.intp)
    values = (np.concatenate(values) if values
              else np.zeros(0, dtype=np.float64))
    return BatchFluxRadiusArgs(values=values, starts=starts, counts=counts,
                               nx=nx, ny=ny, grid_edges=grid_edges,
                               kronflux=kronflux, max_radius=max_radius,
                               use_exact=use_exact, subpixels=subpixels)


def _solve(args, fraction):
    return batch_flux_radius_solve(
        args.values, starts=args.starts, counts=args.counts, nx=args.nx,
        ny=args.ny, grid_edges=args.grid_edges, kronflux=args.kronflux,
        max_radius=args.max_radius, fraction=fraction,
        use_exact=args.use_exact, subpixels=args.subpixels)


def _assert_args_equal(args, expected):
    assert args.values.dtype == np.float64
    assert args.values.flags.c_contiguous
    assert len(args.counts) == len(expected)
    entries = _unpack_args(args)
    for entry, ref in zip(entries, expected, strict=True):
        if ref is None:
            assert entry is None
            continue
        assert entry is not None
        assert_array_equal(entry[0], ref[0])
        assert entry[0].dtype == np.float64
        assert entry[0].flags.c_contiguous
        assert entry[1] == ref[1]
        assert entry[2] == ref[2]
        assert entry[3] == ref[3]


def _reference_solve(args_list, fraction):
    # Verbatim port of the pre-change ``flux_radius`` solve loop and
    # ``_flux_radius_fcn``, operating on the optimizer-args entries
    def fcn(radius, clean_data, grid_params, normflux):
        xmin_e, xmax_e, ymin_e, ymax_e, nx, ny, exact, subpx = \
            grid_params
        weights = circular_overlap_grid(xmin_e, xmax_e, ymin_e,
                                        ymax_e, nx, ny, radius,
                                        exact, subpx)
        return 1.0 - (np.sum(clean_data * weights) / normflux)

    radius = []
    for entry in args_list:
        if entry is None:
            radius.append(np.nan)
            continue
        clean_data, grid_params, kronflux, max_radius = entry
        normflux = kronflux * fraction
        found = False
        min_radius = 0.1
        max_radius_delta = 0.1 * max_radius
        while max_radius > min_radius and found is False:
            try:
                result = root_scalar(
                    fcn, args=(clean_data, grid_params, normflux),
                    bracket=[min_radius, max_radius],
                    method='brentq')
                result = result.root
                found = True
            except ValueError:
                max_radius -= max_radius_delta
        if found is False:
            result = np.nan
        radius.append(result)
    return np.array(radius)


@pytest.fixture(scope='module')
def scene():
    return make_batch_scene()


@pytest.mark.parametrize('method', ['correct', 'mask', 'none'])
@pytest.mark.parametrize('fraction', [0.2, 0.5, 0.9])
def test_matches_reference(scene, method, fraction):
    cat = make_catalog(scene, aperture_mask_method=method)
    args = cat._flux_radius_optimizer_args
    expected = _reference_solve(_unpack_args(args), fraction)
    result = _solve(args, fraction)
    assert_allclose(result, expected, rtol=1e-12, atol=RADIUS_ATOL,
                    equal_nan=True)


def test_none_entries(scene):
    cat = make_catalog(scene)
    entries = _unpack_args(cat._flux_radius_optimizer_args)
    entries[0] = None
    result = _solve(_pack_entries(entries), 0.5)
    assert np.isnan(result[0])
    assert_allclose(result, _reference_solve(entries, 0.5), rtol=1e-12,
                    atol=RADIUS_ATOL, equal_nan=True)


def test_solve_length_guard(scene):
    cat = make_catalog(scene)
    args = cat._flux_radius_optimizer_args
    with pytest.raises(ValueError, match='same length as counts'):
        batch_flux_radius_solve(
            args.values, starts=args.starts, counts=args.counts[:-1],
            nx=args.nx, ny=args.ny, grid_edges=args.grid_edges,
            kronflux=args.kronflux, max_radius=args.max_radius,
            fraction=0.5, use_exact=1, subpixels=1)


def test_solve_grid_edges_guard(scene):
    cat = make_catalog(scene)
    args = cat._flux_radius_optimizer_args
    grid_edges = np.ascontiguousarray(args.grid_edges[:, :3])
    with pytest.raises(ValueError, match='grid_edges must have 4 columns'):
        _solve(args._replace(grid_edges=grid_edges), 0.5)


def test_solve_buffer_guard(scene):
    """
    Test that a region that runs past the end of the packed buffer, a
    negative start, and a negative count are each rejected
    """
    cat = make_catalog(scene)
    args = cat._flux_radius_optimizer_args
    match = 'must lie within the values buffer'
    with pytest.raises(ValueError, match=match):
        _solve(args._replace(values=args.values[:-1]), 0.5)
    starts = args.starts.copy()
    starts[0] = -1
    with pytest.raises(ValueError, match=match):
        _solve(args._replace(starts=starts), 0.5)
    counts = args.counts.copy()
    counts[0] = -1
    with pytest.raises(ValueError, match=match):
        _solve(args._replace(counts=counts), 0.5)


def test_all_skipped(scene):
    cat = make_catalog(scene)
    n_src = cat.n_labels
    args = _pack_entries([None] * n_src)
    assert args.values.size == 0
    assert np.all(np.isnan(_solve(args, 0.5)))


def test_bracket_shrink_and_no_solution(scene):
    # Force the shrink path: negative data beyond a ring makes the
    # enclosed flux non-monotonic, so the initial bracket has equal
    # signs at both ends
    cat = make_catalog(scene)
    entry = [e for e in _unpack_args(cat._flux_radius_optimizer_args)
             if e is not None][0]
    clean_data = entry[0].copy()
    yc = entry[0].shape[0] / 2
    xc = entry[0].shape[1] / 2
    yy, xx = np.mgrid[0:entry[0].shape[0], 0:entry[0].shape[1]]
    rr = np.hypot(xx - xc, yy - yc)
    clean_data[rr > 3] = -np.abs(clean_data[rr > 3]) - 1.0
    forced = [[np.ascontiguousarray(clean_data), entry[1], entry[2],
               entry[3]]]
    assert_allclose(_solve(_pack_entries(forced), 0.5),
                    _reference_solve(forced, 0.5), rtol=1e-12,
                    atol=RADIUS_ATOL, equal_nan=True)

    # A milder outer ring, where shrinking the bracket does reveal a
    # sign change and a root is found on a later retry
    shrink_data = entry[0].copy()
    shrink_data[rr > 3] = -1.0
    shrink = [[np.ascontiguousarray(shrink_data), entry[1], entry[2],
               entry[3]]]
    expected = _reference_solve(shrink, 0.5)
    assert np.isfinite(expected[0])
    assert_allclose(_solve(_pack_entries(shrink), 0.5),
                    expected, rtol=1e-12, atol=RADIUS_ATOL)

    # And a hopeless case that shrinks to no solution -> NaN
    hopeless = [[np.ascontiguousarray(np.full_like(clean_data,
                                                   -1.0)),
                 entry[1], entry[2], entry[3]]]
    assert np.isnan(_solve(_pack_entries(hopeless), 0.5)[0])


@pytest.mark.parametrize('method', ['correct', 'mask', 'none'])
@pytest.mark.parametrize('with_mask', [True, False])
def test_prepare_matches_reference(scene, method, with_mask):
    cat = make_catalog(scene, aperture_mask_method=method,
                       with_mask=with_mask)
    expected = _reference_optimizer_args(cat)
    assert sum(entry is not None for entry in expected) > 0
    _assert_args_equal(cat._flux_radius_optimizer_args, expected)


@pytest.mark.parametrize('kwargs', [{'method': 'center'},
                                    {'method': 'subpixel', 'subpixels': 5}])
def test_prepare_overlap_methods(scene, kwargs):
    cat = make_catalog(scene)
    cat._aperture_mask_kwargs['flux_radius'] = kwargs
    _assert_args_equal(cat._flux_radius_optimizer_args,
                       _reference_optimizer_args(cat))
    assert_allclose(cat.flux_radius(0.5).value,
                    _reference_solve(_reference_optimizer_args(cat), 0.5),
                    rtol=1e-12, atol=RADIUS_ATOL, equal_nan=True)


def test_prepare_local_background(scene):
    cat = SourceCatalog(scene['data'], scene['segm'], error=scene['error'],
                        mask=scene['mask'], local_bkg_width=6)
    assert np.any(cat._local_background != 0)
    _assert_args_equal(cat._flux_radius_optimizer_args,
                       _reference_optimizer_args(cat))


def test_prepare_skipped_sources(scene):
    # A non-finite centroid, a zero Kron flux, and an off-image centroid
    # have no solution, and an oversized bounding box is skipped
    cat = make_catalog(scene)
    _ = cat._kron_photometry
    xcen = cat.x_centroid.copy()
    xcen[0] = np.nan
    xcen[1] = 5000.0
    cat.__dict__['x_centroid'] = xcen
    kron = cat._kron_photometry.copy()
    kron[2, 0] = 0.0
    cat.__dict__['_kron_photometry'] = kron
    expected = _reference_optimizer_args(cat)
    assert expected[0] is None
    assert expected[1] is None
    assert expected[2] is None
    _assert_args_equal(cat._flux_radius_optimizer_args, expected)

    cat = make_catalog(scene)
    _ = cat._kron_photometry
    cat.__dict__['_max_circular_kron_radius'] = np.full(cat.n_labels,
                                                        2000.0)
    assert np.all(cat._flux_radius_optimizer_args.counts == 0)
    assert cat._flux_radius_optimizer_args.values.size == 0


@pytest.mark.parametrize('n_threads', [2, 3, 8])
def test_n_threads(scene, n_threads):
    # The chunked preparation (whose packed buffers are merged with
    # rebased starts) and the chunked solve give identical results
    kwargs = {'error': scene['error'], 'mask': scene['mask']}
    cat1 = SourceCatalog(scene['data'], scene['segm'], **kwargs)
    cat2 = SourceCatalog(scene['data'], scene['segm'], n_threads=n_threads,
                         **kwargs)
    # Skip the first and last sources (non-finite centroid) so that
    # zero-count sources fall at the chunk boundaries
    for cat in (cat1, cat2):
        _ = cat._kron_photometry
        xcen = cat.x_centroid.copy()
        xcen[0] = np.nan
        xcen[-1] = np.nan
        cat.__dict__['x_centroid'] = xcen
    args1 = cat1._flux_radius_optimizer_args
    args2 = cat2._flux_radius_optimizer_args
    assert args1.counts[0] == 0
    assert args1.counts[-1] == 0
    assert np.any(args1.counts > 0)
    for name in ('values', 'starts', 'counts', 'nx', 'ny', 'grid_edges',
                 'kronflux', 'max_radius'):
        assert_array_equal(getattr(args2, name), getattr(args1, name))
    assert args2.use_exact == args1.use_exact
    assert args2.subpixels == args1.subpixels
    assert_array_equal(cat2.flux_radius(0.5), cat1.flux_radius(0.5))
    assert_array_equal(cat2.flux_radius(0.9), cat1.flux_radius(0.9))


def _prepare_inputs(cat):
    arrays = cat._get_batch_arrays()
    n_src = cat.n_labels
    return {'data': arrays['data'], 'mask': arrays['mask'],
            'segm': arrays['segm'],
            'labels': np.ascontiguousarray(cat.labels, dtype=np.intp),
            'xcen': np.ascontiguousarray(cat.x_centroid, dtype=float),
            'ycen': np.ascontiguousarray(cat.y_centroid, dtype=float),
            'local_bkg': np.zeros(n_src),
            'kronflux': np.ones(n_src),
            'max_radius': np.full(n_src, 5.0),
            'skip': np.zeros(n_src, dtype=np.uint8),
            'seg_method': 3, 'use_exact': 1, 'subpixels': 1,
            'max_aper_size': 1_000_000}


def _call_prepare(inp):
    return batch_flux_radius_prepare(inp.pop('data'), **inp)


@pytest.mark.parametrize('name', ['xcen', 'ycen', 'local_bkg', 'kronflux',
                                  'max_radius', 'skip'])
def test_prepare_length_guard(scene, name):
    cat = make_catalog(scene)
    inp = _prepare_inputs(cat)
    inp[name] = inp[name][:-1]
    with pytest.raises(ValueError, match='same length as labels'):
        _call_prepare(inp)


@pytest.mark.parametrize('name', ['mask', 'segm'])
def test_prepare_shape_guard(scene, name):
    cat = make_catalog(scene)
    inp = _prepare_inputs(cat)
    inp[name] = inp[name][:-1, :]
    with pytest.raises(ValueError, match='same shape as data'):
        _call_prepare(inp)


def test_prepare_thread_safety(scene):
    cat = make_catalog(scene)
    inp = _prepare_inputs(cat)
    expected = _call_prepare(dict(inp))

    def run(_):
        return _call_prepare(dict(inp))

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(run, range(8)))
    for result in results:
        _assert_args_equal(result, _unpack_args(expected))


def test_catalog_flux_radius(scene):
    cat = make_catalog(scene)
    expected = _reference_solve(
        _unpack_args(cat._flux_radius_optimizer_args), 0.5)
    result = cat.flux_radius(0.5)
    assert result.unit == u.pix
    assert_allclose(result.value, expected, rtol=1e-12, atol=RADIUS_ATOL,
                    equal_nan=True)

    # Named property
    cat.flux_radius(0.3, name='r30')
    assert_allclose(cat.r30.value,
                    _reference_solve(
                        _unpack_args(cat._flux_radius_optimizer_args),
                        0.3),
                    rtol=1e-12, atol=RADIUS_ATOL, equal_nan=True)
