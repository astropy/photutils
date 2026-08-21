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
from photutils.segmentation.tests._batch_scene import (make_batch_scene,
                                                       make_catalog)
from photutils.segmentation.utils import _mask_to_mirrored_value


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


def _assert_args_equal(args, expected):
    assert len(args) == len(expected)
    for entry, ref in zip(args, expected, strict=True):
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
    expected = _reference_solve(args, fraction)
    result = batch_flux_radius_solve(args, fraction=fraction)
    assert_allclose(result, expected, rtol=1e-12, equal_nan=True)


def test_none_entries(scene):
    cat = make_catalog(scene)
    args = list(cat._flux_radius_optimizer_args)
    args[0] = None
    result = batch_flux_radius_solve(args, fraction=0.5)
    assert np.isnan(result[0])
    assert_allclose(result, _reference_solve(args, 0.5), rtol=1e-12,
                    equal_nan=True)


def test_bracket_shrink_and_no_solution(scene):
    # Force the shrink path: negative data beyond a ring makes the
    # enclosed flux non-monotonic, so the initial bracket has equal
    # signs at both ends
    cat = make_catalog(scene)
    entry = [e for e in cat._flux_radius_optimizer_args
             if e is not None][0]
    clean_data = entry[0].copy()
    yc = entry[0].shape[0] / 2
    xc = entry[0].shape[1] / 2
    yy, xx = np.mgrid[0:entry[0].shape[0], 0:entry[0].shape[1]]
    rr = np.hypot(xx - xc, yy - yc)
    clean_data[rr > 3] = -np.abs(clean_data[rr > 3]) - 1.0
    forced = [[np.ascontiguousarray(clean_data), entry[1], entry[2],
               entry[3]]]
    assert_allclose(batch_flux_radius_solve(forced, fraction=0.5),
                    _reference_solve(forced, 0.5), rtol=1e-12,
                    equal_nan=True)

    # A milder outer ring, where shrinking the bracket does reveal a
    # sign change and a root is found on a later retry
    shrink_data = entry[0].copy()
    shrink_data[rr > 3] = -1.0
    shrink = [[np.ascontiguousarray(shrink_data), entry[1], entry[2],
               entry[3]]]
    expected = _reference_solve(shrink, 0.5)
    assert np.isfinite(expected[0])
    assert_allclose(batch_flux_radius_solve(shrink, fraction=0.5),
                    expected, rtol=1e-12)

    # And a hopeless case that shrinks to no solution -> NaN
    hopeless = [[np.ascontiguousarray(np.full_like(clean_data,
                                                   -1.0)),
                 entry[1], entry[2], entry[3]]]
    assert np.isnan(batch_flux_radius_solve(hopeless,
                                            fraction=0.5)[0])


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
                    rtol=1e-12, equal_nan=True)


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
    assert all(entry is None for entry in cat._flux_radius_optimizer_args)


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
        _assert_args_equal(result, expected)


def test_catalog_flux_radius(scene):
    cat = make_catalog(scene)
    expected = _reference_solve(cat._flux_radius_optimizer_args, 0.5)
    result = cat.flux_radius(0.5)
    assert result.unit == u.pix
    assert_allclose(result.value, expected, rtol=1e-12,
                    equal_nan=True)

    # Named property
    cat.flux_radius(0.3, name='r30')
    assert_allclose(cat.r30.value,
                    _reference_solve(
                        cat._flux_radius_optimizer_args, 0.3),
                    rtol=1e-12, equal_nan=True)
