# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the batch centroid_win Cython driver.
"""

import math
import warnings
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
from astropy.stats import gaussian_fwhm_to_sigma
from numpy.testing import assert_allclose

from photutils.aperture._segmentation import SEG_METHOD_CODES
from photutils.segmentation import SourceCatalog
from photutils.segmentation._batch_catalog import batch_centroid_win
from photutils.segmentation.tests._batch_scene import (make_batch_scene,
                                                       make_catalog)
from photutils.segmentation.utils import _mask_to_mirrored_value


def _reference_iterate_centroid_win(label, xcen, ycen, rad_hl,
                                    nan_hl_source, *, data_arr,
                                    mask_arr, error_arr, segm_data,
                                    data_shape, do_correct,
                                    do_segm_mask, compute_err,
                                    max_aper_size,
                                    aperture_mask_method):
    """
    Compute the windowed centroid for a single source.

    This is a verbatim port of the per-source Python implementation that
    the batch Cython driver replaces. It is the numerical reference for
    the driver.

    Returns
    -------
    result : tuple of float
        Tuple of (xcen, ycen, weighted_flux, cen_mom_xx, cen_mom_yy,
        cen_mom_xy, err_sum, err_var_x, err_var_y, err_cov_xy).
    """
    nan_result = (np.nan, np.nan, 0.0, 0.0, 0.0, 0.0,
                  np.nan, np.nan, np.nan, np.nan)
    if nan_hl_source or math.isnan(xcen) or math.isnan(ycen):
        return nan_result

    sigma = 2.0 * rad_hl * gaussian_fwhm_to_sigma
    inv_2sigma2 = -1.0 / (2.0 * sigma * sigma)
    radius = 4.0 * sigma
    radius_sq = radius * radius

    # Compute the full (unclipped) bounding box for the aperture
    # using the initial centroid. The radius is fixed, so the
    # bbox size stays the same across iterations even if the
    # center shifts slightly.
    bbox_halfsize = int(radius + 1.5)
    full_ny = full_nx = 2 * bbox_halfsize + 1

    # OOM guard
    if full_ny * full_nx > max_aper_size:
        return nan_result

    # Cache for cutout data when the integer bbox doesn't change
    prev_ixcen = prev_iycen = None
    cached_data = cached_mask = cached_var = None

    max_iters = 16
    centroid_threshold = 0.0001
    iter_ = 0
    dcen = 1.0
    weighted_flux = 0.0
    dx_mom = dy_mom = 0.0
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        while iter_ < max_iters and dcen > centroid_threshold:
            # Compute integer bounding box
            ixmin = int(xcen + 0.5) - bbox_halfsize
            ixmax = ixmin + full_nx
            iymin = int(ycen + 0.5) - bbox_halfsize
            iymax = iymin + full_ny

            # Clip to data boundaries
            slc_y = slice(max(0, iymin), min(data_shape[0], iymax))
            slc_x = slice(max(0, ixmin), min(data_shape[1], ixmax))
            if (slc_y.start >= slc_y.stop
                    or slc_x.start >= slc_x.stop):
                xcen = np.nan
                ycen = np.nan
                break

            cur_ixcen = int(xcen + 0.5)
            cur_iycen = int(ycen + 0.5)

            # Recompute cutout data only when the integer center
            # changes to avoid redundant _mask_to_mirrored_value
            # calls
            if cur_ixcen != prev_ixcen or cur_iycen != prev_iycen:
                prev_ixcen = cur_ixcen
                prev_iycen = cur_iycen

                data = data_arr[slc_y, slc_x].astype(float)
                data_mask = ~np.isfinite(data)
                if mask_arr is not None:
                    data_mask |= mask_arr[slc_y, slc_x]

                cutout_xycen = (xcen - max(0, ixmin),
                                ycen - max(0, iymin))

                if do_segm_mask:
                    seg_cut = segm_data[slc_y, slc_x]
                    segm_mask = ((seg_cut != label)
                                 & (seg_cut != 0))
                    if aperture_mask_method == 'mask':
                        data_mask = data_mask | segm_mask

                if do_correct:
                    data = _mask_to_mirrored_value(
                        data, segm_mask, cutout_xycen,
                        mask=data_mask)

                cached_data = data
                cached_mask = data_mask

                if compute_err:
                    var = error_arr[slc_y, slc_x].astype(float)**2
                    if do_correct:
                        # Pixels replaced by mirrored data values
                        # also use the mirrored pixel's variance
                        var = _mask_to_mirrored_value(
                            var, segm_mask, cutout_xycen,
                            mask=data_mask)
                    var[data_mask] = 0.0
                    cached_var = var

            # Centroid position in cutout coordinates
            cx = xcen - max(0, ixmin)
            cy = ycen - max(0, iymin)

            ny = slc_y.stop - slc_y.start
            nx = slc_x.stop - slc_x.start

            # Build coordinate grids relative to centroid
            # (reused for circle mask, Gaussian, and moments)
            xvals = np.arange(nx) - cx
            yvals = np.arange(ny) - cy
            xx = xvals[np.newaxis, :]
            yy = yvals[:, np.newaxis]

            # Inline binary circle mask
            rr2 = xx * xx + yy * yy
            aper_weights = (rr2 <= radius_sq).astype(float)

            # Inline Gaussian weight
            gweight = np.exp(rr2 * inv_2sigma2)

            # Apply weights and mask
            weighted = (cached_data * aper_weights * gweight)
            weighted[cached_mask] = 0.0

            # Inline moment computation
            weighted_flux = np.sum(weighted)
            dx_mom = np.sum(weighted * xx) / weighted_flux
            dy_mom = np.sum(weighted * yy) / weighted_flux

            dcen = math.sqrt(dx_mom * dx_mom
                             + dy_mom * dy_mom)
            xcen += dx_mom * 2.0
            ycen += dy_mom * 2.0
            iter_ += 1

        # Compute the windowed central 2nd-order moments (for
        # the fallback checks) and the raw error
        # sums from the last iteration, relative to
        # the pre-update center.
        cen_mom_xx = cen_mom_yy = cen_mom_xy = 0.0
        err_sum = err_var_x = err_var_y = err_cov_xy = np.nan
        if np.isfinite(weighted_flux) and weighted_flux > 0:
            cen_mom_xx = (np.sum(weighted * xx * xx)
                          / weighted_flux - dx_mom * dx_mom)
            cen_mom_yy = (np.sum(weighted * yy * yy)
                          / weighted_flux - dy_mom * dy_mom)
            cen_mom_xy = (np.sum(weighted * xx * yy)
                          / weighted_flux - dx_mom * dy_mom)

            if compute_err:
                weighted_var = ((aper_weights * gweight)**2
                                * cached_var)
                err_sum = np.sum(weighted_var)
                err_var_x = np.sum(weighted_var * xx * xx)
                err_var_y = np.sum(weighted_var * yy * yy)
                err_cov_xy = np.sum(weighted_var * xx * yy)

    return (xcen, ycen, weighted_flux, cen_mom_xx,
            cen_mom_yy, cen_mom_xy,
            err_sum, err_var_x, err_var_y, err_cov_xy)


@pytest.fixture(scope='module')
def scene():
    return make_batch_scene()


def _driver_inputs(cat):
    """
    Build the batch_centroid_win inputs for a catalog, mirroring the
    wiring in SourceCatalog._centroid_win_results.
    """
    arrays = cat._get_batch_arrays()
    radius_hl = cat.flux_radius(0.5).value.copy()
    nan_hl = ~np.isfinite(radius_hl)
    small = np.isfinite(radius_hl) & (radius_hl < 0.5)
    radius_hl[small] = 0.5
    sigma = 2.0 * radius_hl * gaussian_fwhm_to_sigma
    xcen0 = np.atleast_1d(cat.x_centroid).astype(np.float64)
    ycen0 = np.atleast_1d(cat.y_centroid).astype(np.float64)
    # np.isnan (not ~np.isfinite) for parity with the previous
    # per-source math.isnan checks
    skip = (nan_hl | np.isnan(xcen0) | np.isnan(ycen0)).astype(np.uint8)
    return {'data': arrays['data'], 'error': arrays['error'],
            'mask': arrays['mask'], 'segm': arrays['segm'],
            'labels': np.ascontiguousarray(cat.labels, dtype=np.intp),
            'xcen0': xcen0, 'ycen0': ycen0, 'sigma': sigma,
            'skip': skip, 'radius_hl': radius_hl, 'nan_hl': nan_hl}


@pytest.mark.parametrize('method', ['correct', 'mask', 'none'])
@pytest.mark.parametrize('with_error', [True, False])
@pytest.mark.parametrize('with_mask', [True, False])
def test_matches_reference(scene, method, with_error, with_mask):
    cat = make_catalog(scene, aperture_mask_method=method,
                       with_error=with_error, with_mask=with_mask)
    inp = _driver_inputs(cat)
    max_aper_size = max(scene['data'].size, 1_000_000)
    result = batch_centroid_win(
        inp['data'], error=inp['error'], mask=inp['mask'],
        segm=inp['segm'], labels=inp['labels'], xcen0=inp['xcen0'],
        ycen0=inp['ycen0'], sigma=inp['sigma'], skip=inp['skip'],
        seg_method=SEG_METHOD_CODES[method],
        compute_err=int(with_error), max_aper_size=max_aper_size)

    assert result.shape == (len(inp['labels']), 10)
    ref_kwargs = {
        'data_arr': scene['data'],
        'mask_arr': scene['mask'] if with_mask else None,
        'error_arr': scene['error'] if with_error else None,
        'segm_data': scene['segm'].data,
        'data_shape': scene['data'].shape,
        'do_correct': method == 'correct',
        'do_segm_mask': method != 'none',
        'compute_err': with_error,
        'max_aper_size': max_aper_size,
        'aperture_mask_method': method,
    }
    for i, label in enumerate(inp['labels']):
        ref = _reference_iterate_centroid_win(
            label, inp['xcen0'][i], inp['ycen0'][i],
            inp['radius_hl'][i], inp['nan_hl'][i], **ref_kwargs)
        # atol covers cancellation-limited near-zero central moments;
        # positions and flux are far from zero and are effectively
        # checked at rtol
        assert_allclose(result[i], np.array(ref), rtol=1e-12,
                        atol=1e-10, equal_nan=True)


def test_skip_rows(scene):
    cat = make_catalog(scene)
    inp = _driver_inputs(cat)
    skip = np.ones_like(inp['skip'])
    result = batch_centroid_win(
        inp['data'], error=inp['error'], mask=inp['mask'],
        segm=inp['segm'], labels=inp['labels'], xcen0=inp['xcen0'],
        ycen0=inp['ycen0'], sigma=inp['sigma'], skip=skip,
        seg_method=3, compute_err=1,
        max_aper_size=scene['data'].size)
    expected = np.array([np.nan, np.nan, 0.0, 0.0, 0.0, 0.0,
                         np.nan, np.nan, np.nan, np.nan])
    for row in result:
        assert_allclose(row, expected, equal_nan=True)


def test_oom_guard(scene):
    cat = make_catalog(scene)
    inp = _driver_inputs(cat)
    result = batch_centroid_win(
        inp['data'], error=inp['error'], mask=inp['mask'],
        segm=inp['segm'], labels=inp['labels'], xcen0=inp['xcen0'],
        ycen0=inp['ycen0'], sigma=inp['sigma'], skip=inp['skip'],
        seg_method=3, compute_err=1, max_aper_size=4)
    assert np.all(np.isnan(result[:, 0]))


def test_compute_err_without_error(scene):
    cat = make_catalog(scene, with_error=False)
    inp = _driver_inputs(cat)
    match = 'error must be provided when compute_err is set'
    with pytest.raises(ValueError, match=match):
        batch_centroid_win(
            inp['data'], error=None, mask=inp['mask'],
            segm=inp['segm'], labels=inp['labels'],
            xcen0=inp['xcen0'], ycen0=inp['ycen0'],
            sigma=inp['sigma'], skip=inp['skip'], seg_method=3,
            compute_err=1, max_aper_size=scene['data'].size)


def _call_driver(inp):
    return batch_centroid_win(
        inp['data'], error=inp['error'], mask=inp['mask'],
        segm=inp['segm'], labels=inp['labels'], xcen0=inp['xcen0'],
        ycen0=inp['ycen0'], sigma=inp['sigma'], skip=inp['skip'],
        seg_method=3, compute_err=1, max_aper_size=inp['data'].size)


@pytest.mark.parametrize('name', ['xcen0', 'ycen0', 'sigma', 'skip'])
def test_length_guard(scene, name):
    cat = make_catalog(scene)
    inp = _driver_inputs(cat)
    inp[name] = np.ascontiguousarray(inp[name][:-1])
    match = f'{name} must have the same length as labels'
    with pytest.raises(ValueError, match=match):
        _call_driver(inp)


@pytest.mark.parametrize('name', ['mask', 'segm', 'error'])
def test_shape_guard(scene, name):
    cat = make_catalog(scene)
    inp = _driver_inputs(cat)
    inp[name] = np.ascontiguousarray(inp[name][:-1])
    match = f'{name} must have the same shape as data'
    with pytest.raises(ValueError, match=match):
        _call_driver(inp)


def test_thread_safety(scene):
    cat = make_catalog(scene)
    inp = _driver_inputs(cat)
    max_aper_size = max(scene['data'].size, 1_000_000)

    def run():
        return batch_centroid_win(
            inp['data'], error=inp['error'], mask=inp['mask'],
            segm=inp['segm'], labels=inp['labels'],
            xcen0=inp['xcen0'], ycen0=inp['ycen0'],
            sigma=inp['sigma'], skip=inp['skip'], seg_method=3,
            compute_err=1, max_aper_size=max_aper_size)

    expected = run()
    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(lambda _: run(), range(16)))
    for res in results:
        assert_allclose(res, expected, rtol=0, atol=0, equal_nan=True)


def test_catalog_centroid_win(scene):
    """
    Test the SourceCatalog.centroid_win property against the batch driver.
    """
    cat = make_catalog(scene)
    results = cat._centroid_win_results
    assert results.shape == (cat.n_labels, 6)

    # Slicing: cached values slice along the source axis
    sub = cat[2:5]
    assert_allclose(sub.centroid_win, cat.centroid_win[2:5],
                    rtol=0, atol=0, equal_nan=True)

    # Scalar catalog
    scalar = cat[3]
    assert scalar.isscalar
    assert_allclose(np.atleast_2d(scalar.centroid_win),
                    cat.centroid_win[3:4], rtol=0, atol=0,
                    equal_nan=True)

    # The full-image cast bundle is shared by reference with slices
    cat._get_batch_arrays()
    assert cat[1:3]._batch_arrays_cache is cat._batch_arrays_cache


def test_batch_arrays_shared_before_build(scene):
    """
    Test that a catalog sliced before the batch arrays are built shares
    them with its parent, whichever of the two builds them first.
    """
    cat = make_catalog(scene)
    child = cat[1:4]
    scalar = cat[2]
    assert 'data' not in cat._batch_arrays_cache
    assert child._batch_arrays_cache is cat._batch_arrays_cache
    assert scalar._batch_arrays_cache is cat._batch_arrays_cache

    # Building on a slice populates the parent (and other slices)
    arrays = scalar._get_batch_arrays()
    assert cat._batch_arrays_cache['data'] is arrays['data']
    assert child._get_batch_arrays() is arrays
    assert cat._get_batch_arrays() is arrays

    # The lazily added arrays are shared the same way
    convdata = child._get_batch_convdata()
    assert cat._get_batch_convdata() is convdata


def test_detection_catalog(scene):
    """
    Test that the detection catalog is used when requested.
    """
    cat_det = make_catalog(scene)
    cat = SourceCatalog(scene['data'] * 1.5, scene['segm'],
                        error=scene['error'], mask=scene['mask'],
                        detection_catalog=cat_det)
    assert_allclose(cat.centroid_win, cat_det.centroid_win,
                    rtol=0, atol=0, equal_nan=True)


def test_catalog_centroid_win_precomputed_slice(scene):
    """
    Test that slicing a catalog after computing centroid_win is
    equivalent to computing centroid_win after slicing the catalog.
    """
    cat1 = make_catalog(scene)
    _ = cat1.centroid_win  # trigger cache, then slice
    cat2 = make_catalog(scene)
    sub2 = cat2[1:4]  # slice, then compute
    assert_allclose(cat1[1:4].centroid_win, sub2.centroid_win,
                    rtol=0, atol=0, equal_nan=True)
