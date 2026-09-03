# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the batch moments Cython kernels.
"""

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
from numpy.testing import assert_allclose

from photutils.segmentation import SegmentationImage, SourceCatalog
from photutils.segmentation._batch_catalog import (batch_central_moments,
                                                   batch_moment_err,
                                                   batch_raw_moments)
from photutils.segmentation.tests._batch_scene import (make_batch_scene,
                                                       make_catalog)


def _moment_cutout(cat, i):
    # The _moment_data_cutouts zeroing rules, per source
    slc = cat.slices[i]
    conv = cat._convolved_data[slc]
    segm = cat._segmentation_image.data[slc]
    bad = (~np.isfinite(conv) | (conv < 0)
           | (segm != cat.labels[i]))
    if cat._mask is not None:
        bad |= cat._mask[slc]
    cutout = conv.astype(float).copy()
    cutout[bad] = 0.0
    return cutout


def _reference_raw(cat):
    result = []
    for i in range(cat.n_labels):
        arr = _moment_cutout(cat, i)
        ny, nx = arr.shape
        y = np.arange(ny, dtype=float)
        x = np.arange(nx, dtype=float)
        yp = np.column_stack([np.ones(ny), y, y * y, y ** 3])
        xp = np.column_stack([np.ones(nx), x, x * x, x ** 3])
        result.append(yp.T @ arr @ xp)
    return np.array(result)


def _reference_central(cat, xcen, ycen):
    result = []
    for i in range(cat.n_labels):
        arr = _moment_cutout(cat, i)
        ny, nx = arr.shape
        yc = np.arange(ny, dtype=float) - ycen[i]
        xc = np.arange(nx, dtype=float) - xcen[i]
        yp = np.column_stack([np.ones(ny), yc, yc * yc, yc ** 3])
        xp = np.column_stack([np.ones(nx), xc, xc * xc, xc ** 3])
        result.append(yp.T @ arr @ xp)
    return np.array(result)


def _driver_inputs(cat):
    arrays = cat._get_batch_arrays()
    convdata = np.ascontiguousarray(cat._convolved_data,
                                    dtype=np.float64)
    slices = cat.slices
    return {
        'convdata': convdata,
        'mask': arrays['mask'],
        'segm': arrays['segm'],
        'labels': np.ascontiguousarray(np.atleast_1d(cat.labels),
                                       dtype=np.intp),
        'bbox_iymin': np.array([s[0].start for s in slices],
                               dtype=np.intp),
        'bbox_iymax': np.array([s[0].stop for s in slices],
                               dtype=np.intp),
        'bbox_ixmin': np.array([s[1].start for s in slices],
                               dtype=np.intp),
        'bbox_ixmax': np.array([s[1].stop for s in slices],
                               dtype=np.intp),
    }


def _reference_moment_err(cat, xcen, ycen):
    # The _centroid_err_cov accumulation rules, per source
    result = []
    for i in range(cat.n_labels):
        slc = cat.slices[i]
        moment_data = _moment_cutout(cat, i)
        err_sq = cat._error[slc].astype(float) ** 2
        total_mask = ((cat._segmentation_image.data[slc]
                       != cat.labels[i])
                      | ~np.isfinite(cat._data[slc]))
        if cat._mask is not None:
            total_mask |= cat._mask[slc]
        err_sq[total_mask | (moment_data == 0)] = 0.0
        yy, xx = np.mgrid[0:err_sq.shape[0], 0:err_sq.shape[1]]
        dx = xx - xcen[i]
        dy = yy - ycen[i]
        result.append((np.sum(err_sq), np.sum(err_sq * dx ** 2),
                       np.sum(err_sq * dy ** 2),
                       np.sum(err_sq * dx * dy)))
    return np.array(result)


def _cutout_centroids(raw):
    with np.errstate(invalid='ignore', divide='ignore'):
        xcen = raw[:, 0, 1] / raw[:, 0, 0]
        ycen = raw[:, 1, 0] / raw[:, 0, 0]
    return xcen, ycen


def _call_raw(inp):
    return batch_raw_moments(inp['convdata'], mask=inp['mask'],
                             segm=inp['segm'], labels=inp['labels'],
                             bbox_iymin=inp['bbox_iymin'],
                             bbox_iymax=inp['bbox_iymax'],
                             bbox_ixmin=inp['bbox_ixmin'],
                             bbox_ixmax=inp['bbox_ixmax'])


@pytest.fixture(scope='module')
def scene():
    return make_batch_scene()


@pytest.mark.parametrize('with_mask', [True, False])
def test_raw_moments(scene, with_mask):
    cat = make_catalog(scene, with_mask=with_mask)
    inp = _driver_inputs(cat)
    result = _call_raw(inp)
    assert result.shape == (cat.n_labels, 4, 4)
    assert_allclose(result, _reference_raw(cat), rtol=1e-12,
                    atol=1e-10, equal_nan=True)


def test_central_moments(scene):
    cat = make_catalog(scene)
    inp = _driver_inputs(cat)
    raw = _reference_raw(cat)
    xcen, ycen = _cutout_centroids(raw)
    assert np.all(np.isfinite(xcen))
    assert np.all(np.isfinite(ycen))
    result = batch_central_moments(
        inp['convdata'], mask=inp['mask'], segm=inp['segm'],
        labels=inp['labels'], bbox_iymin=inp['bbox_iymin'],
        bbox_iymax=inp['bbox_iymax'], bbox_ixmin=inp['bbox_ixmin'],
        bbox_ixmax=inp['bbox_ixmax'], xcen=xcen, ycen=ycen)
    assert_allclose(result, _reference_central(cat, xcen, ycen),
                    rtol=1e-12, atol=1e-10, equal_nan=True)


def test_central_moments_nan_centroid(scene):
    # A fully-masked source has zero total flux -> NaN centroid. The
    # central moments are then NaN everywhere except [0, 0], which
    # holds the (zero) flux sum
    cat = make_catalog(scene)
    inp = _driver_inputs(cat)
    n = cat.n_labels
    xcen = np.full(n, np.nan)
    ycen = np.full(n, np.nan)
    result = batch_central_moments(
        inp['convdata'], mask=inp['mask'], segm=inp['segm'],
        labels=inp['labels'], bbox_iymin=inp['bbox_iymin'],
        bbox_iymax=inp['bbox_iymax'], bbox_ixmin=inp['bbox_ixmin'],
        bbox_ixmax=inp['bbox_ixmax'], xcen=xcen, ycen=ycen)
    expected = _reference_central(cat, xcen, ycen)
    assert_allclose(result, expected, rtol=1e-12, atol=1e-10,
                    equal_nan=True)
    # The reference is NaN everywhere except the plain flux sum
    assert np.all(np.isfinite(result[:, 0, 0]))
    assert np.all(np.isnan(result[:, 1:, :]))
    assert np.all(np.isnan(result[:, :, 1:]))


def test_moment_err(scene):
    cat = make_catalog(scene)
    inp = _driver_inputs(cat)
    error = np.ascontiguousarray(scene['error'], dtype=np.float64)
    raw = _reference_raw(cat)
    xcen, ycen = _cutout_centroids(raw)
    result = batch_moment_err(
        error, convdata=inp['convdata'], mask=inp['mask'],
        segm=inp['segm'], labels=inp['labels'],
        bbox_iymin=inp['bbox_iymin'], bbox_iymax=inp['bbox_iymax'],
        bbox_ixmin=inp['bbox_ixmin'], bbox_ixmax=inp['bbox_ixmax'],
        xcen=xcen, ycen=ycen)
    assert result.shape == (cat.n_labels, 4)
    assert_allclose(result, _reference_moment_err(cat, xcen, ycen),
                    rtol=1e-12, atol=1e-10, equal_nan=True)


def _edge_catalog():
    """
    Make a small catalog whose sources exercise every pixel-inclusion
    rule of the moment kernels.
    """
    rng = np.random.default_rng(42)
    ny = nx = 12
    data = np.abs(rng.normal(2.0, 0.5, (ny, nx)))
    segmdata = np.zeros((ny, nx), dtype=int)
    segmdata[2:6, 2:6] = 1
    segmdata[7:11, 6:11] = 2
    convdata = data.copy()
    convdata[3, 3] = -1.5  # negative convolved value in a segment
    convdata[4, 4] = np.nan  # non-finite convolved value
    convdata[8, 8] = np.inf
    convdata[9, 9] = 0.0  # zero flux weight, excluded from the errors
    # Non-finite data with a finite convolved value is included in
    # the moments and excluded from the errors
    data[5, 5] = np.nan
    data[9, 7] = np.nan
    mask = np.zeros((ny, nx), dtype=bool)
    mask[2, 2] = True  # input-masked pixel in a segment
    mask[8, 9] = True
    error = np.full((ny, nx), 0.2)
    error[::3, ::2] = 0.5
    return SourceCatalog(data, SegmentationImage(segmdata),
                         convolved_data=convdata, error=error,
                         mask=mask)


def _reference_err_pixels(cat, i):
    """
    Boolean cutout mask of the pixels included in the error sums.
    """
    slc = cat.slices[i]
    moment_data = _moment_cutout(cat, i)
    included = np.isfinite(cat._data[slc])
    included &= cat._segmentation_image.data[slc] == cat.labels[i]
    if cat._mask is not None:
        included &= ~cat._mask[slc]
    return included & (moment_data != 0)


def test_edge_case_inclusion_rules():
    cat = _edge_catalog()
    inp = _driver_inputs(cat)
    raw = batch_raw_moments(
        inp['convdata'], mask=inp['mask'], segm=inp['segm'],
        labels=inp['labels'], bbox_iymin=inp['bbox_iymin'],
        bbox_iymax=inp['bbox_iymax'], bbox_ixmin=inp['bbox_ixmin'],
        bbox_ixmax=inp['bbox_ixmax'])
    reference_raw = _reference_raw(cat)
    # The excluded pixels must actually change the result, otherwise
    # this test would not discriminate the inclusion rules
    for i in range(cat.n_labels):
        slc = cat.slices[i]
        in_segment = (cat._segmentation_image.data[slc]
                      == cat.labels[i])
        assert (np.count_nonzero(_moment_cutout(cat, i))
                < np.count_nonzero(in_segment))

    assert_allclose(raw, reference_raw, rtol=1e-12, atol=1e-10,
                    equal_nan=True)

    xcen, ycen = _cutout_centroids(reference_raw)
    central = batch_central_moments(
        inp['convdata'], mask=inp['mask'], segm=inp['segm'],
        labels=inp['labels'], bbox_iymin=inp['bbox_iymin'],
        bbox_iymax=inp['bbox_iymax'], bbox_ixmin=inp['bbox_ixmin'],
        bbox_ixmax=inp['bbox_ixmax'], xcen=xcen, ycen=ycen)
    assert_allclose(central, _reference_central(cat, xcen, ycen),
                    rtol=1e-12, atol=1e-10, equal_nan=True)

    error = np.ascontiguousarray(cat._error, dtype=np.float64)
    err = batch_moment_err(
        error, convdata=inp['convdata'], mask=inp['mask'],
        segm=inp['segm'], labels=inp['labels'],
        bbox_iymin=inp['bbox_iymin'], bbox_iymax=inp['bbox_iymax'],
        bbox_ixmin=inp['bbox_ixmin'], bbox_ixmax=inp['bbox_ixmax'],
        xcen=xcen, ycen=ycen)
    reference_err = _reference_moment_err(cat, xcen, ycen)
    # The non-finite data pixel is excluded from the errors but not
    # from the moments, so the error sums use fewer pixels
    n_moment = np.array([np.count_nonzero(_moment_cutout(cat, i))
                         for i in range(cat.n_labels)])
    n_err = np.array([np.count_nonzero(_reference_err_pixels(cat, i))
                      for i in range(cat.n_labels)])
    assert np.all(n_err < n_moment)

    assert_allclose(err, reference_err, rtol=1e-12, atol=1e-10,
                    equal_nan=True)


def test_catalog_moments(scene):
    cat = make_catalog(scene)
    assert_allclose(cat.moments, _reference_raw(cat), rtol=1e-12,
                    atol=1e-10)
    raw = cat._array('moments')
    with np.errstate(invalid='ignore'):
        xcen = raw[:, 0, 1] / raw[:, 0, 0]
        ycen = raw[:, 1, 0] / raw[:, 0, 0]
    assert_allclose(cat.moments_central,
                    _reference_central(cat, xcen, ycen), rtol=1e-12,
                    atol=1e-10, equal_nan=True)
    # Downstream shape properties still finite where expected
    assert np.all(np.isfinite(np.atleast_1d(
        cat.semimajor_axis.value)[~np.isnan(
            np.atleast_1d(cat.semimajor_axis.value))]))


def test_catalog_centroid_err(scene):
    # _centroid_err_cov = normalized batch_moment_err accumulators.
    # Rebuild it from the reference accumulation of test_moment_err
    cat = make_catalog(scene)
    cov = cat._centroid_err_cov
    assert cov.shape == (cat.n_labels, 2, 2)
    raw = _reference_raw(cat)
    m00 = raw[:, 0, 0]
    with np.errstate(invalid='ignore'):
        xcen = raw[:, 0, 1] / m00
        ycen = raw[:, 1, 0] / m00
    for i in range(cat.n_labels):
        if not np.isfinite(m00[i]) or m00[i] <= 0:
            assert np.all(np.isnan(cov[i]))
            continue
        slc = cat.slices[i]
        moment_data = _moment_cutout(cat, i)
        err_sq = scene['error'][slc].astype(float) ** 2
        total_mask = ((cat._segmentation_image.data[slc]
                       != cat.labels[i])
                      | ~np.isfinite(scene['data'][slc])
                      | scene['mask'][slc])
        err_sq[total_mask | (moment_data == 0)] = 0.0
        yy, xx = np.mgrid[0:err_sq.shape[0], 0:err_sq.shape[1]]
        dx = xx - xcen[i]
        dy = yy - ycen[i]
        norm = 1.0 / m00[i] ** 2
        var_x = np.sum(err_sq * dx ** 2) * norm
        var_y = np.sum(err_sq * dy ** 2) * norm
        cov_xy = np.sum(err_sq * dx * dy) * norm
        if np.atleast_1d(cat._singular_covariance_mask)[i]:
            corr = np.sum(err_sq) * (1.0 / 12.0) * norm
            if var_x * var_y - cov_xy ** 2 < corr ** 2:
                var_x += corr
                var_y += corr
        assert_allclose(cov[i], [[var_x, cov_xy], [cov_xy, var_y]],
                        rtol=1e-12, atol=1e-10)


BBOX_NAMES = ['bbox_iymin', 'bbox_iymax', 'bbox_ixmin', 'bbox_ixmax']


def _guard_inputs(scene):
    """
    Build the driver inputs for the input-validation tests, including
    the centroid and error arrays.
    """
    cat = make_catalog(scene)
    inp = _driver_inputs(cat)
    xcen, ycen = _cutout_centroids(_reference_raw(cat))
    inp['xcen'] = xcen
    inp['ycen'] = ycen
    inp['error'] = np.ascontiguousarray(scene['error'],
                                        dtype=np.float64)
    return inp


def _call_central(inp):
    return batch_central_moments(
        inp['convdata'], mask=inp['mask'], segm=inp['segm'],
        labels=inp['labels'], bbox_iymin=inp['bbox_iymin'],
        bbox_iymax=inp['bbox_iymax'], bbox_ixmin=inp['bbox_ixmin'],
        bbox_ixmax=inp['bbox_ixmax'], xcen=inp['xcen'],
        ycen=inp['ycen'])


def _call_err(inp):
    return batch_moment_err(
        inp['error'], convdata=inp['convdata'], mask=inp['mask'],
        segm=inp['segm'], labels=inp['labels'],
        bbox_iymin=inp['bbox_iymin'], bbox_iymax=inp['bbox_iymax'],
        bbox_ixmin=inp['bbox_ixmin'], bbox_ixmax=inp['bbox_ixmax'],
        xcen=inp['xcen'], ycen=inp['ycen'])


@pytest.mark.parametrize('name', BBOX_NAMES)
def test_raw_moments_length_guard(scene, name):
    inp = _guard_inputs(scene)
    inp[name] = np.ascontiguousarray(inp[name][:-1])
    match = f'{name} must have the same length as labels'
    with pytest.raises(ValueError, match=match):
        _call_raw(inp)


@pytest.mark.parametrize('name', ['mask', 'segm'])
def test_raw_moments_shape_guard(scene, name):
    inp = _guard_inputs(scene)
    inp[name] = np.ascontiguousarray(inp[name][:-1])
    match = f'{name} must have the same shape as convdata'
    with pytest.raises(ValueError, match=match):
        _call_raw(inp)


@pytest.mark.parametrize('name', [*BBOX_NAMES, 'xcen', 'ycen'])
def test_central_moments_length_guard(scene, name):
    inp = _guard_inputs(scene)
    inp[name] = np.ascontiguousarray(inp[name][:-1])
    match = f'{name} must have the same length as labels'
    with pytest.raises(ValueError, match=match):
        _call_central(inp)


@pytest.mark.parametrize('name', ['mask', 'segm'])
def test_central_moments_shape_guard(scene, name):
    inp = _guard_inputs(scene)
    inp[name] = np.ascontiguousarray(inp[name][:-1])
    match = f'{name} must have the same shape as convdata'
    with pytest.raises(ValueError, match=match):
        _call_central(inp)


@pytest.mark.parametrize('name', [*BBOX_NAMES, 'xcen', 'ycen'])
def test_moment_err_length_guard(scene, name):
    inp = _guard_inputs(scene)
    inp[name] = np.ascontiguousarray(inp[name][:-1])
    match = f'{name} must have the same length as labels'
    with pytest.raises(ValueError, match=match):
        _call_err(inp)


@pytest.mark.parametrize('name', ['convdata', 'mask', 'segm'])
def test_moment_err_shape_guard(scene, name):
    inp = _guard_inputs(scene)
    inp[name] = np.ascontiguousarray(inp[name][:-1])
    match = f'{name} must have the same shape as error'
    with pytest.raises(ValueError, match=match):
        _call_err(inp)


def test_thread_safety(scene):
    cat = make_catalog(scene)
    inp = _driver_inputs(cat)

    expected = _call_raw(inp)
    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(lambda _: _call_raw(inp),
                                    range(16)))
    for res in results:
        assert_allclose(res, expected, rtol=0, atol=0, equal_nan=True)
