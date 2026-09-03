# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the batch quadratic centroid path.
"""

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from photutils.segmentation import SegmentationImage, SourceCatalog
from photutils.segmentation._batch_catalog import batch_quad_boxes
from photutils.segmentation.tests._batch_scene import (make_batch_scene,
                                                       make_catalog)


def _centroid_quad_var(coeffs, xm_rel, ym_rel, pinv, box_var):
    # Verbatim port of the previous ``_centroid_quad_var`` method
    c10, c01, c11, c20, c02 = coeffs[1:]
    det = 4.0 * c20 * c02 - c11 * c11
    grad_x = np.array(
        [0.0,
         -2.0 * c02 / det,
         c11 / det,
         (c01 + 2.0 * c11 * xm_rel) / det,
         -4.0 * c02 * xm_rel / det,
         (-2.0 * c10 - 4.0 * c20 * xm_rel) / det])
    grad_y = np.array(
        [0.0,
         c11 / det,
         -2.0 * c20 / det,
         (c10 + 2.0 * c11 * ym_rel) / det,
         (-2.0 * c01 - 4.0 * c02 * ym_rel) / det,
         -4.0 * c20 * ym_rel / det])
    u_x = grad_x @ pinv
    u_y = grad_y @ pinv
    var_x = np.sum(u_x**2 * box_var)
    var_y = np.sum(u_y**2 * box_var)
    cov_xy = np.sum(u_x * u_y * box_var)
    return var_x, var_y, cov_xy


def _reference_centroid_quad_results(cat):
    """
    Compute the quadratic centroid results of each source.

    This is a verbatim port of the per-source Python implementation
    of ``_centroid_quad_results`` that the batch path replaces. It is
    the numerical reference for the batch path.
    """
    xi = np.arange(3)
    x, y = np.meshgrid(xi, xi)
    x = x.ravel()
    y = y.ravel()
    coeff_matrix = np.empty((9, 6), dtype=float)
    coeff_matrix[:, 0] = 1
    coeff_matrix[:, 1] = x
    coeff_matrix[:, 2] = y
    coeff_matrix[:, 3] = x * y
    coeff_matrix[:, 4] = x * x
    coeff_matrix[:, 5] = y * y
    pinv = np.linalg.pinv(coeff_matrix)

    compute_err = cat._error is not None

    _nan = np.nan
    nan_result = (_nan, _nan, _nan, _nan, _nan)
    results = []

    for cutout, error_cutout, mask in zip(cat._data_cutouts,
                                          cat._error_cutouts,
                                          cat._cutout_total_masks,
                                          strict=True):
        ny, nx = cutout.shape

        if ny < 3 or nx < 3:
            results.append(nan_result)
            continue

        if np.all(mask):
            results.append(nan_result)
            continue

        cutout = np.array(cutout, dtype=float)
        cutout[mask] = 0.0

        yidx, xidx = np.unravel_index(np.argmax(cutout), cutout.shape)

        if xidx == 0 or xidx == nx - 1 or yidx == 0 or yidx == ny - 1:
            results.append((float(xidx), float(yidx), _nan, _nan,
                            _nan))
            continue

        xidx0 = xidx - 1
        yidx0 = yidx - 1
        cutout_flat = cutout[yidx0:yidx0 + 3, xidx0:xidx0 + 3].ravel()

        c = pinv @ cutout_flat
        c10, c01, c11, c20, c02 = c[1], c[2], c[3], c[4], c[5]

        det = 4.0 * c20 * c02 - c11 * c11
        if det <= 0 or c20 > 0:
            results.append(nan_result)
            continue

        xm_rel = (c01 * c11 - 2.0 * c02 * c10) / det
        ym_rel = (c10 * c11 - 2.0 * c20 * c01) / det
        xm = xm_rel + xidx0
        ym = ym_rel + yidx0

        if not (0.0 < xm < (nx - 1.0) and 0.0 < ym < (ny - 1.0)):
            results.append(nan_result)
            continue

        var_x = var_y = cov_xy = _nan
        if compute_err:
            box_var = (error_cutout[yidx0:yidx0 + 3, xidx0:xidx0 + 3]
                       .astype(float).ravel()**2)
            box_var[mask[yidx0:yidx0 + 3,
                         xidx0:xidx0 + 3].ravel()] = 0.0
            var_x, var_y, cov_xy = _centroid_quad_var(
                c, xm_rel, ym_rel, pinv, box_var)

        results.append((xm, ym, var_x, var_y, cov_xy))

    results = np.array(results)

    nan_mask = (np.isnan(results[:, 0]) | np.isnan(results[:, 1]))
    if np.any(nan_mask):
        cutout_centroid = cat._array('cutout_centroid')
        results[nan_mask, 0:2] = cutout_centroid[nan_mask]
        iso_cov = cat._centroid_err_cov
        results[nan_mask, 2] = iso_cov[nan_mask, 0, 0]
        results[nan_mask, 3] = iso_cov[nan_mask, 1, 1]
        results[nan_mask, 4] = iso_cov[nan_mask, 0, 1]

    return results


@pytest.fixture(scope='module')
def scene():
    return make_batch_scene()


@pytest.mark.parametrize('with_error', [True, False])
@pytest.mark.parametrize('with_mask', [True, False])
def test_matches_reference(scene, with_error, with_mask):
    cat = make_catalog(scene, with_error=with_error, with_mask=with_mask)
    expected = _reference_centroid_quad_results(cat)
    assert_allclose(cat._centroid_quad_results, expected, rtol=1e-12,
                    equal_nan=True)
    assert_allclose(cat.centroid_quad_err[:, 0],
                    np.sqrt(expected[:, 2]), rtol=1e-12, equal_nan=True)


def _edge_case_catalog():
    # Sources exercising every status of the box gather: a cutout
    # smaller than 3x3, a fully masked source, a peak on the cutout
    # edge, a negative source whose peak is a masked (zero) pixel, a
    # rejected (non-maximum) fit, a normal source, and a corner peak
    data = np.zeros((60, 60))
    yy, xx = np.mgrid[0:60, 0:60]
    segm_data = np.zeros(data.shape, dtype=int)
    error = np.full(data.shape, 0.3)
    mask = np.zeros(data.shape, dtype=bool)

    # Label 1: 2x2 segment (too small)
    data[2:4, 2:4] = 5.0
    segm_data[2:4, 2:4] = 1
    # Label 2: fully masked 5x5 segment
    data[2:7, 10:15] = 5.0
    segm_data[2:7, 10:15] = 2
    mask[2:7, 10:15] = True
    # Label 3: peak on the cutout edge (a ramp)
    data[10:15, 2:7] = xx[10:15, 2:7]
    segm_data[10:15, 2:7] = 3
    # Label 4: negative source with one masked interior pixel, whose
    # zero value becomes the peak of the zero-filled cutout
    data[10:17, 10:17] = -5.0 - ((xx[10:17, 10:17] - 13) ** 2
                                 + (yy[10:17, 10:17] - 13) ** 2)
    segm_data[10:17, 10:17] = 4
    mask[12, 12] = True
    # Label 5: an interior peak whose 3x3 box has high corners, so the
    # fitted quadratic has a positive x curvature and is rejected
    data[22:25, 22:25] = [[8.0, 0.0, 8.0], [0.0, 10.0, 0.0],
                          [8.0, 0.0, 8.0]]
    segm_data[20:27, 20:27] = 5
    # Label 6: normal Gaussian source
    data[30:45, 30:45] = 20.0 * np.exp(
        -((xx[30:45, 30:45] - 37.3) ** 2 + (yy[30:45, 30:45] - 37.6) ** 2)
        / 8.0)
    segm_data[30:45, 30:45] = 6
    # Label 7: peak at the cutout corner with a NaN data value
    data[50:55, 50:55] = 1.0
    data[50, 50] = 10.0
    data[52, 52] = np.nan
    segm_data[50:55, 50:55] = 7

    return SourceCatalog(data, SegmentationImage(segm_data), error=error,
                         mask=mask)


def test_edge_cases():
    cat = _edge_case_catalog()
    expected = _reference_centroid_quad_results(cat)
    assert_allclose(cat._centroid_quad_results, expected, rtol=1e-12,
                    equal_nan=True)

    # Check that the intended branches were exercised: the box
    # statuses, the rejected fit, and the fallback to the isophotal
    # centroid
    arrays = cat._get_batch_arrays()
    iymin, iymax, ixmin, ixmax = cat._get_batch_bboxes()
    status, peak, boxes, box_var = batch_quad_boxes(
        arrays['data'], error=arrays['error'], mask=arrays['mask'],
        segm=arrays['segm'],
        labels=np.ascontiguousarray(cat.labels, dtype=np.intp),
        bbox_iymin=iymin, bbox_iymax=iymax, bbox_ixmin=ixmin,
        bbox_ixmax=ixmax, compute_err=1)
    assert_array_equal(status, [1, 2, 3, 0, 0, 0, 3])
    assert_array_equal(peak[0], [-1, -1])
    assert_array_equal(peak[3], [2, 2])  # the masked (zero) pixel
    assert np.all(boxes[3] <= 0)
    assert box_var[3, 4] == 0  # the masked center pixel
    assert np.all(box_var[5] == 0.3**2)
    # Edge peaks report the peak position without errors
    assert_allclose(cat._centroid_quad_results[2, :2], peak[2])
    assert np.all(np.isnan(cat._centroid_quad_results[2, 2:]))
    assert_allclose(cat._centroid_quad_results[6, :2], peak[6])
    # The rejected fit falls back to the isophotal centroid
    assert_array_equal(peak[4], [3, 3])
    assert_allclose(cat._centroid_quad_results[4, :2],
                    cat.cutout_centroid[4])


def _driver_inputs(cat):
    arrays = cat._get_batch_arrays()
    iymin, iymax, ixmin, ixmax = cat._get_batch_bboxes()
    return {'data': arrays['data'], 'error': arrays['error'],
            'mask': arrays['mask'], 'segm': arrays['segm'],
            'labels': np.ascontiguousarray(cat.labels, dtype=np.intp),
            'bbox_iymin': iymin, 'bbox_iymax': iymax,
            'bbox_ixmin': ixmin, 'bbox_ixmax': ixmax, 'compute_err': 1}


def _call_driver(inp):
    return batch_quad_boxes(inp.pop('data'), **inp)


@pytest.mark.parametrize('name', ['bbox_iymin', 'bbox_iymax', 'bbox_ixmin',
                                  'bbox_ixmax'])
def test_length_guard(scene, name):
    cat = make_catalog(scene)
    inp = _driver_inputs(cat)
    inp[name] = inp[name][:-1]
    match = f'{name} must have the same length as labels'
    with pytest.raises(ValueError, match=match):
        _call_driver(inp)


@pytest.mark.parametrize('name', ['error', 'mask', 'segm'])
def test_shape_guard(scene, name):
    cat = make_catalog(scene)
    inp = _driver_inputs(cat)
    inp[name] = inp[name][:-1, :]
    match = f'{name} must have the same shape as data'
    with pytest.raises(ValueError, match=match):
        _call_driver(inp)


def test_compute_err_without_error(scene):
    cat = make_catalog(scene)
    inp = _driver_inputs(cat)
    inp['error'] = None
    match = 'error must be provided when compute_err is set'
    with pytest.raises(ValueError, match=match):
        _call_driver(inp)


def test_no_error_boxes(scene):
    cat = make_catalog(scene)
    inp = _driver_inputs(cat)
    inp['error'] = None
    inp['compute_err'] = 0
    _, _, _, box_var = _call_driver(inp)
    assert np.all(box_var == 0)


def test_thread_safety(scene):
    cat = make_catalog(scene)
    inp = _driver_inputs(cat)
    expected = _call_driver(dict(inp))

    def run(_):
        return _call_driver(dict(inp))

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(run, range(8)))
    for result in results:
        for got, want in zip(result, expected, strict=True):
            assert_array_equal(got, want)
