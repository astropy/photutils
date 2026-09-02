# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the batch Kron radius Cython kernel.
"""

import math
import warnings
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import astropy.units as u
import numpy as np
import pytest
from numpy.testing import assert_allclose

from photutils.segmentation import SegmentationImage, SourceCatalog
from photutils.segmentation._batch_catalog import batch_kron_radius
from photutils.segmentation.tests._batch_scene import (make_batch_scene,
                                                       make_catalog)
from photutils.segmentation.utils import _mask_to_mirrored_value


def _reference_measured_kron_radius(cat):
    """
    Compute the unscaled measured Kron radius of each source.

    This is a verbatim port of the per-source Python implementation
    of ``_measured_kron_radius`` that the batch Cython kernel
    replaces. It is the numerical reference for the kernel.
    """
    scale = 6.0

    xcen_arr = cat._array('x_centroid')
    ycen_arr = cat._array('y_centroid')
    a_arr = cat._array('semimajor_axis').value * scale
    b_arr = cat._array('semiminor_axis').value * scale
    theta_arr = cat._array('orientation').to_value(u.radian)
    cxx_arr = cat._array('ellipse_cxx').value
    cxy_arr = cat._array('ellipse_cxy').value
    cyy_arr = cat._array('ellipse_cyy').value
    all_masked = cat._all_masked

    data_full = cat._data
    data_shape = data_full.shape
    mask_full = cat._mask
    segm_data = cat._segmentation_image.data
    max_size = max(data_full.size, 1_000_000)
    kron_min = cat.kron_params[1]
    min_circ_radius = (cat.kron_params[2]
                       if len(cat.kron_params) == 3 else 0.0)
    aperture_mask_method = cat.aperture_mask_method

    kron_radius = []
    for (label, xc, yc, a, b, theta, cxx_, cxy_, cyy_,
         masked) in zip(cat.labels, xcen_arr, ycen_arr, a_arr, b_arr,
                        theta_arr, cxx_arr, cxy_arr, cyy_arr,
                        all_masked, strict=True):
        if masked or not (math.isfinite(xc) and math.isfinite(yc)
                          and math.isfinite(a) and math.isfinite(b)
                          and math.isfinite(theta)):
            kron_radius.append(np.nan)
            continue

        use_circular = (a == 0 and b == 0)
        if use_circular:
            if min_circ_radius <= 0:
                kron_radius.append(np.nan)
                continue
            half_w = min_circ_radius
            half_h = min_circ_radius
        else:
            cos_theta = math.cos(theta)
            sin_theta = math.sin(theta)
            half_w = math.sqrt(a * a * cos_theta * cos_theta
                               + b * b * sin_theta * sin_theta)
            half_h = math.sqrt(a * a * sin_theta * sin_theta
                               + b * b * cos_theta * cos_theta)

        ixmin = math.floor(xc - half_w + 0.5)
        ixmax = math.floor(xc + half_w + 0.5) + 1
        iymin = math.floor(yc - half_h + 0.5)
        iymax = math.floor(yc + half_h + 0.5) + 1

        if (ixmax - ixmin) * (iymax - iymin) > max_size:
            kron_radius.append(np.nan)
            continue

        dx_min = max(0, -ixmin)
        dy_min = max(0, -iymin)
        dx_max = max(0, ixmax - data_shape[1])
        dy_max = max(0, iymax - data_shape[0])
        lg_xmin = ixmin + dx_min
        lg_xmax = ixmax - dx_max
        lg_ymin = iymin + dy_min
        lg_ymax = iymax - dy_max
        if lg_xmin >= lg_xmax or lg_ymin >= lg_ymax:
            kron_radius.append(np.nan)
            continue

        slc_lg = (slice(lg_ymin, lg_ymax), slice(lg_xmin, lg_xmax))
        data = data_full[slc_lg].astype(float)

        data_mask = ~np.isfinite(data)
        if mask_full is not None:
            data_mask |= mask_full[slc_lg]

        if aperture_mask_method != 'none':
            seg_cut = segm_data[slc_lg]
            segm_mask = (seg_cut != label) & (seg_cut != 0)
            if aperture_mask_method == 'mask':
                mask = data_mask | segm_mask
            else:
                mask = data_mask
            if aperture_mask_method == 'correct':
                cutout_xycen = (xc - max(0, ixmin), yc - max(0, iymin))
                data = _mask_to_mirrored_value(data, segm_mask,
                                               cutout_xycen,
                                               mask=mask)
        else:
            mask = data_mask

        ny, nx = data.shape
        xval = np.arange(nx) - (xc - lg_xmin)
        yval = np.arange(ny) - (yc - lg_ymin)
        yy = yval[:, np.newaxis]
        xx = xval[np.newaxis, :]

        rr_sq = cxx_ * xx * xx + cxy_ * xx * yy + cyy_ * yy * yy
        rr = np.sqrt(np.maximum(rr_sq, 0.0))

        if use_circular:
            dx = xx
            dy = yy
            pixel_mask = ((dx * dx + dy * dy)
                          <= min_circ_radius * min_circ_radius) & ~mask
        else:
            pixel_mask = (rr <= scale) & ~mask

        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            flux_numer = np.sum(data[pixel_mask] * rr[pixel_mask])
            flux_denom = np.sum(data[pixel_mask])

        if flux_numer <= 0 or flux_denom <= 0:
            kron_radius.append(kron_min)
            continue

        kron_radius.append(flux_numer / flux_denom)

    return np.array(kron_radius)


@pytest.fixture(scope='module')
def scene():
    return make_batch_scene()


@pytest.mark.parametrize('method', ['correct', 'mask', 'none'])
@pytest.mark.parametrize('with_mask', [True, False])
def test_matches_reference(scene, method, with_mask):
    cat = make_catalog(scene, aperture_mask_method=method,
                       with_mask=with_mask)
    expected = _reference_measured_kron_radius(cat)
    assert_allclose(cat._measured_kron_radius, expected, rtol=1e-12,
                    equal_nan=True)
    assert np.any(np.isfinite(expected))


def test_negative_data_gives_minimum_radius(scene):
    # A negative background within the measurement ellipse makes the
    # Kron numerator or denominator non-positive, which gives the
    # minimum Kron radius
    data = scene['data'].copy()
    data[scene['segm'].data == 0] = -20.0
    cat = SourceCatalog(data, scene['segm'], kron_params=(2.5, 1.7))
    expected = _reference_measured_kron_radius(cat)
    assert np.any(expected == 1.7)
    assert_allclose(cat._measured_kron_radius, expected, rtol=1e-12,
                    equal_nan=True)


def test_nonfinite_and_masked_sources(scene):
    # A completely masked source or a non-finite centroid is NaN
    mask = scene['mask'].copy()
    segm = scene['segm']
    slc = segm.slices[0]
    mask[slc] |= segm.data[slc] == segm.labels[0]
    cat = SourceCatalog(scene['data'], segm, mask=mask)
    xcen = cat.x_centroid.copy()
    xcen[1] = np.nan
    cat.__dict__['x_centroid'] = xcen
    expected = _reference_measured_kron_radius(cat)
    assert np.isnan(expected[0])
    assert np.isnan(expected[1])
    assert_allclose(cat._measured_kron_radius, expected, rtol=1e-12,
                    equal_nan=True)


@pytest.mark.parametrize('kron_params', [(2.5, 1.4, 5.0), (2.5, 1.4)])
def test_circular_fallback(scene, kron_params):
    # Zero semimajor and semiminor axes use the circle of minimum
    # radius, or give NaN if there is no minimum circular radius
    cat = make_catalog(scene)
    cat.kron_params = kron_params
    n_src = cat.n_labels
    zero = np.zeros(n_src) << u.pix
    cxx = np.ones(n_src) / (u.pix * u.pix)
    cxy = np.zeros(n_src) / (u.pix * u.pix)
    with (patch.object(type(cat), 'semimajor_axis',
                       new_callable=lambda: property(lambda _self: zero)),
          patch.object(type(cat), 'semiminor_axis',
                       new_callable=lambda: property(lambda _self: zero)),
          patch.object(type(cat), 'ellipse_cxx',
                       new_callable=lambda: property(lambda _self: cxx)),
          patch.object(type(cat), 'ellipse_cyy',
                       new_callable=lambda: property(lambda _self: cxx)),
          patch.object(type(cat), 'ellipse_cxy',
                       new_callable=lambda: property(lambda _self: cxy))):
        expected = _reference_measured_kron_radius(cat)
        result = cat._measured_kron_radius
    if len(kron_params) == 3:
        assert np.any(np.isfinite(expected))
    else:
        assert np.all(np.isnan(expected))
    assert_allclose(result, expected, rtol=1e-12, equal_nan=True)


def test_oom_guard_and_off_image(scene):
    cat = make_catalog(scene)
    n_src = cat.n_labels
    huge = np.full(n_src, 1e6) << u.pix
    with patch.object(type(cat), 'semimajor_axis',
                      new_callable=lambda: property(lambda _self: huge)):
        assert np.all(np.isnan(cat._measured_kron_radius))

    cat = make_catalog(scene)
    cat.__dict__['x_centroid'] = np.full(n_src, 5000.0)
    cat.__dict__['y_centroid'] = np.full(n_src, 5000.0)
    assert np.all(np.isnan(cat._measured_kron_radius))


def test_partially_off_image():
    # An ellipse whose bounding box extends beyond every data edge
    data = np.zeros((15, 15))
    yy, xx = np.mgrid[0:15, 0:15]
    data += 10.0 * np.exp(-((xx - 7) ** 2 + (yy - 7) ** 2) / 32.0)
    segm = SegmentationImage((data > 0.5).astype(int))
    cat = SourceCatalog(data, segm)
    expected = _reference_measured_kron_radius(cat)
    assert np.isfinite(expected[0])
    assert_allclose(cat._measured_kron_radius, expected, rtol=1e-12)


def _driver_inputs(cat):
    arrays = cat._get_batch_arrays()
    n_src = cat.n_labels
    return {'data': arrays['data'], 'mask': arrays['mask'],
            'segm': arrays['segm'],
            'labels': np.ascontiguousarray(cat.labels, dtype=np.intp),
            'xcen': np.ascontiguousarray(cat.x_centroid, dtype=float),
            'ycen': np.ascontiguousarray(cat.y_centroid, dtype=float),
            'semimajor': np.ascontiguousarray(cat.semimajor_axis.value),
            'semiminor': np.ascontiguousarray(cat.semiminor_axis.value),
            'theta': np.ascontiguousarray(
                cat.orientation.to_value(u.radian)),
            'cxx': np.ascontiguousarray(cat.ellipse_cxx.value),
            'cxy': np.ascontiguousarray(cat.ellipse_cxy.value),
            'cyy': np.ascontiguousarray(cat.ellipse_cyy.value),
            'skip': np.zeros(n_src, dtype=np.uint8),
            'seg_method': 3, 'scale': 6.0, 'min_circ_radius': 0.0,
            'max_aper_size': 1_000_000}


def _call_driver(inp):
    return batch_kron_radius(inp.pop('data'), **inp)


@pytest.mark.parametrize('name', ['xcen', 'ycen', 'semimajor', 'semiminor',
                                  'theta', 'cxx', 'cxy', 'cyy', 'skip'])
def test_length_guard(scene, name):
    cat = make_catalog(scene)
    inp = _driver_inputs(cat)
    inp[name] = inp[name][:-1]
    with pytest.raises(ValueError, match='same length as labels'):
        _call_driver(inp)


@pytest.mark.parametrize('name', ['mask', 'segm'])
def test_shape_guard(scene, name):
    cat = make_catalog(scene)
    inp = _driver_inputs(cat)
    inp[name] = inp[name][:-1, :]
    with pytest.raises(ValueError, match='same shape as data'):
        _call_driver(inp)


def test_skip_rows(scene):
    cat = make_catalog(scene)
    inp = _driver_inputs(cat)
    inp['skip'][0] = 1
    result = _call_driver(inp)
    assert np.all(np.isnan(result[0]))
    assert np.all(np.isfinite(result[1:]))


def test_thread_safety(scene):
    cat = make_catalog(scene)
    inp = _driver_inputs(cat)
    expected = _call_driver(dict(inp))

    def run(_):
        return _call_driver(dict(inp))

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(run, range(8)))
    for result in results:
        assert_allclose(result, expected, rtol=0, atol=0)


def test_catalog_kron_radius(scene):
    # The catalog kron_radius applies the minimum radius and the
    # measurement-scale limit to the measured value
    cat = make_catalog(scene)
    measured = _reference_measured_kron_radius(cat)
    expected = measured.copy()
    expected[expected > 6.0] = np.nan
    expected[expected < cat.kron_params[1]] = cat.kron_params[1]
    assert_allclose(cat.kron_radius.value, expected, rtol=1e-12,
                    equal_nan=True)
