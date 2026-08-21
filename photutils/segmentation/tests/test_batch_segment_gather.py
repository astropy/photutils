# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the batch segment pixel gather.
"""

from concurrent.futures import ThreadPoolExecutor

import astropy.units as u
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from photutils.morphology import gini as gini_func
from photutils.segmentation import SegmentationImage, SourceCatalog
from photutils.segmentation._batch_catalog import batch_segment_gather
from photutils.segmentation.tests._batch_scene import (make_batch_scene,
                                                       make_catalog)


def _reference_values(masked_cutouts):
    """
    Get a list of 1D arrays of the unmasked values of each masked
    cutout.

    This is a verbatim port of the previous ``_get_values`` method,
    applied to the public masked cutouts, that the batch segment
    gather replaces. It is the reference for the gather.
    """
    values = []
    for arr in masked_cutouts:
        compressed = arr.compressed()
        if len(compressed) == 0:
            compressed = np.array([np.nan])
        values.append(compressed)
    return values


def _assert_values_equal(values, expected):
    assert len(values) == len(expected)
    for got, want in zip(values, expected, strict=True):
        assert got.dtype == np.float64
        assert_array_equal(got, want)


@pytest.fixture(scope='module')
def scene():
    return make_batch_scene()


@pytest.mark.parametrize('with_error', [True, False])
@pytest.mark.parametrize('with_mask', [True, False])
@pytest.mark.parametrize('with_background', [True, False])
def test_matches_reference(scene, with_error, with_mask, with_background):
    background = None
    if with_background:
        rng = np.random.default_rng(1)
        background = rng.normal(0.5, 0.1, scene['data'].shape)
    cat = SourceCatalog(scene['data'], scene['segm'],
                        error=scene['error'] if with_error else None,
                        mask=scene['mask'] if with_mask else None,
                        background=background)
    _assert_values_equal(cat._data_values,
                         _reference_values(cat.data_cutout_masked))
    if with_error:
        _assert_values_equal(cat._error_values,
                             _reference_values(cat.error_cutout_masked))
    else:
        assert all(value is None for value in cat._error_values)
    if with_background:
        _assert_values_equal(
            cat._background_values,
            _reference_values(cat.background_cutout_masked))
    else:
        assert all(value is None for value in cat._background_values)
    assert_array_equal(cat._all_masked,
                       [np.all(mask) for mask in cat._cutout_total_masks])


def test_all_masked_source(scene):
    # A completely masked source gathers a single NaN and is flagged as
    # all masked; a source with a single unmasked pixel is not
    mask = scene['mask'].copy()
    segm = scene['segm']
    slc = segm.slices[0]
    mask[slc] |= segm.data[slc] == segm.labels[0]
    slc = segm.slices[1]
    single = segm.data[slc] == segm.labels[1]
    single[np.flatnonzero(single.ravel())[0] // single.shape[1],
           np.flatnonzero(single.ravel())[0] % single.shape[1]] = False
    mask[slc] |= single
    cat = SourceCatalog(scene['data'], segm, mask=mask,
                        error=scene['error'])
    expected = _reference_values(cat.data_cutout_masked)
    assert expected[0].size == 1
    assert np.isnan(expected[0][0])
    assert expected[1].size == 1
    assert np.isfinite(expected[1][0])
    _assert_values_equal(cat._data_values, expected)
    _assert_values_equal(cat._error_values,
                         _reference_values(cat.error_cutout_masked))
    assert cat._all_masked[0]
    assert not cat._all_masked[1]
    assert_array_equal(cat._all_masked,
                       [np.all(mask) for mask in cat._cutout_total_masks])
    assert np.isnan(cat.segment_flux[0])
    assert np.isnan(cat.area.value[0])
    assert cat.area.value[1] == 1


def test_properties(scene):
    # The pixel-statistics properties computed from the gathered values
    # match the same reductions of the reference values
    rng = np.random.default_rng(2)
    background = rng.normal(0.5, 0.1, scene['data'].shape)
    cat = SourceCatalog(scene['data'], scene['segm'], error=scene['error'],
                        mask=scene['mask'], background=background)
    ref = _reference_values(cat.data_cutout_masked)
    ref_err = _reference_values(cat.error_cutout_masked)
    ref_bkg = _reference_values(cat.background_cutout_masked)
    # The catalog sums with ufunc.reduceat (sequential) rather than
    # np.sum (pairwise), so the sums agree to rounding only
    assert_allclose(cat.segment_flux,
                    np.array([np.sum(arr) for arr in ref]), rtol=1e-14)
    assert_array_equal(cat.min_value, np.array([np.min(arr) for arr in ref]))
    assert_array_equal(cat.max_value, np.array([np.max(arr) for arr in ref]))
    assert_array_equal(cat.area.value,
                       np.array([arr.size for arr in ref], dtype=float))
    assert_allclose(cat.segment_flux_err,
                    np.sqrt([np.sum(arr**2) for arr in ref_err]),
                    rtol=1e-15)
    assert_allclose(cat.background_sum,
                    np.array([np.sum(arr) for arr in ref_bkg]), rtol=1e-14)
    assert_allclose(cat.background_mean,
                    np.array([np.mean(arr) for arr in ref_bkg]),
                    rtol=1e-15)


def test_scalar_catalog(scene):
    cat = make_catalog(scene)[2]
    assert cat.isscalar
    _assert_values_equal(cat._data_values,
                         _reference_values([cat.data_cutout_masked]))
    assert cat._all_masked.shape == (1,)
    assert not cat._all_masked[0]


def test_sliced_catalog_equals_parent(scene):
    parent = make_catalog(scene)
    _ = parent._data_values
    _ = parent._all_masked
    child = parent[[0, 3, 5]]
    _assert_values_equal(child._data_values,
                         [parent._data_values[i] for i in (0, 3, 5)])
    assert_array_equal(child._all_masked, parent._all_masked[[0, 3, 5]])
    fresh = make_catalog(scene)[[0, 3, 5]]
    _assert_values_equal(fresh._data_values, child._data_values)


def _reference_gini(cat):
    # The previous per-source gini implementation
    return np.array([gini_func(arr) for arr in cat._data_values])


def _reference_segment_area(cat):
    # The previous per-source segment_area implementation
    areas = []
    for label, slices in zip(cat.labels, cat._slices_iter, strict=True):
        areas.append(np.count_nonzero(
            cat._segmentation_image[slices] == label))
    return np.array(areas)


@pytest.mark.parametrize('with_mask', [True, False])
def test_gini_matches_reference(scene, with_mask):
    cat = make_catalog(scene, with_mask=with_mask)
    expected = _reference_gini(cat)
    assert np.all(np.isfinite(expected))
    assert_allclose(cat.gini, expected, rtol=1e-12)


def test_gini_edge_cases():
    # A completely masked source (NaN), a single-pixel source (0), a
    # zero-valued source (0), a source with negative values (absolute
    # values are used), and a normal source
    data = np.zeros((30, 30))
    yy, xx = np.mgrid[0:30, 0:30]
    segm_data = np.zeros(data.shape, dtype=int)
    mask = np.zeros(data.shape, dtype=bool)
    data[2:5, 2:5] = 3.0
    segm_data[2:5, 2:5] = 1
    mask[2:5, 2:5] = True
    data[8, 8] = 5.0
    segm_data[8, 8] = 2
    segm_data[12:15, 12:15] = 3
    data[18:24, 18:24] = -np.exp(-((xx[18:24, 18:24] - 20.5) ** 2
                                   + (yy[18:24, 18:24] - 20.5) ** 2) / 3.0)
    segm_data[18:24, 18:24] = 4
    data[2:9, 20:27] = np.exp(-((xx[2:9, 20:27] - 23) ** 2
                                + (yy[2:9, 20:27] - 5) ** 2) / 4.0)
    segm_data[2:9, 20:27] = 5
    cat = SourceCatalog(data, SegmentationImage(segm_data), mask=mask)
    expected = _reference_gini(cat)
    assert np.isnan(expected[0])
    assert expected[1] == 0.0
    assert expected[2] == 0.0
    assert 0 < expected[3] < 1
    assert_allclose(cat.gini, expected, rtol=1e-12, equal_nan=True)


def test_segment_area_matches_reference(scene):
    cat = make_catalog(scene)
    expected = _reference_segment_area(cat)
    assert_array_equal(cat.segment_area.value, expected)
    assert cat.segment_area.unit == u.pix**2
    # Unaffected by the mask, unlike area
    assert np.all(cat.segment_area.value >= cat.area.value)
    child = cat[[2, 0, 5]]
    assert_array_equal(child.segment_area.value, expected[[2, 0, 5]])
    scalar = make_catalog(scene)[4]
    assert scalar.segment_area.value == expected[4]


def _driver_inputs(cat):
    arrays = cat._get_batch_arrays()
    iymin, iymax, ixmin, ixmax = cat._get_batch_bboxes()
    return {'values': arrays['data'], 'mask': arrays['mask'],
            'segm': arrays['segm'],
            'labels': np.ascontiguousarray(cat.labels, dtype=np.intp),
            'bbox_iymin': iymin, 'bbox_iymax': iymax,
            'bbox_ixmin': ixmin, 'bbox_ixmax': ixmax}


def _call_driver(inp):
    return batch_segment_gather(inp.pop('values'), **inp)


def test_driver_outputs(scene):
    cat = make_catalog(scene)
    packed, offsets, counts = _call_driver(_driver_inputs(cat))
    assert offsets[0] == 0
    assert offsets[-1] == packed.size
    assert_array_equal(np.diff(offsets), np.maximum(counts, 1))
    assert_array_equal(counts,
                       [np.count_nonzero(~mask)
                        for mask in cat._cutout_total_masks])


@pytest.mark.parametrize('name', ['bbox_iymin', 'bbox_iymax', 'bbox_ixmin',
                                  'bbox_ixmax'])
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
    with pytest.raises(ValueError, match='same shape as values'):
        _call_driver(inp)


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
