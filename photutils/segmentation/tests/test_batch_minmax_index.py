# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the batch minimum and maximum value position kernel.
"""

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from photutils.segmentation import SegmentationImage, SourceCatalog
from photutils.segmentation._batch_catalog import batch_minmax_index
from photutils.segmentation.tests._batch_scene import (make_batch_scene,
                                                       make_catalog)


def _reference_cutout_index(cat, argfunc):
    """
    The cutout-frame position of an extreme value of each source.

    This is a verbatim port of the per-source ``cutout_min_value_index``
    and ``cutout_max_value_index`` loops that the batch kernel
    replaces. It is the reference for the kernel.
    """
    idx = []
    for arr in cat._array('data_cutout_masked'):
        if np.all(arr.mask):
            idx.append((np.nan, np.nan))
        else:
            idx.append(np.unravel_index(argfunc(arr), arr.shape))
    return np.array(idx)


def _reference_index(cat, cutout_index):
    """
    The verbatim port of the previous ``min_value_index`` and
    ``max_value_index`` loops, adding the bounding-box origin.
    """
    out = []
    for idx, slc in zip(cutout_index, cat.slices, strict=True):
        out.append((idx[0] + slc[0].start, idx[1] + slc[1].start))
    return np.array(out)


def _assert_index_equal(result, expected):
    assert result.shape == expected.shape
    assert result.dtype.kind == expected.dtype.kind
    assert_array_equal(result, expected)


@pytest.fixture(scope='module')
def scene():
    return make_batch_scene()


@pytest.mark.parametrize('with_mask', [True, False])
def test_matches_reference(scene, with_mask):
    cat = make_catalog(scene, with_mask=with_mask)
    cutout_min = _reference_cutout_index(cat, np.argmin)
    cutout_max = _reference_cutout_index(cat, np.argmax)
    _assert_index_equal(cat.cutout_min_value_index, cutout_min)
    _assert_index_equal(cat.cutout_max_value_index, cutout_max)
    _assert_index_equal(cat.min_value_index, _reference_index(cat, cutout_min))
    _assert_index_equal(cat.max_value_index, _reference_index(cat, cutout_max))
    assert_array_equal(cat.min_value_xindex, cat.min_value_index[:, 1])
    assert_array_equal(cat.min_value_yindex, cat.min_value_index[:, 0])
    assert_array_equal(cat.max_value_xindex, cat.max_value_index[:, 1])
    assert_array_equal(cat.max_value_yindex, cat.max_value_index[:, 0])
    # Integer positions when no source is completely masked
    assert cat.min_value_index.dtype.kind == 'i'
    # The positions hold the extreme values
    data = scene['data']
    ymin, xmin = cat.min_value_index.T
    ymax, xmax = cat.max_value_index.T
    assert_array_equal(data[ymin, xmin], cat.min_value)
    assert_array_equal(data[ymax, xmax], cat.max_value)


def test_ties_and_all_masked():
    """
    Test the first occurrence of a repeated extreme, a completely
    masked source (NaN positions and a float result), and a source
    whose extremes lie on the bounding-box edges.
    """
    data = np.zeros((30, 30))
    segm_data = np.zeros(data.shape, dtype=int)
    mask = np.zeros(data.shape, dtype=bool)
    # Label 1: constant source, so every pixel ties
    data[2:6, 2:7] = 4.0
    segm_data[2:6, 2:7] = 1
    # Label 2: completely masked
    data[10:14, 10:14] = 1.0
    segm_data[10:14, 10:14] = 2
    mask[10:14, 10:14] = True
    # Label 3: minimum at the top-left and maximum at the bottom-right
    # corner of the bounding box, with a masked would-be maximum
    yy, xx = np.mgrid[0:6, 0:8]
    data[18:24, 18:26] = xx + yy
    data[20, 21] = 100.0
    mask[20, 21] = True
    segm_data[18:24, 18:26] = 3
    # Label 4: repeated minimum whose first occurrence is not first in
    # the (non-rectangular) segment's bounding box
    data[2:7, 15:22] = 5.0
    data[3, 16] = -1.0
    data[5, 20] = -1.0
    segm_data[2:7, 15:22] = 4
    segm_data[2, 15:17] = 0
    cat = SourceCatalog(data, SegmentationImage(segm_data), mask=mask)

    cutout_min = _reference_cutout_index(cat, np.argmin)
    cutout_max = _reference_cutout_index(cat, np.argmax)
    _assert_index_equal(cat.cutout_min_value_index, cutout_min)
    _assert_index_equal(cat.cutout_max_value_index, cutout_max)
    _assert_index_equal(cat.min_value_index, _reference_index(cat, cutout_min))
    _assert_index_equal(cat.max_value_index, _reference_index(cat, cutout_max))

    # A completely masked source gives NaN positions and a float array
    assert cat.min_value_index.dtype.kind == 'f'
    assert np.all(np.isnan(cat.min_value_index[1]))
    assert np.all(np.isnan(cat.cutout_max_value_index[1]))
    # The constant source reports its first pixel
    assert_array_equal(cat.cutout_min_value_index[0], [0, 0])
    assert_array_equal(cat.cutout_max_value_index[0], [0, 0])
    # The masked would-be maximum is skipped
    assert_array_equal(cat.max_value_index[2], [23, 25])
    assert_array_equal(cat.min_value_index[2], [18, 18])
    assert_array_equal(cat.min_value_index[3], [3, 16])


def test_sliced_and_scalar_catalog(scene):
    parent = make_catalog(scene)
    expected_min = parent.min_value_index
    expected_cutout_max = parent.cutout_max_value_index
    child = parent[[4, 1, 6]]
    assert_array_equal(child.min_value_index, expected_min[[4, 1, 6]])
    assert_array_equal(child.cutout_max_value_index,
                       expected_cutout_max[[4, 1, 6]])
    fresh = make_catalog(scene)[[4, 1, 6]]
    assert_array_equal(fresh.min_value_index, expected_min[[4, 1, 6]])
    scalar = make_catalog(scene)[3]
    assert scalar.isscalar
    assert_array_equal(scalar.min_value_index, expected_min[3])
    assert_array_equal(scalar.cutout_max_value_index,
                       expected_cutout_max[3])
    assert scalar.min_value_xindex == expected_min[3, 1]
    assert scalar.max_value_yindex == parent.max_value_index[3, 0]


def _driver_inputs(cat):
    arrays = cat._get_batch_arrays()
    iymin, iymax, ixmin, ixmax = cat._get_batch_bboxes()
    return {'values': arrays['data'], 'mask': arrays['mask'],
            'segm': arrays['segm'],
            'labels': np.ascontiguousarray(cat.labels, dtype=np.intp),
            'bbox_iymin': iymin, 'bbox_iymax': iymax,
            'bbox_ixmin': ixmin, 'bbox_ixmax': ixmax}


def _call_driver(inp):
    return batch_minmax_index(inp.pop('values'), **inp)


def test_driver_outputs(scene):
    cat = make_catalog(scene)
    index = _call_driver(_driver_inputs(cat))
    assert index.shape == (cat.n_labels, 4)
    assert index.dtype == np.intp
    assert_array_equal(index[:, :2], cat.cutout_min_value_index)
    assert_array_equal(index[:, 2:], cat.cutout_max_value_index)


@pytest.mark.parametrize('name', ['bbox_iymin', 'bbox_iymax', 'bbox_ixmin',
                                  'bbox_ixmax'])
def test_length_guard(scene, name):
    cat = make_catalog(scene)
    inp = _driver_inputs(cat)
    inp[name] = inp[name][:-1]
    match = f'{name} must have the same length as labels'
    with pytest.raises(ValueError, match=match):
        _call_driver(inp)


@pytest.mark.parametrize('name', ['mask', 'segm'])
def test_shape_guard(scene, name):
    cat = make_catalog(scene)
    inp = _driver_inputs(cat)
    inp[name] = inp[name][:-1, :]
    match = f'{name} must have the same shape as values'
    with pytest.raises(ValueError, match=match):
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
        assert_array_equal(result, expected)
