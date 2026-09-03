# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the batch perimeter kernel.
"""

from concurrent.futures import ThreadPoolExecutor

import astropy.units as u
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from photutils.segmentation import SegmentationImage, SourceCatalog
from photutils.segmentation._batch_catalog import batch_perimeter
from photutils.segmentation.tests._batch_scene import (make_batch_scene,
                                                       make_catalog)


def _reference_perimeter(cat):
    """
    Compute the perimeter of each source segment.

    This is a verbatim port of the per-source Python implementation
    of ``perimeter`` that the batch Cython kernel replaces. It is the
    numerical reference for the kernel.
    """
    size = 34
    weights = np.zeros(size, dtype=float)
    weights[[5, 7, 15, 17, 25, 27]] = 1.0
    weights[[21, 33]] = np.sqrt(2.0)
    weights[[13, 23]] = (1 + np.sqrt(2.0)) / 2.0

    perimeter = []
    for mask in cat._cutout_total_masks:
        if np.all(mask):
            perimeter.append(np.nan)
            continue

        ny, nx = mask.shape

        padded = np.zeros((ny + 2, nx + 2), dtype=np.int8)
        padded[1:-1, 1:-1] = ~mask

        p = padded
        eroded = (p[1:-1, 1:-1] & p[:-2, 1:-1] & p[2:, 1:-1]
                  & p[1:-1, :-2] & p[1:-1, 2:])

        border = np.zeros((ny + 2, nx + 2), dtype=np.int8)
        border[1:-1, 1:-1] = padded[1:-1, 1:-1] & ~eroded

        b = border
        conv = (10 * b[:-2, :-2] + 2 * b[:-2, 1:-1]
                + 10 * b[:-2, 2:] + 2 * b[1:-1, :-2]
                + b[1:-1, 1:-1] + 2 * b[1:-1, 2:]
                + 10 * b[2:, :-2] + 2 * b[2:, 1:-1]
                + 10 * b[2:, 2:])

        hist = np.bincount(conv.ravel(), minlength=size)
        perimeter.append(hist[:size] @ weights)

    return np.array(perimeter) * u.pix


@pytest.fixture(scope='module')
def scene():
    return make_batch_scene()


@pytest.mark.parametrize('with_mask', [True, False])
def test_matches_reference(scene, with_mask):
    cat = make_catalog(scene, with_mask=with_mask)
    expected = _reference_perimeter(cat)
    assert np.all(np.isfinite(expected))
    assert_allclose(cat.perimeter, expected, rtol=1e-12)


def test_shapes():
    # Hand-checkable shapes: a single pixel, a 2x2 square, a 5x5
    # square, a diagonal line, an annulus with a masked hole, a fully
    # masked source, and an edge-touching source
    data = np.ones((40, 40))
    segm_data = np.zeros(data.shape, dtype=int)
    mask = np.zeros(data.shape, dtype=bool)
    segm_data[2, 2] = 1
    segm_data[5:7, 5:7] = 2
    segm_data[10:15, 10:15] = 3
    for k in range(5):
        segm_data[20 + k, 20 + k] = 4
    segm_data[25:32, 2:9] = 5
    mask[27:30, 4:7] = True
    segm_data[33:36, 33:36] = 6
    mask[33:36, 33:36] = True
    segm_data[37:40, 0:3] = 7
    segm = SegmentationImage(segm_data)
    cat = SourceCatalog(data, segm, mask=mask)
    expected = _reference_perimeter(cat)
    assert_allclose(cat.perimeter, expected, rtol=1e-12, equal_nan=True)
    # A single pixel has a zero perimeter and the fully masked source
    # has none
    assert cat.perimeter[0] == 0 * u.pix
    assert np.isnan(cat.perimeter[5])
    # The annulus perimeter includes the inner hole
    square = SourceCatalog(data, SegmentationImage(
        np.where(segm_data == 5, 5, 0))).perimeter[0]
    assert cat.perimeter[4] > square


def test_scalar_and_sliced_catalog(scene):
    parent = make_catalog(scene)
    expected = _reference_perimeter(parent)
    child = parent[[1, 4, 6]]
    assert_array_equal(child.perimeter.value,
                       parent.perimeter.value[[1, 4, 6]])
    assert_allclose(child.perimeter, expected[[1, 4, 6]], rtol=1e-12)
    scalar = make_catalog(scene)[3]
    assert scalar.isscalar
    assert_allclose(scalar.perimeter, expected[3], rtol=1e-12)
    assert scalar.perimeter == parent.perimeter[3]


def test_batch_bboxes_cached_and_sliced(scene):
    """
    Test that the cached bounding-box array is sliced along the source
    axis, including for scalar catalogs, and matches a fresh catalog.
    """
    cat = make_catalog(scene)
    bboxes = cat._batch_bboxes
    assert bboxes.shape == (cat.n_labels, 4)
    assert bboxes.dtype == np.intp
    iymin, iymax, ixmin, ixmax = cat._get_batch_bboxes()
    for col in (iymin, iymax, ixmin, ixmax):
        assert col.flags.c_contiguous
    assert_array_equal(np.column_stack((iymin, iymax, ixmin, ixmax)),
                       bboxes)
    slices = cat.slices
    assert_array_equal(iymin, [slc[0].start for slc in slices])
    assert_array_equal(ixmax, [slc[1].stop for slc in slices])

    child = cat[[1, 4, 6]]
    assert_array_equal(child._batch_bboxes, bboxes[[1, 4, 6]])
    scalar = cat[3]
    assert scalar._batch_bboxes.shape == (1, 4)
    assert_array_equal(scalar._batch_bboxes[0], bboxes[3])
    fresh = make_catalog(scene)[3]
    assert_array_equal(fresh._batch_bboxes, scalar._batch_bboxes)


def _driver_inputs(cat):
    arrays = cat._get_batch_arrays()
    iymin, iymax, ixmin, ixmax = cat._get_batch_bboxes()
    return {'mask': arrays['mask'], 'segm': arrays['segm'],
            'labels': np.ascontiguousarray(cat.labels, dtype=np.intp),
            'bbox_iymin': iymin, 'bbox_iymax': iymax,
            'bbox_ixmin': ixmin, 'bbox_ixmax': ixmax}


def _call_driver(inp):
    return batch_perimeter(inp.pop('mask'), **inp)


def test_histogram(scene):
    # Every bounding-box pixel is counted in the histogram (values of
    # 34 and above are dropped)
    cat = make_catalog(scene)
    hist = _call_driver(_driver_inputs(cat))
    assert hist.shape == (cat.n_labels, 34)
    iymin, iymax, ixmin, ixmax = cat._get_batch_bboxes()
    assert np.all(hist.sum(axis=1) <= (iymax - iymin) * (ixmax - ixmin))


@pytest.mark.parametrize('name', ['bbox_iymin', 'bbox_iymax', 'bbox_ixmin',
                                  'bbox_ixmax'])
def test_length_guard(scene, name):
    cat = make_catalog(scene)
    inp = _driver_inputs(cat)
    inp[name] = inp[name][:-1]
    match = f'{name} must have the same length as labels'
    with pytest.raises(ValueError, match=match):
        _call_driver(inp)


def test_shape_guard(scene):
    cat = make_catalog(scene)
    inp = _driver_inputs(cat)
    inp['segm'] = inp['segm'][:-1, :]
    match = 'segm must have the same shape as mask'
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
