# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the cutouts module.
"""

import numpy as np
import pytest
from astropy.nddata.utils import PartialOverlapError
from numpy.testing import assert_equal

from ...aperture import BoundingBox
from ...datasets import make_100gaussians_image
from ..cutouts import CutoutImage


def test_cutout():
    data = make_100gaussians_image()
    shape = (24, 57)
    yxpos = (100, 51)
    cutout = CutoutImage(data, yxpos, shape)
    assert cutout.position == yxpos
    assert cutout.input_shape == shape
    assert cutout.mode == 'trim'
    assert np.isnan(cutout.fill_value)
    assert not cutout.copy

    assert cutout.data.shape == shape
    assert_equal(cutout.__array__(), cutout.data)

    assert isinstance(cutout.bbox_original, BoundingBox)
    assert isinstance(cutout.bbox_cutout, BoundingBox)
    assert cutout.slices_original == (slice(88, 112, None),
                                      slice(23, 80, None))
    assert cutout.slices_cutout == (slice(0, 24, None), slice(0, 57, None))

    assert_equal(cutout.xyorigin, np.array((23, 88)))

    assert f'Shape: {shape}' in repr(cutout)
    assert f'Shape: {shape}' in str(cutout)


def test_cutout_partial_overlap():
    data = make_100gaussians_image()
    shape = (24, 57)

    # mode = 'trim'
    cutout = CutoutImage(data, (11, 10), shape)
    assert cutout.input_shape == shape
    assert cutout.shape == (23, 39)

    # mode = 'strict'
    with pytest.raises(PartialOverlapError):
        CutoutImage(data, (11, 10), shape, mode='strict')

    # mode = 'partial'
    cutout = CutoutImage(data, (11, 10), shape, mode='partial')
    assert cutout.input_shape == shape
    assert cutout.shape == shape

    assert (cutout.bbox_original
            == BoundingBox(ixmin=0, ixmax=39, iymin=0, iymax=23))
    assert (cutout.bbox_cutout
            == BoundingBox(ixmin=18, ixmax=57, iymin=1, iymax=24))

    assert cutout.slices_original == (slice(0, 23, None), slice(0, 39, None))
    assert cutout.slices_cutout == (slice(1, 24, None), slice(18, 57, None))

    assert_equal(cutout.xyorigin, np.array((-18, -1)))


def test_cutout_copy():
    data = make_100gaussians_image()

    cutout1 = CutoutImage(data, (1, 1), (3, 3), copy=True)
    cutout1.data[0, 0] = np.nan
    assert not np.isnan(data[0, 0])

    cutout2 = CutoutImage(data, (1, 1), (3, 3), copy=False)
    cutout2.data[0, 0] = np.nan
    assert np.isnan(data[0, 0])
