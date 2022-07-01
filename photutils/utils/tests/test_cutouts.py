# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the cutouts module.
"""

import numpy as np
from numpy.testing import assert_equal

from ...aperture import BoundingBox
from ...datasets import make_100gaussians_image
from ..cutouts import CutoutImage


def test_cutout():
    data = make_100gaussians_image()
    shape = (24, 57)
    cutout = CutoutImage(data, (100, 51), shape)
    assert cutout.data.shape == shape
    assert_equal(cutout.__array__(), cutout.data)

    assert isinstance(cutout.bbox_original, BoundingBox)
    assert isinstance(cutout.bbox_cutout, BoundingBox)
    assert cutout.slices_original == (slice(88, 112, None),
                                      slice(23, 80, None))
    assert cutout.slices_cutout == (slice(0, 24, None), slice(0, 57, None))

    assert f'Shape: {shape}' in repr(cutout)
    assert f'Shape: {shape}' in str(cutout)


def test_cutout_copy():
    data = make_100gaussians_image()

    cutout1 = CutoutImage(data, (1, 1), (3, 3), copy=True)
    cutout1.data[0, 0] = np.nan
    assert not np.isnan(data[0, 0])

    cutout2 = CutoutImage(data, (1, 1), (3, 3), copy=False)
    cutout2.data[0, 0] = np.nan
    assert np.isnan(data[0, 0])
