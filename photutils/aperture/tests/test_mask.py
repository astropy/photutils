# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the mask module.
"""

import astropy.units as u
import numpy as np
from numpy.testing import assert_allclose, assert_almost_equal
import pytest

from ..bounding_box import BoundingBox
from ..circle import CircularAperture
from ..mask import ApertureMask

try:
    import matplotlib  # noqa
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


POSITIONS = [(-20, -20), (-20, 20), (20, -20), (60, 60)]


def test_mask_input_shapes():
    with pytest.raises(ValueError):
        mask_data = np.ones((10, 10))
        bbox = BoundingBox(5, 10, 5, 10)
        ApertureMask(mask_data, bbox)


def test_mask_array():
    mask_data = np.ones((10, 10))
    bbox = BoundingBox(5, 15, 5, 15)
    mask = ApertureMask(mask_data, bbox)
    data = np.array(mask)
    assert_allclose(data, mask.data)


def test_mask_cutout_shape():
    mask_data = np.ones((10, 10))
    bbox = BoundingBox(5, 15, 5, 15)
    mask = ApertureMask(mask_data, bbox)

    with pytest.raises(ValueError):
        mask.cutout(np.arange(10))

    with pytest.raises(ValueError):
        mask._overlap_slices((10,))

    with pytest.raises(ValueError):
        mask.to_image((10,))


def test_mask_cutout_copy():
    data = np.ones((50, 50))
    aper = CircularAperture((25, 25), r=10.)
    mask = aper.to_mask()
    cutout = mask.cutout(data, copy=True)
    data[25, 25] = 100.
    assert cutout[10, 10] == 1.


def test_mask_cutout_copy_quantity():
    data = np.ones((50, 50)) * u.adu
    aper = CircularAperture((25, 25), r=10.)
    mask = aper.to_mask()
    cutout = mask.cutout(data, copy=True)
    assert cutout.unit == data.unit

    data[25, 25] = 100. * u.adu
    assert cutout[10, 10].value == 1.


@pytest.mark.parametrize('position', POSITIONS)
def test_mask_cutout_no_overlap(position):
    data = np.ones((50, 50))
    aper = CircularAperture(position, r=10.)
    mask = aper.to_mask()

    cutout = mask.cutout(data)
    assert cutout is None

    weighted_data = mask.multiply(data)
    assert weighted_data is None

    image = mask.to_image(data.shape)
    assert image is None


@pytest.mark.parametrize('position', POSITIONS)
def test_mask_cutout_partial_overlap(position):
    data = np.ones((50, 50))
    aper = CircularAperture(position, r=30.)
    mask = aper.to_mask()

    cutout = mask.cutout(data)
    assert cutout.shape == mask.shape

    weighted_data = mask.multiply(data)
    assert weighted_data.shape == mask.shape

    image = mask.to_image(data.shape)
    assert image.shape == data.shape


def test_mask_multiply():
    radius = 10.
    data = np.ones((50, 50))
    aper = CircularAperture((25, 25), r=radius)
    mask = aper.to_mask()
    data_weighted = mask.multiply(data)
    assert_almost_equal(np.sum(data_weighted), radius**2 * np.pi)

    # test that multiply() returns a copy
    data[25, 25] = 100.
    assert data_weighted[10, 10] == 1.


def test_mask_multiply_quantity():
    radius = 10.
    data = np.ones((50, 50)) * u.adu
    aper = CircularAperture((25, 25), r=radius)
    mask = aper.to_mask()

    data_weighted = mask.multiply(data)
    assert data_weighted.unit == u.adu
    assert_almost_equal(np.sum(data_weighted.value), radius**2 * np.pi)

    # test that multiply() returns a copy
    data[25, 25] = 100. * u.adu
    assert data_weighted[10, 10].value == 1.
