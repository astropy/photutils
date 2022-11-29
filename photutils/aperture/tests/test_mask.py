# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the mask module.
"""

import astropy.units as u
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_almost_equal

from photutils.aperture.bounding_box import BoundingBox
from photutils.aperture.circle import CircularAnnulus, CircularAperture
from photutils.aperture.mask import ApertureMask
from photutils.aperture.rectangle import RectangularAnnulus

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


def test_mask_get_overlap_slices():
    aper = CircularAperture((5, 5), r=10.0)
    mask = aper.to_mask()
    slc = ((slice(0, 16, None), slice(0, 16, None)),
           (slice(5, 21, None), slice(5, 21, None)))
    assert mask.get_overlap_slices((25, 25)) == slc


def test_mask_cutout_shape():
    mask_data = np.ones((10, 10))
    bbox = BoundingBox(5, 15, 5, 15)
    mask = ApertureMask(mask_data, bbox)

    with pytest.raises(ValueError):
        mask.cutout(np.arange(10))

    with pytest.raises(ValueError):
        mask.to_image((10,))


def test_mask_cutout_copy():
    data = np.ones((50, 50))
    aper = CircularAperture((25, 25), r=10.0)
    mask = aper.to_mask()
    cutout = mask.cutout(data, copy=True)
    data[25, 25] = 100.0
    assert cutout[10, 10] == 1.0

    # test quantity data
    data2 = np.ones((50, 50)) * u.adu
    cutout2 = mask.cutout(data2, copy=True)
    assert cutout2.unit == data2.unit
    data2[25, 25] = 100.0 * u.adu
    assert cutout2[10, 10].value == 1.0


@pytest.mark.parametrize('position', POSITIONS)
def test_mask_cutout_no_overlap(position):
    data = np.ones((50, 50))
    aper = CircularAperture(position, r=10.0)
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
    aper = CircularAperture(position, r=30.0)
    mask = aper.to_mask()

    cutout = mask.cutout(data)
    assert cutout.shape == mask.shape

    weighted_data = mask.multiply(data)
    assert weighted_data.shape == mask.shape

    image = mask.to_image(data.shape)
    assert image.shape == data.shape


def test_mask_multiply():
    radius = 10.0
    data = np.ones((50, 50))
    aper = CircularAperture((25, 25), r=radius)
    mask = aper.to_mask()
    data_weighted = mask.multiply(data)
    assert_almost_equal(np.sum(data_weighted), np.pi * radius**2)

    # test that multiply() returns a copy
    data[25, 25] = 100.0
    assert data_weighted[10, 10] == 1.0


def test_mask_multiply_quantity():
    radius = 10.0
    data = np.ones((50, 50)) * u.adu
    aper = CircularAperture((25, 25), r=radius)
    mask = aper.to_mask()
    data_weighted = mask.multiply(data)
    assert data_weighted.unit == u.adu
    assert_almost_equal(np.sum(data_weighted.value), np.pi * radius**2)

    # test that multiply() returns a copy
    data[25, 25] = 100.0 * u.adu
    assert data_weighted[10, 10].value == 1.0


@pytest.mark.parametrize('value', (np.nan, np.inf))
def test_mask_nonfinite_fill_value(value):
    aper = CircularAnnulus((0, 0), 10, 20)
    data = np.ones((101, 101)).astype(int)
    cutout = aper.to_mask().cutout(data, fill_value=value)
    assert ~np.isfinite(cutout[0, 0])


def test_mask_multiply_fill_value():
    aper = CircularAnnulus((0, 0), 10, 20)
    data = np.ones((101, 101)).astype(int)
    cutout = aper.to_mask().multiply(data, fill_value=np.nan)
    xypos = ((20, 20), (5, 5), (5, 35), (35, 5), (35, 35))
    for x, y in xypos:
        assert np.isnan(cutout[y, x])


def test_mask_nonfinite_in_bbox():
    """
    Regression test that non-finite data values outside of the mask but
    within the bounding box are set to zero.
    """
    data = np.ones((101, 101))
    data[33, 33] = np.nan
    data[67, 67] = np.inf
    data[33, 67] = -np.inf
    data[22, 22] = np.nan
    data[22, 23] = np.inf

    radius = 20.0
    aper1 = CircularAperture((50, 50), r=radius)
    aper2 = CircularAperture((5, 5), r=radius)

    wdata1 = aper1.to_mask(method='exact').multiply(data)
    assert_allclose(np.sum(wdata1), np.pi * radius**2)

    wdata2 = aper2.to_mask(method='exact').multiply(data)
    assert_allclose(np.sum(wdata2), 561.6040111923013)


def test_mask_get_values():
    aper = CircularAnnulus(((0, 0), (50, 50), (100, 100)), 10, 20)
    data = np.ones((101, 101))
    values = [mask.get_values(data) for mask in aper.to_mask()]
    shapes = [val.shape for val in values]
    sums = [np.sum(val) for val in values]
    assert shapes[0] == (278,)
    assert shapes[1] == (1068,)
    assert shapes[2] == (278,)
    sums_expected = (245.621534, 942.477796, 245.621534)
    assert_allclose(sums, sums_expected)


def test_mask_get_values_no_overlap():
    aper = CircularAperture((-100, -100), r=3)
    data = np.ones((51, 51))
    values = aper.to_mask().get_values(data)
    assert values.shape == (0,)


def test_mask_get_values_mask():
    aper = CircularAperture((24.5, 24.5), r=10.0)
    data = np.ones((51, 51))
    mask = aper.to_mask()
    with pytest.raises(ValueError):
        mask.get_values(data, mask=np.ones(3))

    arr = mask.get_values(data, mask=None)
    assert_allclose(np.sum(arr), 100.0 * np.pi)

    data_mask = np.zeros(data.shape, dtype=bool)
    data_mask[25:] = True
    arr2 = mask.get_values(data, mask=data_mask)
    assert_allclose(np.sum(arr2), 100.0 * np.pi / 2.0)


def test_rectangular_annulus_hin():
    aper = RectangularAnnulus((25, 25), 2, 4, 20, h_in=18, theta=0)
    mask = aper.to_mask(method='center')
    assert mask.data.shape == (21, 5)
    assert np.count_nonzero(mask.data) == 40
