# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the core module.
"""

import astropy.units as u
import numpy as np
import pytest
from numpy.testing import assert_allclose

from photutils.datasets import make_wcs
from photutils.morphology import data_properties


def test_data_properties():
    """
    Test basics of ``data_properties`` with and without a mask.
    """
    data = np.ones((2, 2)).astype(float)
    mask = np.array([[False, False], [True, True]])
    props = data_properties(data, mask=None)
    props2 = data_properties(data, mask=mask)
    properties = ['xcentroid', 'ycentroid']
    result = [getattr(props, i) for i in properties]
    result2 = [getattr(props2, i) for i in properties]
    assert_allclose([0.5, 0.5], result, rtol=0, atol=1.0e-6)
    assert_allclose([0.5, 0.0], result2, rtol=0, atol=1.0e-6)
    assert props.area.value == 4.0
    assert props2.area.value == 2.0

    wcs = make_wcs(data.shape)
    props = data_properties(data, mask=None, wcs=wcs)
    assert props.sky_centroid is not None


def test_data_properties_invalid_data_shape():
    """
    Test that data must be a 2D array.
    """
    match = 'data must be a 2D array'
    with pytest.raises(ValueError, match=match):
        data_properties(np.ones(10))  # 1D

    with pytest.raises(ValueError, match=match):
        data_properties([1, 2, 3])  # 1D list

    with pytest.raises(ValueError, match=match):
        data_properties(np.ones((3, 3, 3)))  # 3D


def test_data_properties_mask_invalid_shape():
    """
    Test that mask must have the same shape as data.
    """
    data = np.ones((10, 10))

    match = 'mask must have the same shape as data'
    with pytest.raises(ValueError, match=match):
        data_properties(data, mask=np.zeros((5, 5), dtype=bool))

    match = 'mask must have the same shape as data'
    with pytest.raises(ValueError, match=match):
        data_properties(data, mask=np.zeros((10, 5), dtype=bool))


def test_data_properties_bkg():
    """
    Test with a scalar and 2D array background.
    """
    data = np.ones((3, 3)).astype(float)
    props = data_properties(data, background=1.0)
    assert props.area.value == 9.0
    assert props.background_sum == 9.0

    bkg_2d = np.full((3, 3), 2.0)
    props2 = data_properties(data, background=bkg_2d)
    assert props2.background_sum == 18.0


def test_data_properties_bkg_invalid():
    """
    Test that invalid background inputs raise ``ValueError``.
    """
    data = np.ones((3, 3))

    match = 'background must be a scalar or a 2D array'
    with pytest.raises(ValueError, match=match):
        data_properties(data, background=[1.0, 2.0])

    with pytest.raises(ValueError, match=match):
        data_properties(data, background=np.ones((2, 2)))


def test_data_properties_all_masked():
    """
    Test that an all-True mask raises ``ValueError``.
    """
    data = np.ones((4, 4))
    mask = np.ones((4, 4), dtype=bool)
    match = 'All pixels in data are masked'
    with pytest.raises(ValueError, match=match):
        data_properties(data, mask=mask)


def test_data_properties_quantity():
    """
    Test that ``~astropy.units.Quantity`` input is accepted.
    """
    data = np.ones((3, 3)) * u.Jy
    props = data_properties(data)
    assert props.area.value == 9.0
