# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the local_background module.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from photutils.aperture import CircularAnnulus
from photutils.background import LocalBackground, MedianBackground


def test_local_background_invalid_radii():
    """
    Test that LocalBackground raises errors for invalid radius values.
    """
    # Test negative inner radius
    match = 'inner_radius must be positive'
    with pytest.raises(ValueError, match=match):
        LocalBackground(-5, 10)

    # Test zero inner radius
    with pytest.raises(ValueError, match=match):
        LocalBackground(0, 10)

    # Test negative outer radius
    match = 'outer_radius must be positive'
    with pytest.raises(ValueError, match=match):
        LocalBackground(5, -10)

    # Test zero outer radius
    with pytest.raises(ValueError, match=match):
        LocalBackground(5, 0)

    # Test outer_radius <= inner_radius
    match = 'outer_radius must be greater than inner_radius'
    with pytest.raises(ValueError, match=match):
        LocalBackground(10, 5)

    # Test equal radii
    with pytest.raises(ValueError, match=match):
        LocalBackground(10, 10)


def test_local_background():
    """
    Test the basic functionality of LocalBackground with a simple
    constant data array.
    """
    data = np.ones((101, 101))
    local_bkg = LocalBackground(5, 10, bkg_estimator=MedianBackground())

    x = np.arange(1, 7) * 10
    y = np.arange(1, 7) * 10
    bkg = local_bkg(data, x, y)
    assert_allclose(bkg, np.ones(len(x)))

    # Test scalar x and y
    bkg2 = local_bkg(data, x[2], y[2])
    assert not isinstance(bkg2, np.ndarray)
    assert_allclose(bkg[2], bkg2)

    bkg3 = local_bkg(data, -100, -100)
    assert np.isnan(bkg3)

    match = "'positions' must not contain any non-finite"
    with pytest.raises(ValueError, match=match):
        _ = local_bkg(data, x[2], np.inf)

    cls_repr = repr(local_bkg)
    assert cls_repr.startswith(local_bkg.__class__.__name__)

    # Test default bkg_estimator
    local_bkg2 = LocalBackground(5, 10, bkg_estimator=None)
    bkg4 = local_bkg2(data, x, y)
    assert_allclose(bkg4, bkg)


def test_local_background_estimator_1d():
    """
    Test that the bkg_estimator can be a 1D function that takes an array
    and returns a scalar.
    """

    def estimator(data):
        assert data.ndim == 1
        return np.nanmedian(data)

    data = np.ones((51, 51))
    local_bkg = LocalBackground(3, 6, bkg_estimator=estimator)
    bkg = local_bkg(data, [10, 20], [10, 20])
    assert_allclose(bkg, np.ones(2))


def test_to_aperture_scalar():
    """
    Test to_aperture method with scalar x and y positions.
    """
    r_in = 5
    r_out = 10
    local_bkg = LocalBackground(r_in, r_out)

    # Test scalar positions
    x = 50.0
    y = 50.0
    aperture = local_bkg.to_aperture(x, y)

    # Check aperture type and properties
    assert isinstance(aperture, CircularAnnulus)
    assert_allclose(aperture.positions, [[x, y]])
    assert_allclose(aperture.r_in, r_in)
    assert_allclose(aperture.r_out, r_out)


def test_to_aperture_array():
    """
    Test to_aperture method with array x and y positions.
    """
    r_in = 7.5
    r_out = 15.2
    local_bkg = LocalBackground(r_in, r_out)

    # Test array positions
    x = np.array([10.0, 20.1, 35.3])
    y = np.array([14.4, 27.2, 33.4])
    xypos = list(zip(x, y, strict=False))
    aperture = local_bkg.to_aperture(x, y)

    # Check aperture type and properties
    assert isinstance(aperture, CircularAnnulus)
    assert_allclose(aperture.positions, xypos)
    assert_allclose(aperture.r_in, r_in)
    assert_allclose(aperture.r_out, r_out)

    # Test list positions
    x = list(x)
    y = list(y)
    aperture2 = local_bkg.to_aperture(x, y)
    assert aperture == aperture2
