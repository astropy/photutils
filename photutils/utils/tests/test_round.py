# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the _round module.
"""

import numpy as np
from numpy.testing import assert_equal

from photutils.utils._round import round_half_away


def test_round():
    """
    Test round_half_away with an array of values.
    """
    a = np.arange(-2, 2, 0.5)
    ar = round_half_away(a)

    result = np.array([-2, -2, -1, -1, 0, 1, 1, 2])

    assert isinstance(ar, np.ndarray)
    assert np.issubdtype(ar.dtype, np.integer)
    assert ar.shape == a.shape
    assert_equal(ar, result)


def test_round_scalar():
    """
    Test round_half_away with scalar inputs.
    """
    a = 0.5
    ar = round_half_away(a)
    assert np.isscalar(ar)
    assert isinstance(ar, int)
    assert ar == 1

    a = -0.5
    ar = round_half_away(a)
    assert np.isscalar(ar)
    assert isinstance(ar, int)
    assert ar == -1


def test_round_nan():
    """
    Test round_half_away with NaN inputs.
    """
    # Scalar NaN returns float NaN
    ar = round_half_away(np.nan)
    assert np.isscalar(ar)
    assert isinstance(ar, float)
    assert np.isnan(ar)

    # Array containing NaN returns float array
    a = np.array([1.5, np.nan, -0.5])
    ar = round_half_away(a)
    assert np.issubdtype(ar.dtype, np.floating)
    assert_equal(ar[0], 2.0)
    assert np.isnan(ar[1])
    assert_equal(ar[2], -1.0)


def test_round_inf():
    """
    Test round_half_away with infinite inputs.
    """
    # Scalar positive infinity returns float inf
    ar = round_half_away(np.inf)
    assert np.isscalar(ar)
    assert isinstance(ar, float)
    assert np.isposinf(ar)

    # Scalar negative infinity returns float -inf
    ar = round_half_away(-np.inf)
    assert np.isscalar(ar)
    assert isinstance(ar, float)
    assert np.isneginf(ar)

    # Array containing infinities returns float array
    a = np.array([np.inf, -np.inf, 1.5])
    ar = round_half_away(a)
    assert np.issubdtype(ar.dtype, np.floating)
    assert np.isposinf(ar[0])
    assert np.isneginf(ar[1])
    assert_equal(ar[2], 2.0)
