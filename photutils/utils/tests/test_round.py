# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the _round module.
"""

import numpy as np
from numpy.testing import assert_equal

from photutils.utils._round import py2intround


def test_round():
    a = np.arange(-2, 2, 0.5)
    ar = py2intround(a)

    result = np.array([-2, -2, -1, -1, 0, 1, 1, 2])

    assert isinstance(ar, np.ndarray)
    assert ar.shape == a.shape
    assert_equal(ar, result)


def test_round_scalar():
    a = 0.5
    ar = py2intround(a)
    assert np.isscalar(ar)
    assert ar == 1.0

    a = -0.5
    ar = py2intround(a)
    assert np.isscalar(ar)
    assert ar == -1.0
