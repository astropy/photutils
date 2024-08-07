# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the _quantity_helpers module.
"""

import astropy.units as u
import numpy as np
import pytest
from numpy.testing import assert_equal

from photutils.utils._quantity_helpers import isscalar, process_quantities


@pytest.mark.parametrize('all_units', [False, True])
def test_units(all_units):
    unit = u.Jy if all_units else 1.0
    arrs = (np.ones(3) * unit, np.ones(3) * unit, np.ones(3) * unit)
    names = ('a', 'b', 'c')

    arrs2, unit2 = process_quantities(arrs, names)
    if all_units:
        assert unit2 == unit
        for (arr, arr2) in zip(arrs, arrs2, strict=True):
            assert_equal(arr.value, arr2)
    else:
        assert unit2 is None
        assert arrs2 == arrs


def test_mixed_units():
    arrs = (np.ones(3) * u.Jy, np.ones(3) * u.km)
    names = ('a', 'b')

    match = 'must all have the same units'
    with pytest.raises(ValueError, match=match):
        _, _ = process_quantities(arrs, names)

    arrs = (np.ones(3) * u.Jy, np.ones(3))
    names = ('a', 'b')
    with pytest.raises(ValueError, match=match):
        _, _ = process_quantities(arrs, names)

    unit = u.Jy
    arrs = (np.ones(3) * unit, np.ones(3), np.ones(3) * unit)
    names = ('a', 'b', 'c')
    with pytest.raises(ValueError, match=match):
        _, _ = process_quantities(arrs, names)

    unit = u.Jy
    arrs = (np.ones(3) * unit, np.ones(3), np.ones(3) * u.km)
    names = ('a', 'b', 'c')
    with pytest.raises(ValueError, match=match):
        _, _ = process_quantities(arrs, names)


def test_inputs():
    match = 'The number of values must match the number of names'
    with pytest.raises(ValueError, match=match):
        _, _ = process_quantities([1, 2, 3], ['a', 'b'])


def test_isscalar():
    assert isscalar(1)
    assert isscalar(1.0 * u.m)
    assert not isscalar([1, 2, 3])
    assert not isscalar([1, 2, 3] * u.m)
