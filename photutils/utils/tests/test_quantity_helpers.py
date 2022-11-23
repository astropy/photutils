# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the _quantity_helpers module.
"""

import astropy.units as u
import numpy as np
import pytest
from numpy.testing import assert_equal

from photutils.utils._quantity_helpers import process_quantities


@pytest.mark.parametrize('all_units', (False, True))
def test_units(all_units):
    if all_units:
        unit = u.Jy
    else:
        unit = 1.0
    arrs = (np.ones(3) * unit, np.ones(3) * unit, np.ones(3) * unit)
    names = ('a', 'b', 'c')

    arrs2, unit2 = process_quantities(arrs, names)
    if all_units:
        assert unit2 == unit
        for (arr, arr2) in zip(arrs, arrs2):
            assert_equal(arr.value, arr2)
    else:
        assert unit2 is None
        assert arrs2 == arrs


def test_mixed_units():
    arrs = (np.ones(3) * u.Jy, np.ones(3) * u.km)
    names = ('a', 'b')
    with pytest.raises(ValueError):
        _, _ = process_quantities(arrs, names)

    arrs = (np.ones(3) * u.Jy, np.ones(3))
    names = ('a', 'b')
    with pytest.raises(ValueError):
        _, _ = process_quantities(arrs, names)

    unit = u.Jy
    arrs = (np.ones(3) * unit, np.ones(3), np.ones(3) * unit)
    names = ('a', 'b', 'c')
    with pytest.raises(ValueError):
        _, _ = process_quantities(arrs, names)

    unit = u.Jy
    arrs = (np.ones(3) * unit, np.ones(3), np.ones(3) * u.km)
    names = ('a', 'b', 'c')
    with pytest.raises(ValueError):
        _, _ = process_quantities(arrs, names)
