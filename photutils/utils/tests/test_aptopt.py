# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

# THIRD-PARTY
import numpy as np
from astropy.tests.helper import pytest
from numpy.testing import assert_allclose

# LOCAL
from ..aptopt import find_center


def test_find_center(seed=1234):
    # Generate normal distribution.
    np.random.seed(seed)
    data = np.hstack([np.random.normal(loc=1000.0, scale=2.5, size=10000),
                      np.random.normal(loc=1010.0, scale=3.0, size=10000)])

    # Build histogram.
    val, bins = np.histogram(data, bins=100)
    loc = (bins[1:] + bins[:-1]) * 0.5

    # Make initial guesses slightly off.
    result_1 = find_center(val, 1001, 2.5, loc=loc)[0]
    result_2 = find_center(val, 1007, 3.0, loc=loc)[0]

    # 0.1%
    assert_allclose(result_1, 1000, rtol=1e-3)
    assert_allclose(result_2, 1010, rtol=1e-3)

    # Ways it could fail
    with pytest.raises(ValueError):
        result = find_center(val, 1001, 0.0, loc=loc)[0]
    with pytest.raises(ValueError):
        result = find_center(val, 900, 2.5, loc=loc)[0]
