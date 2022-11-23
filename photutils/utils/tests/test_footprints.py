# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the footprints module.
"""

import numpy as np
import pytest
from numpy.testing import assert_equal

from photutils.utils.footprints import circular_footprint


def test_footprints():
    footprint = circular_footprint(1)
    result = np.array([[0, 1, 0],
                       [1, 1, 1],
                       [0, 1, 0]])
    assert_equal(footprint, result)

    footprint = circular_footprint(2)
    result = np.array([[0, 0, 1, 0, 0],
                       [0, 1, 1, 1, 0],
                       [1, 1, 1, 1, 1],
                       [0, 1, 1, 1, 0],
                       [0, 0, 1, 0, 0]])
    assert_equal(footprint, result)

    with pytest.raises(ValueError):
        circular_footprint(5.1)

    with pytest.raises(ValueError):
        circular_footprint(0)

    with pytest.raises(ValueError):
        circular_footprint(-1)

    with pytest.raises(ValueError):
        circular_footprint(np.inf)

    with pytest.raises(ValueError):
        circular_footprint(np.nan)
