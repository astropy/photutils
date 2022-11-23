# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the parameters module.
"""

import numpy as np
import pytest
from numpy.testing import assert_equal

from photutils.utils._parameters import as_pair


def test_as_pair():
    assert_equal(as_pair('myparam', 4), (4, 4))

    assert_equal(as_pair('myparam', (3, 4)), (3, 4))

    assert_equal(as_pair('myparam', 0), (0, 0))

    with pytest.raises(ValueError):
        as_pair('myparam', 0, lower_bound=(0, 1))

    with pytest.raises(ValueError):
        as_pair('myparam', (1, np.nan))

    with pytest.raises(ValueError):
        as_pair('myparam', (1, np.inf))

    with pytest.raises(ValueError):
        as_pair('myparam', (3, 4), check_odd=True)

    with pytest.raises(ValueError):
        as_pair('myparam', 4, check_odd=True)
