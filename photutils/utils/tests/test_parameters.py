# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the parameters module.
"""

import numpy as np
import pytest
from numpy.testing import assert_equal

from photutils.utils._parameters import SigmaClipSentinelDefault, as_pair


def test_as_pair():
    """
    Test as_pair with various inputs and validation options.
    """
    assert_equal(as_pair('myparam', 4), (4, 4))

    assert_equal(as_pair('myparam', (3, 4)), (3, 4))

    assert_equal(as_pair('myparam', 0), (0, 0))

    match = 'must be > 0'
    with pytest.raises(ValueError, match=match):
        as_pair('myparam', 0, lower_bound=(0, 0))

    match = 'must be a finite value'
    with pytest.raises(ValueError, match=match):
        as_pair('myparam', (1, np.nan))

    with pytest.raises(ValueError, match=match):
        as_pair('myparam', (1, np.inf))

    # Test check_odd=True success cases
    assert_equal(as_pair('myparam', (3, 5), check_odd=True), (3, 5))
    assert_equal(as_pair('myparam', 3, check_odd=True), (3, 3))

    match = 'must have an odd value for both axes'
    with pytest.raises(ValueError, match=match):
        as_pair('myparam', (3, 4), check_odd=True)

    with pytest.raises(ValueError, match=match):
        as_pair('myparam', 4, check_odd=True)

    # Test len(value) != 2 (e.g., 3 elements)
    match = 'must have 1 or 2 elements'
    with pytest.raises(ValueError, match=match):
        as_pair('myparam', (1, 2, 3))

    # Test value.ndim != 1 (e.g., 2D input with 2 elements)
    match = 'must be 1D'
    with pytest.raises(ValueError, match=match):
        as_pair('myparam', np.array([[1, 2]]))

    # Test non-integer dtype
    match = 'must have integer values'
    with pytest.raises(ValueError, match=match):
        as_pair('myparam', 1.5)

    # Test lower_bound with wrong length
    match = 'lower_bound must contain only 2 elements'
    with pytest.raises(ValueError, match=match):
        as_pair('myparam', 1, lower_bound=(0,))

    # Test upper_bound clipping
    result = as_pair('myparam', (10, 20), upper_bound=(5, 15))
    assert_equal(result, (5, 15))

    # Test inclusive lower_bound (>= bound)
    result = as_pair('myparam', 0, lower_bound=(0, 1))
    assert_equal(result, (0, 0))


def test_sigmaclip_sentinel_repr():
    """
    Test SigmaClipSentinelDefault __repr__ output.
    """
    sentinel = SigmaClipSentinelDefault(sigma=3.0, maxiters=10)
    result = repr(sentinel)
    assert 'SigmaClip' in result
    assert '3.0' in result
    assert '10' in result
