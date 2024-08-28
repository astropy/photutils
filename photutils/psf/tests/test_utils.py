# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the utils module.
"""

import numpy as np
import pytest

from photutils.psf.utils import _interpolate_missing_data
from photutils.utils._optional_deps import HAS_SCIPY


@pytest.mark.skipif(not HAS_SCIPY, reason='scipy is required')
def test_interpolate_missing_data():
    data = np.arange(100).reshape(10, 10)
    mask = np.zeros_like(data, dtype=bool)
    mask[5, 5] = True

    data_int = _interpolate_missing_data(data, mask, method='nearest')
    assert 54 <= data_int[5, 5] <= 56

    data_int = _interpolate_missing_data(data, mask, method='cubic')
    assert 54 <= data_int[5, 5] <= 56

    match = "'data' must be a 2D array."
    with pytest.raises(ValueError, match=match):
        _interpolate_missing_data(np.arange(10), mask)

    match = "'mask' and 'data' must have the same shape."
    with pytest.raises(ValueError, match=match):
        _interpolate_missing_data(data, mask[1:, :])

    match = 'Unsupported interpolation method'
    with pytest.raises(ValueError, match=match):
        _interpolate_missing_data(data, mask, method='invalid')
