# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the _overlap module.
"""

import numpy as np
import pytest
from astropy.nddata import NoOverlapError, PartialOverlapError
from astropy.nddata import overlap_slices as astropy_overlap_slices

from photutils.utils._overlap import overlap_slices


@pytest.mark.parametrize('mode', ['partial', 'trim', 'strict'])
@pytest.mark.parametrize('small_shape', [(5, 7), [5, 7], np.array([5, 7])])
def test_overlap_slices(mode, small_shape):
    """
    Test that the wrapper matches the astropy function for any
    array-like small-array shape.
    """
    large_shape = (20, 30)
    position = (10, 12)
    result = overlap_slices(large_shape, small_shape, position, mode=mode)
    expected = astropy_overlap_slices(large_shape, (5, 7), position,
                                      mode=mode)
    assert result == expected


def test_overlap_slices_default_mode():
    """
    Test that the default mode is 'partial'.
    """
    result = overlap_slices((20, 30), np.array([5, 7]), (0, 0))
    expected = astropy_overlap_slices((20, 30), (5, 7), (0, 0),
                                      mode='partial')
    assert result == expected

    match = 'Arrays overlap only partially'
    with pytest.raises(PartialOverlapError, match=match):
        overlap_slices((20, 30), np.array([5, 7]), (0, 0), mode='strict')


@pytest.mark.parametrize('mode', ['partial', 'trim', 'strict'])
@pytest.mark.parametrize('position', [(50, -16), (-16, 50), (-16, -16)])
def test_overlap_slices_edge_no_overlap(mode, position):
    """
    Test that an array-valued small-array shape raises NoOverlapError
    when the small array ends exactly at the lower edge of the large
    array.
    """
    match = 'Arrays do not overlap'
    with pytest.raises(NoOverlapError, match=match):
        overlap_slices((100, 100), np.array([31, 31]), position, mode=mode)
