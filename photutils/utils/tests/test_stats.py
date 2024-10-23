# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the _stats module.
"""

import astropy.units as u
import numpy as np
import pytest
from numpy.testing import assert_equal

from photutils.utils._optional_deps import HAS_BOTTLENECK
from photutils.utils._stats import (nanmax, nanmean, nanmedian, nanmin, nanstd,
                                    nansum, nanvar)

funcs = [(nansum, np.nansum), (nanmean, np.nanmean),
         (nanmedian, np.nanmedian), (nanstd, np.nanstd),
         (nanvar, np.nanvar), (nanmin, np.nanmin), (nanmax, np.nanmax)]


@pytest.mark.skipif(not HAS_BOTTLENECK, reason='bottleneck is required')
@pytest.mark.parametrize('func', funcs)
@pytest.mark.parametrize('axis', [None, 0, 1, (0, 1), (1, 2), (2, 1),
                                  (0, 1, 2), (3, 1), (0, 3), (2, 0)])
@pytest.mark.parametrize('use_units', [False, True])
def test_nan_funcs(func, axis, use_units):
    arr = np.ones((5, 3, 8, 9))
    if use_units:
        arr <<= u.m

    result1 = func[0](arr, axis=axis)
    result2 = func[1](arr, axis=axis)
    assert_equal(result1, result2)
