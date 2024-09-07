# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the _stats module.
"""

import astropy.units as u
import numpy as np
import pytest
from numpy.testing import assert_equal

from photutils.utils._optional_deps import HAS_BOTTLENECK
from photutils.utils._stats import (_nanmax, _nanmean, _nanmedian, _nanmin,
                                    _nanstd, _nansum, _nanvar)

funcs = [(_nansum, np.nansum), (_nanmean, np.nanmean),
         (_nanmedian, np.nanmedian), (_nanstd, np.nanstd),
         (_nanvar, np.nanvar), (_nanmin, np.nanmin), (_nanmax, np.nanmax)]


@pytest.mark.skipif(not HAS_BOTTLENECK, reason='bottleneck is required')
@pytest.mark.parametrize('func', funcs)
@pytest.mark.parametrize('axis', [None, 0, 1, (0, 1), (1, 2), (2, 1),
                                  (0, 1, 2), (3, 1), (0, 3), (2, 0)])
@pytest.mark.parametrize('use_units', [False, True])
def test_nanmean(func, axis, use_units):
    arr = np.ones((5, 3, 8, 9))
    if use_units:
        arr <<= u.m

    result1 = func[0](arr, axis=axis)
    result2 = func[1](arr, axis=axis)
    assert_equal(result1, result2)
