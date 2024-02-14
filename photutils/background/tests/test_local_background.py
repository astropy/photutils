# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the local_background module.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from photutils.background import LocalBackground, MedianBackground


def test_local_background():
    data = np.ones((101, 101))
    local_bkg = LocalBackground(5, 10, bkg_estimator=MedianBackground())

    x = np.arange(1, 7) * 10
    y = np.arange(1, 7) * 10
    bkg = local_bkg(data, x, y)
    assert_allclose(bkg, np.ones(len(x)))

    # test scalar x and y
    bkg2 = local_bkg(data, x[2], y[2])
    assert not isinstance(bkg2, np.ndarray)
    assert_allclose(bkg[2], bkg2)

    bkg3 = local_bkg(data, -100, -100)
    assert np.isnan(bkg3)

    with pytest.raises(ValueError):
        _ = local_bkg(data, x[2], np.inf)

    cls_repr = repr(local_bkg)
    assert cls_repr.startswith(local_bkg.__class__.__name__)
