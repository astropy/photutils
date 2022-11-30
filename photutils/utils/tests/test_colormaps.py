# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the colormaps module.
"""

import pytest
from numpy.testing import assert_allclose

from photutils.utils._optional_deps import HAS_MATPLOTLIB
from photutils.utils.colormaps import make_random_cmap


@pytest.mark.skipif(not HAS_MATPLOTLIB, reason='matplotlib is required')
def test_colormap():
    ncolors = 100
    cmap = make_random_cmap(ncolors, seed=0)
    assert len(cmap.colors) == ncolors
    assert cmap.colors.shape == (100, 4)
    assert_allclose(cmap.colors[0], [0.36951484, 0.42125961, 0.65984082, 1.0])
