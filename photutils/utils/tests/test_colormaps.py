# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the colormaps module.
"""

from numpy.testing import assert_allclose
import pytest

from ..colormaps import make_random_cmap
from .._optional_deps import HAS_MATPLOTLIB  # noqa


@pytest.mark.skipif('not HAS_MATPLOTLIB')
def test_colormap():
    ncolors = 100
    cmap = make_random_cmap(ncolors, seed=0)
    assert len(cmap.colors) == ncolors
    assert_allclose(cmap.colors[0], [0.36951484, 0.42125961, 0.65984082])
