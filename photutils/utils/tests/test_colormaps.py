# Licensed under a 3-clause BSD style license - see LICENSE.rst

from numpy.testing import assert_allclose
import pytest

from ..colormaps import random_cmap

try:
    import matplotlib    # noqa
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


@pytest.mark.skipif('not HAS_MATPLOTLIB')
def test_colormap():
    ncolors = 100
    cmap = random_cmap(ncolors, random_state=12345)
    assert len(cmap.colors) == ncolors
    assert_allclose(cmap.colors[0], [0.9234715, 0.64837165, 0.76454726])
