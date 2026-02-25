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
    """
    Test make_random_cmap with default parameters.
    """
    ncolors = 100
    cmap = make_random_cmap(ncolors, seed=0)
    assert len(cmap.colors) == ncolors
    assert cmap.colors.shape == (100, 4)
    assert_allclose(cmap.colors[0], [0.36951484, 0.42125961, 0.65984082, 1.0])


@pytest.mark.skipif(not HAS_MATPLOTLIB, reason='matplotlib is required')
def test_colormap_ncolors_one():
    """
    Test make_random_cmap with ncolors=1.
    """
    cmap = make_random_cmap(1, seed=0)
    assert len(cmap.colors) == 1
    assert cmap.colors.shape == (1, 4)


def test_colormap_ncolors_invalid():
    """
    Test make_random_cmap with invalid ncolors.
    """
    match = 'ncolors must be at least 1'
    with pytest.raises(ValueError, match=match):
        make_random_cmap(0)
    with pytest.raises(ValueError, match=match):
        make_random_cmap(-1)
