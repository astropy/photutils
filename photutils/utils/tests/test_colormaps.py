# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the colormaps module.
"""

import pytest
from astropy.utils.exceptions import AstropyDeprecationWarning
from numpy.testing import assert_allclose

from photutils.utils._optional_deps import HAS_MATPLOTLIB
from photutils.utils.colormaps import make_random_cmap


@pytest.mark.skipif(not HAS_MATPLOTLIB, reason='matplotlib is required')
def test_colormap():
    """
    Test make_random_cmap with default parameters.
    """
    n_colors = 100
    cmap = make_random_cmap(n_colors=n_colors, seed=0)
    assert len(cmap.colors) == n_colors
    assert cmap.colors.shape == (100, 4)
    assert_allclose(cmap.colors[0], [0.36951484, 0.42125961, 0.65984082, 1.0])


@pytest.mark.skipif(not HAS_MATPLOTLIB, reason='matplotlib is required')
def test_colormap_n_colors_one():
    """
    Test make_random_cmap with n_colors=1.
    """
    cmap = make_random_cmap(n_colors=1, seed=0)
    assert len(cmap.colors) == 1
    assert cmap.colors.shape == (1, 4)


def test_colormap_n_colors_invalid():
    """
    Test make_random_cmap with invalid n_colors.
    """
    match = 'n_colors must be at least 1'
    with pytest.raises(ValueError, match=match):
        make_random_cmap(n_colors=0)
    with pytest.raises(ValueError, match=match):
        make_random_cmap(n_colors=-1)


@pytest.mark.skipif(not HAS_MATPLOTLIB, reason='matplotlib is required')
def test_colormap_ncolors_deprecated():
    """
    Test that using the deprecated ``ncolors`` keyword raises a
    deprecation warning.
    """
    match = "'ncolors' was deprecated"
    with pytest.warns(AstropyDeprecationWarning, match=match):
        cmap = make_random_cmap(ncolors=10, seed=0)
    assert len(cmap.colors) == 10
