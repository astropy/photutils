# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for deprecation warnings when optional arguments are passed
positionally.
"""

import numpy as np
import pytest
from astropy.utils.exceptions import AstropyDeprecationWarning

from photutils.utils._optional_deps import HAS_MATPLOTLIB
from photutils.utils.colormaps import make_random_cmap
from photutils.utils.cutouts import CutoutImage
from photutils.utils.footprints import circular_footprint
from photutils.utils.interpolation import ShepardIDWInterpolator


class TestMakeRandomCmapPositionalKwargs:
    """
    Test make_random_cmap warns for positional optional args.
    """

    @pytest.mark.skipif(not HAS_MATPLOTLIB, reason='matplotlib is required')
    def test_positional_warns(self):
        match = 'make_random_cmap'
        with pytest.warns(AstropyDeprecationWarning, match=match):
            make_random_cmap(100)

    @pytest.mark.skipif(not HAS_MATPLOTLIB, reason='matplotlib is required')
    def test_keyword_no_warning(self):
        make_random_cmap(n_colors=100)


class TestCutoutImagePositionalKwargs:
    """
    Test CutoutImage.__init__ warns for positional optional args.
    """

    def setup_method(self):
        self.data = np.arange(100).reshape(10, 10).astype(float)

    def test_positional_warns(self):
        match = '__init__'
        with pytest.warns(AstropyDeprecationWarning, match=match):
            CutoutImage(self.data, (5, 5), (3, 3), 'trim')

    def test_keyword_no_warning(self):
        CutoutImage(self.data, (5, 5), (3, 3), mode='trim')


class TestCircularFootprintPositionalKwargs:
    """
    Test circular_footprint warns for positional optional args.
    """

    def test_positional_warns(self):
        match = 'circular_footprint'
        with pytest.warns(AstropyDeprecationWarning, match=match):
            circular_footprint(3, float)

    def test_keyword_no_warning(self):
        circular_footprint(3, dtype=float)


class TestShepardIDWInterpolatorPositionalKwargs:
    """
    Test ShepardIDWInterpolator warns for positional optional args.
    """

    def setup_method(self):
        rng = np.random.default_rng(0)
        self.coords = rng.random((100, 2))
        self.values = np.sin(self.coords[:, 0] + self.coords[:, 1])

    def test_init_positional_warns(self):
        weights = np.ones(100)
        match = 'init'
        with pytest.warns(AstropyDeprecationWarning, match=match):
            ShepardIDWInterpolator(self.coords, self.values, weights)

    def test_init_keyword_no_warning(self):
        weights = np.ones(100)
        ShepardIDWInterpolator(self.coords, self.values, weights=weights)

    def test_call_positional_warns(self):
        interp = ShepardIDWInterpolator(self.coords, self.values)
        match = '__call__'
        with pytest.warns(AstropyDeprecationWarning, match=match):
            interp([0.5, 0.6], 4)

    def test_call_keyword_no_warning(self):
        interp = ShepardIDWInterpolator(self.coords, self.values)
        interp([0.5, 0.6], n_neighbors=4)
