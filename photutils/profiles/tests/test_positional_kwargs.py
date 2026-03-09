# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for deprecation warnings when optional arguments are passed
positionally.
"""

import numpy as np
import pytest
from astropy.modeling.models import Gaussian2D
from astropy.utils.exceptions import AstropyDeprecationWarning

from photutils.profiles import CurveOfGrowth, RadialProfile
from photutils.utils._optional_deps import HAS_MATPLOTLIB


@pytest.fixture
def profile_data():
    """
    Create a simple radial profile for testing.
    """
    gmodel = Gaussian2D(42.1, 50, 50, 4.7, 4.7, 0)
    yy, xx = np.mgrid[0:100, 0:100]
    data = gmodel(xx, yy)
    xycen = (50, 50)
    return data, xycen


class TestProfileBaseNormalizePositionalKwargs:
    """
    Test that ProfileBase.normalize warns for positional optional args.
    """

    def test_positional_warns(self, profile_data):
        data, xycen = profile_data
        radii = np.arange(1, 20)
        cog = CurveOfGrowth(data, xycen, radii)
        match = 'normalize'
        with pytest.warns(AstropyDeprecationWarning, match=match):
            cog.normalize('max')

    def test_keyword_no_warning(self, profile_data):
        data, xycen = profile_data
        radii = np.arange(1, 20)
        cog = CurveOfGrowth(data, xycen, radii)
        cog.normalize(method='max')


class TestProfileBasePlotPositionalKwargs:
    """
    Test that ProfileBase.plot warns for positional optional args.
    """

    @pytest.mark.skipif(not HAS_MATPLOTLIB, reason='matplotlib is required')
    def test_positional_warns(self, profile_data):
        data, xycen = profile_data
        edge_radii = np.arange(20)
        rp = RadialProfile(data, xycen, edge_radii)
        match = 'plot'
        with pytest.warns(AstropyDeprecationWarning, match=match):
            rp.plot(None)

    @pytest.mark.skipif(not HAS_MATPLOTLIB, reason='matplotlib is required')
    def test_keyword_no_warning(self, profile_data):
        data, xycen = profile_data
        edge_radii = np.arange(20)
        rp = RadialProfile(data, xycen, edge_radii)
        rp.plot(ax=None)


class TestProfileBasePlotErrorPositionalKwargs:
    """
    Test that ProfileBase.plot_error warns for positional optional args.
    """

    @pytest.mark.skipif(not HAS_MATPLOTLIB, reason='matplotlib is required')
    def test_positional_warns(self, profile_data):
        data, xycen = profile_data
        error = np.ones_like(data)
        edge_radii = np.arange(20)
        rp = RadialProfile(data, xycen, edge_radii, error=error)
        match = 'plot_error'
        with pytest.warns(AstropyDeprecationWarning, match=match):
            rp.plot_error(None)

    @pytest.mark.skipif(not HAS_MATPLOTLIB, reason='matplotlib is required')
    def test_keyword_no_warning(self, profile_data):
        data, xycen = profile_data
        error = np.ones_like(data)
        edge_radii = np.arange(20)
        rp = RadialProfile(data, xycen, edge_radii, error=error)
        rp.plot_error(ax=None)
