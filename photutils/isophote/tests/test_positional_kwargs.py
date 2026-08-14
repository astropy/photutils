# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for deprecation warnings when optional arguments are passed
positionally.
"""

import numpy as np
import pytest
from astropy.utils.exceptions import AstropyDeprecationWarning

from photutils.isophote.ellipse import Ellipse
from photutils.isophote.geometry import EllipseGeometry
from photutils.isophote.model import build_ellipse_model
from photutils.isophote.sample import CentralEllipseSample, EllipseSample
from photutils.isophote.tests.make_test_data import make_test_image


class TestEllipsePositionalKwargs:
    """
    Test Ellipse.__init__, fit_image, fit_isophote for positional
    optional args.
    """

    def setup_method(self):
        # A small image with a capped fit is sufficient (and fast)
        # for testing the deprecation warnings
        self.data = make_test_image(nx=128, ny=128, sma=10.0, seed=0)

    def test_init_positional_warns(self):
        geometry = EllipseGeometry(x0=64, y0=64, sma=10, eps=0.2,
                                   pa=np.pi / 2)
        match = '__init__'
        with pytest.warns(AstropyDeprecationWarning, match=match):
            Ellipse(self.data, geometry)

    def test_init_keyword_no_warning(self):
        geometry = EllipseGeometry(x0=64, y0=64, sma=10, eps=0.2,
                                   pa=np.pi / 2)
        Ellipse(self.data, geometry=geometry)

    def test_fit_image_positional_warns(self):
        ellipse = Ellipse(self.data)
        match = 'fit_image'
        with pytest.warns(AstropyDeprecationWarning, match=match):
            ellipse.fit_image(10.0, maxsma=15.0)

    def test_fit_image_keyword_no_warning(self):
        ellipse = Ellipse(self.data)
        ellipse.fit_image(sma0=10.0, maxsma=15.0)

    def test_fit_isophote_positional_warns(self):
        ellipse = Ellipse(self.data)
        match = 'fit_isophote'
        with pytest.warns(AstropyDeprecationWarning, match=match):
            ellipse.fit_isophote(10.0, 0.1)

    def test_fit_isophote_keyword_no_warning(self):
        ellipse = Ellipse(self.data)
        ellipse.fit_isophote(10.0, step=0.1)


class TestEllipseGeometryPositionalKwargs:
    """
    Test EllipseGeometry.__init__ and find_center.
    """

    def test_init_positional_warns(self):
        match = '__init__'
        with pytest.warns(AstropyDeprecationWarning, match=match):
            EllipseGeometry(0.0, 0.0, 100.0, 0.0, 0.0, 0.2)

    def test_init_keyword_no_warning(self):
        EllipseGeometry(0.0, 0.0, 100.0, 0.0, 0.0, astep=0.2)

    def test_find_center_positional_warns(self):
        data = make_test_image(seed=0)
        geometry = EllipseGeometry(x0=256, y0=256, sma=10, eps=0.2,
                                   pa=np.pi / 2)
        match = 'find_center'
        with pytest.warns(AstropyDeprecationWarning, match=match):
            geometry.find_center(data, 0.5)

    def test_find_center_keyword_no_warning(self):
        data = make_test_image(seed=0)
        geometry = EllipseGeometry(x0=256, y0=256, sma=10, eps=0.2,
                                   pa=np.pi / 2)
        geometry.find_center(data, threshold=0.5)


@pytest.fixture(name='small_isolist', scope='module')
def fixture_small_isolist():
    """
    Fit isophotes to a small simulated galaxy image.

    A small image with a capped fit is sufficient (and fast) for
    testing the deprecation warnings, and the fit is shared across
    tests.
    """
    data = make_test_image(nx=128, ny=128, sma=10.0, seed=0)
    ellipse = Ellipse(data)
    isolist = ellipse.fit_image(sma0=10.0, maxsma=15.0)
    return data.shape, isolist


class TestBuildEllipseModelPositionalKwargs:
    """
    Test build_ellipse_model.
    """

    def test_positional_warns(self, small_isolist):
        shape, isolist = small_isolist
        match = 'build_ellipse_model'
        with pytest.warns(AstropyDeprecationWarning, match=match):
            build_ellipse_model(shape, isolist, 0.0)

    def test_keyword_no_warning(self, small_isolist):
        shape, isolist = small_isolist
        build_ellipse_model(shape, isolist, fill=0.0)


class TestEllipseSamplePositionalKwargs:
    """
    Test EllipseSample.__init__ and update.
    """

    def setup_method(self):
        self.data = make_test_image(seed=0)

    def test_init_positional_warns(self):
        match = '__init__'
        with pytest.warns(AstropyDeprecationWarning, match=match):
            EllipseSample(self.data, 40.0, 256.0)

    def test_init_keyword_no_warning(self):
        EllipseSample(self.data, 40.0, x0=256.0)

    def test_update_positional_warns(self):
        sample = EllipseSample(self.data, 40.0)
        fix = np.array([False, False, False, False])
        match = 'update'
        with pytest.warns(AstropyDeprecationWarning, match=match):
            sample.update(fix)

    def test_update_keyword_no_warning(self):
        sample = EllipseSample(self.data, 40.0)
        fix = np.array([False, False, False, False])
        sample.update(fixed_parameters=fix)


class TestCentralEllipseSamplePositionalKwargs:
    """
    Test CentralEllipseSample.update.
    """

    def test_update_positional_warns(self):
        data = make_test_image(seed=0)
        sample = CentralEllipseSample(data, 0.0)
        fix = np.array([False, False, False, False])
        match = 'update'
        with pytest.warns(AstropyDeprecationWarning, match=match):
            sample.update(fix)

    def test_update_keyword_no_warning(self):
        data = make_test_image(seed=0)
        sample = CentralEllipseSample(data, 0.0)
        fix = np.array([False, False, False, False])
        sample.update(fixed_parameters=fix)
