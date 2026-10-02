# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Common test fixtures for the psf module tests.
"""

import sys
import threading
import weakref
from itertools import product

import numpy as np
import pytest
from astropy.modeling.models import Gaussian2D
from astropy.nddata import NDData
from scipy.interpolate import RectBivariateSpline

from photutils.psf import GriddedPSFModel


@pytest.fixture(name='short_switch_interval')
def fixture_short_switch_interval():
    """
    Fixture that makes the interpreter switch between threads very
    often, so that a race between threads shows up on builds with the
    GIL instead of only on free-threaded builds.
    """
    interval = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)
    yield
    sys.setswitchinterval(interval)


@pytest.fixture(name='psfmodel')
def fixture_griddedpsf_data():
    psfs = []
    yy, xx = np.mgrid[0:101, 0:101]
    for i in range(16):
        theta = np.deg2rad(i * 10.0)
        gmodel = Gaussian2D(1, 50, 50, 10, 5, theta=theta)
        psfs.append(gmodel(xx, yy))

    xgrid = [0, 40, 160, 200]
    ygrid = [0, 60, 140, 200]
    meta = {}
    meta['grid_xypos'] = list(product(xgrid, ygrid))
    meta['oversampling'] = 4

    nddata = NDData(psfs, meta=meta)
    return GriddedPSFModel(nddata)


class SplineBuilds:
    """
    A record of the spline objects that the PSF image models build.

    Attributes
    ----------
    count : int
        The number of splines built. A test can reset it to zero.
    """

    def __init__(self):
        self.count = 0
        self._refs = []
        self._lock = threading.Lock()

    def record(self, spline):
        """
        Record a new spline object.
        """
        with self._lock:
            self.count += 1
            self._refs.append(weakref.ref(spline))

    @property
    def n_alive(self):
        """
        The number of recorded spline objects that still exist.
        """
        return sum(ref() is not None for ref in self._refs)


@pytest.fixture(name='spline_builds')
def fixture_spline_builds(monkeypatch):
    """
    Fixture that records the `~scipy.interpolate.RectBivariateSpline`
    objects built by the `ImagePSF` and `GriddedPSFModel` models.
    """
    builds = SplineBuilds()

    class CountedSpline(RectBivariateSpline):
        def __init__(self, *args, **kwargs):
            builds.record(self)
            super().__init__(*args, **kwargs)

    for module in ('image_models', 'gridded_models'):
        monkeypatch.setattr(f'photutils.psf.{module}.RectBivariateSpline',
                            CountedSpline)
    return builds


@pytest.fixture(name='public_spline')
def fixture_public_spline():
    """
    Fixture that returns a stand-in for
    `~scipy.interpolate.RectBivariateSpline` that has only its public
    interface, without the undocumented ``tck`` and ``degrees``
    attributes.
    """
    class PublicSpline:
        def __init__(self, *args, **kwargs):
            self._spline = RectBivariateSpline(*args, **kwargs)

        def __call__(self, *args, **kwargs):
            return self._spline(*args, **kwargs)

        def get_knots(self):
            return self._spline.get_knots()

        def get_coeffs(self):
            return self._spline.get_coeffs()

        def partial_derivative(self, dx, dy):
            return self._spline.partial_derivative(dx, dy)

    return PublicSpline
