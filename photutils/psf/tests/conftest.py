# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Common test fixtures for the psf module tests.
"""

import sys
from itertools import product

import numpy as np
import pytest
from astropy.modeling.models import Gaussian2D
from astropy.nddata import NDData

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
