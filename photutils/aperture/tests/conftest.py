# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Shared pytest fixtures for the aperture tests.
"""

import numpy as np
import pytest
from astropy.wcs import WCS

from photutils.datasets import make_4gaussians_image

# The shape of the small uniform image used by the quality-flag tests
UNIT_SHAPE = (25, 25)


@pytest.fixture(name='data')
def fixture_data():
    """
    A 2D image containing four Gaussian sources on a noisy background.

    The image is deterministic and must be treated as read-only by
    tests.
    """
    return make_4gaussians_image()


@pytest.fixture(name='unit_data')
def fixture_unit_data():
    """
    A small uniform image of ones.

    A fresh array is returned for each test, so it may be modified.
    """
    return np.ones(UNIT_SHAPE)


@pytest.fixture(name='unit_mask')
def fixture_unit_mask():
    """
    An all-`False` boolean mask matching the ``unit_data`` shape.

    A fresh array is returned for each test, so it may be modified.
    """
    return np.zeros(UNIT_SHAPE, dtype=bool)


@pytest.fixture
def tan_wcs():
    """
    Return a simple gnomonic (TAN) WCS used for aperture round-trip
    tests.
    """
    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [50.5, 50.5]
    wcs.wcs.cdelt = [-0.001, 0.001]
    wcs.wcs.crval = [10.0, 30.0]
    wcs.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    return wcs
