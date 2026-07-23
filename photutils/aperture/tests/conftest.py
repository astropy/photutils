# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Shared pytest fixtures for the aperture tests.
"""

import pytest
from astropy.wcs import WCS

from photutils.datasets import make_4gaussians_image


@pytest.fixture(name='data')
def fixture_data():
    """
    A 2D image containing four Gaussian sources on a noisy background.

    The image is deterministic and must be treated as read-only by
    tests.
    """
    return make_4gaussians_image()


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
