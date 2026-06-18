# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Shared pytest fixtures for the aperture tests.
"""

import pytest
from astropy.wcs import WCS


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
