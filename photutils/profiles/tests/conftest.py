# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Common test fixtures for the profiles module tests.
"""

import numpy as np
import pytest
from astropy.modeling.models import Gaussian2D


@pytest.fixture(name='profile_data')
def fixture_profile_data():
    """
    Fixture that generates a 2D Gaussian profile with error and mask
    arrays for testing the curve-of-growth classes.
    """
    xsize = 101
    ysize = 80
    xcen = (xsize - 1) / 2
    ycen = (ysize - 1) / 2
    xycen = (xcen, ycen)

    sig = 10.0
    model = Gaussian2D(21., xcen, ycen, sig, sig)
    y, x = np.mgrid[0:ysize, 0:xsize]
    data = model(x, y)

    error = 10.0 * np.sqrt(data)
    mask = np.zeros(data.shape, dtype=bool)
    mask[:int(ycen), :int(xcen)] = True

    return xycen, data, error, mask
