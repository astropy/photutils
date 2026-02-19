# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Pytest configuration and shared fixtures for psf_matching tests.
"""

import numpy as np
import pytest
from astropy.modeling.models import Gaussian2D


def _make_gaussian_psf(size, std):
    """
    Make a centered, normalized 2D Gaussian PSF.
    """
    cen = (size - 1) / 2.0
    yy, xx = np.mgrid[0:size, 0:size]
    model = Gaussian2D(1.0, cen, cen, std, std)
    psf = model(xx, yy)
    return psf / psf.sum()


@pytest.fixture(name='psf1')
def psf1():
    """
    Narrow Gaussian PSF (source).
    """
    return _make_gaussian_psf(25, 3.0)


@pytest.fixture(name='psf2')
def psf2():
    """
    Broad Gaussian PSF (target).
    """
    return _make_gaussian_psf(25, 5.0)
