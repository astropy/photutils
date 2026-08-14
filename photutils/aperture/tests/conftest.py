# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Shared pytest fixtures and helpers for the aperture tests.
"""

import numpy as np
import pytest
from astropy.wcs import WCS

from photutils.aperture import CircularAperture
from photutils.datasets import make_4gaussians_image

# The shape of the small uniform image used by the quality-flag tests
UNIT_SHAPE = (25, 25)


class NoBatchCircularAperture(CircularAperture):
    """
    A `CircularAperture` subclass that does not opt in to the batch
    Cython driver (it is not decorated with
    ``_enable_batch_photometry``), forcing the mask-based code path.
    """


def make_scene():
    """
    Build a deterministic scene with a target source (label 1) and a
    bright neighbor source (label 2), on a nonzero background.

    Returns
    -------
    data, segm : 2D `~numpy.ndarray`
        The image and the matching segmentation image.
    """
    data = np.ones((50, 50))
    segm = np.zeros((50, 50), dtype=int)

    # Target source (label 1)
    data[18:25, 18:25] = 10.0
    segm[18:25, 18:25] = 1

    # Bright neighbor source (label 2)
    data[20:25, 26:32] = 100.0
    segm[20:25, 26:32] = 2

    return data, segm


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
