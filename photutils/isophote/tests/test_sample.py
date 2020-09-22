# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the sample module.
"""

import numpy as np
import pytest

from .make_test_data import make_test_image
from ..integrator import BILINEAR, MEAN, MEDIAN, NEAREST_NEIGHBOR
from ..isophote import Isophote
from ..sample import EllipseSample

DEFAULT_FIX = np.array([False, False, False, False])

DATA = make_test_image(background=100., i0=0., noise=10., seed=0)


# the median is not so good at estimating rms
@pytest.mark.parametrize('integrmode, amin, amax',
                         [(NEAREST_NEIGHBOR, 7., 15.),
                          (BILINEAR, 7., 15.),
                          (MEAN, 7., 15.),
                          (MEDIAN, 6., 15.)])
def test_scatter(integrmode, amin, amax):
    """
    Check that the pixel standard deviation can be reliably estimated
    from the rms scatter and the sector area.

    The test data is just a flat image with noise, no galaxy. We define
    the noise rms and then compare how close the pixel std dev estimated
    at extraction matches this input noise.
    """

    sample = EllipseSample(DATA, 50., astep=0.2, integrmode=integrmode)
    sample.update(DEFAULT_FIX)
    iso = Isophote(sample, 0, True, 0)

    assert iso.pix_stddev < amax
    assert iso.pix_stddev > amin


def test_coordinates():
    sample = EllipseSample(DATA, 50.)
    sample.update(DEFAULT_FIX)
    x, y = sample.coordinates()

    assert isinstance(x, np.ndarray)
    assert isinstance(y, np.ndarray)


def test_sclip():
    sample = EllipseSample(DATA, 50., nclip=3)
    sample.update(DEFAULT_FIX)
    x, y = sample.coordinates()

    assert isinstance(x, np.ndarray)
    assert isinstance(y, np.ndarray)
