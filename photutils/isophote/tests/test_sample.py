# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the sample module.
"""

import numpy as np
import pytest

from photutils.isophote.geometry import EllipseGeometry
from photutils.isophote.integrator import (BILINEAR, MEAN, MEDIAN,
                                           NEAREST_NEIGHBOR)
from photutils.isophote.isophote import Isophote
from photutils.isophote.sample import EllipseSample
from photutils.isophote.tests.make_test_data import make_test_image

DEFAULT_FIX = np.array([False, False, False, False])

DATA = make_test_image(background=100.0, i0=0.0, noise=10.0, seed=0)


# the median is not so good at estimating rms
@pytest.mark.parametrize(('integrmode', 'amin', 'amax'),
                         [(NEAREST_NEIGHBOR, 7.0, 15.0),
                          (BILINEAR, 7.0, 15.0),
                          (MEAN, 7.0, 15.0),
                          (MEDIAN, 6.0, 15.0)])
def test_scatter(integrmode, amin, amax):
    """
    Check that the pixel standard deviation can be reliably estimated
    from the rms scatter and the sector area.

    The test data is just a flat image with noise, no galaxy. We define
    the noise rms and then compare how close the pixel std dev estimated
    at extraction matches this input noise.
    """
    sample = EllipseSample(DATA, 50.0, astep=0.2, integrmode=integrmode)
    sample.update(fixed_parameters=DEFAULT_FIX)
    iso = Isophote(sample, 0, valid=True, stop_code=0)

    assert iso.pix_stddev < amax
    assert iso.pix_stddev > amin


def test_coordinates():
    sample = EllipseSample(DATA, 50.0)
    sample.update(fixed_parameters=DEFAULT_FIX)
    x, y = sample.coordinates()

    assert isinstance(x, np.ndarray)
    assert isinstance(y, np.ndarray)


def test_sclip():
    sample = EllipseSample(DATA, 50.0, n_clip=3)
    sample.update(fixed_parameters=DEFAULT_FIX)
    x, y = sample.coordinates()

    assert isinstance(x, np.ndarray)
    assert isinstance(y, np.ndarray)


def test_geometry_sma_override_resets_cached_attributes():
    geometry = EllipseGeometry(25.0, 25.0, 10.0, 0.3, np.pi / 4.0,
                               astep=0.2)
    sample = EllipseSample(DATA, 20.0, geometry=geometry)
    expected_geometry = EllipseGeometry(25.0, 25.0, 20.0, 0.3,
                                        np.pi / 4.0, astep=0.2)

    assert sample.geometry.sma == 20.0
    assert geometry.sma == 10.0
    np.testing.assert_allclose(sample.geometry._area_factor,
                               expected_geometry._area_factor)
    np.testing.assert_allclose(sample.geometry.sector_angular_width,
                               expected_geometry.sector_angular_width)
    np.testing.assert_allclose(sample.geometry.initial_polar_angle,
                               expected_geometry.initial_polar_angle)
    np.testing.assert_allclose(sample.geometry.initial_polar_radius,
                               expected_geometry.initial_polar_radius)
