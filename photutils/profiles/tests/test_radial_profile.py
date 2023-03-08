# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the core module.
"""

import astropy.units as u
import numpy as np
import pytest
from astropy.modeling.models import Gaussian2D
from numpy.testing import assert_equal

from photutils.aperture import CircularAnnulus, CircularAperture
from photutils.profiles import RadialProfile


@pytest.fixture(name='profile_data')
def fixture_profile_data():
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


def test_radial_profile(profile_data):
    xycen, data, _, _ = profile_data

    min_radius = 0.0
    max_radius = 35.0
    radius_step = 1.0
    rp1 = RadialProfile(data, xycen, min_radius, max_radius, radius_step,
                        error=None, mask=None)

    assert_equal(rp1.radius, np.arange(36))
    assert rp1.area.shape == (36,)
    assert rp1.profile.shape == (36,)
    assert rp1.profile_error.shape == (0,)
    assert rp1.area[0] > 0.0

    assert len(rp1.apertures) == 36
    assert isinstance(rp1.apertures[0], CircularAperture)
    assert isinstance(rp1.apertures[1], CircularAnnulus)

    min_radius = 0.7
    max_radius = 35.0
    radius_step = 1.0
    rp2 = RadialProfile(data, xycen, min_radius, max_radius, radius_step,
                        error=None, mask=None)
    assert isinstance(rp2.apertures[0], CircularAnnulus)


def test_radial_profile_unit(profile_data):
    xycen, data, error, _ = profile_data

    min_radius = 0.0
    max_radius = 35.0
    radius_step = 1.0
    unit = u.Jy
    rp1 = RadialProfile(data << unit, xycen, min_radius, max_radius,
                        radius_step, error=error << unit, mask=None)
    assert rp1.profile.unit == unit
    assert rp1.profile_error.unit == unit

    with pytest.raises(ValueError):
        RadialProfile(data << unit, xycen, min_radius, max_radius, radius_step,
                      error=error, mask=None)


def test_radial_profile_error(profile_data):
    xycen, data, error, _ = profile_data

    min_radius = 0.0
    max_radius = 35.0
    radius_step = 1.0
    rp1 = RadialProfile(data, xycen, min_radius, max_radius, radius_step,
                        error=error, mask=None)

    assert_equal(rp1.radius, np.arange(36))
    assert rp1.area.shape == (36,)
    assert rp1.profile.shape == (36,)
    assert rp1.profile_error.shape == (36,)

    assert len(rp1.apertures) == 36
    assert isinstance(rp1.apertures[0], CircularAperture)
    assert isinstance(rp1.apertures[1], CircularAnnulus)
