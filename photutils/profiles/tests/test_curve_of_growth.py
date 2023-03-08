# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the core module.
"""

import astropy.units as u
import numpy as np
import pytest
from astropy.modeling.models import Gaussian2D
from astropy.utils.exceptions import AstropyUserWarning
from numpy.testing import assert_allclose, assert_equal

from photutils.aperture import CircularAperture
from photutils.profiles import CurveOfGrowth
from photutils.utils._optional_deps import HAS_MATPLOTLIB


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


def test_curve_of_growth(profile_data):
    xycen, data, _, _ = profile_data

    min_radius = 0.0
    max_radius = 35.0
    radius_step = 1.0
    cg1 = CurveOfGrowth(data, xycen, min_radius, max_radius, radius_step,
                        error=None, mask=None)

    assert_equal(cg1.radius, np.arange(36))
    assert cg1.area.shape == (36,)
    assert cg1.profile.shape == (36,)
    assert cg1.profile_error.shape == (0,)
    assert cg1.area[0] == 0.0

    assert len(cg1.apertures) == 36
    assert cg1.apertures[0] is None
    assert isinstance(cg1.apertures[1], CircularAperture)

    min_radius = 1.0
    max_radius = 35.0
    radius_step = 1.0
    cg2 = CurveOfGrowth(data, xycen, min_radius, max_radius, radius_step,
                        error=None, mask=None)
    assert cg2.area[0] > 0.0
    assert isinstance(cg2.apertures[0], CircularAperture)


def test_curve_of_growth_units(profile_data):
    xycen, data, error, _ = profile_data

    min_radius = 0.0
    max_radius = 35.0
    radius_step = 1.0
    unit = u.Jy
    cg1 = CurveOfGrowth(data << unit, xycen, min_radius, max_radius,
                        radius_step, error=error << unit, mask=None)

    assert cg1.profile.unit == unit
    assert cg1.profile_error.unit == unit

    with pytest.raises(ValueError):
        CurveOfGrowth(data << unit, xycen, min_radius, max_radius, radius_step,
                      error=error, mask=None)


def test_curve_of_growth_error(profile_data):
    xycen, data, error, _ = profile_data

    min_radius = 0.0
    max_radius = 35.0
    radius_step = 1.0
    cg1 = CurveOfGrowth(data, xycen, min_radius, max_radius, radius_step,
                        error=error, mask=None)

    assert cg1.profile.shape == (36,)
    assert cg1.profile_error.shape == (36,)


def test_curve_of_growth_mask(profile_data):
    xycen, data, error, mask = profile_data

    min_radius = 0.0
    max_radius = 35.0
    radius_step = 1.0
    cg1 = CurveOfGrowth(data, xycen, min_radius, max_radius, radius_step,
                        error=error, mask=None)
    cg2 = CurveOfGrowth(data, xycen, min_radius, max_radius, radius_step,
                        error=error, mask=mask)

    assert cg1.profile.sum() > cg2.profile.sum()
    assert np.sum(cg1.profile_error**2) > np.sum(cg2.profile_error**2)


def test_curve_of_growth_normalize(profile_data):
    xycen, data, _, _ = profile_data

    min_radius = 0.0
    max_radius = 35.0
    radius_step = 1.0
    cg1 = CurveOfGrowth(data, xycen, min_radius, max_radius, radius_step,
                        error=None, mask=None)

    profile1 = cg1.profile
    cg1.normalize()
    profile2 = cg1.profile
    assert np.mean(profile2) < np.mean(profile1)

    cg1.normalize(method='sum')
    profile3 = cg1.profile
    assert np.mean(profile3) < np.mean(profile1)

    with pytest.raises(ValueError):
        cg1.normalize(method='invalid')

    cg1.__dict__['profile'] -= np.max(cg1.__dict__['profile'])
    msg = 'The profile cannot be normalized'
    with pytest.warns(AstropyUserWarning, match=msg):
        cg1.normalize(method='max')


def test_curve_of_growth_inputs(profile_data):
    xycen, data, error, _ = profile_data

    min_radius = 0.0
    max_radius = 35.0
    radius_step = 1.0

    with pytest.raises(ValueError):
        CurveOfGrowth(data, xycen, -1, max_radius, radius_step, error=None,
                      mask=None)
    with pytest.raises(ValueError):
        CurveOfGrowth(data, xycen, 10.0, 1.0, radius_step, error=None,
                      mask=None)
    with pytest.raises(ValueError):
        CurveOfGrowth(data, xycen, min_radius, max_radius, -1.0, error=None,
                      mask=None)
    with pytest.raises(ValueError):
        CurveOfGrowth(data, xycen, min_radius, max_radius, radius_step,
                      error=np.ones((3, 3)), mask=None)
    with pytest.raises(ValueError):
        CurveOfGrowth(data, xycen, min_radius, max_radius, radius_step,
                      error=None, mask=np.ones((3, 3)))

    with pytest.raises(ValueError):
        unit1 = u.Jy
        unit2 = u.km
        CurveOfGrowth(data << unit1, xycen, min_radius, max_radius,
                      radius_step, error=error << unit2)


@pytest.mark.skipif(not HAS_MATPLOTLIB, reason='matplotlib is required')
def test_curve_of_growth_plot(profile_data):
    xycen, data, error, _ = profile_data

    min_radius = 0.0
    max_radius = 35.0
    radius_step = 1.0

    cg1 = CurveOfGrowth(data, xycen, min_radius, max_radius, radius_step,
                        error=None, mask=None)
    cg1.plot()
    with pytest.warns(AstropyUserWarning, match='Errors were not input'):
        cg1.plot_error()

    cg2 = CurveOfGrowth(data, xycen, min_radius, max_radius, radius_step,
                        error=error, mask=None)
    cg2.plot()
    pc1 = cg2.plot_error()
    assert_allclose(pc1.get_facecolor(), [[0.5, 0.5, 0.5, 0.3]])
    pc2 = cg2.plot_error(facecolor='blue')
    assert_allclose(pc2.get_facecolor(), [[0, 0, 1, 1]])

    unit = u.Jy
    cg3 = CurveOfGrowth(data << unit, xycen, min_radius, max_radius,
                        radius_step, error=error << unit, mask=None)
    cg3.plot()
    cg3.plot_error()
