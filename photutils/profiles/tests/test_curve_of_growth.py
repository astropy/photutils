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

    radii = np.arange(1, 37)
    cg1 = CurveOfGrowth(data, xycen, radii, error=None, mask=None)

    assert_equal(cg1.radius, radii)
    assert cg1.area.shape == (36,)
    assert cg1.profile.shape == (36,)
    assert cg1.profile_error.shape == (0,)
    assert_allclose(cg1.area[0], np.pi)

    assert len(cg1.apertures) == 36
    assert isinstance(cg1.apertures[0], CircularAperture)

    radii = np.arange(1, 36)
    cg2 = CurveOfGrowth(data, xycen, radii, error=None, mask=None)
    assert cg2.area[0] > 0.0
    assert isinstance(cg2.apertures[0], CircularAperture)


def test_curve_of_growth_units(profile_data):
    xycen, data, error, _ = profile_data

    radii = np.arange(1, 36)
    unit = u.Jy
    cg1 = CurveOfGrowth(data << unit, xycen, radii, error=error << unit,
                        mask=None)

    assert cg1.profile.unit == unit
    assert cg1.profile_error.unit == unit

    match = 'must all have the same units'
    with pytest.raises(ValueError, match=match):
        CurveOfGrowth(data << unit, xycen, radii, error=error, mask=None)


def test_curve_of_growth_error(profile_data):
    xycen, data, error, _ = profile_data

    radii = np.arange(1, 36)
    cg1 = CurveOfGrowth(data, xycen, radii, error=error, mask=None)

    assert cg1.profile.shape == (35,)
    assert cg1.profile_error.shape == (35,)


def test_curve_of_growth_mask(profile_data):
    xycen, data, error, mask = profile_data

    radii = np.arange(1, 36)
    cg1 = CurveOfGrowth(data, xycen, radii, error=error, mask=None)
    cg2 = CurveOfGrowth(data, xycen, radii, error=error, mask=mask)

    assert cg1.profile.sum() > cg2.profile.sum()
    assert np.sum(cg1.profile_error**2) > np.sum(cg2.profile_error**2)


def test_curve_of_growth_normalize(profile_data):
    xycen, data, _, _ = profile_data

    radii = np.arange(1, 36)
    cg1 = CurveOfGrowth(data, xycen, radii, error=None, mask=None)
    cg2 = CurveOfGrowth(data, xycen, radii, error=None, mask=None)

    profile1 = cg1.profile
    cg1.normalize()
    profile2 = cg1.profile
    assert np.mean(profile2) < np.mean(profile1)

    cg1.unnormalize()
    assert_allclose(cg1.profile, cg2.profile)

    cg1.normalize(method='sum')
    cg1.normalize(method='max')
    cg1.unnormalize()
    assert_allclose(cg1.profile, cg2.profile)

    cg1.normalize(method='max')
    cg1.normalize(method='sum')
    cg1.normalize(method='max')
    cg1.normalize(method='max')
    cg1.unnormalize()
    assert_allclose(cg1.profile, cg2.profile)

    cg1.normalize(method='sum')
    profile3 = cg1.profile
    assert np.mean(profile3) < np.mean(profile1)

    cg1.unnormalize()
    assert_allclose(cg1.profile, cg2.profile)

    match = 'invalid method, must be "max" or "sum"'
    with pytest.raises(ValueError, match=match):
        cg1.normalize(method='invalid')

    cg1.__dict__['profile'] -= np.max(cg1.__dict__['profile'])
    match = 'The profile cannot be normalized'
    with pytest.warns(AstropyUserWarning, match=match):
        cg1.normalize(method='max')


def test_curve_of_growth_interp(profile_data):
    xycen, data, error, _ = profile_data
    radii = np.arange(1, 36)
    cg1 = CurveOfGrowth(data, xycen, radii, error=None, mask=None)
    cg1.normalize()
    ee_radii = np.array([0, 5, 10, 20, 25, 50], dtype=float)
    ee_vals = cg1.calc_ee_at_radius(ee_radii)
    ee_expected = np.array([np.nan, 0.1176754, 0.39409357, 0.86635049,
                            0.95805792, np.nan])
    assert_allclose(ee_vals, ee_expected, rtol=1e-6)

    rr = cg1.calc_radius_at_ee(ee_vals)
    ee_radii[[0, -1]] = np.nan
    assert_allclose(rr, ee_radii, rtol=1e-6)

    radii = np.linspace(0.1, 36, 200)
    cg1 = CurveOfGrowth(data, xycen, radii, error=None, mask=None,
                        method='center')
    ee_vals = cg1.calc_ee_at_radius(ee_radii)
    match = 'The curve-of-growth profile is not monotonically increasing'
    with pytest.raises(ValueError, match=match):
        cg1.calc_radius_at_ee(ee_vals)


def test_curve_of_growth_inputs(profile_data):
    xycen, data, error, _ = profile_data

    match = 'radii must be > 0'
    radii = np.arange(10)
    with pytest.raises(ValueError, match=match):
        CurveOfGrowth(data, xycen, radii, error=None, mask=None)

    match = 'radii must be a 1D array and have at least two values'
    with pytest.raises(ValueError, match=match):
        CurveOfGrowth(data, xycen, [1], error=None, mask=None)
    with pytest.raises(ValueError, match=match):
        CurveOfGrowth(data, xycen, np.arange(1, 7).reshape(2, 3), error=None,
                      mask=None)

    match = 'radii must be strictly increasing'
    radii = np.arange(1, 10)[::-1]
    with pytest.raises(ValueError, match=match):
        CurveOfGrowth(data, xycen, radii, error=None, mask=None)

    unit1 = u.Jy
    unit2 = u.km
    radii = np.arange(1, 36)
    match = 'must all have the same units'
    with pytest.raises(ValueError, match=match):
        CurveOfGrowth(data << unit1, xycen, radii, error=error << unit2)


@pytest.mark.skipif(not HAS_MATPLOTLIB, reason='matplotlib is required')
def test_curve_of_growth_plot(profile_data):
    xycen, data, error, _ = profile_data

    radii = np.arange(1, 36)
    cg1 = CurveOfGrowth(data, xycen, radii, error=None, mask=None)
    cg1.plot()
    match = 'Errors were not input'
    with pytest.warns(AstropyUserWarning, match=match):
        cg1.plot_error()

    cg2 = CurveOfGrowth(data, xycen, radii, error=error, mask=None)
    cg2.plot()
    pc1 = cg2.plot_error()
    assert_allclose(pc1.get_facecolor(), [[0.5, 0.5, 0.5, 0.3]])
    pc2 = cg2.plot_error(facecolor='blue')
    assert_allclose(pc2.get_facecolor(), [[0, 0, 1, 1]])

    unit = u.Jy
    cg3 = CurveOfGrowth(data << unit, xycen, radii, error=error << unit,
                        mask=None)
    cg3.plot()
    cg3.plot_error()
