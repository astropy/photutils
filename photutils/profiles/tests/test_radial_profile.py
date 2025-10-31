# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the core module.
"""

import astropy.units as u
import numpy as np
import pytest
from astropy.modeling.models import Gaussian1D, Gaussian2D
from astropy.utils.exceptions import AstropyUserWarning
from numpy.testing import assert_allclose, assert_equal

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

    edge_radii = np.arange(36)
    rp1 = RadialProfile(data, xycen, edge_radii, error=None, mask=None)

    assert_equal(rp1.radius, np.arange(35) + 0.5)
    assert rp1.area.shape == (35,)
    assert rp1.profile.shape == (35,)
    assert rp1.profile_error.shape == (0,)
    assert rp1.area[0] > 0.0

    assert len(rp1.apertures) == 35
    assert isinstance(rp1.apertures[0], CircularAperture)
    assert isinstance(rp1.apertures[1], CircularAnnulus)

    edge_radii = np.arange(36) + 0.1
    rp2 = RadialProfile(data, xycen, edge_radii, error=None, mask=None)
    assert isinstance(rp2.apertures[0], CircularAnnulus)


def test_radial_profile_normalization(profile_data):
    xycen, data, error, _ = profile_data

    edge_radii = np.arange(36)
    rp1 = RadialProfile(data, xycen, edge_radii, error=error, mask=None)

    profile = rp1.profile
    profile_error = rp1.profile_error
    data_profile = rp1.data_profile
    rp1.normalize()
    assert np.max(rp1.profile) == 1.0
    assert np.max(rp1.profile_error) <= np.max(profile_error)
    assert np.max(rp1.data_profile) <= np.max(data_profile)

    rp1.unnormalize()
    assert_allclose(rp1.profile, profile)
    assert_allclose(rp1.profile_error, profile_error)
    assert_allclose(rp1.data_profile, data_profile)


def test_radial_profile_data(profile_data):
    xycen, data, _, _ = profile_data

    edge_radii = np.arange(36)
    rp1 = RadialProfile(data, xycen, edge_radii, error=None, mask=None)

    data_radius = rp1.data_radius
    data_profile = rp1.data_profile
    assert np.max(data_radius) <= np.max(edge_radii)
    assert data_radius.shape == data_profile.shape
    assert np.min(data_profile) >= np.min(data)
    assert np.max(data_profile) <= np.max(data)


def test_radial_profile_inputs(profile_data):
    xycen, data, _, _ = profile_data

    match = 'minimum radii must be >= 0'
    edge_radii = np.arange(-1, 10)
    with pytest.raises(ValueError, match=match):
        RadialProfile(data, xycen, edge_radii, error=None, mask=None)

    match = 'radii must be a 1D array and have at least two values'
    edge_radii = [1]
    with pytest.raises(ValueError, match=match):
        RadialProfile(data, xycen, edge_radii, error=None, mask=None)

    edge_radii = np.arange(6).reshape(2, 3)
    with pytest.raises(ValueError, match=match):
        RadialProfile(data, xycen, edge_radii, error=None, mask=None)

    match = 'radii must be strictly increasing'
    edge_radii = np.arange(10)[::-1]
    with pytest.raises(ValueError, match=match):
        RadialProfile(data, xycen, edge_radii, error=None, mask=None)

    match = 'error must have the same shape as data'
    edge_radii = np.arange(10)
    with pytest.raises(ValueError, match=match):
        RadialProfile(data, xycen, edge_radii, error=np.ones(3), mask=None)

    match = 'mask must have the same shape as data'
    edge_radii = np.arange(10)
    mask = np.ones(3, dtype=bool)
    with pytest.raises(ValueError, match=match):
        RadialProfile(data, xycen, edge_radii, error=None, mask=mask)


def test_radial_profile_gaussian(profile_data):
    xycen, data, _, _ = profile_data

    edge_radii = np.arange(36)
    rp1 = RadialProfile(data, xycen, edge_radii, error=None, mask=None)

    assert isinstance(rp1.gaussian_fit, Gaussian1D)
    assert rp1.gaussian_profile.shape == (35,)
    assert rp1.gaussian_fwhm < 23.6

    edge_radii = np.arange(201)
    rp2 = RadialProfile(data, xycen, edge_radii, error=None, mask=None)
    assert isinstance(rp2.gaussian_fit, Gaussian1D)
    assert rp2.gaussian_profile.shape == (200,)
    assert rp2.gaussian_fwhm < 23.6


def test_radial_profile_unit(profile_data):
    xycen, data, error, _ = profile_data

    edge_radii = np.arange(36)
    unit = u.Jy
    rp1 = RadialProfile(data << unit, xycen, edge_radii, error=error << unit,
                        mask=None)
    assert rp1.profile.unit == unit
    assert rp1.profile_error.unit == unit

    match = 'must all have the same units'
    with pytest.raises(ValueError, match=match):
        RadialProfile(data << unit, xycen, edge_radii, error=error, mask=None)


def test_radial_profile_error(profile_data):
    xycen, data, error, _ = profile_data

    edge_radii = np.arange(36)
    rp1 = RadialProfile(data, xycen, edge_radii, error=error, mask=None)

    assert_equal(rp1.radius, np.arange(35) + 0.5)
    assert rp1.area.shape == (35,)
    assert rp1.profile.shape == (35,)
    assert rp1.profile_error.shape == (35,)

    assert len(rp1.apertures) == 35
    assert isinstance(rp1.apertures[0], CircularAperture)
    assert isinstance(rp1.apertures[1], CircularAnnulus)


def test_radial_profile_normalize_nan(profile_data):
    """
    If the profile has NaNs (e.g., aperture outside the image), make
    sure the normalization ignores them.
    """
    xycen, data, _, _ = profile_data

    edge_radii = np.arange(101)
    rp1 = RadialProfile(data, xycen, edge_radii)
    rp1.normalize()
    assert not np.isnan(rp1.profile[0])


def test_radial_profile_nonfinite(profile_data):
    xycen, data, error, _ = profile_data
    data2 = data.copy()
    data2[40, 40] = np.nan
    mask = ~np.isfinite(data2)

    edge_radii = np.arange(36)
    rp1 = RadialProfile(data, xycen, edge_radii, error=None, mask=mask)

    rp2 = RadialProfile(data2, xycen, edge_radii, error=error, mask=mask)
    assert_allclose(rp1.profile, rp2.profile)

    match = 'Input data contains non-finite values'
    with pytest.warns(AstropyUserWarning, match=match):
        rp3 = RadialProfile(data2, xycen, edge_radii, error=error, mask=None)
    assert_allclose(rp1.profile, rp3.profile)

    error2 = error.copy()
    error2[40, 40] = np.inf
    with pytest.warns(AstropyUserWarning, match=match):
        rp4 = RadialProfile(data, xycen, edge_radii, error=error2, mask=None)
    assert_allclose(rp1.profile, rp4.profile)
