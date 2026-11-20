# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the sample module.
"""

import astropy.units as u
import numpy as np
import pytest
from numpy.testing import assert_array_equal

from photutils.isophote.geometry import EllipseGeometry
from photutils.isophote.integrator import (BILINEAR, MEAN, MEDIAN,
                                           NEAREST_NEIGHBOR)
from photutils.isophote.isophote import Isophote
from photutils.isophote.sample import EllipseSample
from photutils.isophote.tests.make_test_data import make_test_image

DEFAULT_FIX = np.array([False, False, False, False])

DATA = make_test_image(background=100.0, i0=0.0, noise=10.0, seed=0)


# The median is not so good at estimating rms.
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


@pytest.mark.parametrize(('x0', 'y0'), [(0.0, 40.0), (40.0, 0.0),
                                        (0.0, 0.0)])
def test_zero_center(x0, y0):
    """
    Test that a center coordinate of zero is not replaced by the image
    center.
    """
    sample = EllipseSample(DATA, 10.0, x0=x0, y0=y0)
    assert sample.geometry.x0 == x0
    assert sample.geometry.y0 == y0


def test_default_center():
    sample = EllipseSample(DATA, 10.0)
    assert sample.geometry.x0 == DATA.shape[1] / 2
    assert sample.geometry.y0 == DATA.shape[0] / 2


@pytest.mark.parametrize('integrmode', [MEAN, MEDIAN])
def test_zero_inner_sma(integrmode):
    """
    Test the area integration modes when the inner bounding ellipse of
    the sectors has zero size.
    """
    data = make_test_image(seed=0)
    sample = EllipseSample(data, 1.0, astep=2.0, linear_growth=True,
                           integrmode=integrmode)
    sample.update(fixed_parameters=DEFAULT_FIX)
    assert sample.actual_points > 0
    assert np.isfinite(sample.mean)


def test_invalid_image():
    match = 'image must not be a Quantity array'
    with pytest.raises(TypeError, match=match):
        EllipseSample(DATA * u.Jy, 10.0)

    match = 'image must be a 2D array'
    with pytest.raises(ValueError, match=match):
        EllipseSample(DATA[0], 10.0)


def test_array_like_image():
    sample = EllipseSample(DATA.tolist(), 10.0)
    assert isinstance(sample.image, np.ndarray)
    expected = EllipseSample(DATA, 10.0)
    assert_array_equal(sample.extract(), expected.extract())


@pytest.mark.parametrize('integrmode', [NEAREST_NEIGHBOR, BILINEAR, MEAN,
                                        MEDIAN])
def test_nonfinite_values(integrmode):
    """
    Test that nonfinite values are excluded from the sample, like
    masked pixels.
    """
    # The nonfinite pixels are on the major and minor axes of the
    # elliptical path.
    data = DATA.copy()
    data[254:259, 294:299] = np.nan
    data[286:291, 254:259] = np.inf
    values = EllipseSample(data, 40.0, integrmode=integrmode).extract()
    assert np.all(np.isfinite(values[2]))

    masked = np.ma.masked_invalid(data)
    expected = EllipseSample(masked, 40.0, integrmode=integrmode).extract()
    assert_array_equal(values, expected)

    unmasked = EllipseSample(DATA, 40.0, integrmode=integrmode).extract()
    assert not np.array_equal(values[2], unmasked[2])
