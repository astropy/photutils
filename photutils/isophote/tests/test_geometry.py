# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the geometry module.
"""

import copy
import pickle  # nosec B403
import warnings

import astropy.units as u
import numpy as np
import pytest
from astropy import log
from astropy.coordinates import Angle
from astropy.modeling.models import Gaussian2D
from numpy.testing import assert_allclose, assert_array_equal
from scipy.integrate import quad

from photutils.aperture import EllipticalAperture
from photutils.datasets import make_noise_image
from photutils.isophote.geometry import EllipseGeometry
from photutils.isophote.tests.make_test_data import make_test_image
from photutils.utils.exceptions import PhotutilsDeprecationWarning


@pytest.mark.parametrize(('astep', 'linear_growth'),
                         [(0.2, False), (20.0, True)])
def test_geometry(astep, linear_growth):
    geometry = EllipseGeometry(255.0, 255.0, 100.0, 0.4, (np.pi / 2) * u.rad,
                               step=astep,
                               linear_growth=linear_growth)

    sma1, sma2 = geometry._bounding_ellipses()
    assert_allclose((sma1, sma2), (90.0, 110.0), atol=0.01)

    # Using an arbitrary angle of 0.5 rad. This is to avoid a polar
    # vector that sits on top of one of the ellipse's axis.
    vertex_x, vertex_y = geometry._initialize_sector_geometry(0.6)
    assert_allclose(geometry._sector_angular_width, 0.0571, atol=0.01)
    assert_allclose(geometry.sector_area, 63.83, atol=0.01)
    assert_allclose(vertex_x, [215.4, 206.6, 213.5, 204.3], atol=0.1)
    assert_allclose(vertex_y, [316.1, 329.7, 312.5, 325.3], atol=0.1)


def test_repr_str():
    """
    Test the repr and str of an EllipseGeometry.
    """
    geometry = EllipseGeometry(np.float64(256.0), 255.5, 20.0, 0.25,
                               40.0 * u.deg, step=0.2, fix_center=True)
    expected = ('<EllipseGeometry(x0=256.0, y0=255.5, sma=20.0, '
                'ellipticity=0.25, orientation=40.0 deg, step=0.2, '
                'linear_growth=False, fix_center=True, '
                'fix_orientation=False, fix_ellipticity=False)>')
    assert repr(geometry) == expected

    expected = ('<photutils.isophote.geometry.EllipseGeometry>\n'
                'x0: 256.0\n'
                'y0: 255.5\n'
                'sma: 20.0\n'
                'ellipticity: 0.25\n'
                'orientation: 40.0 deg\n'
                'step: 0.2\n'
                'linear_growth: False\n'
                'fix_center: True\n'
                'fix_orientation: False\n'
                'fix_ellipticity: False')
    assert str(geometry) == expected


def test_to_polar_integer_input():
    """
    Test that integer coordinates and centers do not truncate the
    polar radii.
    """
    geometry = EllipseGeometry(0, 0, 10.0, 0.3, 0.5 * u.rad)
    x = np.array([[1, 2], [3, 4]])
    y = np.array([[1, 1], [2, 2]])
    radius, angle = geometry.to_polar(x, y)
    expected = EllipseGeometry(0.0, 0.0, 10.0, 0.3, 0.5 * u.rad).to_polar(
        x.astype(float), y.astype(float))
    assert_allclose(radius, expected[0])
    assert_allclose(angle, expected[1])


def test_to_polar():
    # Trivial case of a circle centered in (0.0, 0.0)
    geometry = EllipseGeometry(0.0, 0.0, 100.0, 0.0, 0.0 * u.rad, step=0.2,
                               linear_growth=False)

    r, p = geometry.to_polar(100.0, 0.0)
    assert_allclose(r, 100.0, atol=0.1)
    assert_allclose(p, 0.0, atol=0.0001)

    r, p = geometry.to_polar(0.0, 100.0)
    assert_allclose(r, 100.0, atol=0.1)
    assert_allclose(p, np.pi / 2.0, atol=0.0001)

    # Vector with length 100.0 at 45 deg angle
    r, p = geometry.to_polar(70.71, 70.71)
    assert_allclose(r, 100.0, atol=0.1)
    assert_allclose(p, np.pi / 4.0, atol=0.0001)

    # Position angle tilted 45 deg from X-axis
    geometry = EllipseGeometry(0.0, 0.0, 100.0, 0.0, (np.pi / 4.0) * u.rad,
                               step=0.2, linear_growth=False)

    r, p = geometry.to_polar(100.0, 0.0)
    assert_allclose(r, 100.0, atol=0.1)
    assert_allclose(p, np.pi * 7.0 / 4.0, atol=0.0001)

    r, p = geometry.to_polar(0.0, 100.0)
    assert_allclose(r, 100.0, atol=0.1)
    assert_allclose(p, np.pi / 4.0, atol=0.0001)

    # Vector with length 100.0 at 45 deg angle
    r, p = geometry.to_polar(70.71, 70.71)
    assert_allclose(r, 100.0, atol=0.1)
    assert_allclose(p, np.pi * 2.0, atol=0.0001)


def test_area():
    # Circle with center at origin
    geometry = EllipseGeometry(0.0, 0.0, 100.0, 0.0, 0.0 * u.rad, step=0.2,
                               linear_growth=False)

    # Sector at 45 deg on circle
    vertex_x, vertex_y = geometry._initialize_sector_geometry(np.deg2rad(45))
    assert_allclose(vertex_x, [65.21, 79.70, 62.03, 75.81], atol=0.01)
    assert_allclose(vertex_y, [62.03, 75.81, 65.21, 79.70], atol=0.01)

    # Sector at 0 deg on circle
    vertex_x, vertex_y = geometry._initialize_sector_geometry(0)
    assert_allclose(vertex_x, [89.97, 109.97, 89.97, 109.96], atol=0.01)
    assert_allclose(vertex_y, [-2.25, -2.75, 2.25, 2.75], atol=0.01)


def test_area2():
    # Circle with center at 100.0, 100.0
    geometry = EllipseGeometry(100.0, 100.0, 100.0, 0.0, 0.0 * u.rad, step=0.2,
                               linear_growth=False)

    # Sector at 45 deg on circle
    vertex_x, vertex_y = geometry._initialize_sector_geometry(np.deg2rad(45))
    assert_allclose(vertex_x, [165.21, 179.70, 162.03, 175.81], atol=0.01)
    assert_allclose(vertex_y, [162.03, 175.81, 165.21, 179.70], atol=0.01)

    # Sector at 225 deg on circle
    vertex_x, vertex_y = geometry._initialize_sector_geometry(np.deg2rad(225))
    assert_allclose(vertex_x, [34.79, 20.30, 37.97, 24.19], atol=0.01)
    assert_allclose(vertex_y, [37.97, 24.19, 34.79, 20.30], atol=0.01)


@pytest.mark.parametrize('phi', [0.0, 0.05, 1.0, np.pi / 2, np.pi, 4.0,
                                 6.2, 2.0 * np.pi])
def test_sector_area(phi):
    """
    Test the elliptical sector area against a numerical integration,
    including sectors that straddle a polar angle of 0, pi, or 2 pi.
    """
    eps = 0.4
    geometry = EllipseGeometry(100.0, 100.0, 20.0, eps, 0.3 * u.rad, step=0.1)
    geometry._initialize_sector_geometry(phi)
    sma1, sma2 = geometry._bounding_ellipses()
    phi1, phi2 = geometry._polar_angle_sector_limits()

    def radius_squared(angle, sma):
        q = 1.0 - eps
        return (sma * q)**2 / ((q * np.cos(angle))**2 + np.sin(angle)**2)

    expected = quad(lambda angle: 0.5 * (radius_squared(angle, sma2)
                                         - radius_squared(angle, sma1)),
                    phi1, phi2)[0]
    assert_allclose(geometry.sector_area, expected, rtol=1e-10)


def test_reset_sma():
    geometry = EllipseGeometry(0.0, 0.0, 100.0, 0.0, 0.0 * u.rad, step=0.2,
                               linear_growth=False)
    sma, step = geometry._reset_sma(0.2)
    assert_allclose(sma, 83.33, atol=0.01)
    assert_allclose(step, -0.1666, atol=0.001)

    geometry = EllipseGeometry(0.0, 0.0, 100.0, 0.0, 0.0 * u.rad, step=20.0,
                               linear_growth=True)
    sma, step = geometry._reset_sma(20.0)
    assert_allclose(sma, 80.0, atol=0.01)
    assert_allclose(step, -20.0, atol=0.01)


def test_update_sma():
    geometry = EllipseGeometry(0.0, 0.0, 100.0, 0.0, 0.0 * u.rad, step=0.2,
                               linear_growth=False)
    sma = geometry._update_sma(0.2)
    assert_allclose(sma, 120.0, atol=0.01)

    geometry = EllipseGeometry(0.0, 0.0, 100.0, 0.0, 0.0 * u.rad, step=20.0,
                               linear_growth=True)
    sma = geometry._update_sma(20.0)
    assert_allclose(sma, 120.0, atol=0.01)


def test_polar_angle_sector_limits():
    geometry = EllipseGeometry(0.0, 0.0, 100.0, 0.3, (np.pi / 4) * u.rad,
                               step=0.2, linear_growth=False)
    geometry._initialize_sector_geometry(np.pi / 3)
    phi1, phi2 = geometry._polar_angle_sector_limits()
    assert_allclose(phi1, 1.022198, atol=0.0001)
    assert_allclose(phi2, 1.072198, atol=0.0001)


def test_bounding_ellipses():
    geometry = EllipseGeometry(0.0, 0.0, 100.0, 0.3, (np.pi / 4) * u.rad,
                               step=0.2, linear_growth=False)
    sma1, sma2 = geometry._bounding_ellipses()
    assert_allclose((sma1, sma2), (90.0, 110.0), atol=0.01)


def test_radius():
    geometry = EllipseGeometry(0.0, 0.0, 100.0, 0.3, (np.pi / 4) * u.rad,
                               step=0.2, linear_growth=False)
    r = geometry.radius(0.0)
    assert_allclose(r, 100.0, atol=0.01)

    r = geometry.radius(np.pi / 2)
    assert_allclose(r, 70.0, atol=0.01)


@pytest.mark.parametrize(('x0', 'y0'), [(3.0, 50.0), (50.0, 96.0),
                                        (6.0, 4.0), (96.0, 97.0)])
def test_find_center_near_edge(x0, y0):
    """
    Test find_center for a galaxy near the image edges, where the
    centerer box extends beyond the image.
    """
    data = make_test_image(nx=100, ny=100, x0=x0, y0=y0, sma=10.0,
                           noise=1.0, seed=0)
    geometry = EllipseGeometry(x0, y0, 10.0, 0.2, 0.1 * u.rad)
    geometry.find_center(data, verbose=False)
    assert abs(geometry.x0 - x0) <= 1
    assert abs(geometry.y0 - y0) <= 1


def test_find_center_nonfinite():
    """
    Test that nonfinite values are excluded by find_center.
    """
    data = make_test_image(nx=100, ny=100, x0=50, y0=50, sma=10.0,
                           noise=1.0, seed=0)
    geometry = EllipseGeometry(52.0, 48.0, 10.0, 0.2, 0.1 * u.rad)
    geometry.find_center(data, verbose=False)

    data[49, 48] = np.nan
    data[51, 53] = np.inf
    geometry_nan = EllipseGeometry(52.0, 48.0, 10.0, 0.2, 0.1 * u.rad)
    geometry_nan.find_center(data, verbose=False)
    assert geometry_nan.x0 == geometry.x0
    assert geometry_nan.y0 == geometry.y0


def test_find_center_masked():
    """
    Test that masked pixels are excluded by find_center.
    """
    data = make_test_image(nx=100, ny=100, x0=50, y0=50, sma=10.0,
                           noise=1.0, seed=0)
    geometry = EllipseGeometry(52.0, 48.0, 10.0, 0.2, 0.1 * u.rad)
    geometry.find_center(data, verbose=False)

    data[46:49, 54:57] = 1.0e4
    geometry_bad = EllipseGeometry(52.0, 48.0, 10.0, 0.2, 0.1 * u.rad)
    geometry_bad.find_center(data, verbose=False)
    assert (geometry_bad.x0, geometry_bad.y0) != (geometry.x0, geometry.y0)

    mask = np.zeros(data.shape, dtype=bool)
    mask[46:49, 54:57] = True
    geometry_masked = EllipseGeometry(52.0, 48.0, 10.0, 0.2, 0.1 * u.rad)
    geometry_masked.find_center(np.ma.masked_array(data, mask=mask),
                                verbose=False)
    assert geometry_masked.x0 == geometry.x0
    assert geometry_masked.y0 == geometry.y0


@pytest.mark.parametrize('pa', [0.3, -0.5])
def test_to_polar_scalar_vectorized(pa):
    """
    Test that the scalar and vectorized polar coordinates agree in all
    quadrants, including at the ellipse center.
    """
    geometry = EllipseGeometry(10.0, 20.0, 5.0, 0.2, pa * u.rad)
    x = np.array([10.0, 15.0, 5.0, 5.0, 15.0, 10.0, 7.0])
    y = np.array([20.0, 23.0, 23.0, 17.0, 17.0, 25.0, 20.0])
    radius, angle = geometry.to_polar(x, y)
    for i, (xi, yi) in enumerate(zip(x, y, strict=True)):
        radius_i, angle_i = geometry.to_polar(float(xi), float(yi))
        assert_allclose(radius_i, radius[0, i])
        assert_allclose(angle_i, angle[0, i], atol=1e-12)

    assert radius[0, 0] == 0
    dx = x[1:] - 10.0
    dy = y[1:] - 20.0
    assert_allclose(radius[0, 1:], np.hypot(dx, dy))
    expected = np.mod(np.arctan2(dy, dx) - pa, 2 * np.pi)
    assert_allclose(angle[0, 1:], expected, atol=1e-12)
    assert np.all((angle >= 0) & (angle < 2 * np.pi))


def test_find_center_outside_image():
    """
    Test that find_center starts from the image center if the initial
    center is outside of the image.
    """
    data = make_test_image(nx=100, ny=100, x0=50, y0=50, sma=10.0,
                           noise=1.0, seed=0)
    geometry = EllipseGeometry(-20.0, 150.0, 10.0, 0.2, 0.1 * u.rad)
    geometry.find_center(data, verbose=False)
    assert abs(geometry.x0 - 50) <= 1
    assert abs(geometry.y0 - 50) <= 1


def test_find_center_below_threshold():
    """
    Test that the center is not changed if the figure of merit is below
    the threshold.
    """
    data = make_test_image(nx=100, ny=100, x0=50, y0=50, sma=10.0,
                           noise=1.0, seed=0)
    geometry = EllipseGeometry(52.0, 48.0, 10.0, 0.2, 0.1 * u.rad)
    with log.log_to_list() as log_list:
        geometry.find_center(data, threshold=1e10)
    assert geometry.x0 == 52.0
    assert geometry.y0 == 48.0
    assert 'below the threshold' in log_list[0].msg


@pytest.mark.parametrize(('astep', 'linear_growth'),
                         [(5.0, True), (2.5, False)])
def test_bounding_ellipses_large_step(astep, linear_growth):
    """
    Test that the inner bounding ellipse is not negative if the step is
    larger than twice the semimajor axis length.
    """
    geometry = EllipseGeometry(10.0, 10.0, 2.0, 0.2, 0.0 * u.rad, step=astep,
                               linear_growth=linear_growth)
    sma1, sma2 = geometry._bounding_ellipses()
    assert sma1 == 0.0
    assert_allclose(sma2, 4.5)


@pytest.mark.parametrize(('x0', 'y0'), [(50.0, 50.0), (50.4, 50.4),
                                        (49.6, 50.3), (50.3, 49.6),
                                        (49.7, 49.8)])
def test_find_center_nearest_pixel(x0, y0):
    """
    Test that find_center returns the pixel nearest to the center of
    a symmetric source.
    """
    yy, xx = np.mgrid[0:101, 0:101]
    data = Gaussian2D(100.0, x0, y0, 6.0, 5.0, theta=0.4)(xx, yy)
    geometry = EllipseGeometry(53.0, 47.0, 10.0, 0.2, 0.1 * u.rad)
    geometry.find_center(data, verbose=False)
    assert geometry.x0 == round(x0)
    assert geometry.y0 == round(y0)


def test_find_center_noise():
    """
    Test that the centerer does not find an object in an image of pure
    noise with the default threshold, as in the STSDAS ellipse task.
    """
    data = make_noise_image((101, 101), distribution='gaussian', mean=0.0,
                            stddev=1.0, seed=0)
    geometry = EllipseGeometry(50.0, 50.0, 10.0, 0.2, 0.1 * u.rad)
    with log.log_to_list() as log_list:
        geometry.find_center(data)
    assert geometry.x0 == 50.0
    assert geometry.y0 == 50.0
    assert 'below the threshold' in log_list[0].msg


def test_find_center_no_valid_data():
    """
    Test that the positions whose circular samples have no valid data
    points are skipped, so that the center is not changed for an image
    without valid data.
    """
    data = np.full((101, 101), np.nan)
    geometry = EllipseGeometry(52.0, 48.0, 10.0, 0.2, 0.1 * u.rad)
    geometry.find_center(data, verbose=False)
    assert geometry.x0 == 52.0
    assert geometry.y0 == 48.0


def test_find_center_flipped():
    """
    Test that find_center has no preferred direction.

    The center found in an image that is flipped along both axes must
    be the flipped center, to within 1 pixel. The circular samples
    start at a fixed polar angle, so they are not symmetric under the
    flip, and the noise changes the figure of merit slightly.
    """
    yy, xx = np.mgrid[0:101, 0:101]
    data = Gaussian2D(100.0, 50.0, 50.0, 6.0, 5.0, theta=0.4)(xx, yy)
    data += make_noise_image(data.shape, distribution='gaussian', mean=0.0,
                             stddev=20.0, seed=9)
    geometry = EllipseGeometry(50.0, 50.0, 10.0, 0.2, 0.1 * u.rad)
    geometry.find_center(data, verbose=False)
    assert abs(geometry.x0 - 50) <= 2
    assert abs(geometry.y0 - 50) <= 2

    geometry_flipped = EllipseGeometry(50.0, 50.0, 10.0, 0.2, 0.1 * u.rad)
    geometry_flipped.find_center(data[::-1, ::-1], verbose=False)
    assert abs(geometry_flipped.x0 - (100 - geometry.x0)) <= 1
    assert abs(geometry_flipped.y0 - (100 - geometry.y0)) <= 1


def test_find_center_flat_image():
    """
    Test that the center is not changed, without any warnings, if the
    image has no scatter.
    """
    geometry = EllipseGeometry(52.0, 48.0, 10.0, 0.2, 0.1 * u.rad)
    geometry.find_center(np.ones((101, 101)), verbose=False)
    assert geometry.x0 == 52.0
    assert geometry.y0 == 48.0


def test_find_center_negative_threshold():
    """
    Test that the center is not changed for a negative threshold if no
    position has a figure of merit.
    """
    geometry = EllipseGeometry(52.0, 48.0, 10.0, 0.2, 0.1 * u.rad)
    geometry.find_center(np.ones((101, 101)), threshold=-1.0, verbose=False)
    assert geometry.x0 == 52.0
    assert geometry.y0 == 48.0


@pytest.mark.parametrize('value', [30.0 * u.deg,
                                   (np.pi / 6) * u.rad,
                                   Angle(30.0, 'deg')])
def test_geometry_orientation_inputs(value):
    geometry = EllipseGeometry(50.0, 50.0, 10.0, 0.3, value)
    assert isinstance(geometry.orientation, u.Quantity)
    assert geometry.orientation.unit == u.deg
    assert_allclose(geometry.orientation.value, 30.0)
    assert_allclose(geometry._orientation_rad, np.pi / 6)


def test_geometry_orientation_bad_unit():
    with pytest.raises(u.UnitsError):
        EllipseGeometry(50.0, 50.0, 10.0, 0.3, 30.0 * u.m)


def _deprecation_records(func):
    """
    Call ``func`` and return its recorded deprecation warnings and its
    result.
    """
    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter('always')
        result = func()
    records = [record for record in records
               if issubclass(record.category, PhotutilsDeprecationWarning)]
    return records, result


# Numbers without units, as Python and NumPy floats and integers
NUMBERS = [0.5, 1, np.float32(0.5), np.float64(0.5), np.int64(1)]
NUMBER_IDS = ['float', 'int', 'float32', 'float64', 'int64']


class TestOrientationUnits:
    """
    Test that the ``orientation`` of a geometry requires units and that
    a number without units is accepted only on the legacy paths.
    """

    number_message = 'orientation as a number without units'

    @pytest.mark.parametrize('value', [30.0 * u.deg, Angle(30.0, 'deg')])
    def test_keyword_quantity(self, value):
        records, geometry = _deprecation_records(
            lambda: EllipseGeometry(50.0, 50.0, 10.0, 0.3,
                                    orientation=value))
        assert records == []
        assert_allclose(geometry._orientation_rad, np.pi / 6)

    @pytest.mark.parametrize('value', [30.0 * u.deg, Angle(30.0, 'deg')])
    def test_positional_quantity(self, value):
        records, geometry = _deprecation_records(
            lambda: EllipseGeometry(50.0, 50.0, 10.0, 0.3, value))
        assert records == []
        assert_allclose(geometry._orientation_rad, np.pi / 6)

    @pytest.mark.parametrize('value', NUMBERS, ids=NUMBER_IDS)
    def test_keyword_number(self, value):
        match = 'orientation must be an angular Quantity, e.g., 30 \\* u.deg'
        with pytest.raises(TypeError, match=match):
            EllipseGeometry(50.0, 50.0, 10.0, 0.3, orientation=value)

    @pytest.mark.parametrize('value', NUMBERS, ids=NUMBER_IDS)
    def test_positional_number(self, value):
        records, geometry = _deprecation_records(
            lambda: EllipseGeometry(50.0, 50.0, 10.0, 0.3, value))
        assert len(records) == 1
        message = str(records[0].message)
        assert self.number_message in message
        assert 'deprecated in version 3.2' in message
        assert 'removed in version 4.0' in message
        assert 'interpreted as radians' in message
        assert records[0].filename == __file__
        expected = EllipseGeometry(50.0, 50.0, 10.0, 0.3, value * u.rad)
        assert geometry._orientation_rad == expected._orientation_rad

    def test_positional_number_with_positional_defaults(self):
        records, geometry = _deprecation_records(
            lambda: EllipseGeometry(50.0, 50.0, 10.0, 0.3, 0.5, 0.2))
        assert len(records) == 2
        messages = [str(record.message) for record in records]
        assert sum(self.number_message in msg for msg in messages) == 1
        assert sum("'step' positionally" in msg for msg in messages) == 1
        assert all(record.filename == __file__ for record in records)
        assert geometry._orientation_rad == 0.5
        assert geometry.step == 0.2

    @pytest.mark.parametrize('value', [0.5, np.pi / 7])
    def test_positional_number_bit_identical(self, value):
        with pytest.warns(PhotutilsDeprecationWarning,
                          match=self.number_message):
            geometry = EllipseGeometry(50.0, 50.0, 10.0, 0.3, value)
        expected = EllipseGeometry(50.0, 50.0, 10.0, 0.3, value * u.rad)
        assert geometry._orientation_rad == expected._orientation_rad
        assert geometry._orientation_rad == value

    def test_pa_keyword_number(self):
        records, geometry = _deprecation_records(
            lambda: EllipseGeometry(50.0, 50.0, 10.0, 0.3, pa=0.5))
        assert len(records) == 1
        assert "'pa' was deprecated" in str(records[0].message)
        assert records[0].filename == __file__
        assert geometry._orientation_rad == 0.5

    def test_pa_keyword_quantity(self):
        records, geometry = _deprecation_records(
            lambda: EllipseGeometry(50.0, 50.0, 10.0, 0.3,
                                    pa=30.0 * u.deg))
        assert len(records) == 1
        assert "'pa' was deprecated" in str(records[0].message)
        assert records[0].filename == __file__
        assert_allclose(geometry._orientation_rad, np.pi / 6)

    @pytest.mark.parametrize(('args', 'kwargs'),
                             [((0.5 * u.rad,), {'pa': 0.5}),
                              ((), {'orientation': 0.5 * u.rad, 'pa': 0.5})])
    def test_pa_and_orientation(self, args, kwargs):
        match = 'cannot specify both "pa" and "orientation"'
        with (pytest.warns(PhotutilsDeprecationWarning, match="'pa'"),
              pytest.raises(TypeError, match=match)):
            EllipseGeometry(50.0, 50.0, 10.0, 0.3, *args, **kwargs)

    def test_non_angular_unit(self):
        with pytest.raises(u.UnitsError):
            EllipseGeometry(50.0, 50.0, 10.0, 0.3, orientation=30.0 * u.m)
        with (pytest.warns(PhotutilsDeprecationWarning, match="'pa'"),
              pytest.raises(u.UnitsError)):
            EllipseGeometry(50.0, 50.0, 10.0, 0.3, pa=30.0 * u.m)

    def test_set_orientation_quantity(self):
        geometry = EllipseGeometry(50.0, 50.0, 10.0, 0.3, 0.5 * u.rad)
        geometry.orientation = 30.0 * u.deg
        assert_allclose(geometry._orientation_rad, np.pi / 6)
        geometry.orientation = Angle(0.25, 'rad')
        assert geometry._orientation_rad == 0.25

    @pytest.mark.parametrize('value', NUMBERS, ids=NUMBER_IDS)
    def test_set_orientation_number(self, value):
        geometry = EllipseGeometry(50.0, 50.0, 10.0, 0.3, 0.5 * u.rad)
        match = 'orientation must be an angular Quantity, e.g., 30 \\* u.deg'
        with pytest.raises(TypeError, match=match):
            geometry.orientation = value
        assert geometry._orientation_rad == 0.5

    def test_set_orientation_non_angular_unit(self):
        geometry = EllipseGeometry(50.0, 50.0, 10.0, 0.3, 0.5 * u.rad)
        with pytest.raises(u.UnitsError):
            geometry.orientation = 30.0 * u.m
        assert geometry._orientation_rad == 0.5

    def test_pa_attribute(self):
        geometry = EllipseGeometry(50.0, 50.0, 10.0, 0.3, 0.5 * u.rad)
        with pytest.warns(PhotutilsDeprecationWarning,
                          match="'pa' attribute") as record:
            geometry.pa = 0.7
        assert len(record) == 1
        assert record[0].filename == __file__
        assert geometry._orientation_rad == 0.7
        with pytest.warns(PhotutilsDeprecationWarning,
                          match="'pa' attribute") as record:
            value = geometry.pa
        assert len(record) == 1
        assert record[0].filename == __file__
        assert isinstance(value, float)
        assert value == 0.7

    def test_orientation_attribute(self):
        geometry = EllipseGeometry(50.0, 50.0, 10.0, 0.3, 30.0 * u.deg)
        assert isinstance(geometry.orientation, u.Quantity)
        assert geometry.orientation.unit == u.deg
        assert_allclose(geometry.orientation.value, 30.0)

    def test_copy_and_pickle(self):
        geometry = EllipseGeometry(50.0, 50.0, 10.0, 0.3, 0.5 * u.rad)

        def clones():
            return (copy.copy(geometry), copy.deepcopy(geometry),
                    pickle.loads(  # noqa: S301
                        pickle.dumps(geometry)))  # nosec B301

        records, result = _deprecation_records(clones)
        assert records == []
        for clone in result:
            assert clone._orientation_rad == 0.5


def test_geometry_deprecated_names():
    with pytest.warns(PhotutilsDeprecationWarning, match="'eps'"):
        geometry = EllipseGeometry(50.0, 50.0, 10.0, eps=0.3,
                                   orientation=0.5 * u.rad)
    with pytest.warns(PhotutilsDeprecationWarning, match="'pa'"):
        geometry = EllipseGeometry(50.0, 50.0, 10.0, 0.3, pa=30.0 * u.deg)
    assert_allclose(geometry._orientation_rad, np.pi / 6)
    with pytest.warns(PhotutilsDeprecationWarning, match="'pa'"):
        geometry = EllipseGeometry(50.0, 50.0, 10.0, 0.3, pa=0.5)
    with pytest.warns(PhotutilsDeprecationWarning, match="'pa'"):
        assert geometry.pa == 0.5
    with pytest.warns(PhotutilsDeprecationWarning, match="'pa'"):
        geometry.pa = 0.7
    assert_allclose(geometry.orientation.to_value(u.rad), 0.7)
    with pytest.warns(PhotutilsDeprecationWarning, match="'eps'"):
        geometry.eps = 0.1
    assert geometry.ellipticity == 0.1
    with pytest.warns(PhotutilsDeprecationWarning, match="'astep'"):
        geometry.astep = 0.2
    assert geometry.step == 0.2


def test_geometry_copy_and_pickle():
    geometry = EllipseGeometry(50.0, 50.0, 10.0, 0.3, 0.5 * u.rad,
                               fix_center=True)
    for clone in (copy.deepcopy(geometry),
                  pickle.loads(  # noqa: S301
                      pickle.dumps(geometry))):  # nosec B301
        assert clone.ellipticity == 0.3
        assert clone._orientation_rad == 0.5
        assert clone.step == geometry.step


@pytest.mark.parametrize(('old', 'new', 'value'),
                         [('astep', 'step', 0.2),
                          ('fix_pa', 'fix_orientation', True),
                          ('fix_eps', 'fix_ellipticity', True)])
def test_geometry_deprecated_keywords(old, new, value):
    with pytest.warns(PhotutilsDeprecationWarning, match=f"'{old}'"):
        geometry = EllipseGeometry(50.0, 50.0, 10.0, 0.3, 0.5 * u.rad,
                                   **{old: value})
    expected = EllipseGeometry(50.0, 50.0, 10.0, 0.3, 0.5 * u.rad,
                               **{new: value})
    assert geometry.step == expected.step
    assert_array_equal(geometry._fix, expected._fix)


@pytest.mark.parametrize(('name', 'args'),
                         [('bounding_ellipses', ()),
                          ('polar_angle_sector_limits', ()),
                          ('update_sma', (0.2,)),
                          ('reset_sma', (0.2,)),
                          ('initialize_sector_geometry', (0.1,))])
def test_geometry_deprecated_methods(name, args):
    # The sector initialization changes the geometry, so use two
    # identical instances.
    geometry = EllipseGeometry(50.0, 50.0, 10.0, 0.3, 0.5 * u.rad)
    geometry2 = copy.deepcopy(geometry)
    geometry._initialize_sector_geometry(0.1)
    geometry2._initialize_sector_geometry(0.1)
    expected = getattr(geometry, f'_{name}')(*args)
    match = f"The 'EllipseGeometry.{name}' method was deprecated"
    with pytest.warns(PhotutilsDeprecationWarning, match=match):
        result = getattr(geometry2, name)(*args)
    assert_allclose(result, expected)


@pytest.mark.parametrize('name', ['sector_angular_width',
                                  'initial_polar_angle',
                                  'initial_polar_radius'])
def test_geometry_deprecated_sector_attributes(name):
    geometry = EllipseGeometry(50.0, 50.0, 10.0, 0.3, 0.5 * u.rad)
    match = (f"The '{name}' attribute was deprecated in version 3.2. It "
             'is internal to the fitter and has no replacement. It will '
             'be removed in version 4.0.')
    with pytest.warns(PhotutilsDeprecationWarning, match=match) as record:
        value = getattr(geometry, name)
    assert record[0].filename == __file__
    assert value == getattr(geometry, f'_{name}')
    with pytest.warns(PhotutilsDeprecationWarning, match=match) as record:
        setattr(geometry, name, 0.1)
    assert record[0].filename == __file__
    assert getattr(geometry, f'_{name}') == 0.1
    assert "'_" not in getattr(EllipseGeometry, name).__doc__


@pytest.mark.parametrize('name', ['sector_angular_width',
                                  'initial_polar_angle',
                                  'initial_polar_radius'])
def test_geometry_deprecated_sector_attributes_zero_sma(name):
    """
    Test the deprecated sector attributes of a geometry with sma = 0.

    The sector attributes are not defined for the central pixel. A
    lookup emits the same single deprecation warning, at the caller
    line, as for sma > 0, and then raises an `AttributeError` that
    names the public attribute, not the private one.
    """
    geometry = EllipseGeometry(50.0, 50.0, 0.0, 0.3, 0.5 * u.rad)
    match = f"The '{name}' attribute was deprecated in version 3.2"
    with (pytest.warns(PhotutilsDeprecationWarning, match=match) as record,
          pytest.raises(AttributeError,
                        match=f"has no attribute '{name}'") as excinfo):
        getattr(geometry, name)
    assert f"'_{name}'" not in str(excinfo.value)
    assert len(record) == 1
    assert record[0].filename == __file__


def test_geometry_fix_flags():
    geometry = EllipseGeometry(50.0, 50.0, 10.0, 0.3, 0.5 * u.rad,
                               fix_orientation=True)
    assert geometry.fix_center is False
    assert geometry.fix_orientation is True
    assert geometry.fix_ellipticity is False
    match = ("The 'fix' attribute was deprecated in version 3.2. Use "
             "'fix_center', 'fix_orientation', and 'fix_ellipticity' "
             'instead. It will be removed in version 4.0.')
    with pytest.warns(PhotutilsDeprecationWarning, match=match) as record:
        assert_array_equal(geometry.fix, [False, False, True, False])
    assert record[0].filename == __file__
    with pytest.warns(PhotutilsDeprecationWarning, match=match) as record:
        geometry.fix = np.array([True, True, False, True])
    assert record[0].filename == __file__
    assert "'_fix'" not in EllipseGeometry.fix.__doc__
    assert geometry.fix_center is True
    assert geometry.fix_orientation is False
    assert geometry.fix_ellipticity is True
    geometry._fix = np.array([True, False, False, False])
    assert geometry.fix_center is False


def test_to_aperture():
    geometry = EllipseGeometry(x0=75.0, y0=60.0, sma=20.0, ellipticity=0.25,
                               orientation=30.0 * u.deg)
    aperture = geometry.to_aperture()
    assert isinstance(aperture, EllipticalAperture)
    assert_allclose(aperture.positions, (75.0, 60.0))
    assert aperture.a == 20.0
    assert aperture.b == 15.0
    assert_allclose(aperture.theta.to_value(u.deg), 30.0)


@pytest.mark.parametrize(('sma', 'ellipticity'), [(0.0, 0.2), (20.0, 1.0)])
def test_to_aperture_invalid(sma, ellipticity):
    geometry = EllipseGeometry(x0=75.0, y0=60.0, sma=sma,
                               ellipticity=ellipticity,
                               orientation=30.0 * u.deg)
    match = 'positive semimajor axis length and an ellipticity smaller'
    with pytest.raises(ValueError, match=match):
        geometry.to_aperture()


@pytest.mark.parametrize('sma', [-1.0, np.nan, np.inf])
def test_invalid_sma(sma):
    match = 'sma must be finite and nonnegative'
    with pytest.raises(ValueError, match=match):
        EllipseGeometry(x0=75.0, y0=60.0, sma=sma, ellipticity=0.2,
                        orientation=30.0 * u.deg)
