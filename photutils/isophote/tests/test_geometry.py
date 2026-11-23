# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the geometry module.
"""

import copy
import pickle

import astropy.units as u
import numpy as np
import pytest
from astropy import log
from astropy.coordinates import Angle
from astropy.modeling.models import Gaussian2D
from numpy.testing import assert_allclose, assert_array_equal
from scipy.integrate import quad

from photutils.datasets import make_noise_image
from photutils.isophote.geometry import IN_MASK, OUT_MASK, EllipseGeometry
from photutils.isophote.tests.make_test_data import make_test_image
from photutils.utils.exceptions import PhotutilsDeprecationWarning


@pytest.mark.parametrize(('astep', 'linear_growth'),
                         [(0.2, False), (20.0, True)])
def test_geometry(astep, linear_growth):
    geometry = EllipseGeometry(255.0, 255.0, 100.0, 0.4, np.pi / 2,
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


def test_to_polar_integer_input():
    """
    Test that integer coordinates and centers do not truncate the
    polar radii.
    """
    geometry = EllipseGeometry(0, 0, 10.0, 0.3, 0.5)
    x = np.array([[1, 2], [3, 4]])
    y = np.array([[1, 1], [2, 2]])
    radius, angle = geometry.to_polar(x, y)
    expected = EllipseGeometry(0.0, 0.0, 10.0, 0.3, 0.5).to_polar(
        x.astype(float), y.astype(float))
    assert_allclose(radius, expected[0])
    assert_allclose(angle, expected[1])


def test_to_polar():
    # Trivial case of a circle centered in (0.0, 0.0)
    geometry = EllipseGeometry(0.0, 0.0, 100.0, 0.0, 0.0, step=0.2,
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
    geometry = EllipseGeometry(0.0, 0.0, 100.0, 0.0, np.pi / 4.0, step=0.2,
                               linear_growth=False)

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
    geometry = EllipseGeometry(0.0, 0.0, 100.0, 0.0, 0.0, step=0.2,
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
    geometry = EllipseGeometry(100.0, 100.0, 100.0, 0.0, 0.0, step=0.2,
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
    geometry = EllipseGeometry(100.0, 100.0, 20.0, eps, 0.3, step=0.1)
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
    geometry = EllipseGeometry(0.0, 0.0, 100.0, 0.0, 0.0, step=0.2,
                               linear_growth=False)
    sma, step = geometry._reset_sma(0.2)
    assert_allclose(sma, 83.33, atol=0.01)
    assert_allclose(step, -0.1666, atol=0.001)

    geometry = EllipseGeometry(0.0, 0.0, 100.0, 0.0, 0.0, step=20.0,
                               linear_growth=True)
    sma, step = geometry._reset_sma(20.0)
    assert_allclose(sma, 80.0, atol=0.01)
    assert_allclose(step, -20.0, atol=0.01)


def test_update_sma():
    geometry = EllipseGeometry(0.0, 0.0, 100.0, 0.0, 0.0, step=0.2,
                               linear_growth=False)
    sma = geometry._update_sma(0.2)
    assert_allclose(sma, 120.0, atol=0.01)

    geometry = EllipseGeometry(0.0, 0.0, 100.0, 0.0, 0.0, step=20.0,
                               linear_growth=True)
    sma = geometry._update_sma(20.0)
    assert_allclose(sma, 120.0, atol=0.01)


def test_polar_angle_sector_limits():
    geometry = EllipseGeometry(0.0, 0.0, 100.0, 0.3, np.pi / 4, step=0.2,
                               linear_growth=False)
    geometry._initialize_sector_geometry(np.pi / 3)
    phi1, phi2 = geometry._polar_angle_sector_limits()
    assert_allclose(phi1, 1.022198, atol=0.0001)
    assert_allclose(phi2, 1.072198, atol=0.0001)


def test_bounding_ellipses():
    geometry = EllipseGeometry(0.0, 0.0, 100.0, 0.3, np.pi / 4, step=0.2,
                               linear_growth=False)
    sma1, sma2 = geometry._bounding_ellipses()
    assert_allclose((sma1, sma2), (90.0, 110.0), atol=0.01)


def test_radius():
    geometry = EllipseGeometry(0.0, 0.0, 100.0, 0.3, np.pi / 4, step=0.2,
                               linear_growth=False)
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
    geometry = EllipseGeometry(x0, y0, 10.0, 0.2, 0.1)
    geometry.find_center(data, verbose=False)
    assert abs(geometry.x0 - x0) <= 1
    assert abs(geometry.y0 - y0) <= 1


def test_find_center_nonfinite():
    """
    Test that nonfinite values are excluded by find_center.
    """
    data = make_test_image(nx=100, ny=100, x0=50, y0=50, sma=10.0,
                           noise=1.0, seed=0)
    geometry = EllipseGeometry(52.0, 48.0, 10.0, 0.2, 0.1)
    geometry.find_center(data, verbose=False)

    data[49, 48] = np.nan
    data[51, 53] = np.inf
    geometry_nan = EllipseGeometry(52.0, 48.0, 10.0, 0.2, 0.1)
    geometry_nan.find_center(data, verbose=False)
    assert geometry_nan.x0 == geometry.x0
    assert geometry_nan.y0 == geometry.y0


def test_find_center_masked():
    """
    Test that masked pixels are excluded by find_center.
    """
    data = make_test_image(nx=100, ny=100, x0=50, y0=50, sma=10.0,
                           noise=1.0, seed=0)
    geometry = EllipseGeometry(52.0, 48.0, 10.0, 0.2, 0.1)
    geometry.find_center(data, verbose=False)

    data[49:52, 49:52] = 1.0e9
    geometry_bad = EllipseGeometry(52.0, 48.0, 10.0, 0.2, 0.1)
    geometry_bad.find_center(data, verbose=False)
    assert (geometry_bad.x0, geometry_bad.y0) != (geometry.x0, geometry.y0)

    mask = np.zeros(data.shape, dtype=bool)
    mask[49:52, 49:52] = True
    geometry_masked = EllipseGeometry(52.0, 48.0, 10.0, 0.2, 0.1)
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
    geometry = EllipseGeometry(10.0, 20.0, 5.0, 0.2, pa)
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
    geometry = EllipseGeometry(-20.0, 150.0, 10.0, 0.2, 0.1)
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
    geometry = EllipseGeometry(52.0, 48.0, 10.0, 0.2, 0.1)
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
    geometry = EllipseGeometry(10.0, 10.0, 2.0, 0.2, 0.0, step=astep,
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
    geometry = EllipseGeometry(53.0, 47.0, 10.0, 0.2, 0.1)
    geometry.find_center(data, verbose=False)
    assert geometry.x0 == round(x0)
    assert geometry.y0 == round(y0)


def test_centerer_masks():
    """
    Test that the centerer masks select a disk and a ring that are
    centered on the central pixel of the box.
    """
    assert IN_MASK.shape == (17, 17)
    assert OUT_MASK.shape == (17, 17)
    yy, xx = np.mgrid[-8:9, -8:9]
    radius = np.hypot(xx, yy)
    assert np.count_nonzero(~IN_MASK) == 37
    assert np.max(radius[~IN_MASK]) <= 3.5
    assert np.count_nonzero(~OUT_MASK) == 48
    assert np.min(radius[~OUT_MASK]) >= 7.5
    assert np.max(radius[~OUT_MASK]) < 8.5
    for mask in (IN_MASK, OUT_MASK):
        assert_array_equal(mask, mask[::-1, ::-1])
        assert_array_equal(mask, mask.T)


def test_find_center_flipped():
    """
    Test that find_center has no preferred direction.

    The center found in an image that is flipped along both axes must
    be the flipped center. The noise makes the figure of merit depend
    on every pixel of the centerer masks.
    """
    yy, xx = np.mgrid[0:101, 0:101]
    data = Gaussian2D(100.0, 50.0, 50.0, 6.0, 5.0, theta=0.4)(xx, yy)
    data += make_noise_image(data.shape, distribution='gaussian', mean=0.0,
                             stddev=20.0, seed=9)
    geometry = EllipseGeometry(50.0, 50.0, 10.0, 0.2, 0.1)
    geometry.find_center(data, verbose=False)
    assert abs(geometry.x0 - 50) <= 2
    assert abs(geometry.y0 - 50) <= 2

    geometry_flipped = EllipseGeometry(50.0, 50.0, 10.0, 0.2, 0.1)
    geometry_flipped.find_center(data[::-1, ::-1], verbose=False)
    assert geometry_flipped.x0 == 100 - geometry.x0
    assert geometry_flipped.y0 == 100 - geometry.y0


def test_find_center_flat_image():
    """
    Test that the center is not changed, without any warnings, if the
    image has no scatter.
    """
    geometry = EllipseGeometry(52.0, 48.0, 10.0, 0.2, 0.1)
    geometry.find_center(np.ones((101, 101)), verbose=False)
    assert geometry.x0 == 52.0
    assert geometry.y0 == 48.0


def test_find_center_negative_threshold():
    """
    Test that the center is not changed for a negative threshold if no
    position has a figure of merit.
    """
    geometry = EllipseGeometry(52.0, 48.0, 10.0, 0.2, 0.1)
    geometry.find_center(np.ones((101, 101)), threshold=-1.0, verbose=False)
    assert geometry.x0 == 52.0
    assert geometry.y0 == 48.0


@pytest.mark.parametrize('value', [np.pi / 6, 30.0 * u.deg,
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


def test_geometry_deprecated_names():
    with pytest.warns(PhotutilsDeprecationWarning, match="'eps'"):
        geometry = EllipseGeometry(50.0, 50.0, 10.0, eps=0.3,
                                   orientation=0.5)
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
    geometry = EllipseGeometry(50.0, 50.0, 10.0, 0.3, 0.5,
                               fix_center=True)
    for clone in (copy.deepcopy(geometry),
                  pickle.loads(pickle.dumps(geometry))):  # noqa: S301
        assert clone.ellipticity == 0.3
        assert clone._orientation_rad == 0.5
        assert clone.step == geometry.step


@pytest.mark.parametrize(('old', 'new', 'value'),
                         [('astep', 'step', 0.2),
                          ('fix_pa', 'fix_orientation', True),
                          ('fix_eps', 'fix_ellipticity', True)])
def test_geometry_deprecated_keywords(old, new, value):
    with pytest.warns(PhotutilsDeprecationWarning, match=f"'{old}'"):
        geometry = EllipseGeometry(50.0, 50.0, 10.0, 0.3, 0.5,
                                   **{old: value})
    expected = EllipseGeometry(50.0, 50.0, 10.0, 0.3, 0.5, **{new: value})
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
    geometry = EllipseGeometry(50.0, 50.0, 10.0, 0.3, 0.5)
    geometry2 = copy.deepcopy(geometry)
    geometry._initialize_sector_geometry(0.1)
    geometry2._initialize_sector_geometry(0.1)
    expected = getattr(geometry, f'_{name}')(*args)
    with pytest.warns(PhotutilsDeprecationWarning, match='version 3.2'):
        result = getattr(geometry2, name)(*args)
    assert_allclose(result, expected)


@pytest.mark.parametrize('name', ['sector_angular_width',
                                  'initial_polar_angle',
                                  'initial_polar_radius'])
def test_geometry_deprecated_sector_attributes(name):
    geometry = EllipseGeometry(50.0, 50.0, 10.0, 0.3, 0.5)
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
    line, as for sma > 0, and then raises an `AttributeError`.
    """
    geometry = EllipseGeometry(50.0, 50.0, 0.0, 0.3, 0.5)
    match = f"The '{name}' attribute was deprecated in version 3.2"
    with (pytest.warns(PhotutilsDeprecationWarning, match=match) as record,
          pytest.raises(AttributeError, match=name)):
        getattr(geometry, name)
    assert len(record) == 1
    assert record[0].filename == __file__


def test_geometry_fix_flags():
    geometry = EllipseGeometry(50.0, 50.0, 10.0, 0.3, 0.5,
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
