# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the geometry module.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from photutils.isophote.geometry import EllipseGeometry


@pytest.mark.parametrize(('astep', 'linear_growth'),
                         [(0.2, False), (20.0, True)])
def test_geometry(astep, linear_growth):
    geometry = EllipseGeometry(255.0, 255.0, 100.0, 0.4, np.pi / 2, astep,
                               linear_growth)

    sma1, sma2 = geometry.bounding_ellipses()
    assert_allclose((sma1, sma2), (90.0, 110.0), atol=0.01)

    # using an arbitrary angle of 0.5 rad. This is to avoid a polar
    # vector that sits on top of one of the ellipse's axis.
    vertex_x, vertex_y = geometry.initialize_sector_geometry(0.6)
    assert_allclose(geometry.sector_angular_width, 0.0571, atol=0.01)
    assert_allclose(geometry.sector_area, 63.83, atol=0.01)
    assert_allclose(vertex_x, [215.4, 206.6, 213.5, 204.3], atol=0.1)
    assert_allclose(vertex_y, [316.1, 329.7, 312.5, 325.3], atol=0.1)


def test_to_polar():
    # trivial case of a circle centered in (0.0, 0.0)
    geometry = EllipseGeometry(0.0, 0.0, 100.0, 0.0, 0.0, 0.2,
                               linear_growth=False)

    r, p = geometry.to_polar(100.0, 0.0)
    assert_allclose(r, 100.0, atol=0.1)
    assert_allclose(p, 0.0, atol=0.0001)

    r, p = geometry.to_polar(0.0, 100.0)
    assert_allclose(r, 100.0, atol=0.1)
    assert_allclose(p, np.pi / 2.0, atol=0.0001)

    # vector with length 100.0 at 45 deg angle
    r, p = geometry.to_polar(70.71, 70.71)
    assert_allclose(r, 100.0, atol=0.1)
    assert_allclose(p, np.pi / 4.0, atol=0.0001)

    # position angle tilted 45 deg from X axis
    geometry = EllipseGeometry(0.0, 0.0, 100.0, 0.0, np.pi / 4.0, 0.2,
                               linear_growth=False)

    r, p = geometry.to_polar(100.0, 0.0)
    assert_allclose(r, 100.0, atol=0.1)
    assert_allclose(p, np.pi * 7.0 / 4.0, atol=0.0001)

    r, p = geometry.to_polar(0.0, 100.0)
    assert_allclose(r, 100.0, atol=0.1)
    assert_allclose(p, np.pi / 4.0, atol=0.0001)

    # vector with length 100.0 at 45 deg angle
    r, p = geometry.to_polar(70.71, 70.71)
    assert_allclose(r, 100.0, atol=0.1)
    assert_allclose(p, np.pi * 2.0, atol=0.0001)


def test_area():
    # circle with center at origin
    geometry = EllipseGeometry(0.0, 0.0, 100.0, 0.0, 0.0, 0.2,
                               linear_growth=False)

    # sector at 45 deg on circle
    vertex_x, vertex_y = geometry.initialize_sector_geometry(
        45.0 / 180.0 * np.pi)
    assert_allclose(vertex_x, [65.21, 79.70, 62.03, 75.81], atol=0.01)
    assert_allclose(vertex_y, [62.03, 75.81, 65.21, 79.70], atol=0.01)

    # sector at 0 deg on circle
    vertex_x, vertex_y = geometry.initialize_sector_geometry(0)
    assert_allclose(vertex_x, [89.97, 109.97, 89.97, 109.96], atol=0.01)
    assert_allclose(vertex_y, [-2.25, -2.75, 2.25, 2.75], atol=0.01)


def test_area2():
    # circle with center at 100.0, 100.0
    geometry = EllipseGeometry(100.0, 100.0, 100.0, 0.0, 0.0, 0.2,
                               linear_growth=False)

    # sector at 45 deg on circle
    vertex_x, vertex_y = geometry.initialize_sector_geometry(
        45.0 / 180.0 * np.pi)
    assert_allclose(vertex_x, [165.21, 179.70, 162.03, 175.81], atol=0.01)
    assert_allclose(vertex_y, [162.03, 175.81, 165.21, 179.70], atol=0.01)

    # sector at 225 deg on circle
    vertex_x, vertex_y = geometry.initialize_sector_geometry(
        225.0 / 180.0 * np.pi)
    assert_allclose(vertex_x, [34.79, 20.30, 37.97, 24.19], atol=0.01)
    assert_allclose(vertex_y, [37.97, 24.19, 34.79, 20.30], atol=0.01)


def test_reset_sma():
    geometry = EllipseGeometry(0.0, 0.0, 100.0, 0.0, 0.0, 0.2,
                               linear_growth=False)
    sma, step = geometry.reset_sma(0.2)
    assert_allclose(sma, 83.33, atol=0.01)
    assert_allclose(step, -0.1666, atol=0.001)

    geometry = EllipseGeometry(0.0, 0.0, 100.0, 0.0, 0.0, 20.0,
                               linear_growth=True)
    sma, step = geometry.reset_sma(20.0)
    assert_allclose(sma, 80.0, atol=0.01)
    assert_allclose(step, -20.0, atol=0.01)


def test_update_sma():
    geometry = EllipseGeometry(0.0, 0.0, 100.0, 0.0, 0.0, 0.2,
                               linear_growth=False)
    sma = geometry.update_sma(0.2)
    assert_allclose(sma, 120.0, atol=0.01)

    geometry = EllipseGeometry(0.0, 0.0, 100.0, 0.0, 0.0, 20.0,
                               linear_growth=True)
    sma = geometry.update_sma(20.0)
    assert_allclose(sma, 120.0, atol=0.01)


def test_polar_angle_sector_limits():
    geometry = EllipseGeometry(0.0, 0.0, 100.0, 0.3, np.pi / 4, 0.2,
                               linear_growth=False)
    geometry.initialize_sector_geometry(np.pi / 3)
    phi1, phi2 = geometry.polar_angle_sector_limits()
    assert_allclose(phi1, 1.022198, atol=0.0001)
    assert_allclose(phi2, 1.072198, atol=0.0001)


def test_bounding_ellipses():
    geometry = EllipseGeometry(0.0, 0.0, 100.0, 0.3, np.pi / 4, 0.2,
                               linear_growth=False)
    sma1, sma2 = geometry.bounding_ellipses()
    assert_allclose((sma1, sma2), (90.0, 110.0), atol=0.01)


def test_radius():
    geometry = EllipseGeometry(0.0, 0.0, 100.0, 0.3, np.pi / 4, 0.2,
                               linear_growth=False)
    r = geometry.radius(0.0)
    assert_allclose(r, 100.0, atol=0.01)

    r = geometry.radius(np.pi / 2)
    assert_allclose(r, 70.0, atol=0.01)
