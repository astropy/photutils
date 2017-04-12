from __future__ import (absolute_import, division, print_function, unicode_literals)

import pytest

import numpy as np

from photutils.isophote.geometry import Geometry, normalize_angle


def _check_geometry(geometry):

    sma1, sma2 = geometry.bounding_ellipses()

    assert (sma1, sma2) == pytest.approx((90.0, 110.0), abs=0.01)

    # using an arbitrary angle of 0.5 rad. This is
    # to avoid a polar vector that sits on top of
    # one of the ellipse's axis.
    vertex_x, vertex_y = geometry.initialize_sector_geometry(0.6)

    assert geometry.sector_angular_width == pytest.approx(0.0571, abs=0.01)
    assert geometry.sector_area == pytest.approx(63.83, abs=0.01)

    assert vertex_x[0] == pytest.approx(215.4, abs=0.1)
    assert vertex_x[1] == pytest.approx(206.6, abs=0.1)
    assert vertex_x[2] == pytest.approx(213.5, abs=0.1)
    assert vertex_x[3] == pytest.approx(204.3, abs=0.1)

    assert vertex_y[0] == pytest.approx(316.1, abs=0.1)
    assert vertex_y[1] == pytest.approx(329.7, abs=0.1)
    assert vertex_y[2] == pytest.approx(312.5, abs=0.1)
    assert vertex_y[3] == pytest.approx(325.3, abs=0.1)


def test_ellipse():

    # Geometrical steps
    geometry = Geometry(255., 255., 100., 0.4, np.pi/2, 0.2, False)

    _check_geometry(geometry)

    # Linear steps
    geometry = Geometry(255., 255., 100., 0.4, np.pi/2, 20., True)

    _check_geometry(geometry)


def test_to_polar():
    # trivial case of a circle centered in (0.,0.)
    geometry = Geometry(0., 0., 100., 0.0, 0., 0.2, False)

    r, p = geometry.to_polar(100., 0.)
    assert (r, p) == pytest.approx((100., 0.), abs=(0.1, 0.0001))

    r, p = geometry.to_polar(0., 100.)
    assert (r, p) == pytest.approx((100., np.pi/2.), abs=(0.1, 0.0001))

    # vector with length 100. at 45 deg angle
    r, p = geometry.to_polar(70.71, 70.71)
    # these have to be tested separately. For some unknown reason, using a combined
    # assert statement as above raises an TypeError: unorderable types: tuple() < int()
    # assert (r, p) == pytest.approx((100., np.pi/4.), abs=(0.1, 0.0001))
    assert r == pytest.approx(100., abs=0.1)
    assert p == pytest.approx(np.pi/4., abs=0.0001)

    # position angle tilted 45 deg from X axis
    geometry = Geometry(0., 0., 100., 0.0, np.pi/4., 0.2, False)

    r, p = geometry.to_polar(100., 0.)
    assert (r, p) == pytest.approx((100., np.pi*7./4), abs=(0.1, 0.0001))

    r, p = geometry.to_polar(0., 100.)
    assert (r, p) == pytest.approx((100., np.pi/4.), abs=(0.1, 0.0001))

    # vector with length 100. at 45 deg angle
    r, p = geometry.to_polar(70.71, 70.71)
    # same error as above
    # assert (r, p) == pytest.approx((100., np.pi*2.), abs=(0.1, 0.0001))
    assert r == pytest.approx(100., abs=0.1)
    assert p == pytest.approx(np.pi*2., abs=0.0001)


def test_area():
    # circle with center at origin
    geometry = Geometry(0., 0., 100., 0.0, 0., 0.2, False)

    # sector at 45 deg on circle
    vertex_x, vertex_y = geometry.initialize_sector_geometry(45./180.*np.pi)

    assert vertex_x[0] == pytest.approx(65.21, abs=0.01)
    assert vertex_x[1] == pytest.approx(79.70, abs=0.01)
    assert vertex_x[2] == pytest.approx(62.03, abs=0.01)
    assert vertex_x[3] == pytest.approx(75.81, abs=0.01)

    assert vertex_y[0] == pytest.approx(62.03, abs=0.01)
    assert vertex_y[1] == pytest.approx(75.81, abs=0.01)
    assert vertex_y[2] == pytest.approx(65.21, abs=0.01)
    assert vertex_y[3] == pytest.approx(79.70, abs=0.01)

    # sector at 0 deg on circle
    vertex_x, vertex_y = geometry.initialize_sector_geometry(0)

    assert vertex_x[0] == pytest.approx(89.97,  abs=0.01)
    assert vertex_x[1] == pytest.approx(109.97, abs=0.01)
    assert vertex_x[2] == pytest.approx(89.97,  abs=0.01)
    assert vertex_x[3] == pytest.approx(109.96, abs=0.01)

    assert vertex_y[0] == pytest.approx(-2.25, abs=0.01)
    assert vertex_y[1] == pytest.approx(-2.75, abs=0.01)
    assert vertex_y[2] == pytest.approx(2.25 , abs=0.01)
    assert vertex_y[3] == pytest.approx(2.75 , abs=0.01)


def test_area2():
    # circle with center at 100.,100.
    geometry = Geometry(100., 100., 100., 0.0, 0., 0.2, False)

    # sector at 45 deg on circle
    vertex_x, vertex_y = geometry.initialize_sector_geometry(45./180.*np.pi)

    assert vertex_x[0] == pytest.approx(165.21, abs=0.01)
    assert vertex_x[1] == pytest.approx(179.70, abs=0.01)
    assert vertex_x[2] == pytest.approx(162.03, abs=0.01)
    assert vertex_x[3] == pytest.approx(175.81, abs=0.01)

    assert vertex_y[0] == pytest.approx(162.03, abs=0.01)
    assert vertex_y[1] == pytest.approx(175.81, abs=0.01)
    assert vertex_y[2] == pytest.approx(165.21, abs=0.01)
    assert vertex_y[3] == pytest.approx(179.70, abs=0.01)

    # sector at 225 deg on circle
    vertex_x, vertex_y = geometry.initialize_sector_geometry(225./180.*np.pi)

    assert vertex_x[0] == pytest.approx(34.791,abs=0.01)
    assert vertex_x[1] == pytest.approx(20.30, abs=0.01)
    assert vertex_x[2] == pytest.approx(37.97, abs=0.01)
    assert vertex_x[3] == pytest.approx(24.19, abs=0.01)

    assert vertex_y[0] == pytest.approx(37.97, abs=0.01)
    assert vertex_y[1] == pytest.approx(24.19, abs=0.01)
    assert vertex_y[2] == pytest.approx(34.79, abs=0.01)
    assert vertex_y[3] == pytest.approx(20.30, abs=0.01)


def test_normalize_angle():
    PI = np.pi

    angle = normalize_angle(PI*10 + PI/5)
    assert angle == pytest.approx(PI/5, abs=0.0001)

    angle = normalize_angle(PI*1.3)
    assert angle == pytest.approx(PI*0.3, abs=0.0001)

    angle = normalize_angle(-PI*10 + PI/5)
    assert angle == pytest.approx(PI/5, abs=0.0001)

    angle = normalize_angle(-PI*1.3)
    assert angle == pytest.approx(PI*0.7, abs=0.0001)

    angle = normalize_angle(-PI*10.3)
    assert angle == pytest.approx(PI*0.7, abs=0.0001)


def test_reset_sma():
    geometry = Geometry(0., 0., 100., 0.0, 0., 0.2, False)

    sma, step = geometry.reset_sma(0.2)
    assert sma == pytest.approx(83.33, abs=0.01)
    assert step == pytest.approx(-0.1666, abs=0.001)

    geometry = Geometry(0., 0., 100., 0.0, 0., 20., True)

    sma, step = geometry.reset_sma(20.)
    assert sma == pytest.approx(80.0, abs=0.01)
    assert step == pytest.approx(-20.0, abs=0.01)


def test_update_sma():
    geometry = Geometry(0., 0., 100., 0.0, 0., 0.2, False)
    sma = geometry.update_sma(0.2)
    assert sma == pytest.approx(120.0, abs=0.01)

    geometry = Geometry(0., 0., 100., 0.0, 0., 20., True)
    sma = geometry.update_sma(20.)
    assert sma == pytest.approx(120.0, abs=0.01)


def test_polar_angle_sector_limits():
    geometry = Geometry(0., 0., 100., 0.3, np.pi/4, 0.2, False)
    geometry.initialize_sector_geometry(np.pi/3)

    phi1, phi2 = geometry.polar_angle_sector_limits()

    assert phi1 == pytest.approx(1.022198, abs=0.0001)
    assert phi2 == pytest.approx(1.072198, abs=0.0001)


def test_bounding_ellipses():
    geometry = Geometry(0., 0., 100., 0.3, np.pi/4, 0.2, False)

    sma1, sma2 = geometry.bounding_ellipses()
    assert (sma1, sma2) == pytest.approx((90.0, 110.0), abs=0.01)


def test_radius():
    geometry = Geometry(0., 0., 100., 0.3, np.pi/4, 0.2, False)

    r = geometry.radius(0.0)
    assert r == pytest.approx(100.0, abs=0.01)

    r = geometry.radius(np.pi/2)
    assert r == pytest.approx(70.0, abs=0.01)





