from __future__ import (absolute_import, division, print_function, unicode_literals)

import unittest

import numpy as np

from ..geometry import Geometry, normalize_angle


class TestGeometry(unittest.TestCase):

    def _check_geometry(self, geometry):

        sma1, sma2 = geometry.bounding_ellipses()

        self.assertAlmostEqual(sma1, 90.0, 3)
        self.assertAlmostEqual(sma2, 110.0, 3)

        # using an arbitrary angle of 0.5 rad. This is
        # to avoid a polar vector that sits on top of
        # one of the ellipse's axis.
        vertex_x, vertex_y = geometry.initialize_sector_geometry(0.6)

        self.assertAlmostEqual(geometry.sector_angular_width, 0.0571, 2)
        self.assertAlmostEqual(geometry.sector_area, 63.83, 2)

        self.assertAlmostEqual(vertex_x[0], 215.4, 1)
        self.assertAlmostEqual(vertex_x[1], 206.6, 1)
        self.assertAlmostEqual(vertex_x[2], 213.5, 1)
        self.assertAlmostEqual(vertex_x[3], 204.3, 1)

        self.assertAlmostEqual(vertex_y[0], 316.1, 1)
        self.assertAlmostEqual(vertex_y[1], 329.7, 1)
        self.assertAlmostEqual(vertex_y[2], 312.5, 1)
        self.assertAlmostEqual(vertex_y[3], 325.3, 1)

    def test_ellipse(self):

        # Geometrical steps
        geometry = Geometry(255., 255., 100., 0.4, np.pi/2, 0.2, False)

        self._check_geometry(geometry)

        # Linear steps
        geometry = Geometry(255., 255., 100., 0.4, np.pi/2, 20., True)

        self._check_geometry(geometry)

    def test_to_polar(self):
        # trivial case of a circle centered in (0.,0.)
        geometry = Geometry(0., 0., 100., 0.0, 0., 0.2, False)

        r, p = geometry.to_polar(100., 0.)
        self.assertAlmostEqual(r, 100., 2)
        self.assertAlmostEqual(p, 0., 4)

        r, p = geometry.to_polar(0., 100.)
        self.assertAlmostEqual(r, 100., 2)
        self.assertAlmostEqual(p, np.pi/2., 4)

        # vector with length 100. at 45 deg angle
        r, p = geometry.to_polar(70.71, 70.71)
        self.assertAlmostEqual(r, 100., 2)
        self.assertAlmostEqual(p, np.pi/4., 4)

        # position angle tilted 45 deg from X axis
        geometry = Geometry(0., 0., 100., 0.0, np.pi/4., 0.2, False)

        r, p = geometry.to_polar(100., 0.)
        self.assertAlmostEqual(r, 100., 2)
        self.assertAlmostEqual(p, np.pi*7./4., 4)

        r, p = geometry.to_polar(0., 100.)
        self.assertAlmostEqual(r, 100., 2)
        self.assertAlmostEqual(p, np.pi/4., 4)

        # vector with length 100. at 45 deg angle
        r, p = geometry.to_polar(70.71, 70.71)
        self.assertAlmostEqual(r, 100., 2)
        self.assertAlmostEqual(p, np.pi*2., 4)

    def test_area(self):
        # circle with center at origin
        geometry = Geometry(0., 0., 100., 0.0, 0., 0.2, False)

        # sector at 45 deg on circle
        vertex_x, vertex_y = geometry.initialize_sector_geometry(45./180.*np.pi)

        self.assertAlmostEqual(vertex_x[0], 65.21, 2)
        self.assertAlmostEqual(vertex_x[1], 79.70, 2)
        self.assertAlmostEqual(vertex_x[2], 62.03, 2)
        self.assertAlmostEqual(vertex_x[3], 75.81, 2)

        self.assertAlmostEqual(vertex_y[0], 62.03, 2)
        self.assertAlmostEqual(vertex_y[1], 75.81, 2)
        self.assertAlmostEqual(vertex_y[2], 65.21, 2)
        self.assertAlmostEqual(vertex_y[3], 79.70, 2)

        # sector at 0 deg on circle
        vertex_x, vertex_y = geometry.initialize_sector_geometry(0)

        self.assertAlmostEqual(vertex_x[0], 89.97, 2)
        self.assertAlmostEqual(vertex_x[1], 109.97, 2)
        self.assertAlmostEqual(vertex_x[2], 89.97, 2)
        self.assertAlmostEqual(vertex_x[3], 109.97, 2)

        self.assertAlmostEqual(vertex_y[0], -2.25, 2)
        self.assertAlmostEqual(vertex_y[1], -2.75, 2)
        self.assertAlmostEqual(vertex_y[2], 2.25, 2)
        self.assertAlmostEqual(vertex_y[3], 2.75, 2)

    def test_area2(self):
        # circle with center at 100.,100.
        geometry = Geometry(100., 100., 100., 0.0, 0., 0.2, False)

        # sector at 45 deg on circle
        vertex_x, vertex_y = geometry.initialize_sector_geometry(45./180.*np.pi)

        self.assertAlmostEqual(vertex_x[0], 165.21, 2)
        self.assertAlmostEqual(vertex_x[1], 179.70, 2)
        self.assertAlmostEqual(vertex_x[2], 162.03, 2)
        self.assertAlmostEqual(vertex_x[3], 175.81, 2)

        self.assertAlmostEqual(vertex_y[0], 162.03, 2)
        self.assertAlmostEqual(vertex_y[1], 175.81, 2)
        self.assertAlmostEqual(vertex_y[2], 165.21, 2)
        self.assertAlmostEqual(vertex_y[3], 179.70, 2)

        # sector at 225 deg on circle
        vertex_x, vertex_y = geometry.initialize_sector_geometry(225./180.*np.pi)

        self.assertAlmostEqual(vertex_x[0], 34.79, 2)
        self.assertAlmostEqual(vertex_x[1], 20.30, 2)
        self.assertAlmostEqual(vertex_x[2], 37.97, 2)
        self.assertAlmostEqual(vertex_x[3], 24.19, 2)

        self.assertAlmostEqual(vertex_y[0], 37.97, 2)
        self.assertAlmostEqual(vertex_y[1], 24.19, 2)
        self.assertAlmostEqual(vertex_y[2], 34.79, 2)
        self.assertAlmostEqual(vertex_y[3], 20.30, 2)

    def test_normalize_angle(self):
        PI = np.pi

        angle = normalize_angle(PI*10 + PI/5)
        self.assertAlmostEqual(angle, PI/5, 4)

        angle = normalize_angle(PI*1.3)
        self.assertAlmostEqual(angle, PI*0.3, 4)

        angle = normalize_angle(-PI*10 + PI/5)
        self.assertAlmostEqual(angle, PI/5, 4)

        angle = normalize_angle(-PI*1.3)
        self.assertAlmostEqual(angle, PI*0.7, 4)

        angle = normalize_angle(-PI*10.3)
        self.assertAlmostEqual(angle, PI*0.7, 4)

    def test_reset_sma(self):
        geometry = Geometry(0., 0., 100., 0.0, 0., 0.2, False)

        sma, step = geometry.reset_sma(0.2)
        self.assertAlmostEqual(sma, 83.33, 2)
        self.assertAlmostEqual(step, -0.1666, 3)

        geometry = Geometry(0., 0., 100., 0.0, 0., 20., True)

        sma, step = geometry.reset_sma(20.)
        self.assertAlmostEqual(sma, 80.0, 2)
        self.assertAlmostEqual(step, -20.0, 2)

    def test_update_sma(self):
        geometry = Geometry(0., 0., 100., 0.0, 0., 0.2, False)

        sma = geometry.update_sma(0.2)
        self.assertAlmostEqual(sma, 120.0, 2)

        geometry = Geometry(0., 0., 100., 0.0, 0., 20., True)

        sma = geometry.update_sma(20.)
        self.assertAlmostEqual(sma, 120.0, 2)

    def test_polar_angle_sector_limits(self):
        geometry = Geometry(0., 0., 100., 0.3, np.pi/4, 0.2, False)
        geometry.initialize_sector_geometry(np.pi/3)

        phi1, phi2 = geometry.polar_angle_sector_limits()
        self.assertAlmostEqual(phi1, 1.022198, 4)
        self.assertAlmostEqual(phi2, 1.072198, 4)

    def test_bounding_ellipses(self):
        geometry = Geometry(0., 0., 100., 0.3, np.pi/4, 0.2, False)

        sma1, sma2 = geometry.bounding_ellipses()
        self.assertAlmostEqual(sma1, 90.0, 2)
        self.assertAlmostEqual(sma2, 110.0, 2)

    def test_radius(self):
        geometry = Geometry(0., 0., 100., 0.3, np.pi/4, 0.2, False)

        r = geometry.radius(0.0)
        self.assertAlmostEqual(r, 100.0, 2)

        r = geometry.radius(np.pi/2)
        self.assertAlmostEqual(r, 70., 2)





