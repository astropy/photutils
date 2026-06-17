# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Validation tests for the ``method='exact'`` mask mode of pixel
apertures.

When the data array is all ones, the photometric sum returned by an
aperture in exact mode must equal the analytic geometric area of the
aperture shape (up to floating-point error). These tests verify this
invariant for circular, elliptical, rectangular, and polygon apertures
(including their annulus counterparts), across a variety of shapes,
rotation angles, sub-pixel positions, and convex/non-convex polygons.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from photutils.aperture import (CircularAnnulus, CircularAperture,
                                EllipticalAnnulus, EllipticalAperture,
                                PolygonAperture, RectangularAnnulus,
                                RectangularAperture, aperture_photometry)


def _phot_sum(aperture, data=None):
    if data is None:
        shape = (101, 101)
        data = np.ones(shape, dtype=float)
    table = aperture_photometry(data, aperture, method='exact')
    return float(table['aperture_sum'][0])


@pytest.mark.parametrize('center', [(50.0, 50.0), (50.3, 50.7),
                                    (49.5, 50.5), (12.123, 87.987)])
@pytest.mark.parametrize('r', [1.0, 3.5, 7.25])
def test_circular_aperture_exact_area(center, r):
    aper = CircularAperture(center, r=r)
    assert_allclose(_phot_sum(aper), aper.area, rtol=1e-6, atol=1e-10)


@pytest.mark.parametrize('center', [(50.0, 50.0), (50.4, 50.6)])
@pytest.mark.parametrize(('r_in', 'r_out'), [(1.0, 3.0), (2.5, 4.75),
                                             (5.0, 8.0)])
def test_circular_annulus_exact_area(center, r_in, r_out):
    aper = CircularAnnulus(center, r_in=r_in, r_out=r_out)
    assert_allclose(_phot_sum(aper), aper.area, rtol=1e-6, atol=1e-10)


@pytest.mark.parametrize('center', [(50.0, 50.0), (50.3, 49.8)])
@pytest.mark.parametrize(('a', 'b'), [(5.0, 3.0), (3.5, 1.25),
                                      (8.0, 8.0)])
@pytest.mark.parametrize('theta_deg', [0.0, 17.5, 45.0, 90.0, 137.0])
def test_elliptical_aperture_exact_area(center, a, b, theta_deg):
    aper = EllipticalAperture(center, a=a, b=b,
                              theta=np.deg2rad(theta_deg))
    assert_allclose(_phot_sum(aper), aper.area, rtol=1e-6, atol=1e-10)


@pytest.mark.parametrize(('a_in', 'a_out', 'b_out'),
                         [(2.0, 5.0, 3.0), (1.0, 4.0, 4.0)])
@pytest.mark.parametrize('theta_deg', [0.0, 30.0, 65.0])
def test_elliptical_annulus_exact_area(a_in, a_out, b_out, theta_deg):
    aper = EllipticalAnnulus((50.0, 50.0), a_in=a_in, a_out=a_out,
                             b_out=b_out, theta=np.deg2rad(theta_deg))
    assert_allclose(_phot_sum(aper), aper.area, rtol=1e-6, atol=1e-10)


@pytest.mark.parametrize('center', [(50.0, 50.0), (50.5, 50.5),
                                    (50.123, 49.876)])
@pytest.mark.parametrize(('w', 'h'), [(4.0, 2.0), (5.5, 5.5),
                                      (10.0, 1.0)])
@pytest.mark.parametrize('theta_deg', [0.0, 22.5, 45.0, 90.0, 117.0])
def test_rectangular_aperture_exact_area(center, w, h, theta_deg):
    aper = RectangularAperture(center, w=w, h=h,
                               theta=np.deg2rad(theta_deg))
    assert_allclose(_phot_sum(aper), aper.area, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize(('w_in', 'w_out', 'h_out'),
                         [(2.0, 6.0, 4.0), (1.5, 5.5, 5.5)])
@pytest.mark.parametrize('theta_deg', [0.0, 33.0, 90.0])
def test_rectangular_annulus_exact_area(w_in, w_out, h_out, theta_deg):
    aper = RectangularAnnulus((50.0, 50.0), w_in=w_in, w_out=w_out,
                              h_out=h_out, theta=np.deg2rad(theta_deg))
    assert_allclose(_phot_sum(aper), aper.area, rtol=1e-12, atol=1e-12)


# Polygon shape library: (name, vertex_offsets) - each polygon is
# centered at the origin so it can be placed at arbitrary positions.
def _triangle():
    # Equilateral triangle with circumradius 4.
    angles = np.pi / 2 + np.arange(3) * 2 * np.pi / 3
    return np.column_stack([4.0 * np.cos(angles), 4.0 * np.sin(angles)])


def _hexagon():
    angles = np.arange(6) * np.pi / 3
    return np.column_stack([5.0 * np.cos(angles), 5.0 * np.sin(angles)])


def _l_shape():
    # Non-convex L-shape (centered).
    return np.array([[-2.0, -2.0], [2.0, -2.0], [2.0, 0.0],
                     [0.0, 0.0], [0.0, 2.0], [-2.0, 2.0]])


def _star_5():
    # Non-convex 5-pointed star.
    verts = []
    for i in range(10):
        r = 5.0 if i % 2 == 0 else 2.0
        a = np.pi / 2 + i * np.pi / 5
        verts.append((r * np.cos(a), r * np.sin(a)))
    return np.array(verts)


def _arrow():
    # Non-convex arrow shape.
    return np.array([[-3.0, -1.0], [1.0, -1.0], [1.0, -2.0],
                     [3.0, 0.0], [1.0, 2.0], [1.0, 1.0],
                     [-3.0, 1.0]])


def _rotate(verts, theta_deg):
    c = np.cos(np.deg2rad(theta_deg))
    s = np.sin(np.deg2rad(theta_deg))
    rot = np.array([[c, -s], [s, c]])
    return verts @ rot.T


POLY_SHAPES = {
    'triangle': _triangle(),
    'hexagon': _hexagon(),
    'l_shape': _l_shape(),
    'star_5': _star_5(),
    'arrow': _arrow(),
}


@pytest.mark.parametrize('shape_name', list(POLY_SHAPES))
@pytest.mark.parametrize('center', [(50.0, 50.0), (50.5, 50.5),
                                    (50.123, 49.876)])
@pytest.mark.parametrize('theta_deg', [0.0, 23.5, 60.0])
def test_polygon_aperture_exact_area(shape_name, center, theta_deg):
    offsets = _rotate(POLY_SHAPES[shape_name], theta_deg)
    aper = PolygonAperture(center, offsets)
    assert_allclose(_phot_sum(aper), aper.area, rtol=1e-12, atol=1e-12)


def test_polygon_aperture_exact_area_multiple_positions():
    """
    Test that the exact area is returned for multiple positions.

    The photometric sum should equal the area regardless of the
    position.
    """
    offsets = _hexagon()
    positions = [(30.0, 40.0), (50.5, 50.5), (70.123, 60.876)]
    aper = PolygonAperture(positions, offsets)
    shape = (101, 101)
    data = np.ones(shape, dtype=float)
    table = aperture_photometry(data, aper, method='exact')
    expected = aper.area
    assert_allclose(table['aperture_sum'], expected, rtol=1e-12,
                    atol=1e-12)
