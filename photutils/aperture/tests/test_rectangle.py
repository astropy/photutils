# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the rectangle module.
"""

import astropy.units as u
import numpy as np
import pytest
from astropy.coordinates import Angle, SkyCoord
from astropy.tests.helper import assert_quantity_allclose
from numpy.testing import assert_allclose

from photutils.aperture import PolygonAperture, SkyPolygonAperture
from photutils.aperture.rectangle import (RectangularAnnulus,
                                          RectangularAperture,
                                          RectangularMaskMixin,
                                          SkyRectangularAnnulus,
                                          SkyRectangularAperture)
from photutils.aperture.tests.test_aperture_common import BaseTestAperture
from photutils.utils._optional_deps import HAS_MATPLOTLIB

POSITIONS = [(10, 20), (30, 40), (50, 60), (70, 80)]
RA, DEC = np.transpose(POSITIONS)
SKYCOORD = SkyCoord(ra=RA, dec=DEC, unit='deg')
UNIT = u.arcsec
RADII = (0.0, -1.0, -np.inf)


class TestRectangularAperture(BaseTestAperture):
    aperture = RectangularAperture(POSITIONS, w=10.0, h=5.0, theta=np.pi / 2.0)

    @staticmethod
    @pytest.mark.parametrize('radius', RADII)
    def test_invalid_params(radius):
        match = "'w' must be a positive scalar"
        with pytest.raises(ValueError, match=match):
            RectangularAperture(POSITIONS, w=radius, h=5.0, theta=np.pi / 2.0)

        match = "'h' must be a positive scalar"
        with pytest.raises(ValueError, match=match):
            RectangularAperture(POSITIONS, w=10.0, h=radius, theta=np.pi / 2.0)

    def test_copy_eq(self):
        aper = self.aperture.copy()
        assert aper == self.aperture
        aper.w = 20.0
        assert aper != self.aperture

    def test_theta(self):
        assert isinstance(self.aperture.theta, u.Quantity)
        assert self.aperture.theta.unit == u.rad

    @pytest.mark.skipif(not HAS_MATPLOTLIB, reason='matplotlib is required')
    def test_to_patch_nonscalar(self):
        """
        Test that _to_patch returns a list for non-scalar apertures.
        """
        patches = self.aperture._to_patch()
        assert isinstance(patches, list)


class TestRectangularAnnulus(BaseTestAperture):
    aperture = RectangularAnnulus(POSITIONS, w_in=10.0, w_out=20.0, h_out=17,
                                  theta=np.pi / 3)

    @staticmethod
    @pytest.mark.parametrize('radius', RADII)
    def test_invalid_params(radius):
        match = "'w_in' must be a positive scalar"
        with pytest.raises(ValueError, match=match):
            RectangularAnnulus(POSITIONS, w_in=radius, w_out=20.0, h_out=17,
                               theta=np.pi / 3)

        match = "'w_out' must be greater than 'w_in'"
        with pytest.raises(ValueError, match=match):
            RectangularAnnulus(POSITIONS, w_in=10.0, w_out=radius, h_out=17,
                               theta=np.pi / 3)

        match = "'h_out' must be a positive scalar"
        with pytest.raises(ValueError, match=match):
            RectangularAnnulus(POSITIONS, w_in=10.0, w_out=20.0, h_out=radius,
                               theta=np.pi / 3)

        match = "'h_in' must be a positive scalar"
        with pytest.raises(ValueError, match=match):
            RectangularAnnulus(POSITIONS, w_in=10.0, w_out=20.0, h_out=17,
                               h_in=radius, theta=np.pi / 3)

    def test_copy_eq(self):
        aper = self.aperture.copy()
        assert aper == self.aperture
        aper.w_in = 2.0
        assert aper != self.aperture

    def test_theta(self):
        assert isinstance(self.aperture.theta, u.Quantity)
        assert self.aperture.theta.unit == u.rad

    def test_h_in_greater_than_h_out(self):
        """
        Test that a ValueError is raised when h_in >= h_out.
        """
        match = "'h_out' must be greater than 'h_in'"
        with pytest.raises(ValueError, match=match):
            RectangularAnnulus(POSITIONS, w_in=10.0, w_out=20.0, h_out=5.0,
                               h_in=8.0, theta=np.pi / 3)

    @pytest.mark.skipif(not HAS_MATPLOTLIB, reason='matplotlib is required')
    def test_to_patch_nonscalar(self):
        """
        Test that _to_patch returns a list for non-scalar apertures.
        """
        patches = self.aperture._to_patch()
        assert isinstance(patches, list)


class TestSkyRectangularAperture(BaseTestAperture):
    aperture = SkyRectangularAperture(SKYCOORD, w=10.0 * UNIT, h=5.0 * UNIT,
                                      theta=30 * u.deg)

    @staticmethod
    @pytest.mark.parametrize('radius', RADII)
    def test_invalid_params(radius):
        match = "'w' must be greater than zero"
        with pytest.raises(ValueError, match=match):
            SkyRectangularAperture(SKYCOORD, w=radius * UNIT, h=5.0 * UNIT,
                                   theta=30 * u.deg)

        match = "'h' must be greater than zero"
        with pytest.raises(ValueError, match=match):
            SkyRectangularAperture(SKYCOORD, w=10.0 * UNIT, h=radius * UNIT,
                                   theta=30 * u.deg)

    def test_copy_eq(self):
        aper = self.aperture.copy()
        assert aper == self.aperture
        aper.w = 20.0 * UNIT
        assert aper != self.aperture


class TestSkyRectangularAnnulus(BaseTestAperture):
    aperture = SkyRectangularAnnulus(SKYCOORD, w_in=10.0 * UNIT,
                                     w_out=20.0 * UNIT, h_out=17.0 * UNIT,
                                     theta=60 * u.deg)

    @staticmethod
    @pytest.mark.parametrize('radius', RADII)
    def test_invalid_params(radius):
        match = "'w_in' must be greater than zero"
        with pytest.raises(ValueError, match=match):
            SkyRectangularAnnulus(SKYCOORD, w_in=radius * UNIT,
                                  w_out=20.0 * UNIT, h_out=17.0 * UNIT,
                                  theta=60 * u.deg)

        match = "'w_out' must be greater than 'w_in'"
        with pytest.raises(ValueError, match=match):
            SkyRectangularAnnulus(SKYCOORD, w_in=10.0 * UNIT,
                                  w_out=radius * UNIT, h_out=17.0 * UNIT,
                                  theta=60 * u.deg)

        match = "'h_out' must be greater than zero"
        with pytest.raises(ValueError, match=match):
            SkyRectangularAnnulus(SKYCOORD, w_in=10.0 * UNIT,
                                  w_out=20.0 * UNIT, h_out=radius * UNIT,
                                  theta=60 * u.deg)

        match = "'h_in' must be greater than zero"
        with pytest.raises(ValueError, match=match):
            SkyRectangularAnnulus(SKYCOORD, w_in=10.0 * UNIT,
                                  w_out=20.0 * UNIT, h_out=17.0 * UNIT,
                                  h_in=radius * UNIT, theta=60 * u.deg)

    def test_copy_eq(self):
        aper = self.aperture.copy()
        assert aper == self.aperture
        aper.w_in = 2.0 * UNIT
        assert aper != self.aperture

    def test_h_in_greater_than_h_out(self):
        """
        Test that a ValueError is raised when h_in >= h_out.
        """
        match = "'h_out' must be greater than 'h_in'"
        with pytest.raises(ValueError, match=match):
            SkyRectangularAnnulus(SKYCOORD, w_in=10.0 * UNIT,
                                  w_out=20.0 * UNIT, h_out=5.0 * UNIT,
                                  h_in=8.0 * UNIT, theta=60 * u.deg)


def test_rectangle_theta_quantity():
    aper1 = RectangularAperture(POSITIONS, w=10.0, h=5.0, theta=np.pi / 2.0)
    theta = u.Quantity(90 * u.deg)
    aper2 = RectangularAperture(POSITIONS, w=10.0, h=5.0, theta=theta)
    theta = Angle(90 * u.deg)
    aper3 = RectangularAperture(POSITIONS, w=10.0, h=5.0, theta=theta)

    assert_quantity_allclose(aper1.theta, aper2.theta)
    assert_quantity_allclose(aper1.theta, aper3.theta)


def test_rectangle_annulus_theta_quantity():
    aper1 = RectangularAnnulus(POSITIONS, w_in=10.0, w_out=20.0, h_out=17,
                               theta=np.pi / 3)
    theta = u.Quantity(60 * u.deg)
    aper2 = RectangularAnnulus(POSITIONS, w_in=10.0, w_out=20.0, h_out=17,
                               theta=theta)
    theta = Angle(60 * u.deg)
    aper3 = RectangularAnnulus(POSITIONS, w_in=10.0, w_out=20.0, h_out=17,
                               theta=theta)

    assert_quantity_allclose(aper1.theta, aper2.theta)
    assert_quantity_allclose(aper1.theta, aper3.theta)


@pytest.mark.parametrize(('method', 'subpixels'),
                         [('exact', 5), ('center', 5), ('subpixel', 8)])
def test_deprecated_rectangular_mask_mixin(method, subpixels):
    """
    Test that the deprecated RectangularMaskMixin.to_mask method works
    with the current _translate_mask_method signature and matches the
    non-deprecated to_mask result.
    """
    aper = RectangularAperture((10.0, 10.0), w=4.0, h=2.0, theta=0.5)
    mask = RectangularMaskMixin.to_mask(aper, method=method,
                                        subpixels=subpixels)
    expected = aper.to_mask(method=method, subpixels=subpixels)
    assert_quantity_allclose(mask.data, expected.data)


def test_rectangular_aperture_to_polygon():
    aper = RectangularAperture((10.0, 20.0), w=4.0, h=2.0)
    poly = aper.to_polygon()
    assert isinstance(poly, PolygonAperture)
    assert poly.n_vertices == 4
    # Area is exact (rectangle is itself a polygon)
    assert_allclose(poly.area, aper.area)


@pytest.mark.parametrize('theta_deg', [0.0, 30.0, 45.0, 90.0, 137.0])
def test_rectangular_aperture_to_polygon_rotation(theta_deg):
    aper = RectangularAperture((10.0, 20.0), w=4.0, h=2.0,
                               theta=np.deg2rad(theta_deg))
    poly = aper.to_polygon()
    assert poly.n_vertices == 4
    assert_allclose(poly.area, aper.area, rtol=1e-12)
    # Bounding-box extents should match the rotated rectangle's.
    expected_x_extent = (0.5 * abs(aper.w * np.cos(aper.theta.value))
                         + 0.5 * abs(aper.h * np.sin(aper.theta.value)))
    expected_y_extent = (0.5 * abs(aper.w * np.sin(aper.theta.value))
                         + 0.5 * abs(aper.h * np.cos(aper.theta.value)))
    assert_allclose(poly._xy_extents[0], expected_x_extent, rtol=1e-12)
    assert_allclose(poly._xy_extents[1], expected_y_extent, rtol=1e-12)


def test_rectangular_aperture_to_polygon_multi_position():
    aper = RectangularAperture([(10.0, 20.0), (30.0, 40.0)],
                               w=4.0, h=3.0, theta=0.5)
    poly = aper.to_polygon()
    assert poly.vertices.shape == (2, 4, 2)


def test_sky_rectangular_aperture_to_polygon():
    pos = SkyCoord(ra=10.0, dec=30.0, unit='deg')
    aper = SkyRectangularAperture(pos, w=4.0 * u.arcsec, h=2.0 * u.arcsec)
    poly = aper.to_polygon()
    assert isinstance(poly, SkyPolygonAperture)
    assert poly.n_vertices == 4
    assert poly.vertices.shape == (4,)
    # At theta=0, the width (4) is along +lat (d_lat) and the height (2)
    # is along +lon (d_lon).
    offsets = poly.vertex_offsets.to_value(u.arcsec)
    assert_allclose(np.sort(np.abs(offsets[:, 0])), [1.0, 1.0, 1.0, 1.0])
    assert_allclose(np.sort(np.abs(offsets[:, 1])), [2.0, 2.0, 2.0, 2.0])


@pytest.mark.parametrize('theta_deg', [0.0, 30.0, 45.0, 90.0, 137.0])
def test_sky_rectangular_aperture_to_polygon_rotation(theta_deg):
    pos = SkyCoord(ra=10.0, dec=30.0, unit='deg')
    aper = SkyRectangularAperture(pos, w=4.0 * u.arcsec, h=2.0 * u.arcsec,
                                  theta=theta_deg * u.deg)
    poly = aper.to_polygon()
    assert poly.n_vertices == 4
    # The perimeter and vertex distances are rotation-invariant.
    assert_allclose(poly.perimeter.to_value(u.arcsec), 2 * (4.0 + 2.0))
    radii = np.hypot(*poly.vertex_offsets.to_value(u.arcsec).T)
    assert_allclose(radii, np.hypot(1.0, 2.0))


def test_sky_rectangular_aperture_to_polygon_multi_position():
    pos = SkyCoord(ra=[10.0, 20.0], dec=[30.0, 40.0], unit='deg')
    aper = SkyRectangularAperture(pos, w=4.0 * u.arcsec, h=3.0 * u.arcsec,
                                  theta=0.5 * u.rad)
    poly = aper.to_polygon()
    assert poly.vertices.shape == (2, 4)


@pytest.mark.parametrize('theta_deg', [0.0, 30.0, 45.0, 90.0, 137.0])
def test_sky_rectangular_aperture_to_polygon_wcs_round_trip(
        theta_deg, tan_wcs):
    """
    Test that converting a SkyRectangularAperture to a polygon and then
    to pixels matches converting to pixels and then to a polygon for all
    rotation angles.

    The two pixel polygons share the same area and bounding-box extents.
    """
    pos = SkyCoord(ra=10.0, dec=30.0, unit='deg')
    aper = SkyRectangularAperture(pos, w=4.0 * u.arcsec, h=2.0 * u.arcsec,
                                  theta=theta_deg * u.deg)
    poly_from_sky = aper.to_polygon().to_pixel(tan_wcs)
    poly_from_pix = aper.to_pixel(tan_wcs).to_polygon()
    assert_allclose(poly_from_sky.area, poly_from_pix.area, rtol=1e-6)
    assert_allclose(poly_from_sky._xy_extents, poly_from_pix._xy_extents,
                    rtol=1e-6)


@pytest.mark.parametrize('theta_deg', [0.0, 30.0, 45.0, 90.0, 137.0])
def test_rectangular_aperture_to_polygon_to_sky_round_trip(theta_deg, tan_wcs):
    """
    Test that converting a RectangularAperture to a polygon and then to
    sky matches converting to sky and then to a polygon for all rotation
    angles.
    """
    # The bounding-box orientation can differ at the ~1e-5 level because
    # the sky shape parameters from ``to_sky`` are derived from a local
    # linear (SVD) approximation, while the polygon ``to_sky`` maps
    # each vertex exactly, so only rotation-invariant quantities are
    # compared.
    aper = RectangularAperture((50.0, 50.0), w=4.0, h=2.0,
                               theta=np.deg2rad(theta_deg))
    sky_from_poly = aper.to_polygon().to_sky(tan_wcs)
    sky_from_aper = aper.to_sky(tan_wcs).to_polygon()
    assert_allclose(sky_from_poly.positions.ra.deg,
                    sky_from_aper.positions.ra.deg)
    assert_allclose(sky_from_poly.positions.dec.deg,
                    sky_from_aper.positions.dec.deg)
    assert_allclose(sky_from_poly.perimeter.to_value(u.arcsec),
                    sky_from_aper.perimeter.to_value(u.arcsec), rtol=1e-6)
    assert_allclose(sky_from_poly.to_pixel(tan_wcs).area,
                    sky_from_aper.to_pixel(tan_wcs).area, rtol=1e-6)
