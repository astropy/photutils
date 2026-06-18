# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the ellipse module.
"""

import astropy.units as u
import numpy as np
import pytest
from astropy.coordinates import Angle, SkyCoord
from astropy.tests.helper import assert_quantity_allclose
from astropy.wcs import WCS
from numpy.testing import assert_allclose

from photutils.aperture import PolygonAperture, SkyPolygonAperture
from photutils.aperture.ellipse import (EllipticalAnnulus, EllipticalAperture,
                                        SkyEllipticalAnnulus,
                                        SkyEllipticalAperture)
from photutils.aperture.tests.test_aperture_common import BaseTestAperture
from photutils.utils._optional_deps import HAS_MATPLOTLIB

POSITIONS = [(10, 20), (30, 40), (50, 60), (70, 80)]
RA, DEC = np.transpose(POSITIONS)
SKYCOORD = SkyCoord(ra=RA, dec=DEC, unit='deg')
UNIT = u.arcsec
RADII = (0.0, -1.0, -np.inf)


class TestEllipticalAperture(BaseTestAperture):
    aperture = EllipticalAperture(POSITIONS, a=10.0, b=5.0, theta=np.pi / 2.0)

    @staticmethod
    @pytest.mark.parametrize('radius', RADII)
    def test_invalid_params(radius):
        match = "'a' must be a positive scalar"
        with pytest.raises(ValueError, match=match):
            EllipticalAperture(POSITIONS, a=radius, b=5.0, theta=np.pi / 2.0)

        match = "'b' must be a positive scalar"
        with pytest.raises(ValueError, match=match):
            EllipticalAperture(POSITIONS, a=10.0, b=radius, theta=np.pi / 2.0)

    def test_copy_eq(self):
        aper = self.aperture.copy()
        assert aper == self.aperture
        aper.a = 20.0
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


class TestEllipticalAnnulus(BaseTestAperture):
    aperture = EllipticalAnnulus(POSITIONS, a_in=10.0, a_out=20.0, b_out=17.0,
                                 theta=np.pi / 3)

    @staticmethod
    @pytest.mark.parametrize('radius', RADII)
    def test_invalid_params(radius):
        match = "'a_in' must be a positive scalar"
        with pytest.raises(ValueError, match=match):
            EllipticalAnnulus(POSITIONS, a_in=radius, a_out=20.0, b_out=17.0,
                              theta=np.pi / 3)

        match = "'a_out' must be greater than 'a_in'"
        with pytest.raises(ValueError, match=match):
            EllipticalAnnulus(POSITIONS, a_in=10.0, a_out=radius, b_out=17.0,
                              theta=np.pi / 3)

        match = "'b_out' must be a positive scalar"
        with pytest.raises(ValueError, match=match):
            EllipticalAnnulus(POSITIONS, a_in=10.0, a_out=20.0, b_out=radius,
                              theta=np.pi / 3)

        match = "'b_in' must be a positive scalar"
        with pytest.raises(ValueError, match=match):
            EllipticalAnnulus(POSITIONS, a_in=10.0, a_out=20.0, b_out=17.0,
                              b_in=radius, theta=np.pi / 3)

    def test_copy_eq(self):
        aper = self.aperture.copy()
        assert aper == self.aperture
        aper.a_in = 2.0
        assert aper != self.aperture

    def test_theta(self):
        assert isinstance(self.aperture.theta, u.Quantity)
        assert self.aperture.theta.unit == u.rad

    def test_b_in_greater_than_b_out(self):
        """
        Test that a ValueError is raised when b_in >= b_out.
        """
        match = "'b_out' must be greater than 'b_in'"
        with pytest.raises(ValueError, match=match):
            EllipticalAnnulus(POSITIONS, a_in=10.0, a_out=20.0, b_out=5.0,
                              b_in=8.0, theta=np.pi / 3)

    @pytest.mark.skipif(not HAS_MATPLOTLIB, reason='matplotlib is required')
    def test_to_patch_nonscalar(self):
        """
        Test that _to_patch returns a list for non-scalar apertures.
        """
        patches = self.aperture._to_patch()
        assert isinstance(patches, list)


class TestSkyEllipticalAperture(BaseTestAperture):
    aperture = SkyEllipticalAperture(SKYCOORD, a=10.0 * UNIT, b=5.0 * UNIT,
                                     theta=30 * u.deg)

    @staticmethod
    @pytest.mark.parametrize('radius', RADII)
    def test_invalid_params(radius):
        match = "'a' must be greater than zero"
        with pytest.raises(ValueError, match=match):
            SkyEllipticalAperture(SKYCOORD, a=radius * UNIT, b=5.0 * UNIT,
                                  theta=30 * u.deg)

        match = "'b' must be greater than zero"
        with pytest.raises(ValueError, match=match):
            SkyEllipticalAperture(SKYCOORD, a=10.0 * UNIT, b=radius * UNIT,
                                  theta=30 * u.deg)

    def test_copy_eq(self):
        aper = self.aperture.copy()
        assert aper == self.aperture
        aper.a = 2.0 * UNIT
        assert aper != self.aperture


class TestSkyEllipticalAnnulus(BaseTestAperture):
    aperture = SkyEllipticalAnnulus(SKYCOORD, a_in=10.0 * UNIT,
                                    a_out=20.0 * UNIT, b_out=17.0 * UNIT,
                                    theta=60 * u.deg)

    @staticmethod
    @pytest.mark.parametrize('radius', RADII)
    def test_invalid_params(radius):
        match = "'a_in' must be greater than zero"
        with pytest.raises(ValueError, match=match):
            SkyEllipticalAnnulus(SKYCOORD, a_in=radius * UNIT,
                                 a_out=20.0 * UNIT, b_out=17.0 * UNIT,
                                 theta=60 * u.deg)

        match = "'a_out' must be greater than 'a_in'"
        with pytest.raises(ValueError, match=match):
            SkyEllipticalAnnulus(SKYCOORD, a_in=10.0 * UNIT,
                                 a_out=radius * UNIT, b_out=17.0 * UNIT,
                                 theta=60 * u.deg)

        match = "'b_out' must be greater than zero"
        with pytest.raises(ValueError, match=match):
            SkyEllipticalAnnulus(SKYCOORD, a_in=10.0 * UNIT, a_out=20.0 * UNIT,
                                 b_out=radius * UNIT, theta=60 * u.deg)

        match = "'b_in' must be greater than zero"
        with pytest.raises(ValueError, match=match):
            SkyEllipticalAnnulus(SKYCOORD, a_in=10.0 * UNIT, a_out=20.0 * UNIT,
                                 b_out=17.0 * UNIT, b_in=radius * UNIT,
                                 theta=60 * u.deg)

    def test_copy_eq(self):
        aper = self.aperture.copy()
        assert aper == self.aperture
        aper.a_in = 2.0 * UNIT
        assert aper != self.aperture

    def test_b_in_greater_than_b_out(self):
        """
        Test that a ValueError is raised when b_in >= b_out.
        """
        match = "'b_out' must be greater than 'b_in'"
        with pytest.raises(ValueError, match=match):
            SkyEllipticalAnnulus(SKYCOORD, a_in=10.0 * UNIT, a_out=20.0 * UNIT,
                                 b_out=5.0 * UNIT, b_in=8.0 * UNIT,
                                 theta=60 * u.deg)


def test_ellipse_theta_quantity():
    aper1 = EllipticalAperture(POSITIONS, a=10.0, b=5.0, theta=np.pi / 2.0)
    theta = u.Quantity(90 * u.deg)
    aper2 = EllipticalAperture(POSITIONS, a=10.0, b=5.0, theta=theta)
    theta = Angle(90 * u.deg)
    aper3 = EllipticalAperture(POSITIONS, a=10.0, b=5.0, theta=theta)

    assert_quantity_allclose(aper1.theta, aper2.theta)
    assert_quantity_allclose(aper1.theta, aper3.theta)


def test_ellipse_annulus_theta_quantity():
    aper1 = EllipticalAnnulus(POSITIONS, a_in=10.0, a_out=20.0, b_out=17.0,
                              theta=np.pi / 3)
    theta = u.Quantity(60 * u.deg)
    aper2 = EllipticalAnnulus(POSITIONS, a_in=10.0, a_out=20.0, b_out=17.0,
                              theta=theta)
    theta = Angle(60 * u.deg)
    aper3 = EllipticalAnnulus(POSITIONS, a_in=10.0, a_out=20.0, b_out=17.0,
                              theta=theta)

    assert_quantity_allclose(aper1.theta, aper2.theta)
    assert_quantity_allclose(aper1.theta, aper3.theta)


def test_elliptical_aperture_to_polygon():
    aper = EllipticalAperture((10.0, 20.0), a=5.0, b=3.0, theta=np.pi / 4)
    poly = aper.to_polygon(n_vertices=200)
    assert isinstance(poly, PolygonAperture)
    assert poly.n_vertices == 200
    assert abs(poly.area - aper.area) / aper.area < 1e-3


def test_elliptical_aperture_to_polygon_multi_position():
    aper = EllipticalAperture([(10.0, 20.0), (30.0, 40.0)],
                              a=4.0, b=2.0, theta=0.5)
    poly = aper.to_polygon(n_vertices=50)
    assert poly.vertices.shape == (2, 50, 2)


def test_elliptical_aperture_to_polygon_invalid_n_vertices():
    aper = EllipticalAperture((0.0, 0.0), a=1.0, b=1.0)
    match = 'n_vertices must be at least 3'
    with pytest.raises(ValueError, match=match):
        aper.to_polygon(n_vertices=2)


def test_sky_elliptical_aperture_to_polygon():
    pos = SkyCoord(ra=10.0, dec=30.0, unit='deg')
    aper = SkyEllipticalAperture(pos, a=5.0 * u.arcsec, b=3.0 * u.arcsec,
                                 theta=30 * u.deg)
    poly = aper.to_polygon(n_vertices=200)
    assert isinstance(poly, SkyPolygonAperture)
    assert poly.n_vertices == 200
    assert poly.vertices.shape == (200,)
    # The maximum and minimum vertex distances equal the semimajor and
    # semiminor axes, respectively.
    radii = np.hypot(*poly.vertex_offsets.to_value(u.arcsec).T)
    assert_allclose(radii.max(), 5.0)
    assert_allclose(radii.min(), 3.0)


def test_sky_elliptical_aperture_to_polygon_multi_position():
    pos = SkyCoord(ra=[10.0, 20.0], dec=[30.0, 40.0], unit='deg')
    aper = SkyEllipticalAperture(pos, a=4.0 * u.arcsec, b=2.0 * u.arcsec,
                                 theta=0.5 * u.rad)
    poly = aper.to_polygon(n_vertices=50)
    assert poly.vertices.shape == (2, 50)


def test_sky_elliptical_aperture_to_polygon_invalid_n_vertices():
    pos = SkyCoord(ra=0.0, dec=0.0, unit='deg')
    aper = SkyEllipticalAperture(pos, a=1.0 * u.arcsec, b=1.0 * u.arcsec)
    match = 'n_vertices must be at least 3'
    with pytest.raises(ValueError, match=match):
        aper.to_polygon(n_vertices=2)


def _make_round_trip_wcs():
    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [50.5, 50.5]
    wcs.wcs.cdelt = [-0.001, 0.001]
    wcs.wcs.crval = [10.0, 30.0]
    wcs.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    return wcs


@pytest.mark.parametrize('theta_deg', [0.0, 30.0, 45.0, 90.0, 137.0])
def test_sky_elliptical_aperture_to_polygon_wcs_round_trip(theta_deg):
    """
    Test that converting a SkyEllipticalAperture to a polygon and then
    to pixels is equivalent to converting to pixels and then to a
    polygon.
    """
    wcs = _make_round_trip_wcs()
    pos = SkyCoord(ra=10.0, dec=30.0, unit='deg')
    aper = SkyEllipticalAperture(pos, a=5.0 * u.arcsec, b=3.0 * u.arcsec,
                                 theta=theta_deg * u.deg)
    poly_from_sky = aper.to_polygon(n_vertices=200).to_pixel(wcs)
    poly_from_pix = aper.to_pixel(wcs).to_polygon(n_vertices=200)
    assert_allclose(poly_from_sky.area, poly_from_pix.area, rtol=1e-6)
    assert_allclose(poly_from_sky._xy_extents, poly_from_pix._xy_extents,
                    rtol=1e-6)


@pytest.mark.parametrize('theta_deg', [0.0, 30.0, 45.0, 90.0, 137.0])
def test_elliptical_aperture_to_polygon_to_sky_round_trip(theta_deg):
    """
    Test that converting an EllipticalAperture to a polygon and then to
    sky is equivalent to converting to sky and then to a polygon.
    """
    # The bounding-box orientation can differ at the ~1e-5 level because
    # the sky shape parameters from ``to_sky`` are derived from a local
    # linear (SVD) approximation, while the polygon ``to_sky`` maps
    # each vertex exactly, so only rotation-invariant quantities are
    # compared.
    wcs = _make_round_trip_wcs()
    aper = EllipticalAperture((50.0, 50.0), a=5.0, b=3.0,
                              theta=np.deg2rad(theta_deg))
    sky_from_poly = aper.to_polygon(n_vertices=200).to_sky(wcs)
    sky_from_aper = aper.to_sky(wcs).to_polygon(n_vertices=200)
    assert_allclose(sky_from_poly.positions.ra.deg,
                    sky_from_aper.positions.ra.deg)
    assert_allclose(sky_from_poly.positions.dec.deg,
                    sky_from_aper.positions.dec.deg)
    assert_allclose(sky_from_poly.perimeter.to_value(u.arcsec),
                    sky_from_aper.perimeter.to_value(u.arcsec), rtol=1e-6)
    assert_allclose(sky_from_poly.to_pixel(wcs).area,
                    sky_from_aper.to_pixel(wcs).area, rtol=1e-6)
