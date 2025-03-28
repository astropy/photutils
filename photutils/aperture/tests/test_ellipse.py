# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the ellipse module.
"""

import astropy.units as u
import numpy as np
import pytest
from astropy.coordinates import Angle, SkyCoord
from astropy.tests.helper import assert_quantity_allclose

from photutils.aperture.ellipse import (EllipticalAnnulus, EllipticalAperture,
                                        SkyEllipticalAnnulus,
                                        SkyEllipticalAperture)
from photutils.aperture.tests.test_aperture_common import BaseTestAperture

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

        match = '"a_out" must be greater than "a_in"'
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

        match = '"a_out" must be greater than "a_in"'
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
