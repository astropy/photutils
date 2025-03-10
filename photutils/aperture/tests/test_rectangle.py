# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the rectangle module.
"""

import astropy.units as u
import numpy as np
import pytest
from astropy.coordinates import Angle, SkyCoord
from astropy.tests.helper import assert_quantity_allclose

from photutils.aperture.rectangle import (RectangularAnnulus,
                                          RectangularAperture,
                                          SkyRectangularAnnulus,
                                          SkyRectangularAperture)
from photutils.aperture.tests.test_aperture_common import BaseTestAperture

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

        match = '"w_out" must be greater than "w_in"'
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

        match = '"w_out" must be greater than "w_in"'
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
