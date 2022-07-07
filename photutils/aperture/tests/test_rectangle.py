# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the rectangle module.
"""

from astropy.coordinates import Angle, SkyCoord
import astropy.units as u
import numpy as np
import pytest

from .test_aperture_common import BaseTestAperture

from ..rectangle import (RectangularAperture, RectangularAnnulus,
                         SkyRectangularAperture, SkyRectangularAnnulus)


POSITIONS = [(10, 20), (30, 40), (50, 60), (70, 80)]
RA, DEC = np.transpose(POSITIONS)
SKYCOORD = SkyCoord(ra=RA, dec=DEC, unit='deg')
UNIT = u.arcsec
RADII = (0.0, -1.0, -np.inf)


class TestRectangularAperture(BaseTestAperture):
    aperture = RectangularAperture(POSITIONS, w=10., h=5., theta=np.pi / 2.)

    @staticmethod
    @pytest.mark.parametrize('radius', RADII)
    def test_invalid_params(radius):
        with pytest.raises(ValueError):
            RectangularAperture(POSITIONS, w=radius, h=5., theta=np.pi / 2.)
        with pytest.raises(ValueError):
            RectangularAperture(POSITIONS, w=10., h=radius, theta=np.pi / 2.)

    def test_copy_eq(self):
        aper = self.aperture.copy()
        assert aper == self.aperture
        aper.w = 20.
        assert aper != self.aperture


class TestRectangularAnnulus(BaseTestAperture):
    aperture = RectangularAnnulus(POSITIONS, w_in=10., w_out=20., h_out=17,
                                  theta=np.pi / 3)

    @staticmethod
    @pytest.mark.parametrize('radius', RADII)
    def test_invalid_params(radius):
        with pytest.raises(ValueError):
            RectangularAnnulus(POSITIONS, w_in=radius, w_out=20., h_out=17,
                               theta=np.pi / 3)
        with pytest.raises(ValueError):
            RectangularAnnulus(POSITIONS, w_in=10., w_out=radius, h_out=17,
                               theta=np.pi / 3)
        with pytest.raises(ValueError):
            RectangularAnnulus(POSITIONS, w_in=10., w_out=20., h_out=radius,
                               theta=np.pi / 3)
        with pytest.raises(ValueError):
            RectangularAnnulus(POSITIONS, w_in=10., w_out=20., h_out=17,
                               h_in=radius, theta=np.pi / 3)

    def test_copy_eq(self):
        aper = self.aperture.copy()
        assert aper == self.aperture
        aper.w_in = 2.
        assert aper != self.aperture


class TestSkyRectangularAperture(BaseTestAperture):
    aperture = SkyRectangularAperture(SKYCOORD, w=10. * UNIT, h=5. * UNIT,
                                      theta=30 * u.deg)

    @staticmethod
    @pytest.mark.parametrize('radius', RADII)
    def test_invalid_params(radius):
        with pytest.raises(ValueError):
            SkyRectangularAperture(SKYCOORD, w=radius * UNIT, h=5. * UNIT,
                                   theta=30 * u.deg)
        with pytest.raises(ValueError):
            SkyRectangularAperture(SKYCOORD, w=10. * UNIT, h=radius * UNIT,
                                   theta=30 * u.deg)

    def test_copy_eq(self):
        aper = self.aperture.copy()
        assert aper == self.aperture
        aper.w = 20. * UNIT
        assert aper != self.aperture


class TestSkyRectangularAnnulus(BaseTestAperture):
    aperture = SkyRectangularAnnulus(SKYCOORD, w_in=10. * UNIT,
                                     w_out=20. * UNIT, h_out=17. * UNIT,
                                     theta=60 * u.deg)

    @staticmethod
    @pytest.mark.parametrize('radius', RADII)
    def test_invalid_params(radius):
        with pytest.raises(ValueError):
            SkyRectangularAnnulus(SKYCOORD, w_in=radius * UNIT,
                                  w_out=20. * UNIT, h_out=17. * UNIT,
                                  theta=60 * u.deg)
        with pytest.raises(ValueError):
            SkyRectangularAnnulus(SKYCOORD, w_in=10. * UNIT,
                                  w_out=radius * UNIT, h_out=17. * UNIT,
                                  theta=60 * u.deg)
        with pytest.raises(ValueError):
            SkyRectangularAnnulus(SKYCOORD, w_in=10. * UNIT, w_out=20. * UNIT,
                                  h_out=radius * UNIT, theta=60 * u.deg)
        with pytest.raises(ValueError):
            SkyRectangularAnnulus(SKYCOORD, w_in=10. * UNIT, w_out=20. * UNIT,
                                  h_out=17. * UNIT, h_in=radius * UNIT,
                                  theta=60 * u.deg)

    def test_copy_eq(self):
        aper = self.aperture.copy()
        assert aper == self.aperture
        aper.w_in = 2. * UNIT
        assert aper != self.aperture


def test_rectangle_theta_quantity():
    aper1 = RectangularAperture(POSITIONS, w=10., h=5., theta=np.pi / 2.)
    theta = u.Quantity(90 * u.deg)
    aper2 = RectangularAperture(POSITIONS, w=10., h=5., theta=theta)
    theta = Angle(90 * u.deg)
    aper3 = RectangularAperture(POSITIONS, w=10., h=5., theta=theta)

    assert aper1._theta_radians == aper2._theta_radians
    assert aper1._theta_radians == aper3._theta_radians


def test_rectangle_annulus_theta_quantity():
    aper1 = RectangularAnnulus(POSITIONS, w_in=10., w_out=20., h_out=17,
                               theta=np.pi / 3)
    theta = u.Quantity(60 * u.deg)
    aper2 = RectangularAnnulus(POSITIONS, w_in=10., w_out=20., h_out=17,
                               theta=theta)
    theta = Angle(60 * u.deg)
    aper3 = RectangularAnnulus(POSITIONS, w_in=10., w_out=20., h_out=17,
                               theta=theta)

    assert aper1._theta_radians == aper2._theta_radians
    assert aper1._theta_radians == aper3._theta_radians
