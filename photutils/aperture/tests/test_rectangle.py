# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the rectangle module.
"""

from astropy.coordinates import SkyCoord
import astropy.units as u
import numpy as np

from .test_aperture_common import BaseTestAperture

from ..rectangle import (RectangularAperture, RectangularAnnulus,
                         SkyRectangularAperture, SkyRectangularAnnulus)


POSITIONS = [(10, 20), (30, 40), (50, 60), (70, 80)]
RA, DEC = np.transpose(POSITIONS)
SKYCOORD = SkyCoord(ra=RA, dec=DEC, unit='deg')
UNIT = u.arcsec


class TestRectangularAperture(BaseTestAperture):
    aperture = RectangularAperture(POSITIONS, w=10., h=5., theta=np.pi/2.)


class TestRectangularAnnulus(BaseTestAperture):
    aperture = RectangularAnnulus(POSITIONS, w_in=10., w_out=20., h_out=17,
                                  theta=np.pi/3)


class TestSkyRectangularAperture(BaseTestAperture):
    aperture = SkyRectangularAperture(SKYCOORD, w=10.*UNIT, h=5.*UNIT,
                                      theta=30*u.deg)


class TestSkyRectangularAnnulus(BaseTestAperture):
    aperture = SkyRectangularAnnulus(SKYCOORD, w_in=10.*UNIT, w_out=20.*UNIT,
                                     h_out=17.*UNIT, theta=60*u.deg)
