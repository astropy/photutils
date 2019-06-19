# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the circle module.
"""

from astropy.coordinates import SkyCoord
import astropy.units as u
import numpy as np

from .test_aperture_common import BaseTestAperture
from ..circle import (CircularAperture, CircularAnnulus, SkyCircularAperture,
                      SkyCircularAnnulus)


POSITIONS = [(10, 20), (30, 40), (50, 60), (70, 80)]
RA, DEC = np.transpose(POSITIONS)
SKYCOORD = SkyCoord(ra=RA, dec=DEC, unit='deg')
UNIT = u.arcsec


class TestCircularAperture(BaseTestAperture):
    aperture = CircularAperture(POSITIONS, r=3.)


class TestCircularAnnulus(BaseTestAperture):
    aperture = CircularAnnulus(POSITIONS, r_in=3., r_out=7.)


class TestSkyCircularAperture(BaseTestAperture):
    aperture = SkyCircularAperture(SKYCOORD, r=3.*UNIT)


class TestSkyCircularAnnulus(BaseTestAperture):
    aperture = SkyCircularAnnulus(SKYCOORD, r_in=3.*UNIT, r_out=7.*UNIT)
