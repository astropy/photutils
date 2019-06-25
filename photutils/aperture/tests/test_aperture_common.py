# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
This module provides base classes for aperture tests.
"""

from astropy.coordinates import SkyCoord
from astropy.tests.helper import assert_quantity_allclose
from numpy.testing import assert_array_equal


class BaseTestApertureParams:
    index = 2
    slc = slice(0, 2)
    expected_slc_len = 2


class BaseTestAperture(BaseTestApertureParams):
    def test_index(self):
        aper = self.aperture[self.index]
        assert isinstance(aper, self.aperture.__class__)
        assert aper.isscalar
        expected_positions = self.aperture.positions[self.index]
        if isinstance(expected_positions, SkyCoord):
            assert_quantity_allclose(aper.positions.ra, expected_positions.ra)
            assert_quantity_allclose(aper.positions.dec,
                                     expected_positions.dec)
        else:
            assert_array_equal(aper.positions, expected_positions)
            for shape_param in aper._shape_params:
                assert (getattr(aper, shape_param) ==
                        getattr(self.aperture, shape_param))

    def test_slice(self):
        aper = self.aperture[self.slc]
        assert isinstance(aper, self.aperture.__class__)
        assert len(aper) == self.expected_slc_len

        expected_positions = self.aperture.positions[self.slc]
        if isinstance(self.aperture.positions, SkyCoord):
            assert_quantity_allclose(aper.positions.ra, expected_positions.ra)
            assert_quantity_allclose(aper.positions.dec,
                                     expected_positions.dec)
        else:
            assert_array_equal(aper.positions, expected_positions)
            for shape_param in aper._shape_params:
                assert (getattr(aper, shape_param) ==
                        getattr(self.aperture, shape_param))
