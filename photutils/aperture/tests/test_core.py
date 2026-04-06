# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the core module.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from photutils.aperture import Aperture, CircularAperture, EllipticalAperture

POSITIONS = [(5, 5), (10, 10), (15, 15)]
SCALAR_POS = (5, 5)


class MinimalAperture(Aperture):
    """
    Minimal concrete Aperture subclass that is neither PixelAperture nor
    SkyAperture.

    Used to exercise bare-Aperture code paths.
    """

    _params = ('positions',)

    @property
    def positions(self):
        """
        Return a fixed single position.
        """
        return np.array([[5.0, 5.0]])


class RaisesOnCompare:
    """
    Helper object that raises TypeError on any != comparison, used to
    exercise the except-TypeError branch in Aperture.__eq__.
    """

    def __ne__(self, other):
        """
        Raise TypeError unconditionally.
        """
        msg = 'incompatible types'
        raise TypeError(msg)


class TestAperture:
    """
    Tests for branches of the Aperture base class not covered elsewhere.
    """

    def test_positions_str_raises_for_unknown_type(self):
        """
        Test that _positions_str raises TypeError when the aperture is
        not a PixelAperture or SkyAperture subclass.
        """
        aper = MinimalAperture()
        match = 'Aperture must be a subclass of PixelAperture or SkyAperture'
        with pytest.raises(TypeError, match=match):
            aper._positions_str()

    def test_eq_different_class(self):
        """
        Test that __eq__ returns False when compared to a different
        class (exercises the isinstance early-return branch).
        """
        aper1 = CircularAperture(SCALAR_POS, r=3)
        aper2 = EllipticalAperture(SCALAR_POS, a=3, b=2, theta=0)
        assert aper1 != aper2

    def test_eq_different_params(self):
        """
        Test that __eq__ returns False when the two instances have
        different _params tuples (exercises the params-mismatch branch).
        """
        aper1 = CircularAperture(SCALAR_POS, r=3)
        aper2 = CircularAperture(SCALAR_POS, r=3)
        # Inject an extended _params tuple at the instance level so that
        # isinstance passes (both CircularAperture instances, same type so
        # Python does not invoke subclass reflection) while the params
        # check diverges, hitting the mismatch return at line 104.
        aper2.__dict__['_params'] = (*CircularAperture._params, 'extra')
        assert aper1 != aper2

    def test_eq_comparison_type_error(self):
        """
        Test that __eq__ returns False (rather than propagating the
        exception) when the position comparison raises TypeError.
        """
        aper1 = CircularAperture(SCALAR_POS, r=3)
        aper2 = CircularAperture(SCALAR_POS, r=3)
        # Bypass the descriptor and inject a position object that raises
        # TypeError on != comparison, mimicking incompatible SkyCoords.
        aper1.__dict__['positions'] = RaisesOnCompare()
        aper2.__dict__['positions'] = RaisesOnCompare()
        assert aper1 != aper2


class TestPixelAperture:
    """
    Tests for branches of the PixelAperture class not covered elsewhere.
    """

    def test_to_mask_invalid_method(self):
        """
        Test that to_mask raises ValueError for an unrecognised
        method string (exercises the invalid-method branch in
        _translate_mask_method).
        """
        aper = CircularAperture(SCALAR_POS, r=3)
        match = 'Invalid mask method'
        with pytest.raises(ValueError, match=match):
            aper.to_mask(method='invalid')

    def test_bbox_multi_position(self):
        """
        Test that the bbox property returns a list for a multi-position
        aperture (exercises the non-scalar branch).
        """
        aper = CircularAperture(POSITIONS, r=3)
        bbox = aper.bbox
        assert isinstance(bbox, list)
        assert len(bbox) == len(POSITIONS)


class TestPixelApertureDoPhotometry:
    """
    Tests for error-handling branches of PixelAperture.do_photometry.
    """

    def setup_method(self):
        """
        Set up a simple scalar aperture and matching data array.
        """
        self.aper = CircularAperture(SCALAR_POS, r=3)
        self.data = np.ones((20, 20))

    def test_do_photometry_1d_data_error(self):
        """
        Test that do_photometry raises ValueError when data is not a
        2D array.
        """
        match = 'data must be a 2D array'
        with pytest.raises(ValueError, match=match):
            self.aper.do_photometry(np.ones(20))

    def test_do_photometry_error_shape_mismatch(self):
        """
        Test that do_photometry raises ValueError when the error array
        does not match the data shape.
        """
        match = 'error and data must have the same shape'
        with pytest.raises(ValueError, match=match):
            self.aper.do_photometry(self.data, error=np.ones((5, 5)))

    def test_do_photometry_unit_mismatch_error(self):
        """
        Test that do_photometry raises ValueError when data and error
        have different units.
        """
        import astropy.units as u

        match = 'they both must have the same units'
        with pytest.raises(ValueError, match=match):
            self.aper.do_photometry(
                self.data * u.Jy,
                error=self.data * u.ct,
            )

    def test_do_photometry_basic(self):
        """
        Test that do_photometry returns the expected aperture sum for a
        uniform data array with no error input.
        """
        sums, errs = self.aper.do_photometry(self.data)
        assert_allclose(sums[0], np.pi * 9, rtol=1e-3)
        assert len(errs) == 0
