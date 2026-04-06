# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the attributes module.
"""

import astropy.units as u
import numpy as np
import pytest
from astropy.coordinates import SkyCoord

from photutils.aperture.attributes import (ApertureAttribute, PixelPositions,
                                           PositiveScalar, PositiveScalarAngle,
                                           ScalarAngle, ScalarAngleOrValue,
                                           SkyCoordPositions)


class MockBase:
    """
    Minimal class using ApertureAttribute as a descriptor.
    """

    attr = ApertureAttribute(doc='a test attribute')


class MockWithLazy:
    """
    Minimal class with _lazyproperties for reset testing.
    """

    attr = ApertureAttribute()
    _lazyproperties = ['cached']  # noqa: RUF012


class MockPixelPos:
    """
    Minimal class using PixelPositions as a descriptor.
    """

    positions = PixelPositions()


class MockSkyCoord:
    """
    Minimal class using SkyCoordPositions as a descriptor.
    """

    positions = SkyCoordPositions()


class MockPositiveScalar:
    """
    Minimal class using PositiveScalar as a descriptor.
    """

    r = PositiveScalar()


class MockScalarAngle:
    """
    Minimal class using ScalarAngle as a descriptor.
    """

    angle = ScalarAngle()


class MockPositiveScalarAngle:
    """
    Minimal class using PositiveScalarAngle as a descriptor.
    """

    angle = PositiveScalarAngle()


class MockScalarAngleOrValue:
    """
    Minimal class using ScalarAngleOrValue as a descriptor.
    """

    theta = ScalarAngleOrValue()


class TestApertureAttribute:
    """
    Tests for the ApertureAttribute base descriptor class.
    """

    def test_class_access(self):
        """
        Test that __get__ with instance=None returns the descriptor.
        """
        desc = MockBase.attr
        assert isinstance(desc, ApertureAttribute)

    def test_doc(self):
        """
        Test that the descriptor docstring is set correctly.
        """
        desc = MockBase.__dict__['attr']
        assert desc.__doc__ == 'a test attribute'

    def test_set_name(self):
        """
        Test that __set_name__ assigns the attribute name correctly.
        """
        desc = MockBase.__dict__['attr']
        assert desc.name == 'attr'

    def test_set_and_get(self):
        """
        Test that setting and getting a value round-trips correctly.
        """
        obj = MockBase()
        obj.attr = 5.0
        assert obj.attr == 5.0

    def test_converts_to_float(self):
        """
        Test that non-Quantity scalars are converted to float.
        """
        obj = MockBase()
        obj.attr = 3
        assert isinstance(obj.attr, float)
        assert obj.attr == 3.0

    def test_quantity_not_converted(self):
        """
        Test that Quantity values are stored without conversion.
        """
        obj = MockBase()
        val = 5.0 * u.m
        obj.attr = val
        assert isinstance(obj.attr, u.Quantity)

    def test_skycoord_not_converted(self):
        """
        Test that SkyCoord values are stored without conversion.
        """
        obj = MockBase()
        val = SkyCoord(ra=10.0, dec=20.0, unit='deg')
        obj.attr = val
        assert isinstance(obj.attr, SkyCoord)

    def test_reset_lazyproperties(self):
        """
        Test that lazyproperties are cleared when the attribute is
        updated.
        """
        obj = MockWithLazy()
        obj.attr = 1.0
        obj.cached = 42
        assert 'cached' in obj.__dict__
        obj.attr = 2.0  # second assignment triggers _reset_lazyproperties
        assert 'cached' not in obj.__dict__

    def test_reset_no_lazyproperties(self):
        """
        Test that AttributeError is silently caught when _lazyproperties
        is absent.
        """
        obj = MockBase()
        obj.attr = 1.0
        # Triggers _reset_lazyproperties; obj has no _lazyproperties
        obj.attr = 2.0
        assert obj.attr == 2.0

    def test_delete(self):
        """
        Test that deleting the attribute removes it from the instance.
        """
        obj = MockBase()
        obj.attr = 5.0
        del obj.attr
        assert 'attr' not in obj.__dict__

    def test_validate_noop(self):
        """
        Test that the base _validate method is a no-op.
        """
        obj = MockBase()
        obj.attr = 99.0  # calls _validate internally without error
        assert obj.attr == 99.0


class TestPixelPositions:
    """
    Tests for the PixelPositions descriptor class.
    """

    def test_single_tuple(self):
        """
        Test that a single (x, y) tuple is accepted.
        """
        obj = MockPixelPos()
        obj.positions = (10, 20)
        # _validate returns the original (non-atleast_2d) array
        assert obj.positions.shape == (2,)
        assert obj.positions[0] == 10.0
        assert obj.positions[1] == 20.0

    def test_list_of_tuples(self):
        """
        Test that a list of (x, y) tuples is accepted.
        """
        obj = MockPixelPos()
        obj.positions = [(1, 2), (3, 4)]
        assert obj.positions.shape == (2, 2)

    def test_ndarray(self):
        """
        Test that a 2D numpy array of positions is accepted.
        """
        obj = MockPixelPos()
        pos = np.array([(5.0, 6.0), (7.0, 8.0)])
        obj.positions = pos
        assert obj.positions.shape == (2, 2)

    def test_zip_input(self):
        """
        Test that a zip of x and y arrays is accepted.
        """
        obj = MockPixelPos()
        obj.positions = zip([1.0, 2.0], [3.0, 4.0], strict=False)
        assert obj.positions.shape == (2, 2)

    def test_reset_lazyproperties(self):
        """
        Test that setting positions twice clears cached lazyproperties.
        """
        obj = MockPixelPos()
        obj.positions = [(1, 2)]
        obj._lazyproperties = ['cached']
        obj.cached = 99
        obj.positions = [(3, 4)]  # triggers reset
        assert 'cached' not in obj.__dict__

    def test_quantity_error(self):
        """
        Test that a Quantity array raises TypeError.
        """
        obj = MockPixelPos()
        match = "'positions' must not be a Quantity"
        with pytest.raises(TypeError, match=match):
            obj.positions = np.array([1.0, 2.0]) * u.m

    def test_bad_type_error(self):
        """
        Test the TypeError path via np.asanyarray(...).astype(float).
        """
        desc = MockPixelPos.__dict__['positions']
        match = "'positions' must not be a Quantity"
        with pytest.raises(TypeError, match=match):
            # Sets cannot be converted to float, triggering except TypeError
            desc._validate([({1, 2}, {3, 4})])

    def test_zip_quantity_error(self):
        """
        Test that TypeError is raised when zip contains Quantity objects.
        """
        obj = MockPixelPos()
        match = "'positions' must not be a Quantity"
        with pytest.raises(TypeError, match=match):
            obj.positions = zip(
                [1.0 * u.m, 2.0 * u.m],
                [3.0 * u.m, 4.0 * u.m],
                strict=False,
            )

    def test_nonfinite_nan_error(self):
        """
        Test that NaN positions raise ValueError.
        """
        obj = MockPixelPos()
        match = "'positions' must not contain any non-finite"
        with pytest.raises(ValueError, match=match):
            obj.positions = [(np.nan, 2.0)]

    def test_nonfinite_inf_error(self):
        """
        Test that infinite positions raise ValueError.
        """
        obj = MockPixelPos()
        match = "'positions' must not contain any non-finite"
        with pytest.raises(ValueError, match=match):
            obj.positions = [(1.0, np.inf)]

    def test_3d_error(self):
        """
        Test that a 3D array raises ValueError.
        """
        obj = MockPixelPos()
        match = "'positions' must be a"
        with pytest.raises(ValueError, match=match):
            obj.positions = np.ones((2, 2, 2))

    def test_wrong_ncols_error(self):
        """
        Test that positions with wrong column count raise ValueError.
        """
        obj = MockPixelPos()
        match = "'positions' must be a"
        with pytest.raises(ValueError, match=match):
            obj.positions = [(1.0, 2.0, 3.0), (4.0, 5.0, 6.0)]


class TestSkyCoordPositions:
    """
    Tests for the SkyCoordPositions descriptor class.
    """

    def test_valid(self):
        """
        Test that a SkyCoord value is accepted.
        """
        obj = MockSkyCoord()
        sky = SkyCoord(ra=10.0, dec=20.0, unit='deg')
        obj.positions = sky
        assert isinstance(obj.positions, SkyCoord)

    def test_invalid_type_error(self):
        """
        Test that a non-SkyCoord value raises TypeError.
        """
        obj = MockSkyCoord()
        match = "'positions' must be a SkyCoord instance"
        with pytest.raises(TypeError, match=match):
            obj.positions = (10.0, 20.0)


class TestPositiveScalar:
    """
    Tests for the PositiveScalar descriptor class.
    """

    def test_valid(self):
        """
        Test that a positive scalar value is accepted.
        """
        obj = MockPositiveScalar()
        obj.r = 3.5
        assert obj.r == 3.5

    def test_zero_error(self):
        """
        Test that zero raises ValueError.
        """
        obj = MockPositiveScalar()
        match = "'r' must be a positive scalar"
        with pytest.raises(ValueError, match=match):
            obj.r = 0.0

    def test_negative_error(self):
        """
        Test that a negative value raises ValueError.
        """
        obj = MockPositiveScalar()
        match = "'r' must be a positive scalar"
        with pytest.raises(ValueError, match=match):
            obj.r = -1.0

    def test_array_error(self):
        """
        Test that a non-scalar array raises ValueError.
        """
        obj = MockPositiveScalar()
        match = "'r' must be a positive scalar"
        with pytest.raises(ValueError, match=match):
            obj.r = np.array([1.0, 2.0])


class TestScalarAngle:
    """
    Tests for the ScalarAngle descriptor class.
    """

    def test_valid_rad(self):
        """
        Test that a scalar radian Quantity is accepted.
        """
        obj = MockScalarAngle()
        obj.angle = 0.5 * u.rad
        assert obj.angle == 0.5 * u.rad

    def test_valid_arcsec(self):
        """
        Test that a scalar arcsec Quantity is accepted.
        """
        obj = MockScalarAngle()
        obj.angle = 30.0 * u.arcsec
        assert obj.angle.unit == u.arcsec

    def test_nonscalar_error(self):
        """
        Test that a non-scalar Quantity raises ValueError.
        """
        obj = MockScalarAngle()
        match = "'angle' must be a scalar"
        with pytest.raises(ValueError, match=match):
            obj.angle = np.array([1.0, 2.0]) * u.arcsec

    def test_non_angle_units_error(self):
        """
        Test that a Quantity with non-angular units raises ValueError.
        """
        obj = MockScalarAngle()
        match = "'angle' must have angular units"
        with pytest.raises(ValueError, match=match):
            obj.angle = 5.0 * u.m

    def test_non_quantity_error(self):
        """
        Test that a plain float raises TypeError.
        """
        obj = MockScalarAngle()
        match = "'angle' must be a scalar angle"
        with pytest.raises(TypeError, match=match):
            obj.angle = 1.0


class TestPositiveScalarAngle:
    """
    Tests for the PositiveScalarAngle descriptor class.
    """

    def test_valid(self):
        """
        Test that a positive scalar angle Quantity is accepted.
        """
        obj = MockPositiveScalarAngle()
        obj.angle = 5.0 * u.arcsec
        assert obj.angle == 5.0 * u.arcsec

    def test_negative_error(self):
        """
        Test that a negative angle raises ValueError.
        """
        obj = MockPositiveScalarAngle()
        match = "'angle' must be greater than zero"
        with pytest.raises(ValueError, match=match):
            obj.angle = -1.0 * u.arcsec

    def test_nonscalar_error(self):
        """
        Test that a non-scalar angle array raises ValueError.
        """
        obj = MockPositiveScalarAngle()
        match = "'angle' must be a scalar"
        with pytest.raises(ValueError, match=match):
            # Single-element array passes > 0 check but fails isscalar
            obj.angle = np.array([5.0]) * u.arcsec

    def test_non_angle_units_error(self):
        """
        Test that a Quantity with non-angular units raises ValueError.
        """
        obj = MockPositiveScalarAngle()
        match = "'angle' must have angular units"
        with pytest.raises(ValueError, match=match):
            obj.angle = 5.0 * u.m

    def test_non_quantity_error(self):
        """
        Test that a plain float raises TypeError.
        """
        obj = MockPositiveScalarAngle()
        match = "'angle' must be a scalar angle"
        with pytest.raises(TypeError, match=match):
            obj.angle = 5.0


class TestScalarAngleOrValue:
    """
    Tests for the ScalarAngleOrValue descriptor class.
    """

    def test_valid_quantity(self):
        """
        Test that a scalar angle Quantity is accepted.
        """
        obj = MockScalarAngleOrValue()
        obj.theta = 30.0 * u.deg
        assert obj.theta == 30.0 * u.deg

    def test_float_converted_to_radian(self):
        """
        Test that a plain float is stored as a radian Quantity.
        """
        obj = MockScalarAngleOrValue()
        obj.theta = 1.5
        assert isinstance(obj.theta, u.Quantity)
        assert obj.theta.unit == u.radian
        assert obj.theta.value == 1.5

    def test_reset_lazyproperties(self):
        """
        Test that setting theta twice clears cached lazyproperties.
        """
        obj = MockScalarAngleOrValue()
        obj.theta = 1.0
        obj._lazyproperties = ['cached']
        obj.cached = 42
        obj.theta = 2.0  # triggers reset
        assert 'cached' not in obj.__dict__

    def test_nonscalar_quantity_error(self):
        """
        Test that a non-scalar angle Quantity raises ValueError.
        """
        obj = MockScalarAngleOrValue()
        match = "'theta' must be a scalar"
        with pytest.raises(ValueError, match=match):
            obj.theta = np.array([1.0, 2.0]) * u.arcsec

    def test_non_angle_units_error(self):
        """
        Test that a Quantity with non-angular units raises ValueError.
        """
        obj = MockScalarAngleOrValue()
        match = "'theta' must have angular units"
        with pytest.raises(ValueError, match=match):
            obj.theta = 5.0 * u.m

    def test_nonscalar_float_error(self):
        """
        Test that a non-scalar float array raises ValueError.
        """
        obj = MockScalarAngleOrValue()
        match = 'If not an angle Quantity'
        with pytest.raises(ValueError, match=match):
            obj.theta = np.array([1.0, 2.0])
