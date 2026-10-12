# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the behavior shared by the profile classes.
"""

import astropy.units as u
import numpy as np
import pytest
from astropy.utils.exceptions import AstropyUserWarning
from numpy.testing import assert_allclose

from photutils.profiles import (CurveOfGrowth, EllipticalCurveOfGrowth,
                                EnsquaredCurveOfGrowth, RadialProfile)

PROFILE_CASES = [
    (RadialProfile, {}),
    (CurveOfGrowth, {}),
    (EnsquaredCurveOfGrowth, {}),
    (EllipticalCurveOfGrowth, {'axis_ratio': 0.5}),
]


@pytest.mark.parametrize(('cls', 'kwargs'), PROFILE_CASES)
class TestProfileInputs:
    def test_invalid_method(self, profile_data, cls, kwargs):
        """
        Test that an invalid ``method`` or ``subpixels`` is rejected
        when the class is created.
        """
        xycen, data, _, _ = profile_data
        radii = np.arange(1, 10)

        match = "Invalid method: 'invalid'"
        with pytest.raises(ValueError, match=match):
            cls(data, xycen, radii, method='invalid', **kwargs)

        match = 'subpixels must be a strictly positive integer'
        with pytest.raises(ValueError, match=match):
            cls(data, xycen, radii, method='subpixel', subpixels=0,
                **kwargs)

        # subpixels is used only by the 'subpixel' method
        cls(data, xycen, radii, method='center', subpixels=0, **kwargs)

    def test_invalid_data_shape(self, profile_data, cls, kwargs):
        """
        Test that non-2D data is rejected when the class is created.
        """
        xycen, data, _, _ = profile_data
        radii = np.arange(1, 10)
        match = 'data must be a 2D array'
        with pytest.raises(ValueError, match=match):
            cls(data[0], xycen, radii, **kwargs)

    def test_unit_mismatch(self, profile_data, cls, kwargs):
        """
        Test that data and error must have the same units.
        """
        xycen, data, error, _ = profile_data
        radii = np.arange(1, 10)
        match = 'must all have the same units'
        with pytest.raises(ValueError, match=match):
            cls(data << u.Jy, xycen, radii, error=error, **kwargs)

    @pytest.mark.parametrize('method', ['max', 'sum'])
    def test_normalize_negative(self, profile_data, cls, kwargs, method):
        """
        Test that a profile with a negative maximum or sum is not
        normalized.
        """
        xycen, data, _, _ = profile_data
        radii = np.arange(1, 10)
        profile = cls(-data, xycen, radii, **kwargs)
        expected = profile.profile.copy()
        assert np.all(expected < 0)

        match = 'The profile cannot be normalized'
        with pytest.warns(AstropyUserWarning, match=match):
            profile.normalize(method=method)
        assert profile.normalization_value == 1.0
        assert_allclose(profile.profile, expected)
