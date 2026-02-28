# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the photutils.detection.core module.
"""

import pytest

from photutils.detection.core import _StarFinderKernel, _validate_brightest


class TestStarFinderKernel:
    """Tests for the _StarFinderKernel class."""

    def test_fwhm_zero(self):
        """Test that fwhm=0 raises a ValueError."""
        match = 'fwhm must be positive'
        with pytest.raises(ValueError, match=match):
            _StarFinderKernel(fwhm=0)

    def test_fwhm_negative(self):
        """Test that a negative fwhm raises a ValueError."""
        match = 'fwhm must be positive'
        with pytest.raises(ValueError, match=match):
            _StarFinderKernel(fwhm=-1)


class TestValidateBrightest:
    """
    Parametrized tests for the _validate_brightest function.
    """

    @pytest.mark.parametrize('brightest', [-1, -0.5, -100])
    def test_brightest_negative(self, brightest):
        """Test that negative brightest values raise ValueError."""
        match = 'brightest must be > 0'
        with pytest.raises(ValueError, match=match):
            _validate_brightest(brightest)

    def test_brightest_zero(self):
        """Test that brightest=0 raises ValueError."""
        match = 'brightest must be > 0'
        with pytest.raises(ValueError, match=match):
            _validate_brightest(0)

    @pytest.mark.parametrize('brightest', [3.1, 2.5, 1.9])
    def test_brightest_not_integer(self, brightest):
        """Test that non-integer brightest values raise ValueError."""
        match = 'brightest must be an integer'
        with pytest.raises(ValueError, match=match):
            _validate_brightest(brightest)

    @pytest.mark.parametrize('brightest', [1, 5, 100])
    def test_brightest_valid(self, brightest):
        """Test that valid brightest values are returned unchanged."""
        assert _validate_brightest(brightest) == brightest

    def test_brightest_none(self):
        """Test that None is a valid brightest value."""
        assert _validate_brightest(None) is None
