# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the photutils.detection.core module.
"""

import numpy as np
import pytest

from photutils.detection.core import (StarFinderCatalogBase, _StarFinderKernel,
                                      _validate_brightest)


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

    def test_normalize_zerosum_false(self):
        """Test kernel with normalize_zerosum=False."""
        kernel = _StarFinderKernel(fwhm=2.0, normalize_zerosum=False)
        # without zero-sum normalization, the kernel sums to a positive value
        assert kernel.data.sum() > 0


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


def _make_minimal_catalog_class():
    """
    Create a minimal concrete subclass of StarFinderCatalogBase that
    does NOT override ``_get_init_attributes`` and does NOT set
    ``default_columns``, so the base-class implementations can be
    tested.
    """

    class _MinimalCatalog(StarFinderCatalogBase):

        @property
        def xcentroid(self):
            return self.cutout_xcentroid

        @property
        def ycentroid(self):
            return self.cutout_ycentroid

        def apply_filters(self):
            return self

    return _MinimalCatalog


class TestStarFinderCatalogBase:
    """Tests for the StarFinderCatalogBase base-class methods."""

    def test_get_init_attributes(self):
        """Test base _get_init_attributes returns expected tuple."""
        catalog_cls = _make_minimal_catalog_class()
        data = np.zeros((11, 11))
        data[5, 5] = 10.0
        kernel = np.ones((3, 3))
        xypos = np.array([[5, 5]])
        cat = catalog_cls(data, xypos, kernel)
        expected = ('data', 'unit', 'kernel', 'brightest', 'peakmax',
                    'cutout_shape')
        assert cat._get_init_attributes() == expected

    def test_to_table_missing_default_columns(self):
        """Test that to_table raises when default_columns is not set."""
        catalog_cls = _make_minimal_catalog_class()
        data = np.zeros((11, 11))
        data[5, 5] = 10.0
        kernel = np.ones((3, 3))
        xypos = np.array([[5, 5]])
        cat = catalog_cls(data, xypos, kernel)
        match = 'default_columns attribute is not set'
        with pytest.raises(AttributeError, match=match):
            cat.to_table()
