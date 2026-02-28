# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the photutils.detection.core module.
"""

import numpy as np
import pytest

from photutils.detection.core import (StarFinderCatalogBase, _StarFinderKernel,
                                      _validate_brightest)


class TestStarFinderKernel:
    """
    Tests for the _StarFinderKernel class.
    """

    def test_fwhm_zero(self):
        """
        Test that fwhm=0 raises a ValueError.
        """
        match = 'fwhm must be positive'
        with pytest.raises(ValueError, match=match):
            _StarFinderKernel(fwhm=0)

    def test_fwhm_negative(self):
        """
        Test that a negative fwhm raises a ValueError.
        """
        match = 'fwhm must be positive'
        with pytest.raises(ValueError, match=match):
            _StarFinderKernel(fwhm=-1)

    def test_normalize_zerosum_false(self):
        """
        Test kernel with normalize_zerosum=False.
        """
        kernel = _StarFinderKernel(fwhm=2.0, normalize_zerosum=False)
        # without zero-sum normalization, the kernel sums to a positive value
        assert kernel.data.sum() > 0

    @pytest.mark.parametrize(('ratio', 'theta'), [
        (0.5, 0.0),
        (0.8, 45.0),
        (1.0, 90.0),
        (0.3, 120.0),
    ])
    def test_elliptical_kernel(self, ratio, theta):
        """
        Test kernel with various ratio and theta values.
        """
        kernel = _StarFinderKernel(fwhm=3.0, ratio=ratio, theta=theta)
        assert kernel.data.shape[0] >= 5
        assert kernel.data.shape[1] >= 5
        # zero-sum kernel
        assert abs(kernel.data.sum()) < 1.0e-10
        # check stored attributes
        assert kernel.ratio == ratio
        assert kernel.theta == theta

    def test_repr(self):
        """
        Test the __repr__ of _StarFinderKernel.
        """
        kernel = _StarFinderKernel(fwhm=3.0, ratio=0.5, theta=30.0)
        r = repr(kernel)
        assert '_StarFinderKernel(' in r
        assert 'fwhm=3.0' in r
        assert 'ratio=0.5' in r
        assert 'theta=30.0' in r
        assert 'sigma_radius=1.5' in r

    def test_str(self):
        """
        Test the __str__ of _StarFinderKernel.
        """
        kernel = _StarFinderKernel(fwhm=3.0, ratio=0.5, theta=30.0)
        s = str(kernel)
        assert 'photutils.detection.core._StarFinderKernel' in s
        assert 'fwhm: 3.0' in s
        assert 'ratio: 0.5' in s
        assert 'theta: 30.0' in s
        assert 'sigma_radius: 1.5' in s


class TestValidateBrightest:
    """
    Parametrized tests for the _validate_brightest function.
    """

    @pytest.mark.parametrize('brightest', [-1, -0.5, -100])
    def test_brightest_negative(self, brightest):
        """
        Test that negative brightest values raise ValueError.
        """
        match = 'brightest must be > 0'
        with pytest.raises(ValueError, match=match):
            _validate_brightest(brightest)

    def test_brightest_zero(self):
        """
        Test that brightest=0 raises ValueError.
        """
        match = 'brightest must be > 0'
        with pytest.raises(ValueError, match=match):
            _validate_brightest(0)

    @pytest.mark.parametrize('brightest', [3.1, 2.5, 1.9])
    def test_brightest_not_integer(self, brightest):
        """
        Test that non-integer brightest values raise ValueError.
        """
        match = 'brightest must be an integer'
        with pytest.raises(ValueError, match=match):
            _validate_brightest(brightest)

    @pytest.mark.parametrize('brightest', [1, 5, 100])
    def test_brightest_valid(self, brightest):
        """
        Test that valid brightest values are returned unchanged.
        """
        assert _validate_brightest(brightest) == brightest

    def test_brightest_none(self):
        """
        Test that None is a valid brightest value.
        """
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
    """
    Tests for the StarFinderCatalogBase base-class methods.
    """

    def test_get_init_attributes(self):
        """
        Test base _get_init_attributes returns expected tuple.
        """
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
        """
        Test that to_table raises when default_columns is not set.
        """
        catalog_cls = _make_minimal_catalog_class()
        data = np.zeros((11, 11))
        data[5, 5] = 10.0
        kernel = np.ones((3, 3))
        xypos = np.array([[5, 5]])
        cat = catalog_cls(data, xypos, kernel)
        match = 'default_columns attribute is not set'
        with pytest.raises(AttributeError, match=match):
            cat.to_table()

    def test_getitem_integer_index(self):
        """
        Test indexing the catalog with an integer index.
        """
        catalog_cls = _make_minimal_catalog_class()
        data = np.zeros((11, 11))
        data[3, 3] = 10.0
        data[7, 7] = 20.0
        kernel = np.ones((3, 3))
        xypos = np.array([[3, 3], [7, 7]])
        cat = catalog_cls(data, xypos, kernel)
        assert len(cat) == 2

        sub = cat[0]
        assert len(sub) == 1
        assert sub.xypos[0, 0] == 3

    def test_getitem_slice(self):
        """
        Test slicing the catalog.
        """
        catalog_cls = _make_minimal_catalog_class()
        data = np.zeros((11, 11))
        data[3, 3] = 10.0
        data[5, 5] = 15.0
        data[7, 7] = 20.0
        kernel = np.ones((3, 3))
        xypos = np.array([[3, 3], [5, 5], [7, 7]])
        cat = catalog_cls(data, xypos, kernel)
        assert len(cat) == 3

        sub = cat[1:]
        assert len(sub) == 2
        assert sub.xypos[0, 0] == 5
        assert sub.xypos[1, 0] == 7

    def test_getitem_fancy_index(self):
        """
        Test indexing with a boolean mask (fancy indexing).
        """
        catalog_cls = _make_minimal_catalog_class()
        data = np.zeros((11, 11))
        data[3, 3] = 10.0
        data[5, 5] = 15.0
        data[7, 7] = 20.0
        kernel = np.ones((3, 3))
        xypos = np.array([[3, 3], [5, 5], [7, 7]])
        cat = catalog_cls(data, xypos, kernel)

        # Force evaluation of a lazyproperty before slicing
        _ = cat.flux

        mask = np.array([True, False, True])
        sub = cat[mask]
        assert len(sub) == 2
        assert sub.xypos[0, 0] == 3
        assert sub.xypos[1, 0] == 7

    def test_roundness(self):
        """
        Test roundness computed on a symmetric source via base class.
        """
        catalog_cls = _make_minimal_catalog_class()
        data = np.zeros((11, 11))
        data[5, 5] = 100.0
        data[4, 5] = 50.0
        data[6, 5] = 50.0
        data[5, 4] = 50.0
        data[5, 6] = 50.0
        kernel = np.ones((3, 3))
        xypos = np.array([[5, 5]])
        cat = catalog_cls(data, xypos, kernel)
        # roundness should be finite for a well-defined source
        assert np.isfinite(cat.roundness[0])

    def test_repr(self):
        """
        Test the __repr__ of StarFinderCatalogBase subclass.
        """
        catalog_cls = _make_minimal_catalog_class()
        data = np.zeros((11, 11))
        data[5, 5] = 10.0
        kernel = np.ones((3, 3))
        xypos = np.array([[5, 5], [3, 3]])
        cat = catalog_cls(data, xypos, kernel)
        r = repr(cat)
        assert '_MinimalCatalog(' in r
        assert 'nsources=2' in r

    def test_str(self):
        """
        Test the __str__ of StarFinderCatalogBase subclass.
        """
        catalog_cls = _make_minimal_catalog_class()
        data = np.zeros((11, 11))
        data[5, 5] = 10.0
        kernel = np.ones((3, 3))
        xypos = np.array([[5, 5]])
        cat = catalog_cls(data, xypos, kernel)
        s = str(cat)
        assert '_MinimalCatalog' in s
        assert 'nsources: 1' in s
