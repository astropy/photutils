# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the photutils.detection.core module.
"""

import numpy as np
import pytest
from astropy.utils.exceptions import AstropyDeprecationWarning

from photutils.detection import DAOStarFinder
from photutils.detection.core import (_DEPR_DEFAULT, StarFinderCatalogBase,
                                      _StarFinderKernel, _validate_n_brightest)


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

    def test_fwhm_nonscalar(self):
        """
        Test that a non-scalar fwhm raises a TypeError.
        """
        match = 'fwhm must be a scalar value'
        with pytest.raises(TypeError, match=match):
            _StarFinderKernel(fwhm=np.array([3.0]))

    def test_normalize_zerosum_false(self):
        """
        Test kernel with normalize_zerosum=False.
        """
        kernel = _StarFinderKernel(fwhm=2.0, normalize_zerosum=False)
        # Without zero-sum normalization, the kernel sums to a positive
        # value
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
        # Zero-sum kernel
        assert abs(kernel.data.sum()) < 1.0e-10
        # Check stored attributes
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

    @pytest.mark.parametrize(('theta', 'expected'), [
        (400.0, 40.0),
        (-30.0, 330.0),
        (360.0, 0.0),
        (0.0, 0.0),
    ])
    def test_theta_normalization(self, theta, expected):
        """
        Test that theta values are normalized to [0, 360).
        """
        kernel = _StarFinderKernel(fwhm=3.0, ratio=0.5, theta=theta)
        assert kernel.theta == expected

    @pytest.mark.parametrize('ratio', [0, -0.5, 1.5])
    def test_invalid_ratio(self, ratio):
        """
        Test that invalid ratio values raise ValueError.
        """
        match = 'ratio must be > 0 and <= 1.0'
        with pytest.raises(ValueError, match=match):
            _StarFinderKernel(fwhm=3.0, ratio=ratio)

    @pytest.mark.parametrize('sigma_radius', [0, -1])
    def test_invalid_sigma_radius(self, sigma_radius):
        """
        Test that non-positive sigma_radius raises ValueError.
        """
        match = 'sigma_radius must be positive'
        with pytest.raises(ValueError, match=match):
            _StarFinderKernel(fwhm=3.0, sigma_radius=sigma_radius)


class TestValidateNBrightest:
    """
    Parametrized tests for the _validate_n_brightest function.
    """

    @pytest.mark.parametrize('n_brightest', [-1, -0.5, -100])
    def test_n_brightest_negative(self, n_brightest):
        """
        Test that negative n_brightest values raise ValueError.
        """
        match = 'n_brightest must be > 0'
        with pytest.raises(ValueError, match=match):
            _validate_n_brightest(n_brightest)

    def test_n_brightest_zero(self):
        """
        Test that n_brightest=0 raises ValueError.
        """
        match = 'n_brightest must be > 0'
        with pytest.raises(ValueError, match=match):
            _validate_n_brightest(0)

    @pytest.mark.parametrize('n_brightest', [3.1, 2.5, 1.9])
    def test_n_brightest_not_integer(self, n_brightest):
        """
        Test that non-integer n_brightest values raise ValueError.
        """
        match = 'n_brightest must be an integer'
        with pytest.raises(ValueError, match=match):
            _validate_n_brightest(n_brightest)

    @pytest.mark.parametrize('n_brightest', [1, 5, 100])
    def test_n_brightest_valid(self, n_brightest):
        """
        Test that valid n_brightest values are returned unchanged.
        """
        assert _validate_n_brightest(n_brightest) == n_brightest

    def test_n_brightest_none(self):
        """
        Test that None is a valid n_brightest value.
        """
        assert _validate_n_brightest(None) is None

    @pytest.mark.parametrize('n_brightest', [True, False])
    def test_n_brightest_bool(self, n_brightest):
        """
        Test that boolean n_brightest values raise TypeError.
        """
        match = 'n_brightest must be an integer'
        with pytest.raises(TypeError, match=match):
            _validate_n_brightest(n_brightest)


def _make_minimal_catalog_class():
    """
    Create a minimal concrete subclass of StarFinderCatalogBase that
    does NOT override ``_get_init_attributes`` and does NOT set
    ``default_columns``, so the base-class implementations can be
    tested.
    """

    class _MinimalCatalog(StarFinderCatalogBase):

        @property
        def x_centroid(self):
            return self.cutout_x_centroid

        @property
        def y_centroid(self):
            return self.cutout_y_centroid

        def apply_filters(self):
            return self

    return _MinimalCatalog


@pytest.fixture(name='minimal_catalog_cls')
def fixture_minimal_catalog_cls():
    """
    Fixture that provides a minimal concrete subclass of
    StarFinderCatalogBase.
    """
    return _make_minimal_catalog_class()


class TestStarFinderCatalogBase:
    """
    Tests for the StarFinderCatalogBase base-class methods.
    """

    def test_get_init_attributes(self, minimal_catalog_cls):
        """
        Test base _get_init_attributes returns expected tuple.
        """
        data = np.zeros((11, 11))
        data[5, 5] = 10.0
        kernel = np.ones((3, 3))
        xypos = np.array([[5, 5]])
        cat = minimal_catalog_cls(data, xypos, kernel)
        expected = ('data', 'unit', 'kernel', 'n_brightest', 'peak_max',
                    'cutout_shape', 'default_columns')
        assert cat._get_init_attributes() == expected

    def test_lazyproperties_class_cache(self, minimal_catalog_cls):
        """
        Test that _lazyproperties is cached on the class and shared
        across instances.
        """
        data = np.zeros((11, 11))
        data[5, 5] = 10.0
        kernel = np.ones((3, 3))
        xypos = np.array([[5, 5]])
        cat1 = minimal_catalog_cls(data, xypos, kernel)
        cat2 = minimal_catalog_cls(data, xypos, kernel)
        result1 = cat1._lazyproperties
        result2 = cat2._lazyproperties
        assert result1 is result2

    def test_to_table_missing_default_columns(self, minimal_catalog_cls):
        """
        Test that to_table raises when default_columns is not set.
        """
        data = np.zeros((11, 11))
        data[5, 5] = 10.0
        kernel = np.ones((3, 3))
        xypos = np.array([[5, 5]])
        cat = minimal_catalog_cls(data, xypos, kernel)
        match = 'default_columns attribute is not set'
        with pytest.raises(AttributeError, match=match):
            cat.to_table()

    def test_to_table_explicit_columns(self, minimal_catalog_cls):
        """
        Test that to_table works with explicit column names.
        """
        data = np.zeros((11, 11))
        data[5, 5] = 10.0
        kernel = np.ones((3, 3))
        xypos = np.array([[5, 5]])
        cat = minimal_catalog_cls(data, xypos, kernel)
        columns = ('id', 'x_centroid', 'y_centroid')
        tbl = cat.to_table(columns=columns)
        assert len(tbl) == 1
        assert tbl.colnames == list(columns)

    def test_getitem_integer_index(self, minimal_catalog_cls):
        """
        Test indexing the catalog with an integer index.
        """
        data = np.zeros((11, 11))
        data[3, 3] = 10.0
        data[7, 7] = 20.0
        kernel = np.ones((3, 3))
        xypos = np.array([[3, 3], [7, 7]])
        cat = minimal_catalog_cls(data, xypos, kernel)
        assert len(cat) == 2

        sub = cat[0]
        assert len(sub) == 1
        assert sub.xypos[0, 0] == 3

    def test_getitem_slice(self, minimal_catalog_cls):
        """
        Test slicing the catalog.
        """
        data = np.zeros((11, 11))
        data[3, 3] = 10.0
        data[5, 5] = 15.0
        data[7, 7] = 20.0
        kernel = np.ones((3, 3))
        xypos = np.array([[3, 3], [5, 5], [7, 7]])
        cat = minimal_catalog_cls(data, xypos, kernel)
        assert len(cat) == 3

        sub = cat[1:]
        assert len(sub) == 2
        assert sub.xypos[0, 0] == 5
        assert sub.xypos[1, 0] == 7

    def test_getitem_fancy_index(self, minimal_catalog_cls):
        """
        Test indexing with a boolean mask (fancy indexing).
        """
        data = np.zeros((11, 11))
        data[3, 3] = 10.0
        data[5, 5] = 15.0
        data[7, 7] = 20.0
        kernel = np.ones((3, 3))
        xypos = np.array([[3, 3], [5, 5], [7, 7]])
        cat = minimal_catalog_cls(data, xypos, kernel)

        # Force evaluation of a lazyproperty before slicing
        _ = cat.flux

        mask = np.array([True, False, True])
        sub = cat[mask]
        assert len(sub) == 2
        assert sub.xypos[0, 0] == 3
        assert sub.xypos[1, 0] == 7

    def test_roundness(self, minimal_catalog_cls):
        """
        Test roundness computed on a symmetric source via base class.
        """
        data = np.zeros((11, 11))
        data[5, 5] = 100.0
        data[4, 5] = 50.0
        data[6, 5] = 50.0
        data[5, 4] = 50.0
        data[5, 6] = 50.0
        kernel = np.ones((3, 3))
        xypos = np.array([[5, 5]])
        cat = minimal_catalog_cls(data, xypos, kernel)
        # roundness should be finite for a well-defined source
        assert np.isfinite(cat.roundness[0])

    def test_repr(self, minimal_catalog_cls):
        """
        Test the __repr__ of StarFinderCatalogBase subclass.
        """
        data = np.zeros((11, 11))
        data[5, 5] = 10.0
        kernel = np.ones((3, 3))
        xypos = np.array([[5, 5], [3, 3]])
        cat = minimal_catalog_cls(data, xypos, kernel)
        r = repr(cat)
        assert '_MinimalCatalog(' in r
        assert 'nsources=2' in r

    def test_str(self, minimal_catalog_cls):
        """
        Test the __str__ of StarFinderCatalogBase subclass.
        """
        data = np.zeros((11, 11))
        data[5, 5] = 10.0
        kernel = np.ones((3, 3))
        xypos = np.array([[5, 5]])
        cat = minimal_catalog_cls(data, xypos, kernel)
        s = str(cat)
        assert '_MinimalCatalog' in s
        assert 'nsources: 1' in s

    def test_make_cutouts_partial_overlap(self, minimal_catalog_cls):
        """
        Test that make_cutouts pads with zeros for sources at image
        edges that only partially overlap the data.
        """
        data = np.ones((10, 10)) * 5.0
        kernel = np.ones((5, 5))
        # Corners and edges: each cutout partially extends outside
        xypos = np.array([[0, 0], [9, 9], [0, 9], [9, 0]])
        cat = minimal_catalog_cls(data, xypos, kernel)
        cutouts = cat.make_cutouts(data)

        assert cutouts.shape == (4, 5, 5)

        # Corner (0,0): only bottom-right 3x3 quadrant is inside image
        c00 = cutouts[0]
        assert np.all(c00[:2, :] == 0.0)  # top rows outside
        assert np.all(c00[:, :2] == 0.0)  # left cols outside
        assert np.all(c00[2:, 2:] == 5.0)  # bottom-right inside

        # Corner (9,9): only top-left 3x3 quadrant is inside image
        c99 = cutouts[1]
        assert np.all(c99[3:, :] == 0.0)  # bottom rows outside
        assert np.all(c99[:, 3:] == 0.0)  # right cols outside
        assert np.all(c99[:3, :3] == 5.0)  # top-left inside

    def test_make_cutouts_fully_inside(self, minimal_catalog_cls):
        """
        Test that make_cutouts returns exact data for a fully inside
        source.
        """
        data = np.arange(100, dtype=float).reshape(10, 10)
        kernel = np.ones((3, 3))
        xypos = np.array([[5, 5]])
        cat = minimal_catalog_cls(data, xypos, kernel)
        cutouts = cat.make_cutouts(data)

        assert cutouts.shape == (1, 3, 3)
        expected = data[4:7, 4:7]
        np.testing.assert_array_equal(cutouts[0], expected)

    def test_select_brightest(self, minimal_catalog_cls):
        """
        Test select_brightest selects the top sources by flux.
        """
        data = np.zeros((21, 21))
        data[5, 5] = 10.0
        data[10, 10] = 50.0
        data[15, 15] = 30.0
        kernel = np.ones((3, 3))
        xypos = np.array([[5, 5], [10, 10], [15, 15]])
        cat = minimal_catalog_cls(data, xypos, kernel, n_brightest=2)
        newcat = cat.select_brightest()
        assert len(newcat) == 2
        # Brightest first
        assert newcat.flux[0] >= newcat.flux[1]

    def test_select_brightest_none(self, minimal_catalog_cls):
        """
        Test that select_brightest with n_brightest=None keeps all sources.
        """
        data = np.zeros((21, 21))
        data[5, 5] = 10.0
        data[10, 10] = 50.0
        data[15, 15] = 30.0
        kernel = np.ones((3, 3))
        xypos = np.array([[5, 5], [10, 10], [15, 15]])
        cat = minimal_catalog_cls(data, xypos, kernel, n_brightest=None)
        newcat = cat.select_brightest()
        assert len(newcat) == 3

    def test_reset_ids(self, minimal_catalog_cls):
        """
        Test that reset_ids renumbers the catalog consecutively.
        """
        data = np.zeros((21, 21))
        data[5, 5] = 10.0
        data[10, 10] = 50.0
        data[15, 15] = 30.0
        kernel = np.ones((3, 3))
        xypos = np.array([[5, 5], [10, 10], [15, 15]])
        cat = minimal_catalog_cls(data, xypos, kernel)
        # Slice to drop the first source
        sub = cat[1:]
        assert sub.id[0] == 2
        sub.reset_ids()
        np.testing.assert_array_equal(sub.id, [1, 2])

    def test_apply_all_filters(self, minimal_catalog_cls):
        """
        Test apply_all_filters chains apply_filters, select_brightest,
        and reset_ids.
        """
        data = np.zeros((21, 21))
        data[5, 5] = 10.0
        data[10, 10] = 50.0
        data[15, 15] = 30.0
        kernel = np.ones((3, 3))
        xypos = np.array([[5, 5], [10, 10], [15, 15]])
        cat = minimal_catalog_cls(data, xypos, kernel, n_brightest=2)
        result = cat.apply_all_filters()
        assert result is not None
        assert len(result) == 2
        # IDs should be reset to [1, 2]
        np.testing.assert_array_equal(result.id, [1, 2])

    def test_getitem_negative_index(self, minimal_catalog_cls):
        """
        Test indexing with a negative integer index.
        """
        data = np.zeros((21, 21))
        data[5, 5] = 10.0
        data[10, 10] = 50.0
        data[15, 15] = 30.0
        kernel = np.ones((3, 3))
        xypos = np.array([[5, 5], [10, 10], [15, 15]])
        cat = minimal_catalog_cls(data, xypos, kernel)
        sub = cat[-1]
        assert len(sub) == 1
        assert sub.xypos[0, 0] == 15

    def test_getitem_empty_boolean_mask(self, minimal_catalog_cls):
        """
        Test indexing with an all-False boolean mask.
        """
        data = np.zeros((21, 21))
        data[5, 5] = 10.0
        data[10, 10] = 50.0
        kernel = np.ones((3, 3))
        xypos = np.array([[5, 5], [10, 10]])
        cat = minimal_catalog_cls(data, xypos, kernel)
        mask = np.array([False, False])
        sub = cat[mask]
        assert len(sub) == 0

    def test_getitem_integer_array(self, minimal_catalog_cls):
        """
        Test indexing with an integer array (fancy indexing).
        """
        data = np.zeros((21, 21))
        data[5, 5] = 10.0
        data[10, 10] = 50.0
        data[15, 15] = 30.0
        kernel = np.ones((3, 3))
        xypos = np.array([[5, 5], [10, 10], [15, 15]])
        cat = minimal_catalog_cls(data, xypos, kernel)
        idx = np.array([2, 0])
        sub = cat[idx]
        assert len(sub) == 2
        assert sub.xypos[0, 0] == 15
        assert sub.xypos[1, 0] == 5

    def test_filter_bounds_none_range(self, minimal_catalog_cls):
        """
        Test that _filter_bounds skips filtering when a range is None.
        """
        data = np.zeros((21, 21))
        data[5, 5] = 10.0
        data[10, 10] = 50.0
        data[15, 15] = 30.0
        kernel = np.ones((3, 3))
        xypos = np.array([[5, 5], [10, 10], [15, 15]])
        cat = minimal_catalog_cls(data, xypos, kernel)
        bounds = [('flux', None)]
        result = cat._filter_bounds(bounds)
        assert len(result) == 3

    def test_filter_bounds_initial_mask(self, minimal_catalog_cls):
        """
        Test _filter_bounds with an initial_mask.
        """
        data = np.zeros((21, 21))
        data[5, 5] = 10.0
        data[10, 10] = 50.0
        data[15, 15] = 30.0
        kernel = np.ones((3, 3))
        xypos = np.array([[5, 5], [10, 10], [15, 15]])
        cat = minimal_catalog_cls(data, xypos, kernel)
        # Pre-exclude the first source
        initial_mask = np.array([False, True, True])
        result = cat._filter_bounds([], initial_mask=initial_mask)
        assert len(result) == 2

    def test_default_columns_preserved_on_slice(self, minimal_catalog_cls):
        """
        Test that default_columns is preserved when slicing.
        """
        data = np.zeros((21, 21))
        data[5, 5] = 10.0
        data[10, 10] = 50.0
        kernel = np.ones((3, 3))
        xypos = np.array([[5, 5], [10, 10]])
        cat = minimal_catalog_cls(data, xypos, kernel)
        cat.default_columns = ('id', 'x_centroid', 'y_centroid')
        sub = cat[0]
        assert sub.default_columns == ('id', 'x_centroid', 'y_centroid')


class TestStarFinderBaseCall:
    """
    Test that StarFinderBase.__call__ delegates to find_stars.
    """

    def test_call_delegates_to_find_stars(self, data):
        """
        Test that __call__ returns the same result as find_stars.
        """
        finder = DAOStarFinder(threshold=5.0, fwhm=2.0)
        tbl_call = finder(data)
        tbl_find = finder.find_stars(data)
        assert len(tbl_call) == len(tbl_find)
        for col in tbl_call.colnames:
            np.testing.assert_array_equal(tbl_call[col], tbl_find[col])


def test_deprecated_attr(data):
    """
    Test that accessing the deprecated attribute on the
    StarFinderCatalogBase raises an warning.
    """
    finder = DAOStarFinder(threshold=5.0, fwhm=2.0)
    cat = finder._get_raw_catalog(data)
    match = 'attribute was deprecated'
    with pytest.warns(AstropyDeprecationWarning, match=match):
        _ = cat.xcentroid


def test_deprecated_default():
    """
    Test repr for _DeprecatedDefault.
    """
    default = _DEPR_DEFAULT
    result = '<deprecated>'
    assert repr(default) == result
    assert str(default) == result
