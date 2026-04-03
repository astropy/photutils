# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the starfinder module.
"""

import astropy.units as u
import numpy as np
import pytest
from astropy.table import Table
from astropy.utils.exceptions import AstropyDeprecationWarning
from numpy.testing import assert_array_equal, assert_equal

from photutils.detection import StarFinder
from photutils.utils.exceptions import NoDetectionsWarning


class TestStarFinder:
    """
    Test the StarFinder class.
    """

    def test_find(self, data, kernel):
        """
        Test basic source detection and unit handling.
        """
        finder1 = StarFinder(1, kernel)
        finder2 = StarFinder(10, kernel)
        tbl1 = finder1(data)
        tbl2 = finder2(data)
        assert isinstance(tbl1, Table)
        assert len(tbl1) == 25
        assert len(tbl2) == 9
        assert tbl1['orientation'].unit == u.deg

        # Test with units
        unit = u.Jy
        finder3 = StarFinder(1 * unit, kernel)
        tbl3 = finder3(data << unit)
        assert isinstance(tbl3, Table)
        assert len(tbl3) == 25
        assert tbl3['flux'].unit == unit
        assert tbl3['max_value'].unit == unit
        assert tbl3['orientation'].unit == u.deg
        for col in tbl3.colnames:
            if col not in ('flux', 'max_value'):
                assert_equal(tbl3[col], tbl1[col])

    def test_inputs(self, kernel):
        """
        Test that invalid inputs raise appropriate errors.
        """
        match = 'min_separation must be >= 0'
        with pytest.raises(ValueError, match=match):
            StarFinder(1, kernel, min_separation=-1)
        match = 'n_brightest must be > 0'
        with pytest.raises(ValueError, match=match):
            StarFinder(1, kernel, n_brightest=-1)
        match = 'n_brightest must be an integer'
        with pytest.raises(ValueError, match=match):
            StarFinder(1, kernel, n_brightest=3.1)

    @pytest.mark.parametrize('ndim', [1, 3])
    def test_kernel_not_2d(self, ndim):
        """
        Test that non-2D kernels raise ValueError.
        """
        bad_kernel = np.ones(5) if ndim == 1 else np.ones((3, 3, 3))
        match = 'kernel must be a 2D array'
        with pytest.raises(ValueError, match=match):
            StarFinder(1, bad_kernel)

    def test_nosources(self, data, kernel):
        """
        Test that no sources returns None with a warning.
        """
        match = 'No sources were found'
        finder = StarFinder(100, kernel)
        with pytest.warns(NoDetectionsWarning, match=match):
            tbl = finder(data)
        assert tbl is None

        data = np.ones((5, 5))
        data[2, 2] = 10.0
        finder = StarFinder(1, kernel)
        with pytest.warns(NoDetectionsWarning, match=match):
            tbl = finder(-data)
        assert tbl is None

    def test_exclude_border(self, data, kernel):
        """
        Test that border sources are excluded.
        """
        data = np.zeros((12, 12))
        data[0:2, 0:2] = 1
        data[9:12, 9:12] = 1
        kernel = np.ones((3, 3))

        finder0 = StarFinder(1, kernel, exclude_border=False)
        finder1 = StarFinder(1, kernel, exclude_border=True)
        tbl0 = finder0(data)
        tbl1 = finder1(data)
        assert len(tbl0) > len(tbl1)

    def test_mask(self, data, kernel):
        """
        Test source detection with a mask.
        """
        starfinder = StarFinder(1, kernel)
        mask = np.zeros(data.shape, dtype=bool)
        mask[0:50] = True
        tbl1 = starfinder(data)
        tbl2 = starfinder(data, mask=mask)
        assert len(tbl1) == 25
        assert len(tbl2) == 13
        assert min(tbl2['y_centroid']) > 50

    def test_mask_int(self, data, kernel):
        """
        Test that an integer mask gives the same result as a boolean
        mask.
        """
        starfinder = StarFinder(1, kernel)
        bool_mask = np.zeros(data.shape, dtype=bool)
        bool_mask[0:50] = True
        int_mask = bool_mask.astype(int)

        tbl_bool = starfinder(data, mask=bool_mask)
        tbl_int = starfinder(data, mask=int_mask)
        assert_array_equal(tbl_bool, tbl_int)

    def test_min_separation(self, data, kernel):
        """
        Test the min_separation parameter.
        """
        finder1 = StarFinder(1, kernel, min_separation=0)
        finder2 = StarFinder(1, kernel, min_separation=10)
        tbl1 = finder1(data)
        tbl2 = finder2(data)
        assert len(tbl1) == 25
        assert len(tbl2) == 20

    def test_min_separation_default(self, kernel):
        """
        Test that the default min_separation (None) gives
        2.5 * (min(kernel.shape) // 2).
        """
        finder = StarFinder(1, kernel)
        assert finder.min_separation == 2.5 * (min(kernel.shape) // 2)

        # Non-square kernel
        rect_kernel = np.ones((3, 7))
        finder2 = StarFinder(1, rect_kernel)
        assert finder2.min_separation == 2.5 * (3 // 2)

        # Previous default behavior
        finder_old = StarFinder(1, kernel, min_separation=5)
        assert finder_old.min_separation == 5

    def test_n_brightest(self, data, kernel):
        """
        Test the n_brightest parameter.
        """
        finder = StarFinder(1, kernel, n_brightest=10)
        tbl = finder(data)
        assert len(tbl) == 10
        fluxes = tbl['flux']
        assert fluxes[0] == np.max(fluxes)

    def test_peak_max(self, data, kernel):
        """
        Test the peak_max parameter.
        """
        finder1 = StarFinder(1, kernel, peak_max=None)
        finder2 = StarFinder(1, kernel, peak_max=11)
        tbl1 = finder1(data)
        tbl2 = finder2(data)
        assert len(tbl1) == 25
        assert len(tbl2) == 16

        match = 'Sources were found, but none pass'
        starfinder = StarFinder(10, kernel, peak_max=5)
        with pytest.warns(NoDetectionsWarning, match=match):
            tbl = starfinder(data)
        assert tbl is None

    def test_peak_max_limit(self):
        """
        Test that the peak_max limit is inclusive.
        """
        data = np.zeros((11, 11))
        x = 5
        y = 6
        kernel = np.array([[0.1, 0.6, 0.1],
                           [0.6, 0.8, 0.6],
                           [0.1, 0.6, 0.1]])
        data[y - 1: y + 2, x - 1: x + 2] = kernel

        finder = StarFinder(threshold=0, kernel=kernel, peak_max=0.8)
        tbl = finder.find_stars(data)

        assert len(tbl) == 1
        assert tbl[0]['max_value'] == 0.8

    def test_single_detected_source(self, data, kernel):
        """
        Test detection and slicing with a single source.
        """
        finder = StarFinder(11.5, kernel, n_brightest=1)
        mask = np.zeros(data.shape, dtype=bool)
        mask[0:50] = True
        tbl = finder(data, mask=mask)
        assert len(tbl) == 1

        # Test slicing with scalar catalog to improve coverage
        cat = finder._get_raw_catalog(data, mask=mask)
        assert cat.isscalar
        flux = cat.flux[0]  # evaluate the flux so it can be sliced
        assert cat[0].flux == flux

    def test_repeated_calls(self, data, kernel):
        """
        Test that calling find_stars twice gives identical results.
        """
        finder = StarFinder(1, kernel)
        tbl1 = finder(data)
        tbl2 = finder(data)
        assert len(tbl1) == len(tbl2)
        for col in tbl1.colnames:
            assert_equal(tbl1[col], tbl2[col])

    def test_quantity_units_mismatch(self, kernel):
        """
        Test that mismatched data/threshold units raise an error.
        """
        data = np.ones((11, 11))
        finder = StarFinder(1 * u.Jy, kernel)
        match = 'must all have the same units'
        with pytest.raises(ValueError, match=match):
            finder(data << u.m)

    def test_quantity_with_negatives(self, data, kernel):
        """
        Test detection with Quantity data containing negatives.
        """
        unit = u.Jy
        data_neg = (data - 5.0) << unit
        finder = StarFinder(1 * unit, kernel)
        tbl = finder(data_neg)
        assert isinstance(tbl, Table)
        assert len(tbl) > 0
        assert tbl['flux'].unit == unit

    def test_data_not_mutated(self, data, kernel):
        """
        Test that input data is not mutated by find_stars.
        """
        data = data - 5.0  # create some negative pixel values
        data_copy = data.copy()
        finder = StarFinder(1, kernel)
        finder(data)
        assert_equal(data, data_copy)

    def test_data_not_mutated_with_mask(self, data, kernel):
        """
        Test that input data is not mutated when a mask is used.
        """
        data = data - 5.0
        data_copy = data.copy()
        mask = np.zeros(data.shape, dtype=bool)
        mask[0:50] = True
        finder = StarFinder(1, kernel)
        finder(data, mask=mask)
        assert_equal(data, data_copy)

    def test_repr(self, kernel):
        """
        Test the __repr__ of StarFinder.
        """
        finder = StarFinder(threshold=5.0, kernel=kernel)
        repr_ = repr(finder)
        assert 'StarFinder(' in repr_
        assert 'threshold=5.0' in repr_
        assert '<array; shape=' in repr_
        assert 'n_brightest=None' in repr_

    def test_str(self, kernel):
        """
        Test the __str__ of StarFinder.
        """
        finder = StarFinder(threshold=5.0, kernel=kernel)
        str_ = str(finder)
        assert 'StarFinder' in str_
        assert 'threshold: 5.0' in str_
        assert '<array; shape=' in str_

    def test_threshold_2d_uniform(self, data, kernel):
        """
        Test that a uniform 2D threshold gives the same results
        as a scalar threshold.
        """
        threshold = 1.0
        finder_scalar = StarFinder(threshold, kernel)
        finder_2d = StarFinder(np.full(data.shape, threshold), kernel)
        tbl_scalar = finder_scalar(data)
        tbl_2d = finder_2d(data)
        assert_array_equal(tbl_scalar, tbl_2d)

    def test_threshold_2d_varying(self, data, kernel):
        """
        Test that a varying 2D threshold detects fewer sources in
        regions with a higher threshold.
        """
        threshold_low = 1.0
        threshold_high = 100.0
        threshold_2d = np.full(data.shape, threshold_low)
        threshold_2d[0:50, :] = threshold_high

        finder_low = StarFinder(threshold_low, kernel)
        finder_2d = StarFinder(threshold_2d, kernel)

        tbl_low = finder_low(data)
        tbl_2d = finder_2d(data)
        assert len(tbl_low) > len(tbl_2d)
        # All 2D sources should be in the lower half
        assert all(tbl_2d['y_centroid'] >= 50)

    def test_threshold_2d_repr(self, kernel):
        """
        Test repr with a 2D threshold array.
        """
        threshold = np.ones((10, 10))
        finder = StarFinder(threshold=threshold, kernel=kernel)
        assert '<array; shape=(10, 10)>' in repr(finder)
        assert '<array; shape=(10, 10)>' in str(finder)

    def test_threshold_2d_with_units(self, data, kernel):
        """
        Test that a 2D threshold with units works correctly.
        """
        unit = u.Jy
        threshold = 1.0
        threshold_2d = np.full(data.shape, threshold) * unit
        finder = StarFinder(threshold_2d, kernel)
        tbl = finder(data << unit)
        assert len(tbl) > 0

    def test_deprecated_brightest(self, kernel):
        """
        Test that the deprecated 'brightest' keyword raises a warning
        and still works.
        """
        match = "'brightest' was deprecated"
        with pytest.warns(AstropyDeprecationWarning, match=match):
            finder = StarFinder(threshold=5.0, kernel=kernel, brightest=5)
        assert finder.n_brightest == 5

    def test_deprecated_peakmax(self, kernel):
        """
        Test that the deprecated 'peakmax' keyword raises a warning
        and still works.
        """
        match = "'peakmax' was deprecated"
        with pytest.warns(AstropyDeprecationWarning, match=match):
            finder = StarFinder(threshold=5.0, kernel=kernel, peakmax=100.0)
        assert finder.peak_max == 100.0
