# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the daofinder module.
"""

import astropy.units as u
import numpy as np
import pytest
from astropy.utils.exceptions import AstropyDeprecationWarning
from numpy.testing import assert_array_equal

from photutils.detection.daofinder import DAOStarFinder
from photutils.utils.exceptions import NoDetectionsWarning


class TestDAOStarFinder:
    """
    Test the DAOStarFinder class.
    """

    def test_find(self, data):
        """
        Test basic source detection and unit handling.
        """
        units = u.Jy
        threshold = 5.0
        fwhm = 1.0
        finder0 = DAOStarFinder(threshold, fwhm)
        finder1 = DAOStarFinder(threshold * units, fwhm)

        tbl0 = finder0(data)
        tbl1 = finder1(data << units)
        assert_array_equal(tbl0, tbl1)

        assert np.min(tbl0['flux']) > 150

        # Test that sources are returned with threshold = 0
        finder = DAOStarFinder(0, fwhm)
        tbl = finder(data)
        assert len(tbl) == 25

    def test_inputs(self):
        """
        Test that invalid inputs raise appropriate errors.
        """
        match = 'fwhm must be a scalar value'
        with pytest.raises(TypeError, match=match):
            DAOStarFinder(threshold=3.0, fwhm=np.ones((2, 2)))

        match = 'fwhm must be positive'
        for fwhm in (-10, 0):
            with pytest.raises(ValueError, match=match):
                DAOStarFinder(threshold=3.0, fwhm=fwhm)

        match = 'ratio must be > 0 and <= 1.0'
        with pytest.raises(ValueError, match=match):
            DAOStarFinder(threshold=3.0, fwhm=2, ratio=-10)

        match = 'sigma_radius must be positive'
        with pytest.raises(ValueError, match=match):
            DAOStarFinder(threshold=3.0, fwhm=2, sigma_radius=-10)

        match = 'n_brightest must be > 0'
        with pytest.raises(ValueError, match=match):
            DAOStarFinder(threshold=10, fwhm=1.5, n_brightest=-1)

        match = 'n_brightest must be an integer'
        with pytest.raises(ValueError, match=match):
            DAOStarFinder(threshold=10, fwhm=1.5, n_brightest=3.1)

        xycoords = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
        match = 'xycoords must be shaped as an Nx2 array'
        with pytest.raises(ValueError, match=match):
            DAOStarFinder(threshold=10, fwhm=1.5, xycoords=xycoords)

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
        finder = DAOStarFinder(threshold=5.0, fwhm=3.0, theta=theta)
        assert finder.theta == expected

    def test_nosources(self, data):
        """
        Test that no sources returns None with a warning.
        """
        match = 'No sources were found'
        finder = DAOStarFinder(threshold=100, fwhm=2)
        with pytest.warns(NoDetectionsWarning, match=match):
            tbl = finder(data)
        assert tbl is None

        finder = DAOStarFinder(threshold=1, fwhm=2)
        with pytest.warns(NoDetectionsWarning, match=match):
            tbl = finder(-data)
        assert tbl is None

    def test_exclude_border(self):
        """
        Test that border sources are excluded.
        """
        data = np.zeros((9, 9))
        data[0, 0] = 1
        data[2, 2] = 1
        data[4, 4] = 1
        data[6, 6] = 1

        finder0 = DAOStarFinder(threshold=0.1, fwhm=0.5, exclude_border=False)
        finder1 = DAOStarFinder(threshold=0.1, fwhm=0.5, exclude_border=True)
        tbl0 = finder0(data)
        tbl1 = finder1(data)
        assert len(tbl0) > len(tbl1)

    def test_mask(self, data):
        """
        Test source detection with a mask.
        """
        finder = DAOStarFinder(threshold=1.0, fwhm=1.5)
        mask = np.zeros(data.shape, dtype=bool)
        mask[0:50, :] = True
        tbl0 = finder(data)
        tbl1 = finder(data, mask=mask)
        assert len(tbl0) > len(tbl1)

    def test_mask_int(self, data):
        """
        Test that an integer mask gives the same result as a boolean
        mask.
        """
        finder = DAOStarFinder(threshold=1.0, fwhm=1.5)
        bool_mask = np.zeros(data.shape, dtype=bool)
        bool_mask[0:50, :] = True
        int_mask = bool_mask.astype(int)

        tbl_bool = finder(data, mask=bool_mask)
        tbl_int = finder(data, mask=int_mask)
        assert_array_equal(tbl_bool, tbl_int)

    def test_xycoords(self, data):
        """
        Test source detection at specified coordinates.
        """
        finder0 = DAOStarFinder(threshold=8.0, fwhm=2)
        tbl0 = finder0(data)
        xycoords = list(zip(tbl0['x_centroid'],
                            tbl0['y_centroid'], strict=True))
        xycoords = np.round(xycoords).astype(int)

        finder1 = DAOStarFinder(threshold=8.0, fwhm=2, xycoords=xycoords)
        tbl1 = finder1(data)
        assert_array_equal(tbl0, tbl1)

    def test_min_separation(self, data):
        """
        Test the min_separation parameter.
        """
        threshold = 1.0
        fwhm = 1.0
        finder1 = DAOStarFinder(threshold, fwhm)
        tbl1 = finder1(data)
        assert finder1.min_separation == 2.5 * fwhm

        finder2 = DAOStarFinder(threshold, fwhm, min_separation=10.0)
        tbl2 = finder2(data)
        assert len(tbl1) > len(tbl2)

        match = 'min_separation must be >= 0'
        with pytest.raises(ValueError, match=match):
            DAOStarFinder(threshold=10, fwhm=1.5, min_separation=-1.0)

    def test_min_separation_default(self):
        """
        Test that the default min_separation (None) gives 2.5 * fwhm.
        """
        fwhm = 2.0
        finder = DAOStarFinder(threshold=1.0, fwhm=fwhm)
        assert finder.min_separation == 2.5 * fwhm

        finder_old = DAOStarFinder(threshold=1.0, fwhm=fwhm,
                                   min_separation=0)
        assert finder_old.min_separation == 0

    def test_brightest_filtering(self, data):
        """
        Test that only the top brightest sources are selected.
        """
        n_brightest = 10
        finder = DAOStarFinder(threshold=1.0, fwhm=1.5,
                               roundness_range=(-np.inf, np.inf),
                               sharpness_range=(-np.inf, np.inf),
                               n_brightest=n_brightest)
        tbl = finder(data)
        assert len(tbl) == n_brightest

        # Combined with peak_max
        peak_max = 8
        finder = DAOStarFinder(threshold=1.0, fwhm=1.5,
                               roundness_range=(-np.inf, np.inf),
                               sharpness_range=(-np.inf, np.inf),
                               n_brightest=n_brightest,
                               peak_max=peak_max)
        tbl = finder(data)
        assert len(tbl) == 5

    def test_sharpness(self, data):
        """
        Test that no sources pass the sharpness criteria.
        """
        finder = DAOStarFinder(threshold=1, fwhm=1.0,
                               sharpness_range=(1.0, 1.0))
        match = 'Sources were found, but none pass'
        with pytest.warns(NoDetectionsWarning, match=match):
            tbl = finder(data)
        assert tbl is None

    @pytest.mark.parametrize('sharpness_range', [0.5, (0.5,), (1, 2, 3)])
    def test_invalid_sharpness_range(self, sharpness_range):
        match = 'sharpness_range must be a 2-element .* tuple'
        with pytest.raises(ValueError, match=match):
            DAOStarFinder(threshold=1, fwhm=1.0,
                          sharpness_range=sharpness_range)

    def test_sharpness_range_none(self, data):
        """
        Test that sharpness_range=None disables sharpness filtering.
        """
        finder_none = DAOStarFinder(threshold=1, fwhm=1.0,
                                    roundness_range=None,
                                    sharpness_range=None)
        tbl_none = finder_none(data)
        assert tbl_none is not None

        finder_strict = DAOStarFinder(threshold=1, fwhm=1.0,
                                      roundness_range=None,
                                      sharpness_range=(1.0, 1.0))
        match = 'Sources were found, but none pass'
        with pytest.warns(NoDetectionsWarning, match=match):
            tbl_strict = finder_strict(data)
        assert tbl_strict is None
        assert len(tbl_none) >= 1

    def test_roundness(self, data):
        """
        Test that no sources pass the roundness criteria.
        """
        match = 'Sources were found, but none pass'
        finder = DAOStarFinder(threshold=1, fwhm=1.0,
                               roundness_range=(1.0, 1.0))
        with pytest.warns(NoDetectionsWarning, match=match):
            tbl = finder(data)
        assert tbl is None

    @pytest.mark.parametrize('roundness_range', [0.5, (0.5,), (1, 2, 3)])
    def test_invalid_roundness_range(self, roundness_range):
        match = 'roundness_range must be a 2-element .* tuple'
        with pytest.raises(ValueError, match=match):
            DAOStarFinder(threshold=1, fwhm=1.0,
                          roundness_range=roundness_range)

    def test_roundness_range_none(self, data):
        """
        Test that roundness_range=None disables roundness filtering.
        """
        finder_none = DAOStarFinder(threshold=1, fwhm=1.0,
                                    sharpness_range=None,
                                    roundness_range=None)
        tbl_none = finder_none(data)
        assert tbl_none is not None

        finder_strict = DAOStarFinder(threshold=1, fwhm=1.0,
                                      sharpness_range=None,
                                      roundness_range=(1.0, 1.0))
        match = 'Sources were found, but none pass'
        with pytest.warns(NoDetectionsWarning, match=match):
            tbl_strict = finder_strict(data)
        assert tbl_strict is None
        assert len(tbl_none) >= 1

    def test_peak_max(self, data):
        """
        Test that no sources pass the peak_max criteria.
        """
        match = 'Sources were found, but none pass'
        finder = DAOStarFinder(threshold=1, fwhm=1.0, peak_max=1.0)
        with pytest.warns(NoDetectionsWarning, match=match):
            tbl = finder(data)
        assert tbl is None

    def test_peak_max_filtering(self, data):
        """
        Test that sources with peak >= peak_max are filtered out.
        """
        peak_max = 8
        finder0 = DAOStarFinder(threshold=1.0, fwhm=1.5,
                                roundness_range=(-np.inf, np.inf),
                                sharpness_range=(-np.inf, np.inf))
        finder1 = DAOStarFinder(threshold=1.0, fwhm=1.5,
                                roundness_range=(-np.inf, np.inf),
                                sharpness_range=(-np.inf, np.inf),
                                peak_max=peak_max)

        tbl0 = finder0(data)
        tbl1 = finder1(data)
        assert len(tbl0) > len(tbl1)
        assert all(tbl1['peak'] <= peak_max)

    def test_single_detected_source(self, data):
        """
        Test detection and slicing with a single source.
        """
        finder = DAOStarFinder(7.9, 2, n_brightest=1)
        mask = np.zeros(data.shape, dtype=bool)
        mask[0:50] = True
        tbl = finder(data, mask=mask)
        assert len(tbl) == 1

        # Test slicing with scalar catalog to improve coverage
        cat = finder._get_raw_catalog(data, mask=mask)
        assert cat.isscalar
        flux = cat.flux[0]  # evaluate the flux so it can be sliced
        assert cat[0].flux == flux

    def test_interval_ends_included(self):
        """
        Test that filter interval endpoints are inclusive.
        """
        # https://github.com/astropy/photutils/issues/1977
        data = np.zeros((46, 64))

        x = 33
        y = 21

        data[y - 1: y + 2, x - 1: x + 2] = [
            [1.0, 2.0, 1.0],
            [2.0, 1.0e20, 2.0],
            [1.0, 2.0, 1.0],
        ]

        finder = DAOStarFinder(
            threshold=0,
            fwhm=2.5,
            roundness_range=(0, 1.0),
            sharpness_range=(0.2, 1.407913491884342),
            peak_max=1.0e20,
        )
        tbl = finder.find_stars(data)

        assert len(tbl) == 1
        assert tbl[0]['roundness1'] < 1.e-15
        assert tbl[0]['roundness2'] == 0.0
        assert tbl[0]['peak'] == 1.0e20

    def test_data_not_mutated(self, data):
        """
        Test that input data is not mutated by find_stars.
        """
        data_copy = data.copy()
        finder = DAOStarFinder(threshold=1.0, fwhm=1.5)
        finder(data)
        assert_array_equal(data, data_copy)

    def test_data_not_mutated_with_mask(self, data):
        """
        Test that input data is not mutated when a mask is used.
        """
        data_copy = data.copy()
        mask = np.zeros(data.shape, dtype=bool)
        mask[0:50] = True
        finder = DAOStarFinder(threshold=1.0, fwhm=1.5)
        finder(data, mask=mask)
        assert_array_equal(data, data_copy)

    def test_repr(self):
        """
        Test the __repr__ of DAOStarFinder.
        """
        finder = DAOStarFinder(threshold=5.0, fwhm=3.0)
        repr_ = repr(finder)
        assert 'DAOStarFinder(' in repr_
        assert 'threshold=5.0' in repr_
        assert 'fwhm=3.0' in repr_
        assert 'ratio=1.0' in repr_
        assert 'xycoords=None' in repr_

    def test_str(self):
        """
        Test the __str__ of DAOStarFinder.
        """
        finder = DAOStarFinder(threshold=5.0, fwhm=3.0)
        str_ = str(finder)
        assert 'DAOStarFinder' in str_
        assert 'threshold: 5.0' in str_
        assert 'fwhm: 3.0' in str_

    def test_repr_with_xycoords(self):
        """
        Test that __repr__ shows array shape when xycoords are provided.
        """
        xycoords = np.array([[5, 5], [10, 10]])
        finder = DAOStarFinder(threshold=5.0, fwhm=3.0,
                               xycoords=xycoords)
        assert '<array; shape=(2, 2)>' in repr(finder)

    def test_threshold_2d_uniform(self, data):
        """
        Test that a uniform 2D threshold gives the same results
        as a scalar threshold.
        """
        threshold = 5.0
        fwhm = 1.0
        finder_scalar = DAOStarFinder(threshold, fwhm)
        finder_2d = DAOStarFinder(np.full(data.shape, threshold), fwhm)
        tbl_scalar = finder_scalar(data)
        tbl_2d = finder_2d(data)
        assert_array_equal(tbl_scalar, tbl_2d)

    def test_threshold_2d_varying(self, data):
        """
        Test that a varying 2D threshold detects fewer sources in
        regions with a higher threshold.
        """
        fwhm = 1.0
        threshold_low = 1.0
        threshold_high = 100.0
        threshold_2d = np.full(data.shape, threshold_low)
        threshold_2d[0:50, :] = threshold_high

        finder_low = DAOStarFinder(threshold_low, fwhm)
        finder_2d = DAOStarFinder(threshold_2d, fwhm)

        tbl_low = finder_low(data)
        tbl_2d = finder_2d(data)
        assert len(tbl_low) > len(tbl_2d)
        # All 2D sources should be in the lower half
        assert all(tbl_2d['y_centroid'] >= 50)

    def test_threshold_2d_repr(self):
        """
        Test repr with a 2D threshold array.
        """
        threshold = np.ones((10, 10))
        finder = DAOStarFinder(threshold=threshold, fwhm=3.0)
        assert '<array; shape=(10, 10)>' in repr(finder)
        assert '<array; shape=(10, 10)>' in str(finder)

    def test_threshold_2d_with_units(self, data):
        """
        Test that a 2D threshold with units works correctly.
        """
        units = u.Jy
        threshold = 5.0
        fwhm = 1.0
        threshold_2d = np.full(data.shape, threshold) * units
        finder = DAOStarFinder(threshold_2d, fwhm)
        tbl = finder(data << units)
        assert len(tbl) > 0

    def test_scale_threshold_default(self, data):
        """
        Test that scale_threshold=True (default) applies rel_err
        scaling.
        """
        threshold = 5.0
        fwhm = 1.5
        finder_default = DAOStarFinder(threshold, fwhm)
        finder_explicit = DAOStarFinder(threshold, fwhm,
                                        scale_threshold=True)
        tbl_default = finder_default(data)
        tbl_explicit = finder_explicit(data)
        assert_array_equal(tbl_default, tbl_explicit)
        # Verify the effective threshold is scaled
        assert finder_default.threshold_eff != threshold

    def test_scale_threshold_false(self, data):
        """
        Test that scale_threshold=False uses the threshold directly.
        """
        threshold = 5.0
        fwhm = 1.5
        finder = DAOStarFinder(threshold, fwhm, scale_threshold=False)
        assert finder.threshold_eff == threshold
        tbl = finder(data)
        assert len(tbl) > 0

    def test_scale_threshold_false_different_results(self, data):
        """
        Test that scale_threshold=False gives different results
        than the default.
        """
        threshold = 5.0
        fwhm = 1.5
        finder_scaled = DAOStarFinder(threshold, fwhm,
                                      scale_threshold=True)
        finder_unscaled = DAOStarFinder(threshold, fwhm,
                                        scale_threshold=False)
        tbl_scaled = finder_scaled(data)
        tbl_unscaled = finder_unscaled(data)
        # Different numbers of sources because effective thresholds
        # differ
        assert len(tbl_scaled) != len(tbl_unscaled)

    def test_scale_threshold_false_with_2d(self, data):
        """
        Test that scale_threshold=False works with a 2D threshold array.
        """
        fwhm = 1.5
        threshold_2d = np.full(data.shape, 5.0)
        finder = DAOStarFinder(threshold_2d, fwhm, scale_threshold=False)
        tbl = finder(data)
        assert len(tbl) > 0

    def test_scale_threshold_in_repr(self):
        """
        Test that scale_threshold appears in repr.
        """
        finder = DAOStarFinder(threshold=5.0, fwhm=3.0,
                               scale_threshold=False)
        assert 'scale_threshold=False' in repr(finder)
        assert 'scale_threshold: False' in str(finder)

    def test_deprecated_sharplo_sharphi(self):
        """
        Test that the deprecated 'sharplo'/'sharphi' keywords raise a
        warning and still work.
        """
        match = "The 'sharplo' and 'sharphi' parameters are deprecated"
        with pytest.warns(AstropyDeprecationWarning, match=match):
            finder = DAOStarFinder(threshold=5.0, fwhm=3.0, sharplo=0.1)
        assert finder.sharpness_range == (0.1, 1.0)

        with pytest.warns(AstropyDeprecationWarning, match=match):
            finder = DAOStarFinder(threshold=5.0, fwhm=3.0, sharphi=2.0)
        assert finder.sharpness_range == (0.2, 2.0)

        with pytest.warns(AstropyDeprecationWarning, match=match):
            finder = DAOStarFinder(threshold=5.0, fwhm=3.0,
                                   sharplo=0.1, sharphi=2.0)
        assert finder.sharpness_range == (0.1, 2.0)

    def test_deprecated_roundlo_roundhi(self):
        """
        Test that the deprecated 'roundlo'/'roundhi' keywords raise a
        warning and still work.
        """
        match = "The 'roundlo' and 'roundhi' parameters are deprecated"
        with pytest.warns(AstropyDeprecationWarning, match=match):
            finder = DAOStarFinder(threshold=5.0, fwhm=3.0, roundlo=-0.5)
        assert finder.roundness_range == (-0.5, 1.0)

        with pytest.warns(AstropyDeprecationWarning, match=match):
            finder = DAOStarFinder(threshold=5.0, fwhm=3.0, roundhi=0.5)
        assert finder.roundness_range == (-1.0, 0.5)

        with pytest.warns(AstropyDeprecationWarning, match=match):
            finder = DAOStarFinder(threshold=5.0, fwhm=3.0,
                                   roundlo=-0.5, roundhi=0.5)
        assert finder.roundness_range == (-0.5, 0.5)

    def test_deprecated_brightest(self):
        """
        Test that the deprecated 'brightest' keyword raises a warning
        and still works.
        """
        match = "'brightest' was deprecated"
        with pytest.warns(AstropyDeprecationWarning, match=match):
            finder = DAOStarFinder(threshold=5.0, fwhm=3.0, brightest=5)
        assert finder.n_brightest == 5

    def test_deprecated_peakmax(self):
        """
        Test that the deprecated 'peakmax' keyword raises a warning
        and still works.
        """
        match = "'peakmax' was deprecated"
        with pytest.warns(AstropyDeprecationWarning, match=match):
            finder = DAOStarFinder(threshold=5.0, fwhm=3.0, peakmax=100.0)
        assert finder.peak_max == 100.0
