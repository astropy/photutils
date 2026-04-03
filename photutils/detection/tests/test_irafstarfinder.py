# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the irafstarfinder module.
"""

import astropy.units as u
import numpy as np
import pytest
from astropy.utils.exceptions import AstropyDeprecationWarning
from numpy.testing import assert_array_equal

from photutils.detection import IRAFStarFinder
from photutils.psf import CircularGaussianPRF
from photutils.utils.exceptions import NoDetectionsWarning


class TestIRAFStarFinder:
    """
    Test the IRAFStarFinder class.
    """

    def test_find(self, data):
        """
        Test basic source detection and unit handling.
        """
        units = u.Jy
        threshold = 5.0
        fwhm = 1.0
        finder0 = IRAFStarFinder(threshold, fwhm)
        finder1 = IRAFStarFinder(threshold * units, fwhm)

        tbl0 = finder0(data)
        tbl1 = finder1(data << units)
        assert_array_equal(tbl0, tbl1)
        assert tbl0['orientation'].unit == u.deg

    def test_inputs(self):
        """
        Test that invalid inputs raise appropriate errors.
        """
        match = 'fwhm must be a scalar value'
        with pytest.raises(TypeError, match=match):
            IRAFStarFinder(threshold=3.0, fwhm=np.ones((2, 2)))

        match = 'fwhm must be positive'
        for fwhm in (-10, 0):
            with pytest.raises(ValueError, match=match):
                IRAFStarFinder(threshold=3.0, fwhm=fwhm)

        match = 'n_brightest must be > 0'
        with pytest.raises(ValueError, match=match):
            IRAFStarFinder(10, 1.5, n_brightest=-1)

        match = 'n_brightest must be an integer'
        with pytest.raises(ValueError, match=match):
            IRAFStarFinder(10, 1.5, n_brightest=3.1)

        match = 'minsep_fwhm must be >= 0'
        with (pytest.warns(AstropyDeprecationWarning),
              pytest.raises(ValueError, match=match)):
            IRAFStarFinder(10, 1.5, minsep_fwhm=-1)

        xycoords = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
        match = 'xycoords must be shaped as an Nx2 array'
        with pytest.raises(ValueError, match=match):
            IRAFStarFinder(threshold=10, fwhm=1.5, xycoords=xycoords)

    def test_nosources(self):
        """
        Test that no sources returns None with a warning.
        """
        data = np.ones((3, 3))
        match = 'No sources were found'
        finder = IRAFStarFinder(threshold=10, fwhm=1)
        with pytest.warns(NoDetectionsWarning, match=match):
            tbl = finder(data)
        assert tbl is None

        data = np.ones((5, 5))
        data[2, 2] = 10.0
        finder = IRAFStarFinder(threshold=0.1, fwhm=0.1, min_separation=0)
        with pytest.warns(NoDetectionsWarning, match=match):
            tbl = finder(-data)
        assert tbl is None

    def test_mask(self, data):
        """
        Test source detection with a mask.
        """
        finder = IRAFStarFinder(threshold=1.0, fwhm=1.5)
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
        finder = IRAFStarFinder(threshold=1.0, fwhm=1.5)
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
        finder0 = IRAFStarFinder(threshold=8.0, fwhm=2)
        tbl0 = finder0(data)
        xycoords = list(zip(tbl0['x_centroid'],
                            tbl0['y_centroid'], strict=True))
        xycoords = np.round(xycoords).astype(int)

        finder1 = IRAFStarFinder(threshold=8.0, fwhm=2, xycoords=xycoords)
        tbl1 = finder1(data)
        assert_array_equal(tbl0, tbl1)

    def test_min_separation(self, data):
        """
        Test the min_separation parameter.
        """
        threshold = 1.0
        fwhm = 1.0
        finder1 = IRAFStarFinder(threshold, fwhm)
        tbl1 = finder1(data)
        assert finder1.min_separation == 2.5 * fwhm

        finder2 = IRAFStarFinder(threshold, fwhm, min_separation=10.0)
        tbl2 = finder2(data)
        assert len(tbl1) > len(tbl2)

        match = 'min_separation must be >= 0'
        with pytest.raises(ValueError, match=match):
            IRAFStarFinder(threshold=10, fwhm=1.5, min_separation=-1.0)

    def test_min_separation_default(self):
        """
        Test that the default min_separation (None) gives 2.5 * fwhm.
        """
        fwhm = 2.0
        finder = IRAFStarFinder(threshold=1.0, fwhm=fwhm)
        assert finder.min_separation == 2.5 * fwhm

        # Previous (< 3.0) default behavior
        old_default = max(2, int(fwhm * 2.5 + 0.5))
        finder_old = IRAFStarFinder(threshold=1.0, fwhm=fwhm,
                                    min_separation=old_default)
        assert finder_old.min_separation == old_default

    def test_n_brightest_filtering(self, data):
        """
        Test that only the top n_brightest sources are selected.
        """
        n_brightest = 10
        finder = IRAFStarFinder(threshold=1.0, fwhm=2,
                                roundness_range=(-np.inf, np.inf),
                                sharpness_range=(-np.inf, np.inf),
                                n_brightest=n_brightest)
        tbl = finder(data)
        assert len(tbl) == n_brightest

    def test_sharpness(self, data):
        """
        Test that no sources pass the sharpness criteria.
        """
        match = 'Sources were found, but none pass'
        finder = IRAFStarFinder(threshold=1, fwhm=1.0,
                                sharpness_range=(2.0, 2.0))
        with pytest.warns(NoDetectionsWarning, match=match):
            tbl = finder(data)
        assert tbl is None

    @pytest.mark.parametrize('sharpness_range', [0.5, (0.5,), (1, 2, 3)])
    def test_invalid_sharpness_range(self, sharpness_range):
        match = 'sharpness_range must be a 2-element .* tuple'
        with pytest.raises(ValueError, match=match):
            IRAFStarFinder(threshold=1, fwhm=1.0,
                           sharpness_range=sharpness_range)

    def test_sharpness_range_none(self, data):
        """
        Test that sharpness_range=None disables sharpness filtering.
        """
        finder_none = IRAFStarFinder(threshold=1, fwhm=2,
                                     roundness_range=None,
                                     sharpness_range=None)
        tbl_none = finder_none(data)
        assert tbl_none is not None

        finder_strict = IRAFStarFinder(threshold=1, fwhm=1.0,
                                       roundness_range=None,
                                       sharpness_range=(2.0, 2.0))
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
        finder = IRAFStarFinder(threshold=1, fwhm=1.0,
                                roundness_range=(1.0, np.inf))
        with pytest.warns(NoDetectionsWarning, match=match):
            tbl = finder(data)
        assert tbl is None

    @pytest.mark.parametrize('roundness_range', [0.5, (0.5,), (1, 2, 3)])
    def test_invalid_roundness_range(self, roundness_range):
        match = 'roundness_range must be a 2-element .* tuple'
        with pytest.raises(ValueError, match=match):
            IRAFStarFinder(threshold=1, fwhm=1.0,
                           roundness_range=roundness_range)

    def test_roundness_range_none(self, data):
        """
        Test that roundness_range=None disables roundness filtering.
        """
        finder_none = IRAFStarFinder(threshold=1, fwhm=2,
                                     sharpness_range=None,
                                     roundness_range=None)
        tbl_none = finder_none(data)
        assert tbl_none is not None

        finder_strict = IRAFStarFinder(threshold=1, fwhm=1.0,
                                       sharpness_range=None,
                                       roundness_range=(1.0, np.inf))
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
        finder = IRAFStarFinder(threshold=1, fwhm=1.0, peak_max=1.0)
        with pytest.warns(NoDetectionsWarning, match=match):
            tbl = finder(data)
        assert tbl is None

    def test_peak_max_filtering(self, data):
        """
        Test that sources with peak >= peak_max are filtered out.
        """
        peak_max = 8
        finder0 = IRAFStarFinder(threshold=1.0, fwhm=2,
                                 roundness_range=(-np.inf, np.inf),
                                 sharpness_range=(-np.inf, np.inf))
        finder1 = IRAFStarFinder(threshold=1.0, fwhm=2,
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
        finder = IRAFStarFinder(8.4, 2, n_brightest=1)
        mask = np.zeros(data.shape, dtype=bool)
        mask[0:50] = True
        tbl = finder(data, mask=mask)
        assert len(tbl) == 1

        # Test slicing with scalar catalog to improve coverage
        cat = finder._get_raw_catalog(data, mask=mask)
        assert cat.isscalar
        flux = cat.flux[0]  # evaluate the flux so it can be sliced
        assert cat[0].flux == flux

    def test_all_border_sources(self):
        """
        Test that all-border sources are excluded correctly.
        """
        model1 = CircularGaussianPRF(flux=100, x_0=1, y_0=1, fwhm=2)
        model2 = CircularGaussianPRF(flux=100, x_0=50, y_0=50, fwhm=2)
        model3 = CircularGaussianPRF(flux=100, x_0=30, y_0=30, fwhm=2)

        threshold = 1
        yy, xx = np.mgrid[:51, :51]
        data = model1(xx, yy)

        # Test single source within the border region
        finder = IRAFStarFinder(threshold=threshold, fwhm=2.0,
                                roundness_range=(-0.1, 0.2),
                                exclude_border=True)
        with pytest.warns(NoDetectionsWarning):
            tbl = finder(data)
        assert tbl is None

        # Test multiple sources all within the border region
        data += model2(xx, yy)
        with pytest.warns(NoDetectionsWarning):
            tbl = finder(data)
        assert tbl is None

        # Test multiple sources with some within the border region
        data += model3(xx, yy)
        tbl = finder(data)
        assert len(tbl) == 1

    def test_interval_ends_included(self):
        """
        Test that filter interval endpoints are inclusive.
        """
        # https://github.com/astropy/photutils/issues/1977
        data = np.zeros((46, 64))

        x = 33
        y = 21
        data[y - 1: y + 2, x - 1: x + 2] = [
            [0.1, 0.6, 0.1],
            [0.6, 0.8, 0.6],
            [0.1, 0.6, 0.1],
        ]

        finder = IRAFStarFinder(
            threshold=0,
            fwhm=2.5,
            roundness_range=(0, 0.2),
            peak_max=0.8,
        )
        tbl = finder.find_stars(data)

        assert len(tbl) == 1
        assert tbl[0]['roundness'] < 1.e-15
        assert tbl[0]['peak'] == 0.8

    def test_data_not_mutated(self, data):
        """
        Test that input data is not mutated by find_stars.
        """
        data_copy = data.copy()
        finder = IRAFStarFinder(threshold=1.0, fwhm=1.5)
        finder(data)
        assert_array_equal(data, data_copy)

    def test_data_not_mutated_with_mask(self, data):
        """
        Test that input data is not mutated when a mask is used.
        """
        data_copy = data.copy()
        mask = np.zeros(data.shape, dtype=bool)
        mask[0:50] = True
        finder = IRAFStarFinder(threshold=1.0, fwhm=1.5)
        finder(data, mask=mask)
        assert_array_equal(data, data_copy)

    def test_repr(self):
        """
        Test the __repr__ of IRAFStarFinder.
        """
        finder = IRAFStarFinder(threshold=5.0, fwhm=3.0)
        repr_ = repr(finder)
        assert 'IRAFStarFinder(' in repr_
        assert 'threshold=5.0' in repr_
        assert 'fwhm=3.0' in repr_
        assert 'min_separation=' in repr_
        assert 'xycoords=None' in repr_

    def test_str(self):
        """
        Test the __str__ of IRAFStarFinder.
        """
        finder = IRAFStarFinder(threshold=5.0, fwhm=3.0)
        str_ = str(finder)
        assert 'IRAFStarFinder' in str_
        assert 'threshold: 5.0' in str_
        assert 'fwhm: 3.0' in str_

    def test_repr_with_xycoords(self):
        """
        Test that __repr__ shows array shape when xycoords are provided.
        """
        xycoords = np.array([[5, 5], [10, 10]])
        finder = IRAFStarFinder(threshold=5.0, fwhm=3.0,
                                xycoords=xycoords)
        assert '<array; shape=(2, 2)>' in repr(finder)

    def test_threshold_2d_uniform(self, data):
        """
        Test that a uniform 2D threshold gives the same results
        as a scalar threshold.
        """
        threshold = 5.0
        fwhm = 1.0
        finder_scalar = IRAFStarFinder(threshold, fwhm)
        finder_2d = IRAFStarFinder(np.full(data.shape, threshold), fwhm)
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

        finder_low = IRAFStarFinder(threshold_low, fwhm)
        finder_2d = IRAFStarFinder(threshold_2d, fwhm)

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
        finder = IRAFStarFinder(threshold=threshold, fwhm=3.0)
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
        finder = IRAFStarFinder(threshold_2d, fwhm)
        tbl = finder(data << units)
        assert len(tbl) > 0

    def test_catalog_intermediate_properties(self, data):
        """
        Test IRAF catalog intermediate properties: sky,
        cutout_data_nosub, cutout_xorigin, cutout_yorigin,
        sharpness.
        """
        finder = IRAFStarFinder(threshold=1.0, fwhm=2.0,
                                sharpness_range=(-np.inf, np.inf),
                                roundness_range=(-np.inf, np.inf))
        cat = finder._get_raw_catalog(data)
        assert cat is not None
        nsrc = len(cat)

        # sky should be finite and have same length as nsources
        sky = cat.sky
        assert sky.shape == (nsrc,)
        assert np.all(np.isfinite(sky))

        # cutout_data_nosub should have shape (nsrc, ky, kx) with no
        # sky subtraction
        cdata = cat.cutout_data_nosub
        assert cdata.ndim == 3
        assert cdata.shape[0] == nsrc

        # cutout_xorigin/cutout_yorigin should be finite 1D arrays
        xorig = cat.cutout_xorigin
        yorig = cat.cutout_yorigin
        assert xorig.shape == (nsrc,)
        assert yorig.shape == (nsrc,)
        assert np.all(np.isfinite(xorig))
        assert np.all(np.isfinite(yorig))

        # sharpness should be finite for detected sources
        sharpness = cat.sharpness
        assert sharpness.shape == (nsrc,)
        assert np.all(np.isfinite(sharpness))

    def test_deprecated_sharplo_sharphi(self):
        """
        Test that the deprecated 'sharplo'/'sharphi' keywords raise a
        warning and still work.
        """
        match = "The 'sharplo' and 'sharphi' parameters are deprecated"
        with pytest.warns(AstropyDeprecationWarning, match=match):
            finder = IRAFStarFinder(threshold=5.0, fwhm=3.0, sharplo=0.3)
        assert finder.sharpness_range == (0.3, 2.0)

        with pytest.warns(AstropyDeprecationWarning, match=match):
            finder = IRAFStarFinder(threshold=5.0, fwhm=3.0, sharphi=3.0)
        assert finder.sharpness_range == (0.5, 3.0)

    def test_deprecated_roundlo_roundhi(self):
        """
        Test that the deprecated 'roundlo'/'roundhi' keywords raise a
        warning and still work.
        """
        match = "The 'roundlo' and 'roundhi' parameters are deprecated"
        with pytest.warns(AstropyDeprecationWarning, match=match):
            finder = IRAFStarFinder(threshold=5.0, fwhm=3.0, roundlo=-0.1)
        assert finder.roundness_range == (-0.1, 0.2)

        with pytest.warns(AstropyDeprecationWarning, match=match):
            finder = IRAFStarFinder(threshold=5.0, fwhm=3.0, roundhi=0.5)
        assert finder.roundness_range == (0.0, 0.5)

    def test_deprecated_brightest(self):
        """
        Test that the deprecated 'brightest' keyword raises a warning
        and still works.
        """
        match = "'brightest' was deprecated"
        with pytest.warns(AstropyDeprecationWarning, match=match):
            finder = IRAFStarFinder(threshold=5.0, fwhm=3.0, brightest=5)
        assert finder.n_brightest == 5

    def test_deprecated_peakmax(self):
        """
        Test that the deprecated 'peakmax' keyword raises a warning
        and still works.
        """
        match = "'peakmax' was deprecated"
        with pytest.warns(AstropyDeprecationWarning, match=match):
            finder = IRAFStarFinder(threshold=5.0, fwhm=3.0, peakmax=100.0)
        assert finder.peak_max == 100.0

    def test_deprecated_minsep_fwhm(self):
        """
        Test that the deprecated 'minsep_fwhm' keyword raises a warning
        and still works.
        """
        fwhm = 3.0
        minsep_fwhm = 2.5
        match = "The 'minsep_fwhm' parameter is deprecated"
        with pytest.warns(AstropyDeprecationWarning, match=match):
            finder = IRAFStarFinder(threshold=5.0, fwhm=fwhm,
                                    minsep_fwhm=minsep_fwhm)
        expected = max(2, int((fwhm * minsep_fwhm) + 0.5))
        assert finder.min_separation == expected

    def test_deprecated_minsep_fwhm_overridden(self):
        """
        Test that min_separation takes priority over minsep_fwhm.
        """
        match = "The 'minsep_fwhm' parameter is deprecated"
        with pytest.warns(AstropyDeprecationWarning, match=match):
            finder = IRAFStarFinder(threshold=5.0, fwhm=3.0,
                                    minsep_fwhm=2.5, min_separation=7.0)
        assert finder.min_separation == 7.0
