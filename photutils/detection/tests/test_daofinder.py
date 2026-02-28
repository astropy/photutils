# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the daofinder module.
"""

import astropy.units as u
import numpy as np
import pytest
from numpy.testing import assert_array_equal

from photutils.detection.daofinder import DAOStarFinder
from photutils.utils.exceptions import NoDetectionsWarning


class TestDAOStarFinder:
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
        match = 'threshold must be a scalar value'
        with pytest.raises(TypeError, match=match):
            DAOStarFinder(threshold=np.ones((2, 2)), fwhm=3.0)

        match = 'fwhm must be a scalar value'
        with pytest.raises(TypeError, match=match):
            DAOStarFinder(threshold=3.0, fwhm=np.ones((2, 2)))

        match = 'fwhm must be positive'
        for fwhm in (-10, 0):
            with pytest.raises(ValueError, match=match):
                DAOStarFinder(threshold=3.0, fwhm=fwhm)

        match = 'ratio must be positive and less than or equal to 1'
        with pytest.raises(ValueError, match=match):
            DAOStarFinder(threshold=3.0, fwhm=2, ratio=-10)

        match = 'sigma_radius must be positive'
        with pytest.raises(ValueError, match=match):
            DAOStarFinder(threshold=3.0, fwhm=2, sigma_radius=-10)

        match = 'brightest must be > 0'
        with pytest.raises(ValueError, match=match):
            DAOStarFinder(threshold=10, fwhm=1.5, brightest=-1)

        match = 'brightest must be an integer'
        with pytest.raises(ValueError, match=match):
            DAOStarFinder(threshold=10, fwhm=1.5, brightest=3.1)

        xycoords = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
        match = 'xycoords must be shaped as an Nx2 array'
        with pytest.raises(ValueError, match=match):
            DAOStarFinder(threshold=10, fwhm=1.5, xycoords=xycoords)

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

    def test_xycoords(self, data):
        """
        Test source detection at specified coordinates.
        """
        finder0 = DAOStarFinder(threshold=8.0, fwhm=2)
        tbl0 = finder0(data)
        xycoords = list(zip(tbl0['xcentroid'], tbl0['ycentroid'], strict=True))
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
        finder2 = DAOStarFinder(threshold, fwhm, min_separation=10.0)
        tbl2 = finder2(data)
        assert len(tbl1) > len(tbl2)

        match = 'min_separation must be >= 0'
        with pytest.raises(ValueError, match=match):
            DAOStarFinder(threshold=10, fwhm=1.5, min_separation=-1.0)

    def test_brightest_filtering(self, data):
        """
        Test that only the top brightest sources are selected.
        """
        brightest = 10
        finder = DAOStarFinder(threshold=1.0, fwhm=1.5, roundlo=-np.inf,
                               roundhi=np.inf, sharplo=-np.inf,
                               sharphi=np.inf, brightest=brightest)
        tbl = finder(data)
        assert len(tbl) == brightest

        # Combined with peakmax
        peakmax = 8
        finder = DAOStarFinder(threshold=1.0, fwhm=1.5, roundlo=-np.inf,
                               roundhi=np.inf, sharplo=-np.inf,
                               sharphi=np.inf, brightest=brightest,
                               peakmax=peakmax)
        tbl = finder(data)
        assert len(tbl) == 5

    def test_sharpness(self, data):
        """
        Test that no sources pass the sharpness criteria.
        """
        match = 'Sources were found, but none pass'
        finder = DAOStarFinder(threshold=1, fwhm=1.0, sharplo=1.0)
        with pytest.warns(NoDetectionsWarning, match=match):
            tbl = finder(data)
        assert tbl is None

    def test_roundness(self, data):
        """
        Test that no sources pass the roundness criteria.
        """
        match = 'Sources were found, but none pass'
        finder = DAOStarFinder(threshold=1, fwhm=1.0, roundlo=1.0)
        with pytest.warns(NoDetectionsWarning, match=match):
            tbl = finder(data)
        assert tbl is None

    def test_peakmax(self, data):
        """
        Test that no sources pass the peakmax criteria.
        """
        match = 'Sources were found, but none pass'
        finder = DAOStarFinder(threshold=1, fwhm=1.0, peakmax=1.0)
        with pytest.warns(NoDetectionsWarning, match=match):
            tbl = finder(data)
        assert tbl is None

    def test_peakmax_filtering(self, data):
        """
        Test that sources with peak >= peakmax are filtered out.
        """
        peakmax = 8
        finder0 = DAOStarFinder(threshold=1.0, fwhm=1.5, roundlo=-np.inf,
                                roundhi=np.inf, sharplo=-np.inf,
                                sharphi=np.inf)
        finder1 = DAOStarFinder(threshold=1.0, fwhm=1.5, roundlo=-np.inf,
                                roundhi=np.inf, sharplo=-np.inf,
                                sharphi=np.inf, peakmax=peakmax)

        tbl0 = finder0(data)
        tbl1 = finder1(data)
        assert len(tbl0) > len(tbl1)
        assert all(tbl1['peak'] <= peakmax)

    def test_single_detected_source(self, data):
        """
        Test detection and slicing with a single source.
        """
        finder = DAOStarFinder(7.9, 2, brightest=1)
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
            roundlo=0,
            sharphi=1.407913491884342,
            peakmax=1.0e20,
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
        r = repr(finder)
        assert 'DAOStarFinder(' in r
        assert 'threshold=5.0' in r
        assert 'fwhm=3.0' in r
        assert 'ratio=1.0' in r
        assert 'xycoords=None' in r

    def test_str(self):
        """
        Test the __str__ of DAOStarFinder.
        """
        finder = DAOStarFinder(threshold=5.0, fwhm=3.0)
        s = str(finder)
        assert 'DAOStarFinder' in s
        assert 'threshold: 5.0' in s
        assert 'fwhm: 3.0' in s

    def test_repr_with_xycoords(self):
        """
        Test that __repr__ shows array shape when xycoords are provided.
        """
        xycoords = np.array([[5, 5], [10, 10]])
        finder = DAOStarFinder(threshold=5.0, fwhm=3.0,
                               xycoords=xycoords)
        r = repr(finder)
        assert '<array; shape=(2, 2)>' in r
