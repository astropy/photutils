# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for IRAFStarFinder.
"""

import astropy.units as u
import numpy as np
import pytest
from numpy.testing import assert_array_equal

from photutils.detection import IRAFStarFinder
from photutils.psf import CircularGaussianPRF
from photutils.utils.exceptions import NoDetectionsWarning


class TestIRAFStarFinder:
    def test_irafstarfind(self, data):
        units = u.Jy
        threshold = 5.0
        fwhm = 1.0
        finder0 = IRAFStarFinder(threshold, fwhm)
        finder1 = IRAFStarFinder(threshold * units, fwhm)

        tbl0 = finder0(data)
        tbl1 = finder1(data << units)
        assert_array_equal(tbl0, tbl1)

    def test_irafstarfind_inputs(self):
        match = 'threshold must be a scalar value'
        with pytest.raises(TypeError, match=match):
            IRAFStarFinder(threshold=np.ones((2, 2)), fwhm=3.0)

        match = 'fwhm must be a scalar value'
        with pytest.raises(TypeError, match=match):
            IRAFStarFinder(threshold=3.0, fwhm=np.ones((2, 2)))

        match = 'brightest must be >= 0'
        with pytest.raises(ValueError, match=match):
            IRAFStarFinder(10, 1.5, brightest=-1)

        match = 'brightest must be an integer'
        with pytest.raises(ValueError, match=match):
            IRAFStarFinder(10, 1.5, brightest=3.1)

        xycoords = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
        match = 'xycoords must be shaped as a Nx2 array'
        with pytest.raises(ValueError, match=match):
            IRAFStarFinder(threshold=10, fwhm=1.5, xycoords=xycoords)

    def test_irafstarfind_nosources(self):
        data = np.ones((3, 3))
        match = 'No sources were found'
        finder = IRAFStarFinder(threshold=10, fwhm=1)
        with pytest.warns(NoDetectionsWarning, match=match):
            tbl = finder(data)
        assert tbl is None

        data = np.ones((5, 5))
        data[2, 2] = 10.0
        finder = IRAFStarFinder(threshold=0.1, fwhm=0.1)
        with pytest.warns(NoDetectionsWarning, match=match):
            tbl = finder(-data)
        assert tbl is None

    def test_irafstarfind_sharpness(self, data):
        """
        Sources found, but none pass the sharpness criteria.
        """
        match = 'Sources were found, but none pass'
        finder = IRAFStarFinder(threshold=1, fwhm=1.0, sharplo=2.0)
        with pytest.warns(NoDetectionsWarning, match=match):
            tbl = finder(data)
        assert tbl is None

    def test_irafstarfind_roundness(self, data):
        """
        Sources found, but none pass the roundness criteria.
        """
        match = 'Sources were found, but none pass'
        finder = IRAFStarFinder(threshold=1, fwhm=1.0, roundlo=1.0)
        with pytest.warns(NoDetectionsWarning, match=match):
            tbl = finder(data)
        assert tbl is None

    def test_irafstarfind_peakmax(self, data):
        """
        Sources found, but none pass the peakmax criteria.
        """
        match = 'Sources were found, but none pass'
        finder = IRAFStarFinder(threshold=1, fwhm=1.0, peakmax=1.0)
        with pytest.warns(NoDetectionsWarning, match=match):
            tbl = finder(data)
        assert tbl is None

    def test_irafstarfind_peakmax_filtering(self, data):
        """
        Regression test that objects with peak >= peakmax are filtered
        out.
        """
        peakmax = 8
        finder0 = IRAFStarFinder(threshold=1.0, fwhm=2, roundlo=-np.inf,
                                 roundhi=np.inf, sharplo=-np.inf,
                                 sharphi=np.inf)
        finder1 = IRAFStarFinder(threshold=1.0, fwhm=2, roundlo=-np.inf,
                                 roundhi=np.inf, sharplo=-np.inf,
                                 sharphi=np.inf, peakmax=peakmax)

        tbl0 = finder0(data)
        tbl1 = finder1(data)
        assert len(tbl0) > len(tbl1)
        assert all(tbl1['peak'] < peakmax)

    def test_irafstarfind_brightest_filtering(self, data):
        """
        Regression test that only top ``brightest`` objects are
        selected.
        """
        brightest = 10
        finder = IRAFStarFinder(threshold=1.0, fwhm=2, roundlo=-np.inf,
                                roundhi=np.inf, sharplo=-np.inf,
                                sharphi=np.inf, brightest=brightest)
        tbl = finder(data)
        assert len(tbl) == brightest

    def test_irafstarfind_mask(self, data):
        """
        Test IRAFStarFinder with a mask.
        """
        finder = IRAFStarFinder(threshold=1.0, fwhm=1.5)
        mask = np.zeros(data.shape, dtype=bool)
        mask[0:50, :] = True
        tbl0 = finder(data)
        tbl1 = finder(data, mask=mask)
        assert len(tbl0) > len(tbl1)

    def test_xycoords(self, data):
        finder0 = IRAFStarFinder(threshold=8.0, fwhm=2)
        tbl0 = finder0(data)
        xycoords = list(zip(tbl0['xcentroid'], tbl0['ycentroid'], strict=True))
        xycoords = np.round(xycoords).astype(int)

        finder1 = IRAFStarFinder(threshold=8.0, fwhm=2, xycoords=xycoords)
        tbl1 = finder1(data)
        assert_array_equal(tbl0, tbl1)

    def test_min_separation(self, data):
        threshold = 1.0
        fwhm = 1.0
        finder1 = IRAFStarFinder(threshold, fwhm)
        tbl1 = finder1(data)
        finder2 = IRAFStarFinder(threshold, fwhm, min_separation=3.0)
        tbl2 = finder2(data)
        assert np.all(tbl1 == tbl2)

        finder3 = IRAFStarFinder(threshold, fwhm, min_separation=10.0)
        tbl3 = finder3(data)
        assert len(tbl2) > len(tbl3)

        match = 'min_separation must be >= 0'
        with pytest.raises(ValueError, match=match):
            IRAFStarFinder(threshold=10, fwhm=1.5, min_separation=-1.0)

    def test_single_detected_source(self, data):
        finder = IRAFStarFinder(8.4, 2, brightest=1)
        mask = np.zeros(data.shape, dtype=bool)
        mask[0:50] = True
        tbl = finder(data, mask=mask)
        assert len(tbl) == 1

        # test slicing with scalar catalog to improve coverage
        cat = finder._get_raw_catalog(data, mask=mask)
        assert cat.isscalar
        flux = cat.flux[0]  # evaluate the flux so it can be sliced
        assert cat[0].flux == flux

    def test_all_border_sources(self):
        model1 = CircularGaussianPRF(flux=100, x_0=1, y_0=1, fwhm=2)
        model2 = CircularGaussianPRF(flux=100, x_0=50, y_0=50, fwhm=2)
        model3 = CircularGaussianPRF(flux=100, x_0=30, y_0=30, fwhm=2)

        threshold = 1
        yy, xx = np.mgrid[:51, :51]
        data = model1(xx, yy)

        # test single source within the border region
        finder = IRAFStarFinder(threshold=threshold, fwhm=2.0, roundlo=-0.1,
                                exclude_border=True)
        with pytest.warns(NoDetectionsWarning):
            tbl = finder(data)
        assert tbl is None

        # test multiple sources all within the border region
        data += model2(xx, yy)
        with pytest.warns(NoDetectionsWarning):
            tbl = finder(data)
        assert tbl is None

        # test multiple sources with some within the border region
        data += model3(xx, yy)
        tbl = finder(data)
        assert len(tbl) == 1

    def test_interval_ends_included(self):
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
            roundlo=0,
            peakmax=0.8,
        )
        tbl = finder.find_stars(data)

        assert len(tbl) == 1
        assert tbl[0]['roundness'] < 1.e-15
        assert tbl[0]['peak'] == 0.8
