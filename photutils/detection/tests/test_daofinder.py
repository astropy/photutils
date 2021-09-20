# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for DAOStarFinder.
"""

import itertools
import os.path as op

from astropy.table import Table
from astropy.tests.helper import catch_warnings
import numpy as np
from numpy.testing import assert_allclose
import pytest

from ..daofinder import DAOStarFinder
from ...datasets import make_100gaussians_image
from ...utils.exceptions import NoDetectionsWarning
from ...utils._optional_deps import HAS_SCIPY  # noqa


DATA = make_100gaussians_image()
THRESHOLDS = [8.0, 10.0]
FWHMS = [1.0, 1.5, 2.0]


@pytest.mark.skipif('not HAS_SCIPY')
class TestDAOStarFinder:
    @pytest.mark.parametrize(('threshold', 'fwhm'),
                             list(itertools.product(THRESHOLDS, FWHMS)))
    def test_daofind(self, threshold, fwhm):
        starfinder = DAOStarFinder(threshold, fwhm, sigma_radius=1.5)
        tbl = starfinder(DATA)
        datafn = f'daofind_test_thresh{threshold:04.1f}_fwhm{fwhm:04.1f}.txt'
        datafn = op.join(op.dirname(op.abspath(__file__)), 'data', datafn)
        tbl_ref = Table.read(datafn, format='ascii')

        assert tbl.colnames == tbl_ref.colnames
        for col in tbl.colnames:
            assert_allclose(tbl[col], tbl_ref[col])

    def test_daofind_threshold_fwhm_inputs(self):
        with pytest.raises(TypeError):
            DAOStarFinder(threshold=np.ones((2, 2)), fwhm=3.)

        with pytest.raises(TypeError):
            DAOStarFinder(threshold=3., fwhm=np.ones((2, 2)))

    def test_daofind_include_border(self):
        starfinder = DAOStarFinder(threshold=10, fwhm=2, sigma_radius=1.5,
                                   exclude_border=False)
        tbl = starfinder(DATA)
        assert len(tbl) == 20

    def test_daofind_exclude_border(self):
        starfinder = DAOStarFinder(threshold=10, fwhm=2, sigma_radius=1.5,
                                   exclude_border=True)
        tbl = starfinder(DATA)
        assert len(tbl) == 19

    def test_daofind_nosources(self):
        data = np.ones((3, 3))
        with catch_warnings(NoDetectionsWarning) as warning_lines:
            starfinder = DAOStarFinder(threshold=10, fwhm=1)
            tbl = starfinder(data)
            assert tbl is None
            assert 'No sources were found.' in str(warning_lines[0].message)

    def test_daofind_sharpness(self):
        """Sources found, but none pass the sharpness criteria."""
        with catch_warnings(NoDetectionsWarning) as warning_lines:
            starfinder = DAOStarFinder(threshold=50, fwhm=1.0, sharplo=1.)
            tbl = starfinder(DATA)
            assert tbl is None
            assert ('Sources were found, but none pass' in
                    str(warning_lines[0].message))

    def test_daofind_roundness(self):
        """Sources found, but none pass the roundness criteria."""
        with catch_warnings(NoDetectionsWarning) as warning_lines:
            starfinder = DAOStarFinder(threshold=50, fwhm=1.0, roundlo=1.)
            tbl = starfinder(DATA)
            assert tbl is None
            assert ('Sources were found, but none pass' in
                    str(warning_lines[0].message))

    def test_daofind_peakmax(self):
        """Sources found, but none pass the peakmax criteria."""
        with catch_warnings(NoDetectionsWarning) as warning_lines:
            starfinder = DAOStarFinder(threshold=50, fwhm=1.0, peakmax=1.0)
            tbl = starfinder(DATA)
            assert tbl is None
            assert ('Sources were found, but none pass' in
                    str(warning_lines[0].message))

    def test_daofind_flux_negative(self):
        """Test handling of negative flux (here created by large sky)."""
        data = np.ones((5, 5))
        data[2, 2] = 10.
        starfinder = DAOStarFinder(threshold=0.1, fwhm=1.0, sky=10)
        tbl = starfinder(data)
        assert not np.isfinite(tbl['mag'])

    def test_daofind_negative_fit_peak(self):
        """
        Regression test that sources with negative fit peaks (i.e.,
        hx/hy<=0) are excluded.
        """
        starfinder = DAOStarFinder(threshold=7., fwhm=1.5, roundlo=-np.inf,
                                   roundhi=np.inf, sharplo=-np.inf,
                                   sharphi=np.inf)
        tbl = starfinder(DATA)
        assert len(tbl) == 102

    def test_daofind_peakmax_filtering(self):
        """
        Regression test that objects with ``peak`` >= ``peakmax`` are
        filtered out.
        """
        peakmax = 20
        starfinder = DAOStarFinder(threshold=7., fwhm=1.5, roundlo=-np.inf,
                                   roundhi=np.inf, sharplo=-np.inf,
                                   sharphi=np.inf, peakmax=peakmax)
        tbl = starfinder(DATA)
        assert len(tbl) == 37
        assert all(tbl['peak'] < peakmax)

    def test_daofind_brightest_filtering(self):
        """
        Regression test that only top ``brightest`` objects are
        selected.
        """
        brightest = 40
        peakmax = 20
        starfinder = DAOStarFinder(threshold=7., fwhm=1.5, roundlo=-np.inf,
                                   roundhi=np.inf, sharplo=-np.inf,
                                   sharphi=np.inf, brightest=brightest)
        tbl = starfinder(DATA)
        # combined with peakmax
        assert len(tbl) == brightest
        starfinder = DAOStarFinder(threshold=7., fwhm=1.5, roundlo=-np.inf,
                                   roundhi=np.inf, sharplo=-np.inf,
                                   sharphi=np.inf, brightest=brightest,
                                   peakmax=peakmax)
        tbl = starfinder(DATA)
        assert len(tbl) == 37

    def test_daofind_mask(self):
        """Test DAOStarFinder with a mask."""
        starfinder = DAOStarFinder(threshold=10, fwhm=1.5)
        mask = np.zeros(DATA.shape, dtype=bool)
        mask[100:200] = True
        tbl1 = starfinder(DATA)
        tbl2 = starfinder(DATA, mask=mask)
        assert len(tbl1) > len(tbl2)

    def test_inputs(self):
        with pytest.raises(ValueError):
            DAOStarFinder(threshold=10, fwhm=1.5, brightest=-1)
        with pytest.raises(ValueError):
            DAOStarFinder(threshold=10, fwhm=1.5, brightest=3.1)

    def test_xycoords(self):
        starfinder1 = DAOStarFinder(threshold=30, fwhm=2, sigma_radius=1.5,
                                    exclude_border=True)
        tbl1 = starfinder1(DATA)

        xycoords = np.array([[145, 169], [395, 187], [427, 211], [11, 224]])
        starfinder2 = DAOStarFinder(threshold=30, fwhm=2, sigma_radius=1.5,
                                    exclude_border=True, xycoords=xycoords)
        tbl2 = starfinder2(DATA)
        assert np.all(tbl1 == tbl2)

    def test_invalid_xycoords(self):
        xycoords = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
        with pytest.raises(ValueError):
            DAOStarFinder(threshold=10, fwhm=1.5, xycoords=xycoords)
