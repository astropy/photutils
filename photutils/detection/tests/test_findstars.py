# Licensed under a 3-clause BSD style license - see LICENSE.rst

import os.path as op
import itertools
import warnings

import numpy as np
from numpy.testing import assert_allclose
import pytest

from astropy.table import Table
from astropy.utils.exceptions import AstropyUserWarning

from ..findstars import DAOStarFinder, IRAFStarFinder
from ...datasets import make_100gaussians_image

try:
    import scipy    # noqa
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


DATA = make_100gaussians_image()
THRESHOLDS = [8.0, 10.0]
FWHMS = [1.0, 1.5, 2.0]
warnings.simplefilter('always', AstropyUserWarning)


@pytest.mark.skipif('not HAS_SCIPY')
class TestDAOStarFinder:
    @pytest.mark.parametrize(('threshold', 'fwhm'),
                             list(itertools.product(THRESHOLDS, FWHMS)))
    def test_daofind(self, threshold, fwhm):
        starfinder = DAOStarFinder(threshold, fwhm, sigma_radius=1.5)
        t = starfinder(DATA)
        datafn = ('daofind_test_thresh{0:04.1f}_fwhm{1:04.1f}'
                  '.txt'.format(threshold, fwhm))
        datafn = op.join(op.dirname(op.abspath(__file__)), 'data', datafn)
        t_ref = Table.read(datafn, format='ascii')

        assert t.colnames == t_ref.colnames
        for col in t.colnames:
            assert_allclose(t[col], t_ref[col])

    def test_daofind_threshold_fwhm_inputs(self):
        with pytest.raises(TypeError):
            DAOStarFinder(threshold=np.ones((2, 2)), fwhm=3.)

        with pytest.raises(TypeError):
            DAOStarFinder(threshold=3., fwhm=np.ones((2, 2)))

    def test_daofind_include_border(self):
        starfinder = DAOStarFinder(threshold=10, fwhm=2, sigma_radius=1.5,
                                   exclude_border=False)
        t = starfinder(DATA)
        assert len(t) == 20

    def test_daofind_exclude_border(self):
        starfinder = DAOStarFinder(threshold=10, fwhm=2, sigma_radius=1.5,
                                   exclude_border=True)
        t = starfinder(DATA)
        assert len(t) == 19

    def test_daofind_nosources(self):
        data = np.ones((3, 3))
        starfinder = DAOStarFinder(threshold=10, fwhm=1)
        t = starfinder(data)
        assert len(t) == 0

    def test_daofind_sharpness(self):
        """Sources found, but none pass the sharpness criteria."""
        starfinder = DAOStarFinder(threshold=50, fwhm=1.0, sharplo=1.)
        t = starfinder(DATA)
        assert len(t) == 0

    def test_daofind_roundness(self):
        """Sources found, but none pass the roundness criteria."""
        starfinder = DAOStarFinder(threshold=50, fwhm=1.0, roundlo=1.)
        t = starfinder(DATA)
        assert len(t) == 0

    def test_daofind_flux_negative(self):
        """Test handling of negative flux (here created by large sky)."""
        data = np.ones((5, 5))
        data[2, 2] = 10.
        starfinder = DAOStarFinder(threshold=0.1, fwhm=1.0, sky=10)
        t = starfinder(data)
        assert not np.isfinite(t['mag'])

    def test_daofind_negative_fit_peak(self):
        """
        Regression test that sources with negative fit peaks (i.e.
        hx/hy<=0) are excluded.
        """

        starfinder = DAOStarFinder(threshold=7., fwhm=1.5, roundlo=-np.inf,
                                   roundhi=np.inf, sharplo=-np.inf,
                                   sharphi=np.inf)
        t = starfinder(DATA)
        assert len(t) == 102

    def test_daofind_peakmax_filtering(self):
        """
        Regression test that objects with ``peak`` >= ``peakmax`` are
        filtered out.
        """

        peakmax = 20
        starfinder = DAOStarFinder(threshold=7., fwhm=1.5, roundlo=-np.inf,
                                   roundhi=np.inf, sharplo=-np.inf,
                                   sharphi=np.inf, peakmax=peakmax)
        t = starfinder(DATA)
        assert len(t) == 37
        assert all(t['peak'] < peakmax)

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
        t = starfinder(DATA)
        # combined with peakmax
        assert len(t) == brightest
        starfinder = DAOStarFinder(threshold=7., fwhm=1.5, roundlo=-np.inf,
                                   roundhi=np.inf, sharplo=-np.inf,
                                   sharphi=np.inf, brightest=brightest,
                                   peakmax=peakmax)
        t = starfinder(DATA)
        assert len(t) == 37

    def test_daofind_mask(self):
        """Test DAOStarFinder with a mask."""

        starfinder = DAOStarFinder(threshold=10, fwhm=1.5)
        mask = np.zeros_like(DATA, dtype=bool)
        mask[100:200] = True
        tbl1 = starfinder(DATA)
        tbl2 = starfinder(DATA, mask=mask)
        assert len(tbl1) > len(tbl2)


@pytest.mark.skipif('not HAS_SCIPY')
class TestIRAFStarFinder:
    @pytest.mark.parametrize(('threshold', 'fwhm'),
                             list(itertools.product(THRESHOLDS, FWHMS)))
    def test_irafstarfind(self, threshold, fwhm):
        starfinder = IRAFStarFinder(threshold, fwhm, sigma_radius=1.5)
        t = starfinder(DATA)
        datafn = ('irafstarfind_test_thresh{0:04.1f}_fwhm{1:04.1f}'
                  '.txt'.format(threshold, fwhm))
        datafn = op.join(op.dirname(op.abspath(__file__)), 'data', datafn)
        t_ref = Table.read(datafn, format='ascii')

        assert t.colnames == t_ref.colnames
        for col in t.colnames:
            assert_allclose(t[col], t_ref[col])

    def test_irafstarfind_threshold_fwhm_inputs(self):
        with pytest.raises(TypeError):
            IRAFStarFinder(threshold=np.ones((2, 2)), fwhm=3.)

        with pytest.raises(TypeError):
            IRAFStarFinder(threshold=3., fwhm=np.ones((2, 2)))

    def test_irafstarfind_nosources(self):
        data = np.ones((3, 3))
        starfinder = IRAFStarFinder(threshold=10, fwhm=1)
        t = starfinder(data)
        assert len(t) == 0

    def test_irafstarfind_sharpness(self):
        """Sources found, but none pass the sharpness criteria."""
        starfinder = IRAFStarFinder(threshold=50, fwhm=1.0, sharplo=2.)
        t = starfinder(DATA)
        assert len(t) == 0

    def test_irafstarfind_roundness(self):
        """Sources found, but none pass the roundness criteria."""
        starfinder = IRAFStarFinder(threshold=50, fwhm=1.0, roundlo=1.)
        t = starfinder(DATA)
        assert len(t) == 0

    def test_irafstarfind_sky(self):
        starfinder = IRAFStarFinder(threshold=25.0, fwhm=2.0, sky=10.)
        t = starfinder(DATA)
        assert len(t) == 4

    def test_irafstarfind_largesky(self):
        starfinder = IRAFStarFinder(threshold=25.0, fwhm=2.0, sky=100.)
        t = starfinder(DATA)
        assert len(t) == 0

    def test_irafstarfind_peakmax_filtering(self):
        """
        Regression test that objects with ``peak`` >= ``peakmax`` are
        filtered out.
        """
        peakmax = 20
        starfinder = IRAFStarFinder(threshold=7., fwhm=2, roundlo=-np.inf,
                                    roundhi=np.inf, sharplo=-np.inf,
                                    sharphi=np.inf, peakmax=peakmax)
        t = starfinder(DATA)
        assert len(t) == 117
        assert all(t['peak'] < peakmax)

    def test_irafstarfind_brightest_filtering(self):
        """
        Regression test that only top ``brightest`` objects are selected.
        """
        brightest = 40
        starfinder = IRAFStarFinder(threshold=7., fwhm=2, roundlo=-np.inf,
                                    roundhi=np.inf, sharplo=-np.inf,
                                    sharphi=np.inf, brightest=brightest)
        t = starfinder(DATA)
        assert len(t) == brightest

    def test_irafstarfind_mask(self):
        """Test IRAFStarFinder with a mask."""

        starfinder = IRAFStarFinder(threshold=10, fwhm=1.5)
        mask = np.zeros_like(DATA, dtype=bool)
        mask[100:200] = True
        tbl1 = starfinder(DATA)
        tbl2 = starfinder(DATA, mask=mask)
        assert len(tbl1) > len(tbl2)
