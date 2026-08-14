# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for deprecation warnings when optional arguments are passed
positionally.
"""

import numpy as np
import pytest
from astropy.stats import SigmaClip
from astropy.utils.exceptions import AstropyDeprecationWarning

from photutils.background import (BiweightLocationBackground,
                                  BiweightScaleBackgroundRMS, LocalBackground,
                                  MADStdBackgroundRMS, MeanBackground,
                                  MedianBackground, MMMBackground,
                                  ModeEstimatorBackground,
                                  SExtractorBackground, StdBackgroundRMS)

BKG_CLASSES = [MeanBackground,
               MedianBackground,
               ModeEstimatorBackground,
               SExtractorBackground,
               BiweightLocationBackground]
BKGRMS_CLASSES = [StdBackgroundRMS,
                  MADStdBackgroundRMS,
                  BiweightScaleBackgroundRMS]


class TestBackgroundBasePositionalKwargs:
    """
    Test that __call__ on BackgroundBase subclasses warns for positional
    optional args.
    """

    def setup_method(self):
        self.data = np.arange(100, dtype=float)

    @pytest.mark.parametrize('cls', BKG_CLASSES)
    def test_call_positional_warns(self, cls):
        bkg = cls()
        match = '__call__'
        with pytest.warns(AstropyDeprecationWarning, match=match):
            bkg(self.data, None)

        match = 'calc_background'
        with pytest.warns(AstropyDeprecationWarning, match=match):
            bkg.calc_background(self.data, None)

    @pytest.mark.parametrize('cls', BKG_CLASSES)
    def test_call_keyword_no_warning(self, cls):
        bkg = cls()
        bkg(self.data, axis=None)


class TestBackgroundRMSBasePositionalKwargs:
    """
    Test that __call__ on BackgroundRMSBase subclasses warns for
    positional optional args.
    """

    def setup_method(self):
        self.data = np.arange(100, dtype=float)

    @pytest.mark.parametrize('cls', BKGRMS_CLASSES)
    def test_call_positional_warns(self, cls):
        bkgrms = cls()
        match = '__call__'
        with pytest.warns(AstropyDeprecationWarning, match=match):
            bkgrms(self.data, None)

        match = 'calc_background_rms'
        with pytest.warns(AstropyDeprecationWarning, match=match):
            bkgrms.calc_background_rms(self.data, None)

    @pytest.mark.parametrize('cls', BKGRMS_CLASSES)
    def test_call_keyword_no_warning(self, cls):
        bkgrms = cls()
        bkgrms(self.data, axis=None)


class TestEstimatorInitPositionalKwargs:
    """
    Test that estimator __init__ methods warn for positional optional
    args.
    """

    @pytest.mark.parametrize(('cls', 'args'), [
        (ModeEstimatorBackground, (3.0, 2.0)),
        (MMMBackground, (SigmaClip(sigma=3.0),)),
        (BiweightLocationBackground, (6.0,)),
        (BiweightScaleBackgroundRMS, (9.0,)),
    ])
    def test_init_positional_warns(self, cls, args):
        match = '__init__'
        with pytest.warns(AstropyDeprecationWarning, match=match):
            cls(*args)

    def test_init_keyword_no_warning(self):
        ModeEstimatorBackground(median_factor=3.0, mean_factor=2.0)
        MMMBackground(sigma_clip=SigmaClip(sigma=3.0))
        BiweightLocationBackground(c=6.0, M=None)
        BiweightScaleBackgroundRMS(c=9.0, M=None)


class TestLocalBackgroundPositionalKwargs:
    """
    Test that LocalBackground warns for positional optional args.
    """

    def setup_method(self):
        self.data = np.ones((101, 101))

    def test_init_positional_warns(self):
        match = '__init__'
        with pytest.warns(AstropyDeprecationWarning, match=match):
            LocalBackground(5, 10, MedianBackground())

    def test_init_keyword_no_warning(self):
        LocalBackground(5, 10, bkg_estimator=MedianBackground())

    def test_call_positional_warns(self):
        local_bkg = LocalBackground(5, 10)
        mask = np.zeros((101, 101), dtype=bool)
        match = '__call__'
        with pytest.warns(AstropyDeprecationWarning, match=match):
            local_bkg(self.data, 50, 50, mask)

    def test_call_keyword_no_warning(self):
        local_bkg = LocalBackground(5, 10)
        mask = np.zeros((101, 101), dtype=bool)
        local_bkg(self.data, 50, 50, mask=mask)
