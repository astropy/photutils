# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import numpy as np
from numpy.testing import assert_allclose
from astropy.tests.helper import pytest
from ...datasets.make import make_noise_image
from ..core import (MeanBackground, MedianBackground, ModeEstimatorBackground,
                    MMMBackground, SExtractorBackground,
                    BiweightLocationBackground, StdBackgroundRMS,
                    MADStdBackgroundRMS, BiweightMidvarianceBackgroundRMS)


BKG = 0.0
STD = 0.5
DATA = make_noise_image((100, 100), type='gaussian', mean=BKG, stddev=STD,
                        random_state=12345)

BKG_CLASS = [MeanBackground, MedianBackground, ModeEstimatorBackground,
             MMMBackground, SExtractorBackground, BiweightLocationBackground]

RMS_CLASS = [StdBackgroundRMS, MADStdBackgroundRMS,
             BiweightMidvarianceBackgroundRMS]


@pytest.mark.parametrize('bkg_class', BKG_CLASS)
def test_background(bkg_class):
    bkg = bkg_class(sigma=3.0)
    assert_allclose(bkg.calc_background(DATA), BKG, atol=1.e-2)
    assert_allclose(bkg(DATA), bkg.calc_background(DATA))


def test_sextrator_background_zero_std():
    data = np.ones((100, 100))
    bkg = SExtractorBackground(sigclip=False)
    assert_allclose(bkg.calc_background(data), 1.0)


def test_sextrator_background_skew():
    data = np.arange(100)
    data[70:] = 1.e7
    bkg = SExtractorBackground(sigclip=False)
    assert_allclose(bkg.calc_background(data), np.median(data))


@pytest.mark.parametrize('rms_class', RMS_CLASS)
def test_background_rms(rms_class):
    bkgrms = rms_class(sigma=3.0)
    assert_allclose(bkgrms.calc_background_rms(DATA), STD, atol=1.e-2)
    assert_allclose(bkgrms(DATA), bkgrms.calc_background_rms(DATA))
