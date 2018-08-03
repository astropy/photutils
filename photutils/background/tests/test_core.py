# Licensed under a 3-clause BSD style license - see LICENSE.rst

import pytest
import numpy as np
from numpy.testing import assert_allclose
from astropy.stats import SigmaClip

from ...datasets.make import make_noise_image
from ..core import (MeanBackground, MedianBackground,
                    ModeEstimatorBackground, MMMBackground,
                    SExtractorBackground, BiweightLocationBackground,
                    StdBackgroundRMS, MADStdBackgroundRMS,
                    BiweightScaleBackgroundRMS)


BKG = 0.0
STD = 0.5
DATA = make_noise_image((100, 100), type='gaussian', mean=BKG, stddev=STD,
                        random_state=12345)

BKG_CLASS0 = [MeanBackground, MedianBackground, ModeEstimatorBackground,
              MMMBackground, SExtractorBackground]
# BiweightLocationBackground cannot handle a constant background
# (astropy.stats.biweight_location needs to be fixed)
BKG_CLASS = BKG_CLASS0 + [BiweightLocationBackground]

RMS_CLASS = [StdBackgroundRMS, MADStdBackgroundRMS,
             BiweightScaleBackgroundRMS]

SIGMA_CLIP = SigmaClip(sigma=3.)


@pytest.mark.parametrize('bkg_class', BKG_CLASS0)
def test_constant_background(bkg_class):
    data = np.ones((100, 100))
    bkg = bkg_class(sigma_clip=SIGMA_CLIP)
    bkgval = bkg.calc_background(data)
    assert not np.ma.isMaskedArray(bkgval)
    assert_allclose(bkgval, 1.0)
    assert_allclose(bkg(data), bkg.calc_background(data))


@pytest.mark.parametrize('bkg_class', BKG_CLASS)
def test_background(bkg_class):
    bkg = bkg_class(sigma_clip=SIGMA_CLIP)
    bkgval = bkg.calc_background(DATA)
    assert not np.ma.isMaskedArray(bkgval)
    assert_allclose(bkgval, BKG, atol=1.e-2)
    assert_allclose(bkg(DATA), bkg.calc_background(DATA))


@pytest.mark.parametrize('bkg_class', BKG_CLASS)
def test_background_axis(bkg_class):
    bkg = bkg_class(sigma_clip=SIGMA_CLIP)

    bkg_arr = bkg.calc_background(DATA, axis=0)
    bkgi = []
    for i in range(100):
        bkgi.append(bkg.calc_background(DATA[:, i]))
    bkgi = np.array(bkgi)
    assert_allclose(bkg_arr, bkgi)

    bkg_arr = bkg.calc_background(DATA, axis=1)
    bkgi = []
    for i in range(100):
        bkgi.append(bkg.calc_background(DATA[i, :]))
    bkgi = np.array(bkgi)
    assert_allclose(bkg_arr, bkgi)


def test_sextrator_background_zero_std():
    data = np.ones((100, 100))
    bkg = SExtractorBackground(sigma_clip=None)
    assert_allclose(bkg.calc_background(data), 1.0)


def test_sextrator_background_skew():
    data = np.arange(100)
    data[70:] = 1.e7
    bkg = SExtractorBackground(sigma_clip=None)
    assert_allclose(bkg.calc_background(data), np.median(data))


@pytest.mark.parametrize('rms_class', RMS_CLASS)
def test_background_rms(rms_class):
    bkgrms = rms_class(sigma_clip=SIGMA_CLIP)
    assert_allclose(bkgrms.calc_background_rms(DATA), STD, atol=1.e-2)
    assert_allclose(bkgrms(DATA), bkgrms.calc_background_rms(DATA))
