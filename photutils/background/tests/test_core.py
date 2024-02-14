# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the core module.
"""

import astropy.units as u
import numpy as np
import pytest
from astropy.stats import SigmaClip
from numpy.testing import assert_allclose

from photutils.background.core import (BiweightLocationBackground,
                                       BiweightScaleBackgroundRMS,
                                       MADStdBackgroundRMS, MeanBackground,
                                       MedianBackground, MMMBackground,
                                       ModeEstimatorBackground,
                                       SExtractorBackground, StdBackgroundRMS)
from photutils.datasets import make_noise_image

BKG = 0.0
STD = 0.5
DATA = make_noise_image((100, 100), distribution='gaussian', mean=BKG,
                        stddev=STD, seed=0)

BKG_CLASS = [MeanBackground, MedianBackground, ModeEstimatorBackground,
             MMMBackground, SExtractorBackground, BiweightLocationBackground]
RMS_CLASS = [StdBackgroundRMS, MADStdBackgroundRMS,
             BiweightScaleBackgroundRMS]

SIGMA_CLIP = SigmaClip(sigma=3.0)


@pytest.mark.parametrize('bkg_class', BKG_CLASS)
def test_constant_background(bkg_class):
    data = np.ones((100, 100))
    bkg = bkg_class(sigma_clip=SIGMA_CLIP)
    bkgval = bkg.calc_background(data)
    assert not np.ma.isMaskedArray(bkgval)
    assert_allclose(bkgval, 1.0)
    assert_allclose(bkg(data), bkg.calc_background(data))

    mask = np.zeros(data.shape, dtype=bool)
    mask[0, 0:10] = True
    data = np.ma.MaskedArray(data, mask=mask)
    bkgval = bkg.calc_background(data)
    assert not np.ma.isMaskedArray(bkgval)
    assert_allclose(bkgval, 1.0)
    assert_allclose(bkg(data), bkg.calc_background(data))


@pytest.mark.parametrize('bkg_class', BKG_CLASS)
def test_background(bkg_class):
    bkg = bkg_class(sigma_clip=SIGMA_CLIP)
    bkgval = bkg.calc_background(DATA)
    assert not np.ma.isMaskedArray(bkgval)
    assert_allclose(bkgval, BKG, atol=0.02)
    assert_allclose(bkg(DATA), bkg.calc_background(DATA))


@pytest.mark.parametrize('bkg_class', BKG_CLASS)
def test_background_nosigmaclip(bkg_class):
    bkg = bkg_class(sigma_clip=None)
    bkgval = bkg.calc_background(DATA)
    assert not np.ma.isMaskedArray(bkgval)
    assert_allclose(bkgval, BKG, atol=0.1)
    assert_allclose(bkg(DATA), bkg.calc_background(DATA))

    # test with masked array
    mask = np.zeros(DATA.shape, dtype=bool)
    mask[0, 0:10] = True
    data = np.ma.MaskedArray(DATA, mask=mask)
    bkgval = bkg.calc_background(data)
    assert not np.ma.isMaskedArray(bkgval)
    assert_allclose(bkgval, BKG, atol=0.1)
    assert_allclose(bkg(data), bkg.calc_background(data))


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


def test_sourceextrator_background_zero_std():
    data = np.ones((100, 100))
    bkg = SExtractorBackground(sigma_clip=None)
    assert_allclose(bkg.calc_background(data), 1.0)


def test_sourceextrator_background_skew():
    data = np.arange(100)
    data[70:] = 1.0e7
    bkg = SExtractorBackground(sigma_clip=None)
    assert_allclose(bkg.calc_background(data), np.median(data))


@pytest.mark.parametrize('rms_class', RMS_CLASS)
def test_background_rms(rms_class):
    bkgrms = rms_class(sigma_clip=SIGMA_CLIP)
    assert_allclose(bkgrms.calc_background_rms(DATA), STD, atol=1.0e-2)
    assert_allclose(bkgrms(DATA), bkgrms.calc_background_rms(DATA))


@pytest.mark.parametrize('rms_class', RMS_CLASS)
def test_background_rms_axis(rms_class):
    bkgrms = rms_class(sigma_clip=SIGMA_CLIP)

    rms_arr = bkgrms.calc_background_rms(DATA, axis=0)
    rmsi = []
    for i in range(100):
        rmsi.append(bkgrms.calc_background_rms(DATA[:, i]))
    rmsi = np.array(rmsi)
    assert_allclose(rms_arr, rmsi)

    rms_arr = bkgrms.calc_background_rms(DATA, axis=1)
    rmsi = []
    for i in range(100):
        rmsi.append(bkgrms.calc_background_rms(DATA[i, :]))
    rmsi = np.array(rmsi)
    assert_allclose(rms_arr, rmsi)


@pytest.mark.parametrize('rms_class', RMS_CLASS)
def test_background_rms_nosigmaclip(rms_class):
    bkgrms = rms_class(sigma_clip=None)
    assert_allclose(bkgrms.calc_background_rms(DATA), STD, atol=1.0e-2)
    assert_allclose(bkgrms(DATA), bkgrms.calc_background_rms(DATA))

    # test with masked array
    mask = np.zeros(DATA.shape, dtype=bool)
    mask[0, 0:10] = True
    data = np.ma.MaskedArray(DATA, mask=mask)
    rms = bkgrms.calc_background_rms(data)
    assert not np.ma.isMaskedArray(bkgrms)
    assert_allclose(rms, STD, atol=0.01)
    assert_allclose(bkgrms(data), bkgrms.calc_background_rms(data))


@pytest.mark.parametrize('bkg_class', BKG_CLASS)
def test_background_masked(bkg_class):
    bkg = bkg_class(sigma_clip=None)
    mask = np.zeros(DATA.shape, dtype=bool)
    mask[0, 0:10] = True
    data = np.ma.MaskedArray(DATA, mask=mask)

    # test masked array with masked=True with axis
    bkgval1 = bkg(data, masked=True, axis=1)
    bkgval2 = bkg.calc_background(data, masked=True, axis=1)
    assert np.ma.isMaskedArray(bkgval1)
    assert_allclose(np.mean(bkgval1), np.mean(bkgval2))
    assert_allclose(np.mean(bkgval1), BKG, atol=0.01)

    # test masked array with masked=False with axis
    bkgval2 = bkg.calc_background(data, masked=False, axis=1)
    assert not np.ma.isMaskedArray(bkgval2)
    assert_allclose(np.nanmean(bkgval2), BKG, atol=0.01)


@pytest.mark.parametrize('rms_class', RMS_CLASS)
def test_background_rms_masked(rms_class):
    bkgrms = rms_class(sigma_clip=None)
    mask = np.zeros(DATA.shape, dtype=bool)
    mask[0, 0:10] = True
    data = np.ma.MaskedArray(DATA, mask=mask)

    # test masked array with masked=True with axis
    rms1 = bkgrms(data, masked=True, axis=1)
    rms2 = bkgrms.calc_background_rms(data, masked=True, axis=1)
    assert np.ma.isMaskedArray(rms1)
    assert_allclose(np.mean(rms1), np.mean(rms2))
    assert_allclose(np.mean(rms1), STD, atol=0.01)

    # test masked array with masked=False with axis
    rms3 = bkgrms.calc_background_rms(data, masked=False, axis=1)
    assert not np.ma.isMaskedArray(rms3)
    assert_allclose(np.nanmean(rms3), STD, atol=0.01)


@pytest.mark.parametrize('bkg_class', BKG_CLASS)
def test_background_axis_tuple(bkg_class):
    bkg = bkg_class(sigma_clip=None)
    bkg_val1 = bkg.calc_background(DATA, axis=None)
    bkg_val2 = bkg.calc_background(DATA, axis=(0, 1))
    assert_allclose(bkg_val1, bkg_val2)


@pytest.mark.parametrize('bkg_class', BKG_CLASS)
def test_background_units(bkg_class):
    data = np.ones((100, 100)) << u.Jy
    bkg = bkg_class(sigma_clip=SIGMA_CLIP)
    bkgval = bkg.calc_background(data)
    assert isinstance(bkgval, u.Quantity)


@pytest.mark.parametrize('rms_class', RMS_CLASS)
def test_background_rms_units(rms_class):
    data = np.ones((100, 100)) << u.Jy
    bkgrms = rms_class(sigma_clip=SIGMA_CLIP)
    rmsval = bkgrms.calc_background_rms(data)
    assert isinstance(rmsval, u.Quantity)


@pytest.mark.parametrize('bkg_class', BKG_CLASS)
def test_background_invalid_sigmaclip(bkg_class):
    with pytest.raises(TypeError):
        bkg_class(sigma_clip=3)


@pytest.mark.parametrize('rms_class', RMS_CLASS)
def test_background_rms_invalid_sigmaclip(rms_class):
    with pytest.raises(TypeError):
        rms_class(sigma_clip=3)


@pytest.mark.parametrize('bkg_class', BKG_CLASS)
def test_background_repr(bkg_class):
    bkg = bkg_class()
    bkg_repr = repr(bkg)
    assert bkg_repr == str(bkg)
    assert bkg_repr.startswith(f'{bkg.__class__.__name__}')


@pytest.mark.parametrize('rms_class', RMS_CLASS)
def test_background_rms_repr(rms_class):
    bkgrms = rms_class()
    rms_repr = repr(bkgrms)
    assert rms_repr == str(bkgrms)
    assert rms_repr.startswith(f'{bkgrms.__class__.__name__}')
