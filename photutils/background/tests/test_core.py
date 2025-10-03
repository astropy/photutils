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
from photutils.utils._stats import nanmean

BACKGROUND_CLASSES = [MeanBackground, MedianBackground,
                      ModeEstimatorBackground, MMMBackground,
                      SExtractorBackground, BiweightLocationBackground]

BACKGROUND_RMS_CLASSES = [StdBackgroundRMS, MADStdBackgroundRMS,
                          BiweightScaleBackgroundRMS]


@pytest.fixture
def bkg_value():
    """
    Expected background value for test data.
    """
    return 12.4


@pytest.fixture
def std_value():
    """
    Expected standard deviation for test data.
    """
    return 0.42


@pytest.fixture
def test_data(bkg_value, std_value):
    """
    Create test data with Gaussian noise.
    """
    return make_noise_image((100, 100), distribution='gaussian',
                            mean=bkg_value, stddev=std_value, seed=0)


@pytest.fixture
def sigma_clip():
    """
    Sigma clipping instance for tests.
    """
    return SigmaClip(sigma=3.0)


@pytest.mark.parametrize('bkg_class', BACKGROUND_CLASSES)
def test_constant_background(bkg_class, sigma_clip):
    data = np.ones((100, 100))
    bkg = bkg_class(sigma_clip=sigma_clip)
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


@pytest.mark.parametrize('bkg_class', BACKGROUND_CLASSES)
def test_background(bkg_class, sigma_clip, test_data, bkg_value):
    bkg = bkg_class(sigma_clip=sigma_clip)
    bkgval = bkg.calc_background(test_data)
    assert not np.ma.isMaskedArray(bkgval)
    assert_allclose(bkgval, bkg_value, atol=0.017)
    assert_allclose(bkg(test_data), bkg.calc_background(test_data))


@pytest.mark.parametrize('bkg_class', BACKGROUND_CLASSES)
def test_background_nosigmaclip(bkg_class, test_data, bkg_value):
    bkg = bkg_class(sigma_clip=None)
    bkgval = bkg.calc_background(test_data)
    assert not np.ma.isMaskedArray(bkgval)
    assert_allclose(bkgval, bkg_value, atol=0.017)
    assert_allclose(bkg(test_data), bkg.calc_background(test_data))

    # test with masked array
    mask = np.zeros(test_data.shape, dtype=bool)
    mask[0, 0:10] = True
    data = np.ma.MaskedArray(test_data, mask=mask)
    bkgval = bkg.calc_background(data)
    assert not np.ma.isMaskedArray(bkgval)
    assert_allclose(bkgval, bkg_value, atol=0.018)
    assert_allclose(bkg(data), bkg.calc_background(data))


@pytest.mark.parametrize('bkg_class', BACKGROUND_CLASSES)
def test_background_axis(bkg_class, sigma_clip, test_data):
    bkg = bkg_class(sigma_clip=sigma_clip)

    bkg_arr = bkg.calc_background(test_data, axis=0)
    bkgi = np.array([bkg.calc_background(test_data[:, i]) for i in range(100)])
    assert_allclose(bkg_arr, bkgi)

    bkg_arr = bkg.calc_background(test_data, axis=1)
    bkgi = []
    for i in range(100):
        bkgi.append(bkg.calc_background(test_data[i, :]))
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


@pytest.mark.parametrize('bkg_class', BACKGROUND_CLASSES)
def test_background_ndim(bkg_class):
    data1 = np.ones((1, 100, 100))
    data2 = np.ones((1, 100 * 100))
    data3 = np.ones((1, 1, 100 * 100))
    data4 = np.ones((1, 1, 1, 100 * 100))

    bkg = bkg_class(sigma_clip=None)
    val = bkg(data1, axis=None)
    assert np.ndim(val) == 0
    val = bkg(data1, axis=(1, 2))
    assert val.shape == (1,)
    val = bkg(data1, axis=-1)
    assert val.shape == (1, 100)
    val = bkg(data2, axis=-1)
    assert val.shape == (1,)
    val = bkg(data3, axis=-1)
    assert val.shape == (1, 1)
    val = bkg(data4, axis=-1)
    assert val.shape == (1, 1, 1)
    val = bkg(data4, axis=(2, 3))
    assert val.shape == (1, 1)
    val = bkg(data4, axis=(1, 2, 3))
    assert val.shape == (1,)
    val = bkg(data4, axis=(0, 1, 2))
    assert val.shape == (10000,)


@pytest.mark.parametrize('bkgrms_class', BACKGROUND_RMS_CLASSES)
def test_background_rms_ndim(bkgrms_class):
    data1 = np.ones((1, 100, 100))
    data2 = np.ones((1, 100 * 100))
    data3 = np.ones((1, 1, 100 * 100))
    data4 = np.ones((1, 1, 1, 100 * 100))

    bkgrms = bkgrms_class(sigma_clip=None)
    val = bkgrms(data1, axis=None)
    assert np.ndim(val) == 0
    val = bkgrms(data1, axis=(1, 2))
    assert val.shape == (1,)
    val = bkgrms(data1, axis=-1)
    assert val.shape == (1, 100)
    val = bkgrms(data2, axis=-1)
    assert val.shape == (1,)
    val = bkgrms(data3, axis=-1)
    assert val.shape == (1, 1)
    val = bkgrms(data4, axis=-1)
    assert val.shape == (1, 1, 1)
    val = bkgrms(data4, axis=(2, 3))
    assert val.shape == (1, 1)
    val = bkgrms(data4, axis=(1, 2, 3))
    assert val.shape == (1,)
    val = bkgrms(data4, axis=(0, 1, 2))
    assert val.shape == (10000,)


@pytest.mark.parametrize('rms_class', BACKGROUND_RMS_CLASSES)
def test_background_rms(rms_class, sigma_clip, test_data, std_value):
    bkgrms = rms_class(sigma_clip=sigma_clip)
    assert_allclose(bkgrms.calc_background_rms(test_data), std_value,
                    atol=0.007)
    assert_allclose(bkgrms(test_data), bkgrms.calc_background_rms(test_data))


@pytest.mark.parametrize('rms_class', BACKGROUND_RMS_CLASSES)
def test_background_rms_axis(rms_class, sigma_clip, test_data):
    bkgrms = rms_class(sigma_clip=sigma_clip)

    rms_arr = bkgrms.calc_background_rms(test_data, axis=0)
    rmsi = np.array([bkgrms.calc_background_rms(test_data[:, i])
                     for i in range(100)])
    assert_allclose(rms_arr, rmsi)

    rms_arr = bkgrms.calc_background_rms(test_data, axis=1)
    rmsi = []
    for i in range(100):
        rmsi.append(bkgrms.calc_background_rms(test_data[i, :]))
    rmsi = np.array(rmsi)
    assert_allclose(rms_arr, rmsi)


@pytest.mark.parametrize('rms_class', BACKGROUND_RMS_CLASSES)
def test_background_rms_nosigmaclip(rms_class, test_data, std_value):
    bkgrms = rms_class(sigma_clip=None)
    assert_allclose(bkgrms.calc_background_rms(test_data), std_value,
                    atol=0.004)
    assert_allclose(bkgrms(test_data), bkgrms.calc_background_rms(test_data))

    # test with masked array
    mask = np.zeros(test_data.shape, dtype=bool)
    mask[0, 0:10] = True
    data = np.ma.MaskedArray(test_data, mask=mask)
    rms = bkgrms.calc_background_rms(data)
    assert not np.ma.isMaskedArray(bkgrms)
    assert_allclose(rms, std_value, atol=0.004)
    assert_allclose(bkgrms(data), bkgrms.calc_background_rms(data))


@pytest.mark.parametrize('bkg_class', BACKGROUND_CLASSES)
def test_background_masked(bkg_class, test_data, bkg_value):
    bkg = bkg_class(sigma_clip=None)
    mask = np.zeros(test_data.shape, dtype=bool)
    mask[0, 0:10] = True
    data = np.ma.MaskedArray(test_data, mask=mask)

    # test masked array with masked=True with axis
    bkgval1 = bkg(data, masked=True, axis=1)
    bkgval2 = bkg.calc_background(data, masked=True, axis=1)
    assert np.ma.isMaskedArray(bkgval1)
    assert_allclose(np.mean(bkgval1), np.mean(bkgval2))
    assert_allclose(np.mean(bkgval1), bkg_value, atol=0.004)

    # test masked array with masked=False with axis
    bkgval2 = bkg.calc_background(data, masked=False, axis=1)
    assert not np.ma.isMaskedArray(bkgval2)
    assert_allclose(nanmean(bkgval2), bkg_value, atol=0.004)


@pytest.mark.parametrize('rms_class', BACKGROUND_RMS_CLASSES)
def test_background_rms_masked(rms_class, test_data, std_value):
    bkgrms = rms_class(sigma_clip=None)
    mask = np.zeros(test_data.shape, dtype=bool)
    mask[0, 0:10] = True
    data = np.ma.MaskedArray(test_data, mask=mask)

    # test masked array with masked=True with axis
    rms1 = bkgrms(data, masked=True, axis=1)
    rms2 = bkgrms.calc_background_rms(data, masked=True, axis=1)
    assert np.ma.isMaskedArray(rms1)
    assert_allclose(np.mean(rms1), np.mean(rms2))
    assert_allclose(np.mean(rms1), std_value, atol=0.04)

    # test masked array with masked=False with axis
    rms3 = bkgrms.calc_background_rms(data, masked=False, axis=1)
    assert not np.ma.isMaskedArray(rms3)
    assert_allclose(nanmean(rms3), std_value, atol=0.04)


@pytest.mark.parametrize('bkg_class', BACKGROUND_CLASSES)
def test_background_axis_tuple(bkg_class, test_data):
    bkg = bkg_class(sigma_clip=None)
    bkg_val1 = bkg.calc_background(test_data, axis=None)
    bkg_val2 = bkg.calc_background(test_data, axis=(0, 1))
    assert_allclose(bkg_val1, bkg_val2)


@pytest.mark.parametrize('bkg_class', BACKGROUND_CLASSES)
def test_background_units(bkg_class, sigma_clip):
    data = np.ones((100, 100)) << u.Jy
    bkg = bkg_class(sigma_clip=sigma_clip)
    bkgval = bkg.calc_background(data)
    assert isinstance(bkgval, u.Quantity)


@pytest.mark.parametrize('rms_class', BACKGROUND_RMS_CLASSES)
def test_background_rms_units(rms_class, sigma_clip):
    data = np.ones((100, 100)) << u.Jy
    bkgrms = rms_class(sigma_clip=sigma_clip)
    rmsval = bkgrms.calc_background_rms(data)
    assert isinstance(rmsval, u.Quantity)


@pytest.mark.parametrize('bkg_class', BACKGROUND_CLASSES)
def test_background_invalid_sigmaclip(bkg_class):
    match = 'sigma_clip must be an astropy SigmaClip instance or None'
    with pytest.raises(TypeError, match=match):
        bkg_class(sigma_clip=3)


@pytest.mark.parametrize('rms_class', BACKGROUND_RMS_CLASSES)
def test_background_rms_invalid_sigmaclip(rms_class):
    match = 'sigma_clip must be an astropy SigmaClip instance or None'
    with pytest.raises(TypeError, match=match):
        rms_class(sigma_clip=3)


@pytest.mark.parametrize('bkg_class', BACKGROUND_CLASSES)
def test_background_repr(bkg_class):
    bkg = bkg_class()
    bkg_repr = repr(bkg)
    assert bkg_repr == str(bkg)
    assert bkg_repr.startswith(f'{bkg.__class__.__name__}')


@pytest.mark.parametrize('rms_class', BACKGROUND_RMS_CLASSES)
def test_background_rms_repr(rms_class):
    bkgrms = rms_class()
    rms_repr = repr(bkgrms)
    assert rms_repr == str(bkgrms)
    assert rms_repr.startswith(f'{bkgrms.__class__.__name__}')


@pytest.mark.parametrize(('median_factor', 'mean_factor'), [
    (3.0, 2.0),  # default
    (2.5, 1.5),  # custom
    (1.0, 0.0),  # edge case
])
def test_mode_estimator_parameters(median_factor, mean_factor):
    """
    Test ModeEstimatorBackground with different parameters.
    """
    rng = np.random.default_rng(0)
    data = rng.standard_normal((100, 100)) + 10
    bkg = ModeEstimatorBackground(median_factor=median_factor,
                                  mean_factor=mean_factor,
                                  sigma_clip=None)
    result = bkg.calc_background(data)

    # Verify the formula is applied correctly
    expected = (median_factor * np.median(data)
                - mean_factor * np.mean(data))
    assert_allclose(result, expected)


@pytest.mark.parametrize('bkg_class', BACKGROUND_CLASSES)
def test_background_all_nan(bkg_class):
    """
    Test behavior with all-NaN data.
    """
    data = np.full((10, 10), np.nan)
    bkg = bkg_class(sigma_clip=None)
    result = bkg.calc_background(data)
    assert np.isnan(result)


@pytest.mark.parametrize('bkg_class', BACKGROUND_CLASSES)
@pytest.mark.parametrize('dtype', [np.int8, np.int16, np.int32, np.int64,
                                   np.float16, np.float32, np.float64])
def test_background_dtypes(bkg_class, dtype):
    """
    Test with various numeric data types.
    """
    data = np.ones((50, 50), dtype=dtype)
    bkg = bkg_class(sigma_clip=None)
    result = bkg.calc_background(data)
    assert_allclose(result, 1.0)


@pytest.mark.parametrize('bkg_class', BACKGROUND_CLASSES)
def test_background_1d(bkg_class):
    """
    Test with 1D arrays.
    """
    rng = np.random.default_rng(0)
    data = rng.standard_normal(1000) + 5.0
    bkg = bkg_class(sigma_clip=None)
    result = bkg.calc_background(data)
    assert_allclose(result, 5.0, atol=0.2)


@pytest.mark.parametrize('bkg_class', BACKGROUND_CLASSES)
def test_background_empty_masked_array(bkg_class):
    """
    Test with completely masked array.
    """
    data = np.ma.array(np.ones((10, 10)), mask=True)
    bkg = bkg_class(sigma_clip=None)
    result = bkg.calc_background(data)
    assert np.isnan(result)


def test_background_sigmaclip_mean():
    """
    Test that sigma clipping removes outliers for MeanBackground.

    Uses MeanBackground because median and biweight are already robust
    to outliers without sigma clipping.
    """
    data = np.ones((100, 100))
    data[0:5, 0:5] = 1000  # Add outliers

    bkg_no_clip = MeanBackground(sigma_clip=None)
    bkg_with_clip = MeanBackground(sigma_clip=SigmaClip(sigma=3.0))

    result_no_clip = bkg_no_clip.calc_background(data)
    result_with_clip = bkg_with_clip.calc_background(data)

    # With clipping should be closer to 1.0
    assert abs(result_with_clip - 1.0) < abs(result_no_clip - 1.0)
    # Mean without clipping should be > 1.0 due to outliers
    assert result_no_clip > 1.0
    # Mean with clipping should be close to 1.0
    assert_allclose(result_with_clip, 1.0, atol=0.01)
