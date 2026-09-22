# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the errors module.
"""

import tracemalloc

import astropy.units as u
import numpy as np
import pytest
from numpy.testing import assert_allclose

from photutils.utils.errors import calc_total_error

SHAPE = (5, 5)
DATAVAL = 2.0
DATA = np.ones(SHAPE) * DATAVAL
BKG_ERROR = np.ones(SHAPE)
EFFGAIN = np.ones(SHAPE) * DATAVAL
BACKGROUND = np.ones(SHAPE)
WRONG_SHAPE = np.ones((2, 2))


def test_error_shape():
    """
    Test that mismatched bkg_error shape raises ValueError.
    """
    match = 'bkg_error must have the same shape as the input data'
    with pytest.raises(ValueError, match=match):
        calc_total_error(DATA, WRONG_SHAPE, EFFGAIN)


def test_gain_shape():
    """
    Test that mismatched effective_gain shape raises ValueError.
    """
    match = 'must have the same shape as the input data'
    with pytest.raises(ValueError, match=match):
        calc_total_error(DATA, BKG_ERROR, WRONG_SHAPE)


@pytest.mark.parametrize('effective_gain', [-1, -100])
def test_gain_negative(effective_gain):
    """
    Test that negative effective_gain raises ValueError.
    """
    match = 'effective_gain must be non-negative everywhere'
    with pytest.raises(ValueError, match=match):
        calc_total_error(DATA, BKG_ERROR, effective_gain)


def test_gain_scalar():
    """
    Test calc_total_error with a scalar effective_gain.
    """
    error_tot = calc_total_error(DATA, BKG_ERROR, 2.0)
    assert_allclose(error_tot, np.sqrt(2.0) * BKG_ERROR)


def test_integer_data():
    """
    Test calc_total_error with integer data and bkg_error inputs.
    """
    data_int = (DATA * 5).astype(int)
    error_tot = calc_total_error(data_int, BKG_ERROR, 2.0)
    assert error_tot.dtype == float
    expected = calc_total_error(data_int.astype(float), BKG_ERROR, 2.0)
    assert_allclose(error_tot, expected)

    error_tot = calc_total_error(data_int, BKG_ERROR.astype(int), 2.0)
    assert_allclose(error_tot, expected)


def test_gain_array():
    """
    Test calc_total_error with an array effective_gain.
    """
    error_tot = calc_total_error(DATA, BKG_ERROR, EFFGAIN)
    assert_allclose(error_tot, np.sqrt(2.0) * BKG_ERROR)


def test_gain_zero():
    """
    Test calc_total_error with zero effective_gain values.
    """
    error_tot = calc_total_error(DATA, BKG_ERROR, 0.0)
    assert_allclose(error_tot, BKG_ERROR)

    effgain = np.copy(EFFGAIN)
    effgain[0, 0] = 0
    effgain[1, 1] = 0
    mask = (effgain == 0)
    error_tot = calc_total_error(DATA, BKG_ERROR, effgain)
    assert_allclose(error_tot[mask], BKG_ERROR[mask])
    assert_allclose(error_tot[~mask], np.sqrt(2))


def test_units():
    """
    Test calc_total_error with Quantity inputs.
    """
    units = u.electron / u.s
    error_tot1 = calc_total_error(DATA * units, BKG_ERROR * units,
                                  EFFGAIN * u.s)
    assert error_tot1.unit == units
    error_tot2 = calc_total_error(DATA, BKG_ERROR, EFFGAIN)
    assert_allclose(error_tot1.value, error_tot2)


def test_error_units():
    """
    Test that mismatched data and bkg_error units raises ValueError.
    """
    units = u.electron / u.s
    match = 'must have the same units'
    with pytest.raises(ValueError, match=match):
        calc_total_error(DATA * units, BKG_ERROR * u.electron,
                         EFFGAIN * u.s)


def test_effgain_units():
    """
    Test that invalid effective_gain units raises UnitsError.
    """
    units = u.electron / u.s
    match = 'it must have count units'
    with pytest.raises(u.UnitsError, match=match):
        calc_total_error(DATA * units, BKG_ERROR * units, EFFGAIN * u.km)


def test_missing_bkgerror_units():
    """
    Test that missing bkg_error units raises ValueError.
    """
    units = u.electron / u.s
    match = 'all must have units'
    with pytest.raises(ValueError, match=match):
        calc_total_error(DATA * units, BKG_ERROR, EFFGAIN * u.s)


def test_missing_effgain_units():
    """
    Test that missing effective_gain units raises ValueError.
    """
    units = u.electron / u.s
    match = 'all must have units'
    with pytest.raises(ValueError, match=match):
        calc_total_error(DATA * units, BKG_ERROR * units,
                         EFFGAIN)


def make_dtype_inputs(shape=(40, 50)):
    """
    Make float64 data (with negative values), bkg_error, and
    effective_gain (with zero values) arrays.
    """
    rng = np.random.default_rng(0)
    data = rng.normal(20.0, 15.0, shape)
    bkg_error = rng.uniform(1.0, 3.0, shape)
    effective_gain = rng.uniform(0.5, 4.0, shape)
    effective_gain[3, 4] = 0.0
    effective_gain[10:12, 20:25] = 0.0
    return data, bkg_error, effective_gain


@pytest.mark.parametrize(
    ('data_dtype', 'bkg_dtype', 'gain_dtype', 'expected'),
    [(np.float32, np.float32, None, np.float32),
     (np.float32, np.float32, np.float32, np.float32),
     (np.float64, np.float64, None, np.float64),
     (np.float32, np.float64, None, np.float64),
     (np.float64, np.float32, None, np.float64),
     (np.float32, np.float32, np.float64, np.float64),
     (np.float16, np.float16, None, np.float32),
     (np.float16, np.float16, np.float16, np.float32),
     (np.float16, np.float32, None, np.float32),
     ('>f4', '>f4', None, np.float32),
     (np.int32, np.float32, None, np.float64),
     (np.float32, np.int16, None, np.float64),
     (np.float32, np.float32, np.int64, np.float64),
     (np.int16, np.int16, None, np.float64)])
def test_output_dtype(data_dtype, bkg_dtype, gain_dtype, expected):
    """
    Test that the output dtype follows the NumPy promotion of the
    floating-point inputs, with a minimum of float32, and that integer
    inputs give float64.

    `None` for ``gain_dtype`` is a Python scalar effective_gain, which
    does not affect the output dtype.
    """
    data, bkg_error, effective_gain = make_dtype_inputs()
    data = data.astype(data_dtype)
    bkg_error = bkg_error.astype(bkg_dtype)
    if gain_dtype is None:
        effective_gain = 2.0
    else:
        effective_gain = np.ceil(effective_gain).astype(gain_dtype)
        effective_gain[0, 0] = 0

    error_tot = calc_total_error(data, bkg_error, effective_gain)
    assert error_tot.dtype == np.dtype(expected)
    assert error_tot.dtype.isnative
    assert error_tot.shape == data.shape

    # The values match the float64 calculation to the output precision
    gain64 = (effective_gain if gain_dtype is None
              else effective_gain.astype(float))
    error_ref = calc_total_error(data.astype(float), bkg_error.astype(float),
                                 gain64)
    assert error_ref.dtype == np.float64
    rtol = 1e-6 if expected is np.float32 else 1e-14
    assert_allclose(error_tot, error_ref, rtol=rtol)


def test_output_dtype_numpy_scalar_gain():
    """
    Test that a NumPy scalar effective_gain does not affect the output
    dtype.
    """
    data, bkg_error, _ = make_dtype_inputs()
    data = data.astype(np.float32)
    bkg_error = bkg_error.astype(np.float32)
    expected = calc_total_error(data, bkg_error, 2.0)
    for gain in (np.float64(2.0), np.float32(2.0), np.int64(2),
                 np.array(2.0)):
        error_tot = calc_total_error(data, bkg_error, gain)
        assert error_tot.dtype == np.float32
        assert_allclose(error_tot, expected, rtol=1e-6)


def test_output_dtype_units():
    """
    Test that the output dtype is preserved for Quantity inputs.
    """
    data, bkg_error, effective_gain = make_dtype_inputs()
    units = u.electron / u.s
    data = data.astype(np.float32) << units
    bkg_error = bkg_error.astype(np.float32) << units
    for gain in (2.0 * u.s, effective_gain.astype(np.float32) << u.s):
        error_tot = calc_total_error(data, bkg_error, gain)
        assert error_tot.unit == units
        assert error_tot.dtype == np.float32
        expected = calc_total_error(data.value, bkg_error.value, gain.value)
        assert_allclose(error_tot.value, expected)


def test_inputs_not_modified():
    """
    Test that the input arrays are not modified.
    """
    data, bkg_error, effective_gain = make_dtype_inputs()
    for dtype in (np.float32, np.float64):
        inputs = [arr.astype(dtype) for arr in (data, bkg_error,
                                                effective_gain)]
        copies = [arr.copy() for arr in inputs]
        calc_total_error(*inputs)
        calc_total_error(inputs[0], inputs[1], 2.0)
        for arr, copy in zip(inputs, copies, strict=True):
            assert_allclose(arr, copy, rtol=0)


def test_scalar_gain_memory():
    """
    Test that a scalar effective_gain is not expanded to a full-size
    float64 image.
    """
    rng = np.random.default_rng(0)
    data = rng.normal(20.0, 15.0, (1024, 1024)).astype(np.float32)
    bkg_error = np.ones(data.shape, dtype=np.float32)

    tracemalloc.start()
    try:
        start = tracemalloc.get_traced_memory()[0]
        error_tot = calc_total_error(data, bkg_error, 2.0)
        peak = tracemalloc.get_traced_memory()[1] - start
    finally:
        tracemalloc.stop()

    assert error_tot.dtype == np.float32
    # The float32 output needs 4 bytes per pixel. A float64 copy of the
    # data plus a float64 gain image would need 16.
    assert peak < 14 * data.size
