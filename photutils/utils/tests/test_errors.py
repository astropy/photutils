# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the errors module.
"""

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
