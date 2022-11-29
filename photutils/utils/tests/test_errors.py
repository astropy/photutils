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
    with pytest.raises(ValueError):
        calc_total_error(DATA, WRONG_SHAPE, EFFGAIN)


def test_gain_shape():
    with pytest.raises(ValueError):
        calc_total_error(DATA, BKG_ERROR, WRONG_SHAPE)


@pytest.mark.parametrize('effective_gain', (-1, -100))
def test_gain_negative(effective_gain):
    with pytest.raises(ValueError):
        calc_total_error(DATA, BKG_ERROR, effective_gain)


def test_gain_scalar():
    error_tot = calc_total_error(DATA, BKG_ERROR, 2.0)
    assert_allclose(error_tot, np.sqrt(2.0) * BKG_ERROR)


def test_gain_array():
    error_tot = calc_total_error(DATA, BKG_ERROR, EFFGAIN)
    assert_allclose(error_tot, np.sqrt(2.0) * BKG_ERROR)


def test_gain_zero():
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
    units = u.electron / u.s
    error_tot1 = calc_total_error(DATA * units, BKG_ERROR * units,
                                  EFFGAIN * u.s)
    assert error_tot1.unit == units
    error_tot2 = calc_total_error(DATA, BKG_ERROR, EFFGAIN)
    assert_allclose(error_tot1.value, error_tot2)


def test_error_units():
    units = u.electron / u.s
    with pytest.raises(ValueError):
        calc_total_error(DATA * units, BKG_ERROR * u.electron,
                         EFFGAIN * u.s)


def test_effgain_units():
    units = u.electron / u.s
    with pytest.raises(u.UnitsError):
        calc_total_error(DATA * units, BKG_ERROR * units, EFFGAIN * u.km)


def test_missing_bkgerror_units():
    units = u.electron / u.s
    with pytest.raises(ValueError):
        calc_total_error(DATA * units, BKG_ERROR, EFFGAIN * u.s)


def test_missing_effgain_units():
    units = u.electron / u.s
    with pytest.raises(ValueError):
        calc_total_error(DATA * units, BKG_ERROR * units,
                         EFFGAIN)
