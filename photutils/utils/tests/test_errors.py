# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
from numpy.testing import assert_allclose
import pytest
import astropy.units as u

from ..errors import calc_total_error


SHAPE = (5, 5)
DATAVAL = 2.
DATA = np.ones(SHAPE) * DATAVAL
MASK = np.zeros_like(DATA, dtype=bool)
MASK[2, 2] = True
BKG_ERROR = np.ones(SHAPE)
EFFGAIN = np.ones(SHAPE) * DATAVAL
BACKGROUND = np.ones(SHAPE)
WRONG_SHAPE = np.ones((2, 2))


class TestCalculateTotalError:
    def test_error_shape(self):
        with pytest.raises(ValueError):
            calc_total_error(DATA, WRONG_SHAPE, EFFGAIN)

    def test_gain_shape(self):
        with pytest.raises(ValueError):
            calc_total_error(DATA, BKG_ERROR, WRONG_SHAPE)

    @pytest.mark.parametrize('effective_gain', (0, -1))
    def test_gain_le_zero(self, effective_gain):
        with pytest.raises(ValueError):
            calc_total_error(DATA, BKG_ERROR, effective_gain)

    def test_gain_scalar(self):
        error_tot = calc_total_error(DATA, BKG_ERROR, 2.)
        assert_allclose(error_tot, np.sqrt(2.) * BKG_ERROR)

    def test_gain_array(self):
        error_tot = calc_total_error(DATA, BKG_ERROR, EFFGAIN)
        assert_allclose(error_tot, np.sqrt(2.) * BKG_ERROR)

    def test_units(self):
        units = u.electron / u.s
        error_tot1 = calc_total_error(DATA * units, BKG_ERROR * units,
                                      EFFGAIN * u.s)
        assert error_tot1.unit == units
        error_tot2 = calc_total_error(DATA, BKG_ERROR, EFFGAIN)
        assert_allclose(error_tot1.value, error_tot2)

    def test_error_units(self):
        units = u.electron / u.s
        with pytest.raises(ValueError):
            calc_total_error(DATA * units, BKG_ERROR * u.electron,
                             EFFGAIN * u.s)

    def test_effgain_units(self):
        units = u.electron / u.s
        with pytest.raises(u.UnitsError):
            calc_total_error(DATA * units, BKG_ERROR * units, EFFGAIN * u.km)

    def test_missing_bkgerror_units(self):
        units = u.electron / u.s
        with pytest.raises(ValueError):
            calc_total_error(DATA * units, BKG_ERROR, EFFGAIN * u.s)

    def test_missing_effgain_units(self):
        units = u.electron / u.s
        with pytest.raises(ValueError):
            calc_total_error(DATA * units, BKG_ERROR * units,
                             EFFGAIN)
