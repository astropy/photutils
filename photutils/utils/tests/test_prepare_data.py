# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import numpy as np
from numpy.testing import assert_allclose
from astropy.tests.helper import pytest
from .. import calc_total_error

SHAPE = (5, 5)
DATAVAL = 2.
DATA = np.ones(SHAPE) * DATAVAL
MASK = np.zeros_like(DATA, dtype=bool)
MASK[2, 2] = True
BKG_ERROR = np.ones(SHAPE)
EFFGAIN = np.ones(SHAPE) * DATAVAL
BACKGROUND = np.ones(SHAPE)
WRONG_SHAPE = np.ones((2, 2))


class TestCalculateTotalError(object):
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
