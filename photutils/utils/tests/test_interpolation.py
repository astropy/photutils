# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import numpy as np
from numpy.testing import assert_allclose
from astropy.tests.helper import pytest
from .. import interpolate_masked_data

SHAPE = (5, 5)
DATA = np.ones(SHAPE) * 2.0
MASK = np.zeros_like(DATA, dtype=bool)
MASK[2, 2] = True
ERROR = np.ones(SHAPE)
BACKGROUND = np.ones(SHAPE)
WRONG_SHAPE = np.ones((2, 2))


class TestInterpolateMaskedData(object):
    def test_mask_shape(self):
        with pytest.raises(ValueError):
            interpolate_masked_data(DATA, WRONG_SHAPE)

    def test_error_shape(self):
        with pytest.raises(ValueError):
            interpolate_masked_data(DATA, MASK, error=WRONG_SHAPE)

    def test_background_shape(self):
        with pytest.raises(ValueError):
            interpolate_masked_data(DATA, MASK, background=WRONG_SHAPE)

    def test_interpolation(self):
        data2 = DATA.copy()
        data2[2, 2] = 100.
        error2 = ERROR.copy()
        error2[2, 2] = 100.
        background2 = BACKGROUND.copy()
        background2[2, 2] = 100.
        data, error, background = interpolate_masked_data(
            data2, MASK, error=error2, background=background2)
        assert_allclose(data, DATA)
        assert_allclose(error, ERROR)
        assert_allclose(background, BACKGROUND)

    def test_interpolation_larger_mask(self):
        data2 = DATA.copy()
        data2[2, 2] = 100.
        error2 = ERROR.copy()
        error2[2, 2] = 100.
        background2 = BACKGROUND.copy()
        background2[2, 2] = 100.
        mask2 = MASK.copy()
        mask2[1:4, 1:4] = True
        data, error, background = interpolate_masked_data(
            data2, MASK, error=error2, background=background2)
        assert_allclose(data, DATA)
        assert_allclose(error, ERROR)
        assert_allclose(background, BACKGROUND)
