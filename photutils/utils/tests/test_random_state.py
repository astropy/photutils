# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from astropy.tests.helper import pytest
import numpy as np
from .. import check_random_state


@pytest.mark.parametrize('seed', [None, np.random, 1,
                                  np.random.RandomState(1)])
def test_seed(seed):
    assert isinstance(check_random_state(seed), np.random.RandomState)


@pytest.mark.parametrize('seed', [1., [1, 2]])
def test_invalid_seed(seed):
    with pytest.raises(ValueError):
        check_random_state(seed)
