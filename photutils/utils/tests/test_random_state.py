# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the random_state module.
"""

import warnings

from astropy.utils.exceptions import AstropyDeprecationWarning
import numpy as np
import pytest

from .. import check_random_state


@pytest.mark.parametrize('seed', [None, np.random, 1,
                                  np.random.RandomState(1)])
def test_seed(seed):
    warnings.simplefilter('ignore', AstropyDeprecationWarning)
    assert isinstance(check_random_state(seed), np.random.RandomState)


@pytest.mark.parametrize('seed', [1., [1, 2]])
def test_invalid_seed(seed):
    warnings.simplefilter('ignore', AstropyDeprecationWarning)
    with pytest.raises(ValueError):
        check_random_state(seed)
