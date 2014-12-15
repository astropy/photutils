# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from .. import mad_std
from .. import check_random_state
from numpy.testing import assert_allclose

def test_mad_std():
    prng = check_random_state(12345)
    data = prng.normal(5, 2, size=(100, 100))
    assert_allclose(mad_std(data), 2.0, rtol=0.05)
