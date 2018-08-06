# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np

from ..non_parametric import gini


def test_gini():
    """
    Test Gini coefficient measurement.
    """

    data_evenly_distributed = np.ones((100, 100))
    data_point_like = np.zeros((100, 100))
    data_point_like[50, 50] = 1

    assert gini(data_evenly_distributed) == 0.
    assert gini(data_point_like) == 1.
