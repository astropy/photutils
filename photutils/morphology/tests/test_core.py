# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
from numpy.testing import assert_allclose
import pytest

from ..core import data_properties

try:
    import skimage    # noqa
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False


XCS = [25.7]
YCS = [26.2]
XSTDDEVS = [3.2, 4.0]
YSTDDEVS = [5.7, 4.1]
THETAS = np.array([30., 45.]) * np.pi / 180.
DATA = np.zeros((3, 3))
DATA[0:2, 1] = 1.
DATA[1, 0:2] = 1.
DATA[1, 1] = 2.


@pytest.mark.skipif('not HAS_SKIMAGE')
def test_data_properties():
    data = np.ones((2, 2)).astype(np.float)
    mask = np.array([[False, False], [True, True]])
    props = data_properties(data, mask=None)
    props2 = data_properties(data, mask=mask)
    properties = ['xcentroid', 'ycentroid', 'area']
    result = [props[i].value for i in properties]
    result2 = [props2[i].value for i in properties]
    assert_allclose([0.5, 0.5, 4.0], result, rtol=0, atol=1.e-6)
    assert_allclose([0.5, 0.0, 2.0], result2, rtol=0, atol=1.e-6)
