# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the core module.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from photutils.datasets import make_wcs
from photutils.morphology import data_properties
from photutils.utils._optional_deps import HAS_SKIMAGE

XCS = [25.7]
YCS = [26.2]
XSTDDEVS = [3.2, 4.0]
YSTDDEVS = [5.7, 4.1]
THETAS = np.array([30.0, 45.0]) * np.pi / 180.0
DATA = np.zeros((3, 3))
DATA[0:2, 1] = 1.0
DATA[1, 0:2] = 1.0
DATA[1, 1] = 2.0


@pytest.mark.skipif(not HAS_SKIMAGE, reason='skimage is required')
def test_data_properties():
    data = np.ones((2, 2)).astype(float)
    mask = np.array([[False, False], [True, True]])
    props = data_properties(data, mask=None)
    props2 = data_properties(data, mask=mask)
    properties = ['xcentroid', 'ycentroid']
    result = [getattr(props, i) for i in properties]
    result2 = [getattr(props2, i) for i in properties]
    assert_allclose([0.5, 0.5], result, rtol=0, atol=1.0e-6)
    assert_allclose([0.5, 0.0], result2, rtol=0, atol=1.0e-6)
    assert props.area.value == 4.0
    assert props2.area.value == 2.0

    wcs = make_wcs(data.shape)
    props = data_properties(data, mask=None, wcs=wcs)
    assert props.sky_centroid is not None


@pytest.mark.skipif(not HAS_SKIMAGE, reason='skimage is required')
def test_data_properties_bkg():
    data = np.ones((3, 3)).astype(float)
    props = data_properties(data, background=1.0)
    assert props.area.value == 9.0
    assert props.background_sum == 9.0

    with pytest.raises(ValueError):
        data_properties(data, background=[1.0, 2.0])
