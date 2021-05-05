# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the _utils module.
"""

import numpy as np
from numpy.testing import assert_allclose

from .._utils import mask_to_mirrored_value


def testmask_to_mirrored_value():
    center = (2.0, 2.0)
    data = np.arange(25).reshape(5, 5)
    mask = np.zeros(data.shape, dtype=bool)
    mask[0, 0] = True
    mask[1, 1] = True
    data_ref = data.copy()
    data_ref[0, 0] = data[4, 4]
    data_ref[1, 1] = data[3, 3]
    mirror_data = mask_to_mirrored_value(data, mask, center)
    assert_allclose(mirror_data, data_ref, rtol=0, atol=1.e-6)


def testmask_to_mirrored_value_range():
    """
    Test mask_to_mirrored_value when mirrored pixels are outside of the
    image.
    """
    center = (3.0, 3.0)
    data = np.arange(25).reshape(5, 5)
    mask = np.zeros(data.shape, dtype=bool)
    mask[0, 0] = True
    mask[1, 1] = True
    mask[2, 2] = True
    data_ref = data.copy()
    data_ref[0, 0] = 0.
    data_ref[1, 1] = 0.
    data_ref[2, 2] = data[4, 4]
    mirror_data = mask_to_mirrored_value(data, mask, center)
    assert_allclose(mirror_data, data_ref, rtol=0, atol=1.e-6)


def testmask_to_mirrored_value_masked():
    """
    Test mask_to_mirrored_value when mirrored pixels are also in the
    replace_mask.
    """
    center = (2.0, 2.0)
    data = np.arange(25).reshape(5, 5)
    mask = np.zeros(data.shape, dtype=bool)
    mask[0, 0] = True
    mask[1, 1] = True
    mask[3, 3] = True
    mask[4, 4] = True
    data_ref = data.copy()
    data_ref[0, 0] = 0.
    data_ref[1, 1] = 0.
    data_ref[3, 3] = 0.
    data_ref[4, 4] = 0.
    mirror_data = mask_to_mirrored_value(data, mask, center)
    mirror_data = mask_to_mirrored_value(data, mask, center)
    assert_allclose(mirror_data, data_ref, rtol=0, atol=1.e-6)


def testmask_to_mirrored_value_mask_keyword():
    """
    Test mask_to_mirrored_value when mirrored pixels are masked (via the
    mask keyword).
    """
    center = (2.0, 2.0)
    data = np.arange(25.).reshape(5, 5)
    replace_mask = np.zeros(data.shape, dtype=bool)
    mask = np.zeros(data.shape, dtype=bool)
    replace_mask[0, 2] = True
    data[4, 2] = np.nan
    mask[4, 2] = True
    result = mask_to_mirrored_value(data, replace_mask, center, mask=mask)
    assert result[0, 2] == 0
