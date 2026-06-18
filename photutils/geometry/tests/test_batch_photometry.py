# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the low-level batch Cython aperture photometry driver.
"""

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from photutils.geometry._batch_photometry import (SHAPE_CIRCLE,
                                                  batch_aperture_sums)


def _batch_inputs():
    """
    Build deterministic data, error, mask, and source positions for the
    batch-driver tests.
    """
    rng = np.random.default_rng(42)
    data = rng.random((80, 80))
    error = rng.random((80, 80)) + 0.1
    mask = np.zeros((80, 80), dtype=np.uint8)
    mask[::7, ::5] = 1
    positions = np.array([[20.0, 25.0], [40.0, 40.0], [55.0, 30.0],
                          [10.0, 60.0], [70.0, 70.0], [35.0, 15.0]])
    return data, error, mask, positions


@pytest.mark.parametrize('use_exact', [1, 0])
def test_readonly_arrays(use_exact):
    """
    Test that the batch driver accepts read-only (non-writeable) data,
    error, positions, and params arrays and returns results identical to
    writeable arrays.

    The data, error, positions, and params arguments are declared as
    ``const`` typed memoryviews so that read-only arrays do not raise a
    ``ValueError``.
    """
    data, error, mask, positions = _batch_inputs()
    params = np.array([8.0], dtype=np.float64)

    expected = batch_aperture_sums(data, error, mask, positions, SHAPE_CIRCLE,
                                   params, 8.0, 8.0, use_exact, 8)

    for arr in (data, error, positions, params):
        arr.setflags(write=False)
    result = batch_aperture_sums(data, error, mask, positions, SHAPE_CIRCLE,
                                 params, 8.0, 8.0, use_exact, 8)

    for res_arr, exp_arr in zip(result, expected, strict=True):
        assert_array_equal(res_arr, exp_arr)
