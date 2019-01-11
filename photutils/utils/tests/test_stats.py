# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
from numpy.testing import assert_allclose
import pytest

from ..stats import std_blocksum
from ...datasets import make_noise_image


def test_std_blocksum():
    stddev = 5
    data = make_noise_image((100, 100), mean=0, stddev=stddev,
                            random_state=12345)
    block_sizes = np.array([5, 7, 10])
    stds = std_blocksum(data, block_sizes)
    expected = np.array([stddev, stddev, stddev])
    assert_allclose(stds / block_sizes, expected, atol=0.2)

    mask = np.zeros_like(data, dtype=np.bool)
    mask[25:50, 25:50] = True
    stds2 = std_blocksum(data, block_sizes, mask=mask)
    assert_allclose(stds2 / block_sizes, expected, atol=0.3)


def test_std_blocksum_mask_shape():
    with pytest.raises(ValueError):
        data = np.ones((10, 10))
        mask = np.ones((2, 2))
        std_blocksum(data, 10, mask=mask)
