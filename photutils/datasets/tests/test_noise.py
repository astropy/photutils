# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the noise module.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from photutils.datasets import apply_poisson_noise, make_noise_image


def test_apply_poisson_noise():
    """
    Test if Poisson noise is applied correctly.
    """
    shape = (100, 100)
    data = np.ones(shape)
    result = apply_poisson_noise(data)
    assert result.shape == shape
    assert_allclose(result.mean(), 1.0, atol=1.0)


def test_apply_poisson_noise_dtype():
    """
    Test if the output data type is the same as the input data type.
    """
    data = np.ones((10, 10), dtype=float) * 5.0
    result = apply_poisson_noise(data, seed=0)
    assert result.dtype == data.dtype


def test_apply_poisson_noise_seed():
    """
    Test if Poisson noise is reproducible with a given seed.
    """
    shape = (100, 100)
    data = np.ones(shape) * 10.0
    result1 = apply_poisson_noise(data, seed=12345)
    result2 = apply_poisson_noise(data, seed=12345)
    assert_allclose(result1, result2)


def test_apply_poisson_noise_negative():
    """
    Test if negative image values raises ValueError.
    """
    shape = (100, 100)
    data = np.zeros(shape) - 1.0
    match = 'data must not contain any negative values'
    with pytest.raises(ValueError, match=match):
        apply_poisson_noise(data)


def test_make_noise_image():
    """
    Test if noise image is generated correctly.
    """
    shape = (100, 100)
    image = make_noise_image(shape, 'gaussian', mean=0.0, stddev=2.0)
    assert image.shape == shape
    assert_allclose(image.mean(), 0.0, atol=1.0)


def test_make_noise_image_poisson():
    """
    Test noise image with Poisson noise.
    """
    shape = (100, 100)
    image = make_noise_image(shape, 'poisson', mean=1.0)
    assert image.shape == shape
    assert_allclose(image.mean(), 1.0, atol=1.0)


def test_make_noise_image_nomean():
    """
    Test invalid inputs to make_noise_image.
    """
    shape = (100, 100)
    match = 'Invalid distribution:'
    with pytest.raises(ValueError, match=match):
        make_noise_image(shape, 'invalid', mean=0, stddev=2.0)

    match = '"mean" must be input'
    with pytest.raises(ValueError, match=match):
        make_noise_image(shape, 'gaussian', stddev=2.0)

    match = '"stddev" must be input for Gaussian noise'
    with pytest.raises(ValueError, match=match):
        make_noise_image(shape, 'gaussian', mean=2.0)


def test_make_noise_image_seed():
    """
    Test if noise image is reproducible with a given seed.
    """
    shape = (100, 100)
    image1 = make_noise_image(shape, 'gaussian', mean=0.0,
                              stddev=2.0, seed=12345)
    image2 = make_noise_image(shape, 'gaussian', mean=0.0,
                              stddev=2.0, seed=12345)
    assert_allclose(image1, image2)
