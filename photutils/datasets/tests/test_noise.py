# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the noise module.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from photutils.datasets import apply_poisson_noise, make_noise_image


def test_apply_poisson_noise():
    shape = (100, 100)
    data = np.ones(shape)
    result = apply_poisson_noise(data)
    assert result.shape == shape
    assert_allclose(result.mean(), 1.0, atol=1.0)


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
    shape = (100, 100)
    image = make_noise_image(shape, 'gaussian', mean=0.0, stddev=2.0)
    assert image.shape == shape
    assert_allclose(image.mean(), 0.0, atol=1.0)


def test_make_noise_image_poisson():
    shape = (100, 100)
    image = make_noise_image(shape, 'poisson', mean=1.0)
    assert image.shape == shape
    assert_allclose(image.mean(), 1.0, atol=1.0)


def test_make_noise_image_nomean():
    """
    Test invalid inputs.
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
