# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the examples module.
"""

from numpy.testing import assert_allclose

from photutils.datasets import make_4gaussians_image, make_100gaussians_image


def test_make_4gaussians_image():
    shape = (100, 200)
    data_sum = 176219.18
    image = make_4gaussians_image()
    assert image.shape == shape
    assert_allclose(image.sum(), data_sum, rtol=1.0e-2)


def test_make_100gaussians_image():
    shape = (300, 500)
    data_sum = 826182.24501251709
    image = make_100gaussians_image()
    assert image.shape == shape
    assert_allclose(image.sum(), data_sum, rtol=1.0e-6)
