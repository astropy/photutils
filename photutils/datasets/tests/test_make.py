# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import numpy as np
from numpy.testing import assert_allclose
from astropy.tests.helper import pytest, assert_quantity_allclose
from astropy.table import Table
import astropy.units as u
from .. import (make_noise_image, make_poisson_noise, make_gaussian_sources,
                make_random_gaussians, make_4gaussians_image,
                make_100gaussians_image)


TABLE = Table()
TABLE['flux'] = [1, 2, 3]
TABLE['x_mean'] = [30, 50, 70.5]
TABLE['y_mean'] = [50, 50, 50.5]
TABLE['x_stddev'] = [1, 2, 3.5]
TABLE['y_stddev'] = [2, 1, 3.5]
TABLE['theta'] = np.array([0., 30, 50]) * np.pi / 180.


def test_make_noise_image():
    shape = (100, 100)
    image = make_noise_image(shape, 'gaussian', mean=0., stddev=2.)
    assert image.shape == shape
    assert_allclose(image.mean(), 0., atol=1.)


def test_make_noise_image_poisson():
    shape = (100, 100)
    image = make_noise_image(shape, 'poisson', mean=1.)
    assert image.shape == shape
    assert_allclose(image.mean(), 1., atol=1.)


def test_make_noise_image_nomean():
    """
    Test if ValueError raises if mean is not input.
    """
    with pytest.raises(ValueError):
        shape = (100, 100)
        make_noise_image(shape, 'gaussian', stddev=2.)


def test_make_noise_image_nostddev():
    """
    Test if ValueError raises if stddev is not input for Gaussian noise.
    """
    with pytest.raises(ValueError):
        shape = (100, 100)
        make_noise_image(shape, 'gaussian', mean=2.)


def test_make_noise_image_unit():
    shape = (100, 100)
    unit = u.electron / u.s
    image = make_noise_image(shape, 'gaussian', mean=0., stddev=2., unit=unit)
    assert image.shape == shape
    assert image.unit == unit
    assert_quantity_allclose(image.mean(), 0.*unit, atol=1.*unit)


def test_make_poisson_noise():
    shape = (100, 100)
    data = np.ones(shape)
    result = make_poisson_noise(data)
    assert result.shape == shape
    assert_allclose(result.mean(), 1., atol=1.)


def test_make_poisson_noise_negative():
    """Test if negative image values raises ValueError."""
    with pytest.raises(ValueError):
        shape = (100, 100)
        data = np.zeros(shape) - 1.
        make_poisson_noise(data)


def test_make_poisson_noise_unit():
    shape = (100, 100)
    unit = u.electron / u.s
    data = np.ones(shape) * unit
    result = make_poisson_noise(data)
    assert result.shape == shape
    assert result.unit == unit
    assert_quantity_allclose(result.mean(), 1.*unit, atol=1.*unit)


def test_make_gaussian_sources():
    shape = (100, 100)
    image = make_gaussian_sources(shape, TABLE)
    assert image.shape == shape
    assert_allclose(image.sum(), TABLE['flux'].sum())


def test_make_gaussian_sources_amplitude():
    table = TABLE.copy()
    table.remove_column('flux')
    table['amplitude'] = [1, 2, 3]
    shape = (100, 100)
    image = make_gaussian_sources(shape, table)
    assert image.shape == shape


def test_make_gaussian_sources_oversample():
    shape = (100, 100)
    image = make_gaussian_sources(shape, TABLE, oversample=10)
    assert image.shape == shape
    assert_allclose(image.sum(), TABLE['flux'].sum())


def test_make_gaussian_sources_parameters():
    with pytest.raises(ValueError):
        table = TABLE.copy()
        table.remove_column('flux')
        shape = (100, 100)
        make_gaussian_sources(shape, table)


def test_make_gaussian_sources_unit():
    shape = (100, 100)
    unit = u.electron / u.s
    image = make_gaussian_sources(shape, TABLE, unit=unit)
    assert image.shape == shape
    assert image.unit == unit
    assert_quantity_allclose(image.sum(), TABLE['flux'].sum()*unit)


def test_make_random_gaussians():
    n_sources = 5
    bounds = [0, 1]
    table = make_random_gaussians(n_sources, bounds, bounds, bounds, bounds,
                                  bounds)
    assert len(table) == n_sources


def test_make_random_gaussians_amplitude():
    n_sources = 5
    bounds = [0, 1]
    table = make_random_gaussians(n_sources, bounds, bounds, bounds, bounds,
                                  bounds, amplitude_range=bounds)
    assert len(table) == n_sources


def test_make_4gaussians_image():
    shape = (100, 200)
    data_sum = 176219.18059091491
    image = make_4gaussians_image()
    assert image.shape == shape
    assert_allclose(image.sum(), data_sum, rtol=1.e-6)


def test_make_100gaussians_image():
    shape = (300, 500)
    data_sum = 826182.24501251709
    image = make_100gaussians_image()
    assert image.shape == shape
    assert_allclose(image.sum(), data_sum, rtol=1.e-6)
