# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the images module.
"""

import numpy as np
import pytest
from astropy.modeling.models import Moffat2D
from astropy.table import Table
from astropy.utils.exceptions import AstropyUserWarning
from numpy.testing import assert_allclose

from photutils.datasets import (make_4gaussians_image, make_100gaussians_image,
                                make_gaussian_prf_sources_image,
                                make_gaussian_sources_image,
                                make_model_sources_image, make_test_psf_data)
from photutils.psf import IntegratedGaussianPRF
from photutils.utils._optional_deps import HAS_SCIPY

SOURCE_TABLE = Table()
SOURCE_TABLE['flux'] = [1, 2, 3]
SOURCE_TABLE['x_mean'] = [30, 50, 70.5]
SOURCE_TABLE['y_mean'] = [50, 50, 50.5]
SOURCE_TABLE['x_stddev'] = [1, 2, 3.5]
SOURCE_TABLE['y_stddev'] = [2, 1, 3.5]
SOURCE_TABLE['theta'] = np.array([0.0, 30, 50]) * np.pi / 180.0

SOURCE_TABLE_PRF = Table()
SOURCE_TABLE_PRF['x_0'] = [30, 50, 70.5]
SOURCE_TABLE_PRF['y_0'] = [50, 50, 50.5]
# Without sigma, make_gaussian_prf_sources_image will default to sigma = 1
# so we can ignore it when converting to amplitude
SOURCE_TABLE_PRF['amplitude'] = np.array([1, 2, 3]) / (2 * np.pi)


def test_make_model_sources_image():
    source_tbl = Table()
    source_tbl['x_0'] = [50, 70, 90]
    source_tbl['y_0'] = [50, 50, 50]
    source_tbl['gamma'] = [1.7, 2.32, 5.8]
    source_tbl['alpha'] = [2.9, 5.7, 4.6]
    model = Moffat2D(amplitude=1)
    image = make_model_sources_image((300, 500), model, source_tbl)
    assert image.sum() > 1


def test_make_gaussian_sources_image():
    shape = (100, 100)
    image = make_gaussian_sources_image(shape, SOURCE_TABLE)
    assert image.shape == shape
    assert_allclose(image.sum(), SOURCE_TABLE['flux'].sum())


def test_make_gaussian_sources_image_amplitude():
    table = SOURCE_TABLE.copy()
    table.remove_column('flux')
    table['amplitude'] = [1, 2, 3]
    shape = (100, 100)
    image = make_gaussian_sources_image(shape, table)
    assert image.shape == shape


def test_make_gaussian_sources_image_oversample():
    shape = (100, 100)
    image = make_gaussian_sources_image(shape, SOURCE_TABLE, oversample=10)
    assert image.shape == shape
    assert_allclose(image.sum(), SOURCE_TABLE['flux'].sum())


@pytest.mark.skipif(not HAS_SCIPY, reason='scipy is required')
def test_make_gaussian_prf_sources_image():
    shape = (100, 100)
    image = make_gaussian_prf_sources_image(shape, SOURCE_TABLE_PRF)
    assert image.shape == shape
    # Without sigma in table, image assumes sigma = 1
    flux = SOURCE_TABLE_PRF['amplitude'] * (2 * np.pi)
    assert_allclose(image.sum(), flux.sum())


def test_make_4gaussians_image():
    shape = (100, 200)
    data_sum = 176219.18059091491
    image = make_4gaussians_image()
    assert image.shape == shape
    assert_allclose(image.sum(), data_sum, rtol=1.0e-6)


def test_make_100gaussians_image():
    shape = (300, 500)
    data_sum = 826182.24501251709
    image = make_100gaussians_image()
    assert image.shape == shape
    assert_allclose(image.sum(), data_sum, rtol=1.0e-6)


@pytest.mark.skipif(not HAS_SCIPY, reason='scipy is required')
def test_make_test_psf_data():
    psf_model = IntegratedGaussianPRF(flux=100, sigma=1.5)
    psf_shape = (5, 5)
    nsources = 10
    shape = (100, 100)
    data, true_params = make_test_psf_data(shape, psf_model, psf_shape,
                                           nsources, flux_range=(500, 1000),
                                           min_separation=10, seed=0)

    assert isinstance(data, np.ndarray)
    assert data.shape == shape
    assert isinstance(true_params, Table)
    assert len(true_params) == nsources
    assert true_params['x'].min() >= 0
    assert true_params['y'].min() >= 0

    match = 'Unable to produce'
    with pytest.warns(AstropyUserWarning, match=match):
        nsources = 100
        make_test_psf_data(shape, psf_model, psf_shape, nsources,
                           flux_range=(500, 1000), min_separation=100, seed=0)
