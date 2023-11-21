# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the make module.
"""

import numpy as np
import pytest
from astropy.modeling.models import Moffat2D
from astropy.table import Table
from astropy.utils.exceptions import AstropyUserWarning
from numpy.testing import assert_allclose

from photutils.datasets import (apply_poisson_noise, make_4gaussians_image,
                                make_100gaussians_image,
                                make_gaussian_prf_sources_image,
                                make_gaussian_sources_image, make_gwcs,
                                make_model_sources_image, make_noise_image,
                                make_random_gaussians_table,
                                make_random_models_table, make_test_psf_data,
                                make_wcs)
from photutils.psf import IntegratedGaussianPRF
from photutils.utils._optional_deps import HAS_GWCS, HAS_SCIPY

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
    """Test if ValueError raises if mean is not input."""

    with pytest.raises(ValueError):
        shape = (100, 100)
        make_noise_image(shape, 'gaussian', stddev=2.0)


def test_make_noise_image_nostddev():
    """
    Test if ValueError raises if stddev is not input for Gaussian noise.
    """

    with pytest.raises(ValueError):
        shape = (100, 100)
        make_noise_image(shape, 'gaussian', mean=2.0)


def test_apply_poisson_noise():
    shape = (100, 100)
    data = np.ones(shape)
    result = apply_poisson_noise(data)
    assert result.shape == shape
    assert_allclose(result.mean(), 1.0, atol=1.0)


def test_apply_poisson_noise_negative():
    """Test if negative image values raises ValueError."""

    with pytest.raises(ValueError):
        shape = (100, 100)
        data = np.zeros(shape) - 1.0
        apply_poisson_noise(data)


def test_make_gaussian_sources_image():
    shape = (100, 100)
    image = make_gaussian_sources_image(shape, SOURCE_TABLE)
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


def test_make_random_gaussians_table():
    n_sources = 5
    param_ranges = dict([('amplitude', [500, 1000]), ('x_mean', [0, 500]),
                         ('y_mean', [0, 300]), ('x_stddev', [1, 5]),
                         ('y_stddev', [1, 5]), ('theta', [0, np.pi])])

    table = make_random_gaussians_table(n_sources, param_ranges, seed=0)
    assert len(table) == n_sources


def test_make_random_gaussians_table_flux():
    n_sources = 5
    param_ranges = dict([('flux', [500, 1000]), ('x_mean', [0, 500]),
                         ('y_mean', [0, 300]), ('x_stddev', [1, 5]),
                         ('y_stddev', [1, 5]), ('theta', [0, np.pi])])
    table = make_random_gaussians_table(n_sources, param_ranges, seed=0)
    assert 'amplitude' in table.colnames
    assert len(table) == n_sources


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


def test_make_random_models_table():
    model = Moffat2D(amplitude=1)
    param_ranges = {'x_0': (0, 300), 'y_0': (0, 500),
                    'gamma': (1, 3), 'alpha': (1.5, 3)}
    source_table = make_random_models_table(10, param_ranges)

    # most of the make_model_sources_image options are exercised in the
    # make_gaussian_sources_image tests
    image = make_model_sources_image((300, 500), model, source_table)
    assert image.sum() > 1


def test_make_wcs():
    shape = (100, 200)
    wcs = make_wcs(shape)

    assert wcs.pixel_shape == shape
    assert wcs.wcs.radesys == 'ICRS'

    wcs = make_wcs(shape, galactic=True)
    assert wcs.wcs.ctype[0] == 'GLON-CAR'
    assert wcs.wcs.ctype[1] == 'GLAT-CAR'


@pytest.mark.skipif(not HAS_GWCS, reason='gwcs is required')
def test_make_gwcs():
    shape = (100, 200)

    wcs = make_gwcs(shape)
    assert wcs.pixel_n_dim == 2
    assert wcs.available_frames == ['detector', 'icrs']
    assert wcs.output_frame.name == 'icrs'
    assert wcs.output_frame.axes_names == ('lon', 'lat')

    wcs = make_gwcs(shape, galactic=True)
    assert wcs.pixel_n_dim == 2
    assert wcs.available_frames == ['detector', 'galactic']
    assert wcs.output_frame.name == 'galactic'
    assert wcs.output_frame.axes_names == ('lon', 'lat')


@pytest.mark.skipif(not HAS_GWCS, reason='gwcs is required')
def test_make_wcs_compare():
    shape = (200, 300)
    wcs = make_wcs(shape)
    gwcs_obj = make_gwcs(shape)
    sc1 = wcs.pixel_to_world((50, 75), (50, 100))
    sc2 = gwcs_obj.pixel_to_world((50, 75), (50, 100))

    assert_allclose(sc1.ra, sc2.ra)
    assert_allclose(sc1.dec, sc2.dec)


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
