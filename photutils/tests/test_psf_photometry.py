# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import division

import numpy as np
from numpy.testing import assert_allclose

from astropy.tests.helper import pytest
from astropy.modeling.models import Gaussian2D
from astropy.convolution.utils import discretize_model

from ..psf import create_prf, DiscretePRF, psf_photometry, GaussianPSF

try:
    from scipy import optimize
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


psf_size = 11
gaussian_width = 1.
image_size = 101

# Position and fluxes of tes sources
positions = [(50, 50), (23, 83), (12, 80), (86, 84)]
fluxes = [np.pi * 10, 3.654, 20., 80 / np.sqrt(3)]

# Create test psf
psf_model = Gaussian2D(1. / (2 * np.pi * gaussian_width ** 2), psf_size // 2,
                       psf_size // 2, gaussian_width, gaussian_width)
test_psf = discretize_model(psf_model, (0, psf_size), (0, psf_size),
                            mode='oversample')

# Set up grid for test image
image = np.zeros((image_size, image_size))

# Add sources to test image
for i, position in enumerate(positions):
    x, y = position
    model = Gaussian2D(fluxes[i] / (2 * np.pi * gaussian_width ** 2),
                       x, y, gaussian_width, gaussian_width)
    image += discretize_model(model, (0, image_size), (0, image_size),
                              mode='oversample')


def test_create_prf_mean():
    """
    Check if create_prf works correctly on simulated data.
    """
    prf = create_prf(image, positions, psf_size, subsampling=1, mode='mean')
    assert_allclose(prf._prf_array[0, 0], test_psf, atol=1E-8)


def test_create_prf_median():
    """
    Check if create_prf works correctly on simulated data.
    """
    prf = create_prf(image, positions, psf_size, subsampling=1, mode='median')
    assert_allclose(prf._prf_array[0, 0], test_psf, atol=1E-8)


def test_create_prf_nan():
    """
    Check if create_prf deals correctly with nan values.
    """
    image_nan = image.copy()
    image_nan[52, 52] = np.nan
    image_nan[52, 48] = np.nan
    prf = create_prf(image_nan, positions, psf_size, subsampling=1)
    assert not np.isnan(prf._prf_array[0, 0]).any()


def test_create_prf_flux():
    """
    Check if create_prf works correctly when fluxes are specified.
    """
    prf = create_prf(image, positions, psf_size, fluxes=fluxes, subsampling=1)
    assert_allclose(np.abs(prf._prf_array[0, 0].sum()), 1)
    assert_allclose(prf._prf_array[0, 0], test_psf, atol=1E-8)


@pytest.mark.skipif('not HAS_SCIPY')
def test_discrete_prf_fit():
    """
    Check if fitting of discrete PSF model works.
    """
    prf = DiscretePRF(test_psf, subsampling=1)
    prf.x_0 = psf_size // 2
    prf.y_0 = psf_size // 2

    # test_psf is normalized to unity
    data = 10 * test_psf
    indices = np.indices(data.shape)
    flux = prf.fit(data, indices)
    assert_allclose(flux, 10, rtol=1E-5)


@pytest.mark.skipif('not HAS_SCIPY')
def test_psf_photometry_discrete():
    """
    Test psf_photometry with discrete PRF model.
    """
    prf = DiscretePRF(test_psf, subsampling=1)
    f = psf_photometry(image, positions, prf)
    assert_allclose(f, fluxes, rtol=1E-6)


@pytest.mark.skipif('not HAS_SCIPY')
def test_psf_photometry_gaussian():
    """
    Test psf_photometry with Gaussian PSF model.
    """
    prf = GaussianPSF(gaussian_width)
    f = psf_photometry(image, positions, prf)
    assert_allclose(f, fluxes, rtol=1E-3)

