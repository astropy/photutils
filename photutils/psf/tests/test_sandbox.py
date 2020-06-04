# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the sandbox module.
"""

from astropy.convolution.utils import discretize_model
from astropy.modeling.models import Gaussian2D
from astropy.table import Table
import numpy as np
from numpy.testing import assert_allclose

from ..sandbox import DiscretePRF


PSF_SIZE = 11
GAUSSIAN_WIDTH = 1.
IMAGE_SIZE = 101

# Position and FLUXES of test sources
INTAB = Table([[50., 23, 12, 86], [50., 83, 80, 84],
               [np.pi * 10, 3.654, 20., 80 / np.sqrt(3)]],
              names=['x_0', 'y_0', 'flux_0'])

# Create test psf
psf_model = Gaussian2D(1. / (2 * np.pi * GAUSSIAN_WIDTH ** 2), PSF_SIZE // 2,
                       PSF_SIZE // 2, GAUSSIAN_WIDTH, GAUSSIAN_WIDTH)
test_psf = discretize_model(psf_model, (0, PSF_SIZE), (0, PSF_SIZE),
                            mode='oversample')

# Set up grid for test image
image = np.zeros((IMAGE_SIZE, IMAGE_SIZE))

# Add sources to test image
for x, y, flux in INTAB:
    model = Gaussian2D(flux / (2 * np.pi * GAUSSIAN_WIDTH ** 2),
                       x, y, GAUSSIAN_WIDTH, GAUSSIAN_WIDTH)
    image += discretize_model(model, (0, IMAGE_SIZE), (0, IMAGE_SIZE),
                              mode='oversample')

# Some tests require an image with wider sources.
WIDE_GAUSSIAN_WIDTH = 3.
WIDE_INTAB = Table([[50, 23.2], [50.5, 1], [10, 20]],
                   names=['x_0', 'y_0', 'flux_0'])
wide_image = np.zeros((IMAGE_SIZE, IMAGE_SIZE))

# Add sources to test image
for x, y, flux in WIDE_INTAB:
    model = Gaussian2D(flux / (2 * np.pi * WIDE_GAUSSIAN_WIDTH ** 2),
                       x, y, WIDE_GAUSSIAN_WIDTH, WIDE_GAUSSIAN_WIDTH)
    wide_image += discretize_model(model, (0, IMAGE_SIZE), (0, IMAGE_SIZE),
                                   mode='oversample')


def test_create_prf_mean():
    """
    Check if create_prf works correctly on simulated data.
    Position input format: list
    """

    prf = DiscretePRF.create_from_image(image,
                                        list(INTAB['x_0', 'y_0'].as_array()),
                                        PSF_SIZE, subsampling=1, mode='mean')
    assert_allclose(prf._prf_array[0, 0], test_psf, atol=1E-8)


def test_create_prf_median():
    """
    Check if create_prf works correctly on simulated data.
    Position input format: astropy.table.Table
    """

    prf = DiscretePRF.create_from_image(image, np.array(INTAB['x_0', 'y_0']),
                                        PSF_SIZE, subsampling=1,
                                        mode='median')
    assert_allclose(prf._prf_array[0, 0], test_psf, atol=1E-8)


def test_create_prf_nan():
    """
    Check if create_prf deals correctly with nan values.
    """

    image_nan = image.copy()
    image_nan[52, 52] = np.nan
    image_nan[52, 48] = np.nan
    prf = DiscretePRF.create_from_image(image, np.array(INTAB['x_0', 'y_0']),
                                        PSF_SIZE, subsampling=1, fix_nan=True)
    assert not np.isnan(prf._prf_array[0, 0]).any()


def test_create_prf_flux():
    """
    Check if create_prf works correctly when FLUXES are specified.
    """

    prf = DiscretePRF.create_from_image(image, np.array(INTAB['x_0', 'y_0']),
                                        PSF_SIZE, subsampling=1,
                                        mode='median', fluxes=INTAB['flux_0'])
    assert_allclose(prf._prf_array[0, 0].sum(), 1)
    assert_allclose(prf._prf_array[0, 0], test_psf, atol=1E-8)
