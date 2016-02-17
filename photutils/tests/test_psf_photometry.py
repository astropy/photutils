# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import division

import numpy as np
from numpy.testing import assert_allclose

from astropy.tests.helper import pytest
from astropy.modeling.models import Gaussian2D
from astropy.convolution.utils import discretize_model
from astropy.table import Table

from ..psf import (create_prf, DiscretePRF, psf_photometry,
                   IntegratedGaussianPSF, subtract_psf)

try:
    from scipy import optimize
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


PSF_SIZE = 11
GAUSSIAN_WIDTH = 1.
IMAGE_SIZE = 101

# Position and FLUXES of test sources
POSITIONS = [(50, 50), (23, 83), (12, 80), (86, 84)]
FLUXES = [np.pi * 10, 3.654, 20., 80 / np.sqrt(3)]

# Create test psf
psf_model = Gaussian2D(1. / (2 * np.pi * GAUSSIAN_WIDTH ** 2), PSF_SIZE // 2,
                       PSF_SIZE // 2, GAUSSIAN_WIDTH, GAUSSIAN_WIDTH)
test_psf = discretize_model(psf_model, (0, PSF_SIZE), (0, PSF_SIZE),
                            mode='oversample')

# Set up grid for test image
image = np.zeros((IMAGE_SIZE, IMAGE_SIZE))

# Add sources to test image
for flux, position in zip(FLUXES, POSITIONS):
    x, y = position
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
    """
    prf = create_prf(image, POSITIONS, PSF_SIZE, subsampling=1, mode='mean')
    assert_allclose(prf._prf_array[0, 0], test_psf, atol=1E-8)


def test_create_prf_median():
    """
    Check if create_prf works correctly on simulated data.
    """
    prf = create_prf(image, POSITIONS, PSF_SIZE, subsampling=1, mode='median')
    assert_allclose(prf._prf_array[0, 0], test_psf, atol=1E-8)


def test_create_prf_nan():
    """
    Check if create_prf deals correctly with nan values.
    """
    image_nan = image.copy()
    image_nan[52, 52] = np.nan
    image_nan[52, 48] = np.nan
    prf = create_prf(image_nan, POSITIONS, PSF_SIZE, subsampling=1,
                     fix_nan=True)
    assert not np.isnan(prf._prf_array[0, 0]).any()


def test_create_prf_flux():
    """
    Check if create_prf works correctly when FLUXES are specified.
    """
    prf = create_prf(image, POSITIONS, PSF_SIZE, fluxes=FLUXES, subsampling=1)
    assert_allclose(prf._prf_array[0, 0].sum(), 1)
    assert_allclose(prf._prf_array[0, 0], test_psf, atol=1E-8)


@pytest.mark.skipif('not HAS_SCIPY')
def test_discrete_prf_fit():
    """
    Check if fitting of discrete PSF model works.
    """
    prf = DiscretePRF(test_psf, subsampling=1)
    prf.x_0 = 50
    prf.y_0 = 50

    # test_psf is normalized to unity
    indices = np.indices(image.shape)
    flux = prf.fit(image, indices)
    assert_allclose(flux, FLUXES[0], rtol=1E-5)


@pytest.mark.skipif('not HAS_SCIPY')
def test_psf_photometry_discrete():
    """
    Test psf_photometry with discrete PRF model.
    """
    prf = DiscretePRF(test_psf, subsampling=1)
    f = psf_photometry(image, POSITIONS, prf)
    assert_allclose(f, FLUXES, rtol=1E-6)


@pytest.mark.skipif('not HAS_SCIPY')
def test_tune_coordinates():
    """
    Test psf_photometry with discrete PRF model and tune_coordinates=True.
    """
    prf = DiscretePRF(test_psf, subsampling=1)
    # Shift all sources by 0.3 pixels
    positions = [(_[0] + 0.3, _[1] + 0.3) for _ in POSITIONS]
    f = psf_photometry(image, positions, prf, tune_coordinates=True)
    assert_allclose(f, FLUXES, rtol=1E-6)


@pytest.mark.skipif('not HAS_SCIPY')
def test_psf_boundary():
    """
    Test psf_photometry with discrete PRF model at the boundary of the data.
    """
    prf = DiscretePRF(test_psf, subsampling=1)
    # Shift all sources by 0.3 pixels
    f = psf_photometry(image, [(1, 1)], prf)
    assert f == 0


@pytest.mark.skipif('not HAS_SCIPY')
def test_psf_boundary_gaussian():
    """
    Test psf_photometry with discrete PRF model at the boundary of the data.
    """
    psf = IntegratedGaussianPSF(GAUSSIAN_WIDTH)
    # Shift all sources by 0.3 pixels
    f = psf_photometry(image, [(1, 1)], psf)
    assert f == 0


@pytest.mark.skipif('not HAS_SCIPY')
def test_psf_photometry_gaussian():
    """
    Test psf_photometry with Gaussian PSF model.
    """
    prf = IntegratedGaussianPSF(GAUSSIAN_WIDTH)
    f = psf_photometry(image, POSITIONS, prf)
    assert_allclose(f, FLUXES, rtol=1E-3)


@pytest.mark.skipif('not HAS_SCIPY')
def test_subtract_psf():
    """
    Test subtract_psf
    """
    prf = DiscretePRF(test_psf, subsampling=1)
    residuals = subtract_psf(image, prf, POSITIONS, FLUXES)
    assert_allclose(residuals, np.zeros_like(image), atol=1E-6)


@pytest.mark.skipif('not HAS_SCIPY')
def test_psf_fitting_data_on_edge():
    """
    No mask is input explicitly here, but source 2 is so close to the edge
    that the subarray that's extracted gets a mask internally.
    """
    psf_guess = IntegratedGaussianPSF(flux=1, sigma=WIDE_GAUSSIAN_WIDTH)
    psf_guess.flux.fixed = psf_guess.x_0.fixed = psf_guess.y_0.fixed = False
    fitshape = (8, 8)
    outtab = psf_photometry(wide_image, WIDE_INTAB, psf_guess, fitshape)
    for n in ['x', 'y', 'flux']:
        assert_allclose(outtab[n + '_0'], outtab[n + '_fit'],
                        rtol=0.05, atol=0.1)


@pytest.mark.skipif('not HAS_SCIPY')
def test_psf_fitting_data_masked():
    """
    There are several ways to input masked data, but we do not test
    all of them here, because the @nddata decorartor and the
    aperture_photometry tests take care of some of them.
    """
    mimage = wide_image.copy()
    mask = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.bool)
    mask[::3, 1::4] = 1
    # Set masked values so high it would be obvious if they were used in fit
    mimage[mask] = 1e5

    psf_guess = IntegratedGaussianPSF(flux=1, sigma=WIDE_GAUSSIAN_WIDTH)
    psf_guess.flux.fixed = psf_guess.x_0.fixed = psf_guess.y_0.fixed = False
    fitshape = (8, 8)
    # This definitely has to fail
    outtab = psf_photometry(mimage, WIDE_INTAB, psf_guess, fitshape)
    for n in ['x', 'y', 'flux']:
        assert not np.allclose(outtab[n + '_0'], outtab[n + '_fit'],
                               rtol=0.05, atol=0.1)

    outtab = psf_photometry(mimage, WIDE_INTAB, psf_guess, fitshape, mask=mask)
    for n in ['x', 'y', 'flux']:
        assert_allclose(outtab[n + '_0'], outtab[n + '_fit'],
                        rtol=0.05, atol=0.1)
