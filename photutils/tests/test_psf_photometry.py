# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import division

import numpy as np

from astropy.tests.helper import pytest
from astropy.modeling.models import Gaussian2DModel
from astropy.nddata.convolution.utils import discretize_model

from photutils.psf import create_prf, DiscretePRF, psf_photometry, GaussianPSF

try:
    from scipy import optimize
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


psf_size = 11
gaussian_width = 1.
image_size = 101.

# Position and fluxes
positions = [(50, 50), (23, 83), (12, 80), (86, 84)]
fluxes = [np.pi * 10, 3.654, 20., 80 / np.sqrt(3)]

# Create test psf
psf_model = Gaussian2DModel(1. / (2 * np.pi * gaussian_width ** 2), 
                            psf_size // 2, psf_size // 2, gaussian_width, gaussian_width)
test_psf = discretize_model(psf_model, (0, psf_size), (0, psf_size), mode='oversample')

# Set up grid for test image
image = np.zeros((image_size, image_size)) 

# Add sources to test image
for i, position in enumerate(positions):
    x, y = position
    model = Gaussian2DModel(fluxes[i] / (2 * np.pi * gaussian_width ** 2), 
                            x, y, gaussian_width, gaussian_width)
    image += discretize_model(model, (0, image_size), (0, image_size), mode='oversample')


def test_create_prf():
    """
    Check if create_prf works correctly on simulated data.
    """
    prf = create_prf(image, positions, psf_size, subsampling=1)
    assert np.all(np.abs(prf._prf_array[0, 0] - test_psf) < 0.02)
  

def test_create_psf_nan():
    """
    Check if create_psf deals correctly with nan values.
    """
    image[52, 52] = np.nan
    image[52, 48] = np.nan
    prf = create_prf(image, positions, psf_size, fix_nan=True, subsampling=1)
    assert not np.isnan(prf._prf_array[0, 0]).any()


def test_create_psf_flux():
    """
    Check if create_psf works correctly when fluxes are specified. 
    """
    prf = create_prf(image, positions, psf_size, fluxes=fluxes, subsampling=1)
    assert np.abs(prf._prf_array[0, 0].sum() - 1) < 0.001
    assert np.all(np.abs(prf._prf_array[0, 0] - test_psf) < 0.1)
    
    
#def test_discrete_psf_fit():
#    """
#    Check if fitting of discrete PSF model works.
#    """
#    prf = DiscretePRF(test_psf, subsampling=1)
#    prf.x_0 = psf_size // 2
#    prf.y_0 = psf_size // 2
#    data = 10 * test_psf
#    indices = np.indices(data.shape) 
#    flux = prf.fit(data, indices)
#    assert np.abs(flux - 10) < 0.1
    
    
def test_psf_photometry_discrete():
    """
    Test psf_photometry with discrete PRF model.
    """
    prf = DiscretePRF(test_psf, subsampling=1)
    f = psf_photometry(image, positions, prf)
    assert np.all(np.abs(fluxes - f) < 1E-12)
    

def test_psf_photometry_gaussian():
    """
    Test psf_photometry with Gaussian PRF model.
    """
    prf = GaussianPSF(gaussian_width)
    f = psf_photometry(image, positions, prf)
    assert np.all(np.abs(fluxes - f) < 1E-5)
    