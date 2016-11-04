# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import division
import numpy as np
from numpy.testing import assert_allclose
from astropy.utils.exceptions import AstropyUserWarning
from astropy.tests.helper import pytest
from astropy.modeling.models import Gaussian2D
from astropy.convolution.utils import discretize_model
from astropy.table import Table
from .. import subtract_psf
from ..sandbox import DiscretePRF

try:
    import scipy
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


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

@pytest.mark.skipif('not HAS_SCIPY')
def test_subtract_psf():
    """Test subtract_psf."""

    prf = DiscretePRF(test_psf, subsampling=1)
    posflux = INTAB.copy()
    for n in posflux.colnames:
        posflux.rename_column(n, n.split('_')[0] + '_fit')
    residuals = subtract_psf(image, prf, posflux)
    assert_allclose(residuals, np.zeros_like(image), atol=1E-4)
    gm = Gaussian2D(x_stddev=3, y_stddev=3)
    xg, yg = np.mgrid[-2:3, -2:3]

    imod_nonorm = FittableImageModel(gm(xg, yg))
    assert np.allclose(imod_nonorm(0, 0), gm(0, 0))

    imod_norm = FittableImageModel(gm(xg, yg), normalize=True)
    assert not np.allclose(imod_norm(0, 0), gm(0, 0))
    assert np.allclose(np.sum(imod_norm(xg, yg)), 1)

    imod_norm2 = FittableImageModel(gm(xg, yg), normalize=True,
                                    normalization_correction=2)
    assert not np.allclose(imod_norm2(0, 0), gm(0, 0))
    assert np.allclose(imod_norm(0, 0), imod_norm2(0, 0)*2)
    assert np.allclose(np.sum(imod_norm2(xg, yg)), 0.5)
