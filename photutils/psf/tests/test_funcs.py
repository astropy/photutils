# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
from numpy.testing import assert_allclose
import pytest
from astropy.modeling.models import Gaussian2D
from astropy.convolution.utils import discretize_model
from astropy.table import Table
from .. import subtract_psf, IntegratedGaussianPRF
from ..sandbox import DiscretePRF
from ..funcs import CullerAndEnderBase

try:
    import scipy    # noqa
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


class TestCullerAndEnder(CullerAndEnderBase):
    """Test Culler and Ender class"""
    def cull_data(self, data, psf_model):
        return data


@pytest.mark.skipif('not HAS_SCIPY')
def test_culler_ender_base():
    """Test SingleObjectModel"""

    igp = IntegratedGaussianPRF(sigma=1.2)
    tab = Table(names=['x_0', 'y_0', 'flux_0', 'object_type'],
                data=[[1, 2], [3, 4], [0.5, 1], ['Star', 'Galaxy']])

    single_object_model = CullerAndEnderBase()
    with pytest.raises(NotImplementedError):
        gpsf = single_object_model(tab, igp, tab)

    single_object_model = TestCullerAndEnder()
    with pytest.raises(NotImplementedError):
        gpsf = single_object_model(tab, igp, tab)
