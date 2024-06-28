# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Fixtures used in tests.
"""

import numpy as np
import pytest
from astropy.modeling.models import Gaussian2D

from photutils.psf import IntegratedGaussianPRF, make_psf_model_image


@pytest.fixture(name='kernel')
def fixture_kernel():
    size = 5
    cen = (size - 1) / 2
    y, x = np.mgrid[0:size, 0:size]
    g = Gaussian2D(1, cen, cen, 1.2, 1.2, theta=0)
    return g(x, y)


@pytest.fixture(name='data')
def fixture_data():
    shape = (101, 101)
    model_shape = (11, 11)
    psf_model = IntegratedGaussianPRF(flux=1, sigma=1.5)
    n_sources = 25
    data, _ = make_psf_model_image(shape, psf_model, n_sources,
                                   model_shape=model_shape,
                                   flux=(100, 200),
                                   min_separation=10,
                                   seed=0,
                                   border_size=(10, 10),
                                   progress_bar=False)
    return data
