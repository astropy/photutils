# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import division
import numpy as np
from numpy.testing import assert_allclose
from astropy.tests.helper import pytest
from astropy.table import Table
from astropy.modeling.models import Gaussian2D
from astropy.stats import gaussian_sigma_to_fwhm
from .. import IntegratedGaussianPRF, PSFMeasure

try:
    import scipy
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


@pytest.mark.xfail('not HAS_SCIPY')
@pytest.mark.parametrize("x_0, y_0, sigma_psf",
                         [(20, 40, 2), (60, 20, 3), (20 , 20, 4)])
def test_psf_measure(x_0, y_0, sigma_psf):
    y, x = np.mgrid[0:100, 0:100]
    g = Gaussian2D(amplitude=50, x_mean=x_0, y_mean=y_0, x_stddev=sigma_psf,
                   y_stddev=sigma_psf)

    positions = Table(names=['xcentroid', 'ycentroid'],
                      data=[[x_0], [y_0]])
    psf_measure = PSFMeasure(fitshape=17)

    est_fwhm = psf_measure(image=g(x,y), positions=positions)
    assert_allclose(est_fwhm, sigma_psf*gaussian_sigma_to_fwhm, rtol=1e-1)

    noise = np.random.normal(size=g(x,y).shape)
    est_fwhm = psf_measure(image=g(x,y) + noise, positions=positions)
    assert_allclose(est_fwhm, sigma_psf*gaussian_sigma_to_fwhm, rtol=1e-1)
