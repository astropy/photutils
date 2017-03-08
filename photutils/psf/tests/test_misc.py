# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Miscellaneous tests for psf functionality that doesn't have another obvious
place to go
"""

from __future__ import division
import numpy as np
from numpy.testing import assert_allclose
from astropy.tests.helper import pytest
from astropy.table import Table
from .. import IntegratedGaussianPRF, prepare_psf_model, get_grouped_psf_model

try:
    import scipy
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


widths = [0.001, 0.01, 0.1, 1]
sigmas = [0.5, 1., 2., 10., 12.34]


@pytest.mark.skipif('not HAS_SCIPY')
@pytest.mark.parametrize('width', widths)
def test_subpixel_gauss_psf(width):
    """
    Test subpixel accuracy of Gaussian PSF by checking the sum of pixels.
    """
    gauss_psf = IntegratedGaussianPRF(width)
    y, x = np.mgrid[-10:11, -10:11]
    assert_allclose(gauss_psf(x, y).sum(), 1)


@pytest.mark.skipif('not HAS_SCIPY')
@pytest.mark.parametrize('sigma', sigmas)
def test_gaussian_psf_integral(sigma):
    """
    Test if Gaussian PSF integrates to unity on larger scales.
    """
    psf = IntegratedGaussianPRF(sigma=sigma)
    y, x = np.mgrid[-100:101, -100:101]
    assert_allclose(psf(y, x).sum(), 1)


@pytest.fixture(scope="module")
def moffimg():
    """
    This fixture requires scipy so don't call it from non-scipy tests
    """
    from scipy import integrate
    from astropy.modeling.models import Moffat2D

    mof = Moffat2D(alpha=4.8)

    # this is the analytic value needed to get a total flux of 1
    mof.amplitude = (mof.alpha-1)/(np.pi*mof.gamma**2)

    # first make sure it really is normalized
    assert (1 - integrate.dblquad(mof, -10, 10,
                                  lambda x: -10, lambda x: 10)[0]) < 1e-6

    # now create an "image" of the PSF
    xg, yg = np.meshgrid(*([np.linspace(-2, 2, 100)]*2))

    return mof, (xg, yg, mof(xg, yg))


@pytest.mark.skipif('not HAS_SCIPY')
def test_moffat_fitting(moffimg):
    """
    Test that the Moffat to be fit in test_psf_adapter is behaving correctly
    """
    from astropy.modeling.fitting import LevMarLSQFitter
    from astropy.modeling.models import Moffat2D

    mof, (xg, yg, img) = moffimg

    # a closeish-but-wrong "guessed Moffat"
    guess_moffat = Moffat2D(x_0=.1, y_0=-.05, gamma=1.05,
                            amplitude=mof.amplitude*1.06, alpha=4.75)

    f = LevMarLSQFitter()

    fit_mof = f(guess_moffat, xg, yg, img)
    assert_allclose(fit_mof.parameters, mof.parameters, rtol=.01, atol=.0005)


# we set the tolerances in flux to be 2-3% because the shape paraameters of
# the guessed version are known to be wrong.
@pytest.mark.parametrize("prepkwargs,tols", [
                         (dict(xname='x_0', yname='y_0', fluxname=None,
                               renormalize_psf=True), (1e-3, .02)),
                         (dict(xname=None, yname=None, fluxname=None,
                               renormalize_psf=True), (1e-3, .02)),
                         (dict(xname=None, yname=None, fluxname=None,
                               renormalize_psf=False), (1e-3, .03)),
                         (dict(xname='x_0', yname='y_0', fluxname='amplitude',
                               renormalize_psf=False), (1e-3, None)),
                         ])
@pytest.mark.skipif('not HAS_SCIPY')
def test_psf_adapter(moffimg, prepkwargs, tols):
    """
    Test that the PSF adapter behaves as expected for fitting (don't worry
    about full-on psf photometry for now)
    """

    from astropy.modeling.fitting import LevMarLSQFitter
    from astropy.modeling.models import Moffat2D

    mof, (xg, yg, img) = moffimg
    f = LevMarLSQFitter()

    # a close-but-wrong "guessed Moffat"
    guess_moffat = Moffat2D(x_0=.1, y_0=-.05, gamma=1.01,
                            amplitude=mof.amplitude*1.01, alpha=4.79)
    if prepkwargs['renormalize_psf']:
        # definitely very wrong, so this ensures the re-normalization
        # stuff works
        guess_moffat.amplitude = 5.

    if prepkwargs['xname'] is None:
        guess_moffat.x_0 = 0
    if prepkwargs['yname'] is None:
        guess_moffat.y_0 = 0

    psfmod = prepare_psf_model(guess_moffat, **prepkwargs)
    xytol, fluxtol = tols

    fit_psfmod = f(psfmod, xg, yg, img)

    if xytol is not None:
        assert np.abs(getattr(fit_psfmod, fit_psfmod.xname)) < xytol
        assert np.abs(getattr(fit_psfmod, fit_psfmod.yname)) < xytol
    if fluxtol is not None:
        assert np.abs(1 - getattr(fit_psfmod, fit_psfmod.fluxname)) < fluxtol

    # ensure the amplitude and shape parameters did *not* change
    assert fit_psfmod.psfmodel.gamma == guess_moffat.gamma
    assert fit_psfmod.psfmodel.alpha == guess_moffat.alpha
    if prepkwargs['fluxname'] is None:
        assert fit_psfmod.psfmodel.amplitude == guess_moffat.amplitude


@pytest.mark.skipif('not HAS_SCIPY')
def test_get_grouped_psf_model():
    igp = IntegratedGaussianPRF(sigma=1.2)
    tab = Table(names=['x_0', 'y_0', 'flux_0'], data=[[1, 2], [3, 4], [0.5, 1]])
    pars_to_set = {'x_0':'x_0', 'y_0':'y_0', 'flux_0':'flux'}

    gpsf = get_grouped_psf_model(igp, tab, pars_to_set)

    assert gpsf.x_0_0 == 1
    assert gpsf.y_0_1 == 4
    assert gpsf.flux_0 == 0.5
    assert gpsf.flux_1 == 1
    assert gpsf.sigma_0 == gpsf.sigma_1 == 1.2
