# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
from numpy.testing import assert_allclose
import pytest

from astropy.modeling.models import Gaussian2D
from astropy.convolution.utils import discretize_model
from astropy.table import Table

from ..models import IntegratedGaussianPRF
from ..sandbox import DiscretePRF
from ..utils import prepare_psf_model, get_grouped_psf_model, subtract_psf

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
def test_prepare_psf_model(moffimg, prepkwargs, tols):
    """
    Test that prepare_psf_model behaves as expected for fitting (don't worry
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
    tab = Table(names=['x_0', 'y_0', 'flux_0'],
                data=[[1, 2], [3, 4], [0.5, 1]])
    pars_to_set = {'x_0': 'x_0', 'y_0': 'y_0', 'flux_0': 'flux'}

    gpsf = get_grouped_psf_model(igp, tab, pars_to_set)

    assert gpsf.x_0_0 == 1
    assert gpsf.y_0_1 == 4
    assert gpsf.flux_0 == 0.5
    assert gpsf.flux_1 == 1
    assert gpsf.sigma_0 == gpsf.sigma_1 == 1.2


@pytest.mark.skipif('not HAS_SCIPY')
def test_subtract_psf():
    """Test subtract_psf."""

    prf = DiscretePRF(test_psf, subsampling=1)
    posflux = INTAB.copy()
    for n in posflux.colnames:
        posflux.rename_column(n, n.split('_')[0] + '_fit')
    residuals = subtract_psf(image, prf, posflux)
    assert_allclose(residuals, np.zeros_like(image), atol=1E-4)
