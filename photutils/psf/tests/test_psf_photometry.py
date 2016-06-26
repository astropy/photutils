# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import division

import numpy as np
from numpy.testing import assert_allclose, assert_equal

import warnings
from astropy.utils.exceptions import AstropyUserWarning

from astropy.tests.helper import pytest
from astropy.modeling.models import Gaussian2D
from astropy.modeling.fitting import SLSQPLSQFitter
from astropy.convolution.utils import discretize_model
from astropy.table import Table

from .. import IntegratedGaussianPRF, psf_photometry, subtract_psf
from ..sandbox import DiscretePRF

try:
    from scipy import optimize
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

if pytest.__version__ >= '2.8':
    HAS_PYTEST_GEQ_28 = True
else:
    HAS_PYTEST_GEQ_28 = False


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
                                        PSF_SIZE, subsampling=1, mode='median')
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
                                        PSF_SIZE, subsampling=1, mode='median',
                                        fluxes=INTAB['flux_0'])
    assert_allclose(prf._prf_array[0, 0].sum(), 1)
    assert_allclose(prf._prf_array[0, 0], test_psf, atol=1E-8)


@pytest.mark.skipif('not HAS_SCIPY')
def test_psf_photometry_discrete():
    """
    Test psf_photometry with discrete PRF model.
    """
    prf = DiscretePRF(test_psf, subsampling=1)
    f = psf_photometry(image, INTAB, prf)
    for n in ['x', 'y', 'flux']:
        assert_allclose(f[n + '_0'], f[n + '_fit'], rtol=1e-6)


@pytest.mark.skipif('not HAS_SCIPY')
def test_tune_coordinates():
    """
    Test psf_photometry with discrete PRF model and coordinates that need
    to be adjusted in the fit.
    """
    prf = DiscretePRF(test_psf, subsampling=1)
    prf.x_0.fixed = False
    prf.y_0.fixed = False
    # Shift all sources by 0.3 pixels
    intab = INTAB.copy()
    intab['x_0'] += 0.3
    f = psf_photometry(image, intab, prf)
    for n in ['x', 'y', 'flux']:
        assert_allclose(f[n + '_0'], f[n + '_fit'], rtol=1e-3)


@pytest.mark.skipif('not HAS_SCIPY')
def test_psf_boundary():
    """
    Test psf_photometry with discrete PRF model at the boundary of the data.
    """
    prf = DiscretePRF(test_psf, subsampling=1)
    f = psf_photometry(image, np.ones((2, 1)), prf)
    assert_allclose(f['flux_fit'], 0, atol=1e-8)


@pytest.mark.skipif('not HAS_SCIPY')
def test_psf_boundary_gaussian():
    """
    Test psf_photometry at the boundary of the data where no source is found.
    This also tests the case were input positions is an array instead
    of a table.
    """
    psf = IntegratedGaussianPRF(GAUSSIAN_WIDTH)
    f = psf_photometry(image, np.ones((2, 1)), psf)
    assert_allclose(f['flux_fit'], 0, atol=1e-8)


@pytest.mark.skipif('not HAS_SCIPY')
def test_psf_photometry_gaussian():
    """
    Test psf_photometry with Gaussian PSF model.
    """
    psf = IntegratedGaussianPRF(sigma=GAUSSIAN_WIDTH)
    f = psf_photometry(image, INTAB, psf)
    for n in ['x', 'y', 'flux']:
        assert_allclose(f[n + '_0'], f[n + '_fit'], rtol=1e-3)


@pytest.mark.skipif('not HAS_SCIPY')
def test_psf_photometry_uncertainties():
    """
    Make sure proper columns are added to store uncertainties on fitted
    parameters.
    """
    psf = IntegratedGaussianPRF(sigma=GAUSSIAN_WIDTH)
    f = psf_photometry(image, INTAB, psf, param_uncert=True)
    assert_equal(f['flux_fit_uncertainty'].all() > 0.1 and
                 f['flux_fit_uncertainty'].all() < 10.0 and 
                 f['x_0_fit_uncertainty'].all() > 0.1 and
                 f['x_0_fit_uncertainty'].all() < 10.0 and
                 f['y_0_fit_uncertainty'].all() > 0.1 and
                 f['y_0_fit_uncertainty'].all() < 10.0, True)

    # test for fixed params
    psf = IntegratedGaussianPRF(sigma=GAUSSIAN_WIDTH)
    psf.flux.fixed = True
    f = psf_photometry(image, INTAB, psf, param_uncert=True)
    assert_equal(f['x_0_fit_uncertainty'].all() > 0.1 and
                 f['x_0_fit_uncertainty'].all() < 10.0 and
                 f['y_0_fit_uncertainty'].all() > 0 and
                 f['y_0_fit_uncertainty'].all() < 10.0, True)
    assert_equal('flux_fit_uncertainty' in f.colnames, False)

# test in case fitter does not have 'param_cov' key
@pytest.mark.skipif('not HAS_PYTEST_GEQ_28')
def test_psf_photometry_uncertainties_warning_check():
    psf = IntegratedGaussianPRF(sigma=GAUSSIAN_WIDTH)
    with pytest.warns(AstropyUserWarning):
        f = psf_photometry(image, INTAB, psf, fitter=SLSQPLSQFitter(),
                           param_uncert=True)
        assert_equal('flux_fit_uncertainty' in f.colnames or\
                     'y_0_fit_uncertainty' in f.colnames or \
                     'x_0_fit_uncertainty' in f.colnames, False)
        # test that AstropyUserWarning is raised
        warnings.warn("uncertainties on fitted parameters cannot be " +
                      "computed because fitter does not contain " +
                      "`param_cov` key in its `fit_info` dictionary.",
                      AstropyUserWarning)

@pytest.mark.skipif('not HAS_SCIPY')
def test_subtract_psf():
    """
    Test subtract_psf
    """
    prf = DiscretePRF(test_psf, subsampling=1)
    posflux = INTAB.copy()
    for n in posflux.colnames:
        posflux.rename_column(n, n.split('_')[0] + '_fit')
    residuals = subtract_psf(image, prf, posflux)
    assert_allclose(residuals, np.zeros_like(image), atol=1E-4)


@pytest.mark.skipif('not HAS_SCIPY')
def test_psf_fitting_data_on_edge():
    """
    No mask is input explicitly here, but source 2 is so close to the edge
    that the subarray that's extracted gets a mask internally.
    """
    psf_guess = IntegratedGaussianPRF(flux=1, sigma=WIDE_GAUSSIAN_WIDTH)
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

    psf_guess = IntegratedGaussianPRF(flux=1, sigma=WIDE_GAUSSIAN_WIDTH)
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
