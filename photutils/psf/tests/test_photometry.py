# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import division
import numpy as np
import astropy
from astropy.table import Table
from astropy.stats import gaussian_sigma_to_fwhm
from astropy.modeling.fitting import LevMarLSQFitter
from astropy.tests.helper import pytest
from numpy.testing import assert_allclose, assert_array_equal, assert_equal
from ..models import IntegratedGaussianPRF
from ...datasets import make_gaussian_sources
from ...datasets import make_noise_image
from ..groupstars import DAOGroup
from ..photometry import DAOPhotPSFPhotometry
from ...detection import DAOStarFinder
from ...background import MedianBackground
from ...background import StdBackgroundRMS

try:
    import scipy
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

if astropy.__version__ < '1.2':
    HAS_MIN_ASTROPY = False
else:
    HAS_MIN_ASTROPY = True


def make_fiducial_phot_obj(std=1, sigma_psf=1):
    """
    Produces a baseline DAOPhotPSFPhotometry object which is then modified as-needed
    in specific tests below
    """
    daofind = DAOStarFinder(threshold=5.0*std, fwhm=sigma_psf*gaussian_sigma_to_fwhm)
    daogroup = DAOGroup(1.5*sigma_psf*gaussian_sigma_to_fwhm)
    median_bkg = MedianBackground(sigma=3.)
    psf_model = IntegratedGaussianPRF(sigma=sigma_psf)
    fitter = LevMarLSQFitter()
    return DAOPhotPSFPhotometry(finder=daofind, group_maker=daogroup, bkg_estimator=median_bkg,
                                psf_model=psf_model, fitter=fitter, niters=1,
                                fitshape=(11, 11))


@pytest.mark.xfail('not HAS_SCIPY or not HAS_MIN_ASTROPY')
def test_complete_photometry_one():
    """
    Tests in an image with a group of two overlapped stars and an
    isolated one.
    """
    sigma_psf = 2.0
    sources = Table()
    sources['flux'] = [800, 1000, 1200]
    sources['x_mean'] = [13, 18, 25]
    sources['y_mean'] = [16, 16, 25]
    sources['x_stddev'] = [sigma_psf, sigma_psf, sigma_psf]
    sources['y_stddev'] = sources['x_stddev']
    sources['theta'] = [0, 0, 0]
    sources['id'] = [1, 2, 3]
    sources['group_id'] = [1, 1, 2]
    tshape = (32, 32)


    # generate image with read-out noise (Gaussian) and
    # background noise (Poisson)
    image = (make_gaussian_sources(tshape, sources) +
             make_noise_image(tshape, type='poisson', mean=6.,
                              random_state=1) +
             make_noise_image(tshape, type='gaussian', mean=0.,
                              stddev=2., random_state=1))

    bkgrms = StdBackgroundRMS(sigma=3.)
    std = bkgrms(image)

    phot_obj = make_fiducial_phot_obj(std, sigma_psf)

    result_tab, residual_image = phot_obj(image)

    assert_allclose(result_tab['x_fit'], sources['x_mean'], rtol=1e-1)
    assert_allclose(result_tab['y_fit'], sources['y_mean'], rtol=1e-1)
    assert_allclose(result_tab['flux_fit'], sources['flux'], rtol=1e-1)
    assert_array_equal(result_tab['id'], sources['id'])
    assert_array_equal(result_tab['group_id'], sources['group_id'])
    assert_allclose(np.mean(residual_image), 0.0, atol=1e1)

    # test fixed photometry
    phot_obj.psf_model.x_0.fixed = True
    phot_obj.psf_model.y_0.fixed = True

    # setting the finder to None is not strictly needed, but is a good test to
    # make sure fixed photometry doesn't try to use the star-finder
    phot_obj.finder = None

    pos = Table(names=['x_0', 'y_0'], data=[sources['x_mean'],
                                            sources['y_mean']])
    result_tab, residual_image = phot_obj(image, pos)

    assert_array_equal(result_tab['x_fit'], sources['x_mean'])
    assert_array_equal(result_tab['y_fit'], sources['y_mean'])
    assert_allclose(result_tab['flux_fit'], sources['flux'], rtol=1e-1)
    assert_array_equal(result_tab['id'], sources['id'])
    assert_array_equal(result_tab['group_id'], sources['group_id'])
    assert_allclose(np.mean(residual_image), 0.0, atol=1e1)


@pytest.mark.xfail('not HAS_SCIPY or not HAS_MIN_ASTROPY')
def test_complete_photometry_two():
    """
    Tests in an image with one single group with four stars.
    """
    sigma_psf = 2.0
    sources = Table()
    sources['flux'] = [700, 800, 700, 800]
    sources['x_mean'] = [12, 17, 12, 17]
    sources['y_mean'] = [15, 15, 20, 20]
    sources['x_stddev'] = sigma_psf*np.ones(4)
    sources['y_stddev'] = sources['x_stddev']
    sources['theta'] = [0, 0, 0, 0]
    sources['id'] = [1, 2, 3, 4]
    sources['group_id'] = [1, 1, 1, 1]
    tshape = (32, 32)

    image = (make_gaussian_sources(tshape, sources) +
             make_noise_image(tshape, type='poisson', mean=6.,
                              random_state=1) +
             make_noise_image(tshape, type='gaussian', mean=0.,
                              stddev=2., random_state=1))

    bkgrms = StdBackgroundRMS(sigma=3.)
    std = bkgrms(image)

    phot_obj = make_fiducial_phot_obj(std, sigma_psf)

    result_tab, residual_image = phot_obj(image)

    assert_allclose(result_tab['x_fit'], sources['x_mean'], rtol=1e-1)
    assert_allclose(result_tab['y_fit'], sources['y_mean'], rtol=1e-1)
    assert_allclose(result_tab['flux_fit'], sources['flux'], rtol=1e-1)
    assert_array_equal(result_tab['id'], sources['id'])
    assert_array_equal(result_tab['group_id'], sources['group_id'])
    assert_allclose(np.mean(residual_image), 0.0, atol=1e1)

    # test fixed photometry
    phot_obj.psf_model.x_0.fixed = True
    phot_obj.psf_model.y_0.fixed = True

    # setting the finder to None is not strictly needed, but is a good test to
    # make sure fixed photometry doesn't try to use the star-finder
    phot_obj.finder = None

    pos = Table(names=['x_0', 'y_0'], data=[sources['x_mean'],
                                            sources['y_mean']])
    result_tab, residual_image = phot_obj(image, pos)

    assert_array_equal(result_tab['x_fit'], sources['x_mean'])
    assert_array_equal(result_tab['y_fit'], sources['y_mean'])
    assert_allclose(result_tab['flux_fit'], sources['flux'], rtol=1e-1)
    assert_array_equal(result_tab['id'], sources['id'])
    assert_array_equal(result_tab['group_id'], sources['group_id'])
    assert_allclose(np.mean(residual_image), 0.0, atol=1e1)


@pytest.mark.xfail('not HAS_SCIPY or not HAS_MIN_ASTROPY')
def test_niters_exceptions():
    phot_obj = make_fiducial_phot_obj()

    # tests that niters is set to an integer even if the user inputs
    # a float
    phot_obj.niters = 1.1
    assert_equal(phot_obj.niters, 1)

    # test that a ValueError is raised if niters <= 0
    with pytest.raises(ValueError):
        phot_obj.niters = 0

    # test that it's OK to set niters to None
    phot_obj.niters = None


@pytest.mark.xfail('not HAS_SCIPY or not HAS_MIN_ASTROPY')
def test_fitshape_exceptions():
    phot_obj = make_fiducial_phot_obj()

    # first make sure setting to a scalar does the right thing (and makes
    # no errors)
    phot_obj.fitshape = 11
    assert np.all(phot_obj.fitshape == (11, 11))

    # test that a ValuError is raised if fitshape has even components
    with pytest.raises(ValueError):
        phot_obj.fitshape = (2, 2)
    with pytest.raises(ValueError):
        phot_obj.fitshape = 2

    # test that a ValuError is raised if fitshape has non positive
    # components
    with pytest.raises(ValueError):
        phot_obj.fitshape = (-1, 0)

    # test that a ValuError is raised if fitshape does not have two
    # components
    with pytest.raises(ValueError):
        phot_obj.fitshape = 2


@pytest.mark.xfail('not HAS_SCIPY or not HAS_MIN_ASTROPY')
def test_aperture_radius_exceptions():
    phot_obj = make_fiducial_phot_obj()

    # test that aperture_radius was set to None by default
    assert_equal(phot_obj.aperture_radius, None)

    # test that a ValuError is raised if aperture_radius is non positive
    with pytest.raises(ValueError):
        phot_obj.aperture_radius = -3
