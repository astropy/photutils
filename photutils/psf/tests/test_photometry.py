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
from ...background import SigmaClip, MedianBackground, StdBackgroundRMS

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
    Produces a baseline DAOPhotPSFPhotometry object which is then
    modified as-needed in specific tests below
    """

    daofind = DAOStarFinder(threshold=5.0*std,
                            fwhm=sigma_psf*gaussian_sigma_to_fwhm)
    daogroup = DAOGroup(1.5*sigma_psf*gaussian_sigma_to_fwhm)
    sigma_clip = SigmaClip(sigma=3.)
    median_bkg = MedianBackground(sigma_clip)
    psf_model = IntegratedGaussianPRF(sigma=sigma_psf)
    fitter = LevMarLSQFitter()
    return DAOPhotPSFPhotometry(finder=daofind, group_maker=daogroup,
                                bkg_estimator=median_bkg,
                                psf_model=psf_model, fitter=fitter, niters=1,
                                fitshape=(11, 11))


sigma_psfs = []

# A group of two overlapped stars and an isolated one
sigma_psfs.append(2)
sources1 = Table()
sources1['flux'] = [800, 1000, 1200]
sources1['x_mean'] = [13, 18, 25]
sources1['y_mean'] = [16, 16, 25]
sources1['x_stddev'] = [sigma_psfs[-1]] * 3
sources1['y_stddev'] = sources1['x_stddev']
sources1['theta'] = [0] * 3
sources1['id'] = [1, 2, 3]
sources1['group_id'] = [1, 1, 2]


# one single group with four stars.
sigma_psfs.append(2)
sources2 = Table()
sources2['flux'] = [700, 800, 700, 800]
sources2['x_mean'] = [12, 17, 12, 17]
sources2['y_mean'] = [15, 15, 20, 20]
sources2['x_stddev'] = [sigma_psfs[-1]] * 4
sources2['y_stddev'] = sources2['x_stddev']
sources2['theta'] = [0] * 4
sources2['id'] = [1, 2, 3, 4]
sources2['group_id'] = [1, 1, 1, 1]


@pytest.mark.xfail('not HAS_SCIPY or not HAS_MIN_ASTROPY')
@pytest.mark.parametrize("sigma_psf, sources",
                         [(sigma_psfs[0], sources1),
                          (sigma_psfs[1], sources2),
                          # these ensure that the test *fails* if the model
                          # PSFs are the wrong shape
                          pytest.mark.xfail((sigma_psfs[0]/1.2, sources1)),
                          pytest.mark.xfail((sigma_psfs[1]*1.2, sources2))])
def test_complete_photometry_oneiter(sigma_psf, sources):
    """
    Tests in an image with a group of two overlapped stars and an
    isolated one.
    """

    img_shape = (32, 32)

    # generate image with read-out noise (Gaussian) and
    # background noise (Poisson)
    image = (make_gaussian_sources(img_shape, sources) +
             make_noise_image(img_shape, type='poisson', mean=6.,
                              random_state=1) +
             make_noise_image(img_shape, type='gaussian', mean=0.,
                              stddev=2., random_state=1))

    sigma_clip = SigmaClip(sigma=3.)
    bkgrms = StdBackgroundRMS(sigma_clip)
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

    # setting the finder to None is not strictly needed, but is a good test
    # to make sure fixed photometry doesn't try to use the star-finder
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
