# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import division
import numpy as np
import astropy
from astropy.table import Table
from astropy.stats import gaussian_sigma_to_fwhm
from astropy.modeling.fitting import LevMarLSQFitter
from astropy.modeling import Parameter, Fittable2DModel
from astropy.tests.helper import pytest
from astropy.utils import minversion
from numpy.testing import assert_allclose, assert_array_equal, assert_equal
from ..models import IntegratedGaussianPRF
from ...datasets import make_gaussian_sources
from ...datasets import make_noise_image
from ..groupstars import DAOGroup
from ..photometry import DAOPhotPSFPhotometry, IterativelySubtractedPSFPhotometry, BasicPSFPhotometry
from ...detection import DAOStarFinder
from ...background import SigmaClip, MedianBackground, StdBackgroundRMS
from ...background import MedianBackground, MMMBackground, SigmaClip
from ...background import StdBackgroundRMS

ASTROPY_GT_1_1_2 = minversion('astropy', '1.1.2') 

try:
    import scipy
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


def make_psf_photometry_objs(std=1, sigma_psf=1):
    """
    Produces baseline photometry objects which are then
    modified as-needed in specific tests below
    """

    daofind = DAOStarFinder(threshold=5.0*std,
                            fwhm=sigma_psf*gaussian_sigma_to_fwhm)
    daogroup = DAOGroup(1.5*sigma_psf*gaussian_sigma_to_fwhm)
    sigma_clip = SigmaClip(sigma=3.)
    median_bkg = MedianBackground(sigma_clip)
    threshold = 5. * std
    fwhm = sigma_psf * gaussian_sigma_to_fwhm
    crit_separation = 1.5 * sigma_psf * gaussian_sigma_to_fwhm

    daofind = DAOStarFinder(threshold=threshold, fwhm=fwhm)
    daogroup = DAOGroup(crit_separation)
    mode_bkg = MMMBackground()
    psf_model = IntegratedGaussianPRF(sigma=sigma_psf)
    fitter = LevMarLSQFitter()

    basic_phot_obj = BasicPSFPhotometry(finder=daofind,
                                        group_maker=daogroup,
                                        bkg_estimator=mode_bkg,
                                        psf_model=psf_model,
                                        fitter=fitter,
                                        fitshape=(11, 11))


    iter_phot_obj = IterativelySubtractedPSFPhotometry(finder=daofind,
                                                       group_maker=daogroup,
                                                       bkg_estimator=mode_bkg,
                                                       psf_model=psf_model,
                                                       fitter=fitter, niters=1,
                                                       fitshape=(11, 11))

    dao_phot_obj = DAOPhotPSFPhotometry(crit_separation=crit_separation,
                                        threshold=threshold, fwhm=fwhm,
                                        psf_model=psf_model, fitshape=(11, 11),
                                        niters=1)

    return (basic_phot_obj, iter_phot_obj, dao_phot_obj)


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

# one faint star and one brither companion
# although they are in the same group, the detection algorithm
# is not able to detect the fainter star, hence photometry should
# be performed with niters > 1 or niters=None
sigma_psfs.append(2)
sources3 = Table()
sources3['flux'] = [10000, 1000]
sources3['x_mean'] = [18, 13]
sources3['y_mean'] = [17, 19]
sources3['x_stddev'] = [sigma_psfs[-1]] * 2
sources3['y_stddev'] = sources3['x_stddev']
sources3['theta'] = [0] * 2
sources3['id'] = [1] * 2
sources3['group_id'] = [1] * 2
sources3['iter_detected'] = [1, 2]


@pytest.mark.xfail('not HAS_SCIPY or not ASTROPY_GT_1_1_2')
@pytest.mark.parametrize("sigma_psf, sources", [(sigma_psfs[2], sources3)])
def test_psf_photometry_niters(sigma_psf, sources):
    img_shape = (32, 32)
    # generate image with read-out noise (Gaussian) and
    # background noise (Poisson)
    image = (make_gaussian_sources(img_shape, sources) +
             make_noise_image(img_shape, type='poisson', mean=6.,
                              random_state=1) +
             make_noise_image(img_shape, type='gaussian', mean=0.,
                              stddev=2., random_state=1))
    cp_image = image.copy()
    sigma_clip = SigmaClip(sigma=3.)
    bkgrms = StdBackgroundRMS(sigma_clip)
    std = bkgrms(image)

    iter_phot_obj = make_psf_photometry_objs(std, sigma_psf)[1]
    iter_phot_obj.niters = None
    result_tab = iter_phot_obj(image)
    residual_image = iter_phot_obj.get_residual_image()

    assert_allclose(result_tab['x_fit'], sources['x_mean'], rtol=1e-1)
    assert_allclose(result_tab['y_fit'], sources['y_mean'], rtol=1e-1)
    assert_allclose(result_tab['flux_fit'], sources['flux'], rtol=1e-1)
    assert_array_equal(result_tab['id'], sources['id'])
    assert_array_equal(result_tab['group_id'], sources['group_id'])
    assert_array_equal(result_tab['iter_detected'], sources['iter_detected'])
    assert_allclose(np.mean(residual_image), 0.0, atol=1e1)

    # make sure image is note overwritten
    assert_array_equal(cp_image, image)


@pytest.mark.xfail('not HAS_SCIPY or not ASTROPY_GT_1_1_2')
@pytest.mark.parametrize("sigma_psf, sources",
                         [(sigma_psfs[0], sources1),
                          (sigma_psfs[1], sources2),
                          # these ensure that the test *fails* if the model
                          # PSFs are the wrong shape
                          pytest.mark.xfail((sigma_psfs[0]/1.2, sources1)),
                          pytest.mark.xfail((sigma_psfs[1]*1.2, sources2))])
def test_psf_photometry_oneiter(sigma_psf, sources):
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
    cp_image = image.copy()

    sigma_clip = SigmaClip(sigma=3.)
    bkgrms = StdBackgroundRMS(sigma_clip)
    std = bkgrms(image)
    phot_objs = make_psf_photometry_objs(std, sigma_psf)

    for phot_proc in phot_objs:
        result_tab = phot_proc(image)
        residual_image = phot_proc.get_residual_image()
        assert_allclose(result_tab['x_fit'], sources['x_mean'], rtol=1e-1)
        assert_allclose(result_tab['y_fit'], sources['y_mean'], rtol=1e-1)
        assert_allclose(result_tab['flux_fit'], sources['flux'], rtol=1e-1)
        assert_array_equal(result_tab['id'], sources['id'])
        assert_array_equal(result_tab['group_id'], sources['group_id'])
        assert_allclose(np.mean(residual_image), 0.0, atol=1e1)

        # test fixed photometry
        phot_proc.psf_model.x_0.fixed = True
        phot_proc.psf_model.y_0.fixed = True

        pos = Table(names=['x_0', 'y_0'], data=[sources['x_mean'],
                                                sources['y_mean']])
        result_tab = phot_proc(image, pos)
        residual_image = phot_proc.get_residual_image()

        assert_array_equal(result_tab['x_fit'], sources['x_mean'])
        assert_array_equal(result_tab['y_fit'], sources['y_mean'])
        assert_allclose(result_tab['flux_fit'], sources['flux'], rtol=1e-1)
        assert_array_equal(result_tab['id'], sources['id'])
        assert_array_equal(result_tab['group_id'], sources['group_id'])
        assert_allclose(np.mean(residual_image), 0.0, atol=1e1)

        # make sure image is note overwritten
        assert_array_equal(cp_image, image)

        # resets fixed positions
        phot_proc.psf_model.x_0.fixed = False
        phot_proc.psf_model.y_0.fixed = False


@pytest.mark.xfail('not HAS_SCIPY')
def test_niters_errors():
    iter_phot_obj = make_psf_photometry_objs()[1]

    # tests that niters is set to an integer even if the user inputs
    # a float
    iter_phot_obj.niters = 1.1
    assert_equal(iter_phot_obj.niters, 1)

    # test that a ValueError is raised if niters <= 0
    with pytest.raises(ValueError):
        iter_phot_obj.niters = 0

    # test that it's OK to set niters to None
    iter_phot_obj.niters = None


@pytest.mark.xfail('not HAS_SCIPY')
def test_fitshape_erros():
    basic_phot_obj = make_psf_photometry_objs()[0]

    # first make sure setting to a scalar does the right thing (and makes
    # no errors)
    basic_phot_obj.fitshape = 11
    assert np.all(basic_phot_obj.fitshape == (11, 11))

    # test that a ValuError is raised if fitshape has even components
    with pytest.raises(ValueError):
        basic_phot_obj.fitshape = (2, 2)
    with pytest.raises(ValueError):
        basic_phot_obj.fitshape = 2

    # test that a ValueError is raised if fitshape has non positive
    # components
    with pytest.raises(ValueError):
        basic_phot_obj.fitshape = (-1, 0)

    # test that a ValueError is raised if fitshape has more than two
    # dimensions
    with pytest.raises(ValueError):
        basic_phot_obj.fitshape = (3, 3, 3)

@pytest.mark.xfail('not HAS_SCIPY')
def test_aperture_radius_errors():
    basic_phot_obj = make_psf_photometry_objs()[0]

    # test that aperture_radius was set to None by default
    assert_equal(basic_phot_obj.aperture_radius, None)

    # test that a ValuError is raised if aperture_radius is non positive
    with pytest.raises(ValueError):
        basic_phot_obj.aperture_radius = -3

@pytest.mark.xfail('not HAS_SCIPY')
def test_finder_erros():
    iter_phot_obj = make_psf_photometry_objs()[1]
    with pytest.raises(ValueError):
        iter_phot_obj.finder = None

    with pytest.raises(ValueError):
        iter_phot_obj = IterativelySubtractedPSFPhotometry(finder=None,
                group_maker=DAOGroup(1), bkg_estimator=MMMBackground(),
                psf_model=IntegratedGaussianPRF(1), fitshape=(11, 11))

@pytest.mark.xfail('not HAS_SCIPY or not ASTROPY_GT_1_1_2')
def test_aperture_radius():
    img_shape = (32, 32)

    # generate image with read-out noise (Gaussian) and
    # background noise (Poisson)
    image = (make_gaussian_sources(img_shape, sources1) +
             make_noise_image(img_shape, type='poisson', mean=6.,
                              random_state=1) +
             make_noise_image(img_shape, type='gaussian', mean=0.,
                              stddev=2., random_state=1))

    basic_phot_obj = make_psf_photometry_objs()[0]
    # test that aperture radius is properly set whenever the PSF model has
    # a `fwhm` attribute
    class PSFModelWithFWHM(Fittable2DModel):
        x_0 = Parameter(default=1)
        y_0 = Parameter(default=1)
        flux = Parameter(default=1)
        fwhm = Parameter(default=5)

        def __init__(self, fwhm=fwhm.default):
            super(PSFModelWithFWHM, self).__init__(fwhm=fwhm)

        def evaluate(self, x, y, x_0, y_0, flux, fwhm):
            return flux / (fwhm * (x - x_0)**2 * (y - y_0)**2)

    psf_model = PSFModelWithFWHM()
    basic_phot_obj.psf_model = psf_model
    result_tab = basic_phot_obj(image)

    assert_equal(basic_phot_obj.aperture_radius, psf_model.fwhm.value)
