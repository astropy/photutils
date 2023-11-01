# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the photometry module.
"""

from contextlib import nullcontext

import astropy
import numpy as np
import pytest
from astropy.convolution.utils import discretize_model
from astropy.modeling import Fittable2DModel, Parameter
from astropy.modeling.fitting import LevMarLSQFitter, SimplexLSQFitter
from astropy.modeling.models import Gaussian2D, Moffat2D
from astropy.stats import SigmaClip, gaussian_sigma_to_fwhm
from astropy.table import Table
from astropy.utils import minversion
from astropy.utils.exceptions import (AstropyDeprecationWarning,
                                      AstropyUserWarning)
from numpy.testing import assert_allclose, assert_array_equal, assert_equal

from photutils.background import MMMBackground, StdBackgroundRMS
from photutils.datasets import (make_gaussian_prf_sources_image,
                                make_noise_image)
from photutils.detection import DAOStarFinder, IRAFStarFinder
from photutils.psf.groupstars import DAOGroup
from photutils.psf.models import FittableImageModel, IntegratedGaussianPRF
from photutils.psf.photometry_depr import (BasicPSFPhotometry,
                                           DAOPhotPSFPhotometry,
                                           IterativelySubtractedPSFPhotometry)
from photutils.psf.utils import prepare_psf_model
from photutils.tests.helper import PYTEST_LT_80
from photutils.utils._optional_deps import HAS_SCIPY


def make_psf_photometry_objs(std=1, sigma_psf=1):
    """
    Produces baseline photometry objects which are then
    modified as-needed in specific tests below
    """
    with pytest.warns(AstropyDeprecationWarning):
        daofind = DAOStarFinder(threshold=5.0 * std,
                                fwhm=sigma_psf * gaussian_sigma_to_fwhm)
        daogroup = DAOGroup(1.5 * sigma_psf * gaussian_sigma_to_fwhm)
        threshold = 5.0 * std
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

        iter_phot_obj = IterativelySubtractedPSFPhotometry(
            finder=daofind, group_maker=daogroup, bkg_estimator=mode_bkg,
            psf_model=psf_model, fitter=fitter, niters=1, fitshape=(11, 11))

        dao_phot_obj = DAOPhotPSFPhotometry(crit_separation=crit_separation,
                                            threshold=threshold, fwhm=fwhm,
                                            psf_model=psf_model,
                                            fitshape=(11, 11),
                                            niters=1)

        return (basic_phot_obj, iter_phot_obj, dao_phot_obj)


sigma_psfs = []

# A group of two overlapped stars and an isolated one
sigma_psfs.append(2)
sources1 = Table()
sources1['flux'] = [800, 1000, 1200]
sources1['x_0'] = [13, 18, 25]
sources1['y_0'] = [16, 16, 25]
sources1['sigma'] = [sigma_psfs[-1]] * 3
sources1['theta'] = [0] * 3
sources1['id'] = [1, 2, 3]
sources1['group_id'] = [1, 1, 2]


# one single group with four stars.
sigma_psfs.append(2)
sources2 = Table()
sources2['flux'] = [700, 800, 700, 800]
sources2['x_0'] = [12, 17, 12, 17]
sources2['y_0'] = [15, 15, 20, 20]
sources2['sigma'] = [sigma_psfs[-1]] * 4
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
sources3['x_0'] = [18, 13]
sources3['y_0'] = [17, 19]
sources3['sigma'] = [sigma_psfs[-1]] * 2
sources3['theta'] = [0] * 2
sources3['id'] = [1] * 2
sources3['group_id'] = [1] * 2
sources3['iter_detected'] = [1, 2]


@pytest.mark.skipif(not HAS_SCIPY, reason='scipy is required')
@pytest.mark.parametrize('sigma_psf, sources', [(sigma_psfs[2], sources3)])
def test_psf_photometry_niters(sigma_psf, sources):
    img_shape = (32, 32)
    # generate image with read-out noise (Gaussian) and
    # background noise (Poisson)
    image = (make_gaussian_prf_sources_image(img_shape, sources)
             + make_noise_image(img_shape, distribution='poisson', mean=6.0,
                                seed=0)
             + make_noise_image(img_shape, distribution='gaussian', mean=0.0,
                                stddev=2.0, seed=0))
    cp_image = image.copy()
    sigma_clip = SigmaClip(sigma=3.0)
    bkgrms = StdBackgroundRMS(sigma_clip)
    std = bkgrms(image)

    phot_obj = make_psf_photometry_objs(std, sigma_psf)[1:3]
    for iter_phot_obj in phot_obj:
        iter_phot_obj.niters = None

        with pytest.warns(AstropyDeprecationWarning):
            result_tab = iter_phot_obj(image)
            residual_image = iter_phot_obj.get_residual_image()

        assert (result_tab['x_0_unc'] < 1.96 * sigma_psf
                / np.sqrt(sources['flux'])).all()
        assert (result_tab['y_0_unc'] < 1.96 * sigma_psf
                / np.sqrt(sources['flux'])).all()
        assert (result_tab['flux_unc'] < 1.96
                * np.sqrt(sources['flux'])).all()

        assert_allclose(result_tab['x_fit'], sources['x_0'], rtol=1e-1)
        assert_allclose(result_tab['y_fit'], sources['y_0'], rtol=1e-1)
        assert_allclose(result_tab['flux_fit'], sources['flux'], rtol=1e-1)
        assert_array_equal(result_tab['id'], sources['id'])
        assert_array_equal(result_tab['group_id'], sources['group_id'])
        assert_array_equal(result_tab['iter_detected'],
                           sources['iter_detected'])
        assert_allclose(np.mean(residual_image), 0.0, atol=1e1)

        # make sure image is note overwritten
        assert_array_equal(cp_image, image)


@pytest.mark.filterwarnings('ignore:Both init_guesses and finder are '
                            'different than None')
@pytest.mark.filterwarnings('ignore:No sources were found')
@pytest.mark.skipif(not HAS_SCIPY, reason='scipy is required')
@pytest.mark.parametrize('sigma_psf, sources',
                         [(sigma_psfs[0], sources1),
                          (sigma_psfs[1], sources2)])
def test_psf_photometry_oneiter(sigma_psf, sources):
    """
    Tests in an image with a group of two overlapped stars and an
    isolated one.
    """

    img_shape = (32, 32)
    # generate image with read-out noise (Gaussian) and
    # background noise (Poisson)
    image = (make_gaussian_prf_sources_image(img_shape, sources)
             + make_noise_image(img_shape, distribution='poisson', mean=6.0,
                                seed=0)
             + make_noise_image(img_shape, distribution='gaussian', mean=0.0,
                                stddev=2.0, seed=0))
    cp_image = image.copy()

    sigma_clip = SigmaClip(sigma=3.0)
    bkgrms = StdBackgroundRMS(sigma_clip)
    std = bkgrms(image)
    phot_objs = make_psf_photometry_objs(std, sigma_psf)

    for phot_proc in phot_objs:
        with pytest.warns(AstropyDeprecationWarning):
            result_tab = phot_proc(image)
            residual_image = phot_proc.get_residual_image()
        assert (result_tab['x_0_unc'] < 1.96 * sigma_psf
                / np.sqrt(sources['flux'])).all()
        assert (result_tab['y_0_unc'] < 1.96 * sigma_psf
                / np.sqrt(sources['flux'])).all()
        assert (result_tab['flux_unc'] < 1.96
                * np.sqrt(sources['flux'])).all()
        assert_allclose(result_tab['x_fit'], sources['x_0'], rtol=1e-1)
        assert_allclose(result_tab['y_fit'], sources['y_0'], rtol=1e-1)
        assert_allclose(result_tab['flux_fit'], sources['flux'], rtol=1e-1)
        assert_array_equal(result_tab['id'], sources['id'])
        assert_array_equal(result_tab['group_id'], sources['group_id'])
        assert_allclose(np.mean(residual_image), 0.0, atol=1e1)

        # test fixed photometry
        phot_proc.psf_model.x_0.fixed = True
        phot_proc.psf_model.y_0.fixed = True

        pos = Table(names=['x_0', 'y_0'], data=[sources['x_0'],
                                                sources['y_0']])
        cp_pos = pos.copy()

        with pytest.warns(AstropyDeprecationWarning):
            result_tab = phot_proc(image, init_guesses=pos)
            residual_image = phot_proc.get_residual_image()
        assert 'x_0_unc' not in result_tab.colnames
        assert 'y_0_unc' not in result_tab.colnames
        assert (result_tab['flux_unc'] < 1.96
                * np.sqrt(sources['flux'])).all()
        assert_array_equal(result_tab['x_fit'], sources['x_0'])
        assert_array_equal(result_tab['y_fit'], sources['y_0'])
        assert_allclose(result_tab['flux_fit'], sources['flux'], rtol=1e-1)
        assert_array_equal(result_tab['id'], sources['id'])
        assert_array_equal(result_tab['group_id'], sources['group_id'])
        assert_allclose(np.mean(residual_image), 0.0, atol=1e1)

        # make sure image is not overwritten
        assert_array_equal(cp_image, image)

        # make sure initial guess table is not modified
        assert_array_equal(cp_pos, pos)

        # resets fixed positions
        phot_proc.psf_model.x_0.fixed = False
        phot_proc.psf_model.y_0.fixed = False


@pytest.mark.skipif(not HAS_SCIPY, reason='scipy is required')
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


@pytest.mark.skipif(not HAS_SCIPY, reason='scipy is required')
def test_fitshape_errors():
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


@pytest.mark.skipif(not HAS_SCIPY, reason='scipy is required')
def test_aperture_radius_errors():
    basic_phot_obj = make_psf_photometry_objs()[0]

    # test that aperture_radius was set to None by default
    assert_equal(basic_phot_obj.aperture_radius, None)

    # test that a ValueError is raised if aperture_radius is non positive
    with pytest.raises(ValueError):
        basic_phot_obj.aperture_radius = -3


@pytest.mark.skipif(not HAS_SCIPY, reason='scipy is required')
def test_finder_errors():
    with pytest.warns(AstropyDeprecationWarning):
        iter_phot_obj = make_psf_photometry_objs()[1]

        with pytest.raises(ValueError):
            iter_phot_obj.finder = None

        with pytest.raises(ValueError):
            iter_phot_obj = IterativelySubtractedPSFPhotometry(
                finder=None, group_maker=DAOGroup(1),
                bkg_estimator=MMMBackground(),
                psf_model=IntegratedGaussianPRF(1), fitshape=(11, 11))


@pytest.mark.skipif(not HAS_SCIPY, reason='scipy is required')
def test_finder_positions_warning():
    basic_phot_obj = make_psf_photometry_objs(sigma_psf=2)[0]
    positions = Table()
    positions['x_0'] = [12.8, 18.2, 25.3]
    positions['y_0'] = [15.7, 16.5, 25.1]

    image = (make_gaussian_prf_sources_image((32, 32), sources1)
             + make_noise_image((32, 32), distribution='poisson', mean=6.0,
                                seed=0))

    match = 'Both init_guesses and finder are different than None'
    ctx1 = pytest.warns(AstropyUserWarning, match=match)
    if PYTEST_LT_80:
        ctx2 = nullcontext()
    else:
        ctx2 = pytest.warns(AstropyDeprecationWarning)
    with ctx1, ctx2:
        result_tab = basic_phot_obj(image=image, init_guesses=positions)
        assert_array_equal(result_tab['x_0'], positions['x_0'])
        assert_array_equal(result_tab['y_0'], positions['y_0'])
        assert_allclose(result_tab['x_fit'], positions['x_0'], rtol=1e-1)
        assert_allclose(result_tab['y_fit'], positions['y_0'], rtol=1e-1)

    basic_phot_obj.finder = None
    with pytest.raises(ValueError):
        result_tab = basic_phot_obj(image=image)


@pytest.mark.filterwarnings('ignore:The fit may be unsuccessful')
@pytest.mark.skipif(not HAS_SCIPY, reason='scipy is required')
def test_aperture_radius():
    img_shape = (32, 32)

    # generate image with read-out noise (Gaussian) and
    # background noise (Poisson)
    image = (make_gaussian_prf_sources_image(img_shape, sources1)
             + make_noise_image(img_shape, distribution='poisson', mean=6.0,
                                seed=0)
             + make_noise_image(img_shape, distribution='gaussian', mean=0.0,
                                stddev=2.0, seed=0))

    basic_phot_obj = make_psf_photometry_objs()[0]

    # test that aperture radius is properly set whenever the PSF model has
    # a `fwhm` attribute
    class PSFModelWithFWHM(Fittable2DModel):
        x_0 = Parameter(default=1)
        y_0 = Parameter(default=1)
        flux = Parameter(default=1)
        fwhm = Parameter(default=5)

        def __init__(self, fwhm=fwhm.default):
            super().__init__(fwhm=fwhm)

        def evaluate(self, x, y, x_0, y_0, flux, fwhm):
            return flux / (fwhm * (x - x_0)**2 * (y - y_0)**2)

    psf_model = PSFModelWithFWHM()
    basic_phot_obj.psf_model = psf_model
    with pytest.warns(AstropyDeprecationWarning):
        basic_phot_obj(image)
    assert_equal(basic_phot_obj.aperture_radius, psf_model.fwhm.value)


PARS_TO_SET_0 = {'x_0': 'x_0', 'y_0': 'y_0', 'flux_0': 'flux'}
PARS_TO_OUTPUT_0 = {'x_fit': 'x_0', 'y_fit': 'y_0', 'flux_fit': 'flux'}
PARS_TO_SET_1 = PARS_TO_SET_0.copy()
PARS_TO_SET_1['sigma_0'] = 'sigma'
PARS_TO_OUTPUT_1 = PARS_TO_OUTPUT_0.copy()
PARS_TO_OUTPUT_1['sigma_fit'] = 'sigma'


@pytest.mark.parametrize('actual_pars_to_set, actual_pars_to_output,'
                         'is_sigma_fixed', [(PARS_TO_SET_0, PARS_TO_OUTPUT_0,
                                             True),
                                            (PARS_TO_SET_1, PARS_TO_OUTPUT_1,
                                             False)])
@pytest.mark.skipif(not HAS_SCIPY, reason='scipy is required')
def test_define_fit_param_names(actual_pars_to_set, actual_pars_to_output,
                                is_sigma_fixed):
    psf_model = IntegratedGaussianPRF()
    psf_model.sigma.fixed = is_sigma_fixed

    basic_phot_obj = make_psf_photometry_objs()[0]
    basic_phot_obj.psf_model = psf_model

    with pytest.warns(AstropyDeprecationWarning):
        basic_phot_obj._define_fit_param_names()
    assert_equal(basic_phot_obj._pars_to_set, actual_pars_to_set)
    assert_equal(basic_phot_obj._pars_to_output, actual_pars_to_output)


# tests previously written to psf_photometry

PSF_SIZE = 11
GAUSSIAN_WIDTH = 1.0
IMAGE_SIZE = 101

# Position and FLUXES of test sources
INTAB = Table([[50.0, 23, 12, 86], [50.0, 83, 80, 84],
               [np.pi * 10, 3.654, 20.0, 80 / np.sqrt(3)]],
              names=['x_0', 'y_0', 'flux_0'])

# Create test psf
psf_model = Gaussian2D(1.0 / (2 * np.pi * GAUSSIAN_WIDTH ** 2), PSF_SIZE // 2,
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
WIDE_GAUSSIAN_WIDTH = 3.0
WIDE_INTAB = Table([[50, 23.2], [50.5, 1], [10, 20]],
                   names=['x_0', 'y_0', 'flux_0'])
wide_image = np.zeros((IMAGE_SIZE, IMAGE_SIZE))

# Add sources to test image
for x, y, flux in WIDE_INTAB:
    model = Gaussian2D(flux / (2 * np.pi * WIDE_GAUSSIAN_WIDTH ** 2),
                       x, y, WIDE_GAUSSIAN_WIDTH, WIDE_GAUSSIAN_WIDTH)
    wide_image += discretize_model(model, (0, IMAGE_SIZE), (0, IMAGE_SIZE),
                                   mode='oversample')


@pytest.mark.skipif(not HAS_SCIPY, reason='scipy is required')
def test_default_aperture_radius():
    """
    Test psf_photometry with non-Gaussian model, such that it raises a
    warning about aperture_radius.
    """

    def tophatfinder(image, mask=None):
        """Simple top hat finder function for use with a top hat PRF."""
        fluxes = np.unique(image[image > 1])
        table = Table(names=['id', 'xcentroid', 'ycentroid', 'flux'],
                      dtype=[int, float, float, float])
        for n, f in enumerate(fluxes):
            ys, xs = np.where(image == f)
            x = np.mean(xs)
            y = np.mean(ys)
            table.add_row([int(n + 1), x, y, f * 9])
        table.sort(['flux'])

        return table

    with pytest.warns(AstropyDeprecationWarning):
        prf = np.zeros((7, 7), float)
        prf[2:5, 2:5] = 1 / 9
        prf = FittableImageModel(prf)

        img = np.zeros((50, 50), float)
        x0 = [38, 20, 35]
        y0 = [20, 5, 40]
        f0 = [50, 100, 200]
        for x, y, f in zip(x0, y0, f0):
            img[y - 1:y + 2, x - 1:x + 2] = f / 9

        intab = Table(data=[[37, 19.6, 34.9], [19.6, 4.5, 40.1]],
                      names=['x_0', 'y_0'])

        basic_phot = BasicPSFPhotometry(group_maker=DAOGroup(2),
                                        bkg_estimator=None, psf_model=prf,
                                        fitshape=7, finder=tophatfinder)
        # Test for init_guesses is None
        match = 'aperture_radius is None and could not be determined'
        with pytest.warns(AstropyUserWarning, match=match):
            results = basic_phot(image=img)
        assert_allclose(results['flux_fit'], f0, rtol=0.05)

        # Have to reset the object or it saves any updates, and we wish to
        # re-verify the aperture_radius assignment
        basic_phot = BasicPSFPhotometry(group_maker=DAOGroup(2),
                                        bkg_estimator=None, psf_model=prf,
                                        fitshape=7)

        # Test for init_guesses is not None, but lacks a flux_0 column
        match = 'aperture_radius is None and could not be determined'
        with pytest.warns(AstropyUserWarning, match=match):
            results = basic_phot(image=img, init_guesses=intab)
        assert_allclose(results['flux_fit'], f0, rtol=0.05)

        iter_phot = IterativelySubtractedPSFPhotometry(finder=tophatfinder,
                                                       group_maker=DAOGroup(2),
                                                       bkg_estimator=None,
                                                       psf_model=prf,
                                                       fitshape=7, niters=2)

        match1 = 'aperture_radius is None and could not be determined'
        ctx1 = pytest.warns(AstropyUserWarning, match=match1)
        if PYTEST_LT_80:
            ctx2 = nullcontext()
        else:
            match2 = 'Both init_guesses and finder are different than None'
            ctx2 = pytest.warns(AstropyUserWarning, match=match2)
        with ctx1, ctx2:
            results = iter_phot(image=img, init_guesses=intab)
            assert_allclose(results['flux_fit'], f0, rtol=0.05)

        iter_phot = IterativelySubtractedPSFPhotometry(
            finder=tophatfinder, group_maker=DAOGroup(2), bkg_estimator=None,
            psf_model=prf, fitshape=7, niters=2)

        # Test for init_guesses is None
        match = 'aperture_radius is None and could not be determined'
        with pytest.warns(AstropyUserWarning, match=match):
            results = iter_phot(image=img)
        assert_allclose(results['flux_fit'], f0, rtol=0.05)


@pytest.mark.skipif(not HAS_SCIPY, reason='scipy is required')
def test_psf_boundary_gaussian():
    """
    Test psf_photometry with discrete PRF model at the boundary of the data.
    """
    with pytest.warns(AstropyDeprecationWarning):
        psf = IntegratedGaussianPRF(GAUSSIAN_WIDTH)

        basic_phot = BasicPSFPhotometry(group_maker=DAOGroup(2),
                                        bkg_estimator=None, psf_model=psf,
                                        fitshape=7)

        intab = Table(data=[[1], [1]], names=['x_0', 'y_0'])
        f = basic_phot(image=image, init_guesses=intab)
        assert_allclose(f['flux_fit'], 0, atol=1e-8)


@pytest.mark.skipif(not HAS_SCIPY, reason='scipy is required')
def test_psf_photometry_gaussian():
    """
    Test psf_photometry with Gaussian PSF model.
    """
    with pytest.warns(AstropyDeprecationWarning):
        psf = IntegratedGaussianPRF(sigma=GAUSSIAN_WIDTH)

        basic_phot = BasicPSFPhotometry(group_maker=DAOGroup(2),
                                        bkg_estimator=None, psf_model=psf,
                                        fitshape=7)
        f = basic_phot(image=image, init_guesses=INTAB)
        for n in ['x', 'y', 'flux']:
            assert_allclose(f[n + '_0'], f[n + '_fit'], rtol=1e-3)


@pytest.mark.skipif(not HAS_SCIPY, reason='scipy is required')
@pytest.mark.parametrize('renormalize_psf', (True, False))
def test_psf_photometry_gaussian2(renormalize_psf):
    """
    Test psf_photometry with Gaussian PSF model from Astropy.
    """
    with pytest.warns(AstropyDeprecationWarning):
        psf = Gaussian2D(1.0 / (2 * np.pi * GAUSSIAN_WIDTH ** 2),
                         PSF_SIZE // 2, PSF_SIZE // 2, GAUSSIAN_WIDTH,
                         GAUSSIAN_WIDTH)
        psf = prepare_psf_model(psf, xname='x_mean', yname='y_mean',
                                renormalize_psf=renormalize_psf)

        basic_phot = BasicPSFPhotometry(group_maker=DAOGroup(2),
                                        bkg_estimator=None, psf_model=psf,
                                        fitshape=7)
        match = ('aperture_radius is None and could not be determined by '
                 'psf_model')
        with pytest.warns(AstropyUserWarning, match=match):
            f = basic_phot(image=image, init_guesses=INTAB)

        for n in ['x', 'y']:
            assert_allclose(f[n + '_0'], f[n + '_fit'], rtol=1e-1)
        assert_allclose(f['flux_0'], f['flux_fit'], rtol=1e-1)


@pytest.mark.skipif(not HAS_SCIPY, reason='scipy is required')
def test_psf_photometry_moffat():
    """
    Test psf_photometry with Moffat PSF model from Astropy.
    """
    with pytest.warns(AstropyDeprecationWarning):
        psf = Moffat2D(1.0 / (2 * np.pi * GAUSSIAN_WIDTH ** 2), PSF_SIZE // 2,
                       PSF_SIZE // 2, 1, 1)
        psf = prepare_psf_model(psf, xname='x_0', yname='y_0',
                                renormalize_psf=False)

        basic_phot = BasicPSFPhotometry(group_maker=DAOGroup(2),
                                        bkg_estimator=None, psf_model=psf,
                                        fitshape=7)
        match = ('aperture_radius is None and could not be determined by '
                 'psf_model')
        with pytest.warns(AstropyUserWarning, match=match):
            f = basic_phot(image=image, init_guesses=INTAB)
        f.pprint(max_width=-1)

        for n in ['x', 'y']:
            assert_allclose(f[n + '_0'], f[n + '_fit'], rtol=1e-3)
        # image was created with a gaussian, so flux won't match exactly
        assert_allclose(f['flux_0'], f['flux_fit'], rtol=1e-1)


@pytest.mark.skipif(not HAS_SCIPY, reason='scipy is required')
def test_psf_fitting_data_on_edge():
    """
    No mask is input explicitly here, but source 2 is so close to the
    edge that the subarray that's extracted gets a mask internally.
    """
    with pytest.warns(AstropyDeprecationWarning):
        psf_guess = IntegratedGaussianPRF(flux=1, sigma=WIDE_GAUSSIAN_WIDTH)
        psf_guess.flux.fixed = psf_guess.x_0.fixed = False
        psf_guess.y_0.fixed = False
        basic_phot = BasicPSFPhotometry(group_maker=DAOGroup(2),
                                        bkg_estimator=None,
                                        psf_model=psf_guess, fitshape=7)

        outtab = basic_phot(image=wide_image, init_guesses=WIDE_INTAB)

        for n in ['x', 'y', 'flux']:
            assert_allclose(outtab[n + '_0'], outtab[n + '_fit'],
                            rtol=0.05, atol=0.1)


@pytest.mark.filterwarnings('ignore:Both init_guesses and finder '
                            'are different than None')
@pytest.mark.skipif(not HAS_SCIPY, reason='scipy is required')
@pytest.mark.parametrize('sigma_psf, sources', [(sigma_psfs[2], sources3)])
def test_psf_extra_output_cols(sigma_psf, sources):
    """
    Test the handling of a non-None extra_output_cols
    """
    with pytest.warns(AstropyDeprecationWarning):
        psf_model = IntegratedGaussianPRF(sigma=sigma_psf)
        tshape = (32, 32)
        image = (make_gaussian_prf_sources_image(tshape, sources)
                 + make_noise_image(tshape, distribution='poisson', mean=6.0,
                                    seed=0)
                 + make_noise_image(tshape, distribution='gaussian', mean=0.0,
                                    stddev=2.0, seed=0))

        init_guess1 = None
        init_guess2 = Table(names=['x_0', 'y_0', 'sharpness', 'roundness1',
                                   'roundness2'],
                            data=[[17.4], [16], [0.4], [0], [0]])
        init_guess3 = Table(names=['x_0', 'y_0'],
                            data=[[17.4], [16]])
        init_guess4 = Table(names=['x_0', 'y_0', 'sharpness'],
                            data=[[17.4], [16], [0.4]])
        for i, init_guesses in enumerate([init_guess1, init_guess2,
                                          init_guess3, init_guess4]):
            dao_phot = DAOPhotPSFPhotometry(crit_separation=8, threshold=40,
                                            fwhm=4 * np.sqrt(2 * np.log(2)),
                                            psf_model=psf_model,
                                            fitshape=(11, 11),
                                            extra_output_cols=['sharpness',
                                                               'roundness1',
                                                               'roundness2'])
            phot_results = dao_phot(image, init_guesses=init_guesses)
            # test that the original required columns are also passed
            # back, as well as extra_output_cols
            assert np.all([name in phot_results.colnames for name in
                           ['x_0', 'y_0']])
            assert np.all([name in phot_results.colnames for name in
                           ['sharpness', 'roundness1', 'roundness2']])
            assert len(phot_results) == 2
            # checks to verify that half-passing init_guesses results
            # in NaN output for extra_output_cols not passed as initial
            # guesses
            if i == 2:  # init_guess3
                assert np.all(np.all(np.isnan(phot_results[o])) for o in
                              ['sharpness', 'roundness1', 'roundness2'])
            if i == 3:  # init_guess4
                assert np.all(np.all(np.isnan(phot_results[o])) for o in
                              ['roundness1', 'roundness2'])
                assert np.all(~np.isnan(phot_results['sharpness']))


@pytest.fixture(params=[2, 3])
def overlap_image(request):
    if request.param == 2:
        close_tab = Table([[50.0, 53.0], [50.0, 50.0], [25.0, 25.0]],
                          names=['x_0', 'y_0', 'flux_0'])
    elif request.param == 3:
        close_tab = Table([[50.0, 55.0, 50.0], [50.0, 50.0, 55.0],
                           [25.0, 25.0, 25.0]], names=['x_0', 'y_0', 'flux_0'])
    else:
        raise ValueError

    # Add sources to test image
    close_image = np.zeros((IMAGE_SIZE, IMAGE_SIZE))
    for x, y, flux in close_tab:
        close_model = Gaussian2D(flux / (2 * np.pi * GAUSSIAN_WIDTH ** 2),
                                 x, y, GAUSSIAN_WIDTH, GAUSSIAN_WIDTH)
        close_image += discretize_model(close_model, (0, IMAGE_SIZE),
                                        (0, IMAGE_SIZE), mode='oversample')
    return close_image


@pytest.mark.skipif(not HAS_SCIPY, reason='scipy is required')
def test_psf_fitting_group(overlap_image):
    """
    Test psf_photometry when two input stars are close and need to be
    fit together.
    """
    with pytest.warns(AstropyDeprecationWarning):
        from photutils.background import MADStdBackgroundRMS

        # There are a few models here that fail, be it something created
        # by EPSFBuilder or simpler the Moffat2D one unprepared_psf =
        # Moffat2D(amplitude=1, gamma=2, alpha=2.8, x_0=0, y_0=0) psf =
        # prepare_psf_model(unprepared_psf, xname='x_0', yname='y_0',
        # fluxname=None)
        psf = prepare_psf_model(Gaussian2D(), renormalize_psf=False)

        psf.fwhm = Parameter('fwhm', 'this is not the way to add this I think')
        psf.fwhm.value = 10

        separation_crit = 10

        # choose low threshold and fwhm to find stars no matter what
        basic_phot = BasicPSFPhotometry(finder=DAOStarFinder(1, 1),
                                        group_maker=DAOGroup(separation_crit),
                                        bkg_estimator=MADStdBackgroundRMS(),
                                        fitter=LevMarLSQFitter(),
                                        psf_model=psf,
                                        fitshape=31)
        # this should not raise AttributeError: Attribute "offset_0_0"
        # not found
        basic_phot(image=overlap_image)


@pytest.mark.skipif(not HAS_SCIPY, reason='scipy is required')
def test_finder_return_none():
    """
    Test psf_photometry with finder that does not return None if no
    sources are detected, to test Iterative PSF fitting.
    """

    def tophatfinder(image, mask=None):
        """Simple top hat finder function for use with a top hat PRF."""
        fluxes = np.unique(image[image > 1])
        table = Table(names=['id', 'xcentroid', 'ycentroid', 'flux'],
                      dtype=[int, float, float, float])
        for n, f in enumerate(fluxes):
            ys, xs = np.where(image == f)
            x = np.mean(xs)
            y = np.mean(ys)
            table.add_row([int(n + 1), x, y, f * 9])
        table.sort(['flux'])

        return table

    with pytest.warns(AstropyDeprecationWarning):
        prf = np.zeros((7, 7), float)
        prf[2:5, 2:5] = 1 / 9
        prf = FittableImageModel(prf)

        img = np.zeros((50, 50), float)
        x0 = [38, 20, 35]
        y0 = [20, 5, 40]
        f0 = [50, 100, 200]
        for x, y, f in zip(x0, y0, f0):
            img[y - 1:y + 2, x - 1:x + 2] = f / 9

        intab = Table(data=[[37, 19.6, 34.9], [19.6, 4.5, 40.1],
                            [45, 103, 210]], names=['x_0', 'y_0', 'flux_0'])

        iter_phot = IterativelySubtractedPSFPhotometry(finder=tophatfinder,
                                                       group_maker=DAOGroup(2),
                                                       bkg_estimator=None,
                                                       psf_model=prf,
                                                       fitshape=7, niters=2,
                                                       aperture_radius=3)

        match = 'Both init_guesses and finder are different than None'
        with pytest.warns(AstropyUserWarning, match=match):
            results = iter_phot(image=img, init_guesses=intab)
        assert_allclose(results['flux_fit'], f0, rtol=0.05)


@pytest.mark.skipif(not HAS_SCIPY, reason='scipy is required')
def test_psf_photometry_uncertainties():
    """
    Test an Astropy fitter that does not return a parameter
    covariance matrix (param_cov). The output table should not
    contain flux_unc, x_0_unc, and y_0_unc columns.
    """
    with pytest.warns(AstropyDeprecationWarning):
        psf = IntegratedGaussianPRF(sigma=GAUSSIAN_WIDTH)

        basic_phot = BasicPSFPhotometry(group_maker=DAOGroup(2),
                                        bkg_estimator=None, psf_model=psf,
                                        fitter=SimplexLSQFitter(),
                                        fitshape=7)
        match = 'The fit may be unsuccessful'
        with pytest.warns(AstropyUserWarning, match=match):
            phot_tbl = basic_phot(image=image, init_guesses=INTAB)
        columns = ('flux_unc', 'x_0_unc', 'y_0_unc')
        for column in columns:
            assert column not in phot_tbl.colnames


@pytest.mark.skipif(not HAS_SCIPY, reason='scipy is required')
def test_re_use_result_as_initial_guess():
    img_shape = (32, 32)
    # generate image with read-out noise (Gaussian) and
    # background noise (Poisson)
    image = (make_gaussian_prf_sources_image(img_shape, sources1)
             + make_noise_image(img_shape, distribution='poisson', mean=6.0,
                                seed=0)
             + make_noise_image(img_shape, distribution='gaussian', mean=0.0,
                                stddev=2.0, seed=0))

    _, _, dao_phot_obj = make_psf_photometry_objs()

    match = 'The fit may be unsuccessful'
    ctx1 = pytest.warns(AstropyUserWarning, match=match)
    if PYTEST_LT_80:
        ctx2 = nullcontext()
    else:
        ctx2 = pytest.warns(AstropyDeprecationWarning)
    with ctx1, ctx2:
        result_table = dao_phot_obj(image)
        result_table['x'] = result_table['x_fit']
        result_table['y'] = result_table['y_fit']
        result_table['flux'] = result_table['flux_fit']

    match1 = 'Both init_guesses and finder are different than None'
    ctx1 = pytest.warns(AstropyUserWarning, match=match1)
    if PYTEST_LT_80:
        ctx2 = nullcontext()
        ctx3 = nullcontext()
        ctx4 = nullcontext()
    else:
        match2 = 'init_guesses contains a "group_id" column'
        match3 = 'The fit may be unsuccessful; check fit_info'
        ctx2 = pytest.warns(AstropyUserWarning, match=match2)
        ctx3 = pytest.warns(AstropyUserWarning, match=match3)
        ctx4 = pytest.warns(AstropyDeprecationWarning)
    with ctx1, ctx2, ctx3, ctx4:
        second_result = dao_phot_obj(image, init_guesses=result_table)
        assert second_result


@pytest.mark.skipif(not HAS_SCIPY, reason='scipy is required')
def test_photometry_mask_nan():
    with pytest.warns(AstropyDeprecationWarning):
        size = 64
        sources1 = Table()
        sources1['flux'] = [800]
        sources1['x_0'] = [size / 2]
        sources1['y_0'] = [size / 2]
        sources1['sigma'] = [6]
        sources1['theta'] = [0]

        img_shape = (size, size)
        data = make_gaussian_prf_sources_image(img_shape, sources1)
        data[30, 20:40] = np.nan

        daogroup = DAOGroup(3.0)
        psf_model = IntegratedGaussianPRF(sigma=2.0)
        psfphot = BasicPSFPhotometry(group_maker=daogroup, finder=None,
                                     bkg_estimator=None, psf_model=psf_model,
                                     fitshape=(11, 11))

        init = Table()
        init['x_0'] = [30]
        init['y_0'] = [30]
        init['flux_0'] = [200.0]

        mask = ~np.isfinite(data)
        tbl = psfphot(data, init_guesses=init, mask=mask)
        assert tbl['x_fit'] != init['x_0']
        assert tbl['y_fit'] != init['y_0']
        assert tbl['flux_fit'] != init['flux_0']

        match = 'Input data contains unmasked non-finite values'
        with pytest.warns(AstropyUserWarning, match=match):
            tbl2 = psfphot(data, init_guesses=init)
            assert tbl == tbl2


@pytest.mark.skipif(not HAS_SCIPY, reason='scipy is required')
def test_photometry_subshape():
    with pytest.warns(AstropyDeprecationWarning):
        size = 21
        cen = (size - 1) // 2
        sigma = 2.0
        sources = Table()
        sources['flux'] = [1]
        sources['x_0'] = [cen]
        sources['y_0'] = [cen]
        sources['sigma'] = [sigma]
        psf = make_gaussian_prf_sources_image((size, size), sources)
        psf_model = FittableImageModel(psf)

        sources = Table()
        sources['flux'] = [2000, 1000]
        sources['x_0'] = [18, 7]
        sources['y_0'] = [17, 25]
        sources['sigma'] = [sigma, sigma]
        shape = (33, 33)
        image = make_gaussian_prf_sources_image(shape, sources)

        daogroup = DAOGroup(crit_separation=8)
        mmm_bkg = MMMBackground()
        iraffind = IRAFStarFinder(threshold=10, fwhm=5, roundlo=-1,
                                  minsep_fwhm=1)
        fitter = LevMarLSQFitter()

        pobj1 = BasicPSFPhotometry(finder=iraffind, group_maker=daogroup,
                                   bkg_estimator=mmm_bkg, psf_model=psf_model,
                                   fitter=fitter, fitshape=(7, 7),
                                   aperture_radius=5, subshape=(3, 3))
        pobj2 = BasicPSFPhotometry(finder=iraffind, group_maker=daogroup,
                                   bkg_estimator=mmm_bkg, psf_model=psf_model,
                                   fitter=fitter, fitshape=(7, 7),
                                   aperture_radius=5, subshape=5)
        pobj3 = BasicPSFPhotometry(finder=iraffind, group_maker=daogroup,
                                   bkg_estimator=mmm_bkg, psf_model=psf_model,
                                   fitter=fitter, fitshape=(7, 7),
                                   aperture_radius=5, subshape=None)
        pobj4 = BasicPSFPhotometry(finder=iraffind, group_maker=daogroup,
                                   bkg_estimator=mmm_bkg, psf_model=psf_model,
                                   fitter=fitter, fitshape=(7, 7),
                                   aperture_radius=5, subshape=7)

        _ = pobj1(image)
        _ = pobj2(image)
        _ = pobj3(image)
        _ = pobj4(image)
        resid1 = pobj1.get_residual_image()
        resid2 = pobj2.get_residual_image()
        resid3 = pobj3.get_residual_image()
        resid4 = pobj4.get_residual_image()
        assert np.sum(resid1) > np.sum(resid2)
        assert_allclose(np.sum(resid3), np.sum(resid4))


@pytest.mark.skipif(not HAS_SCIPY, reason='scipy is required')
def test_subshape_invalid():
    basic_phot_obj = make_psf_photometry_objs()[0]

    with pytest.raises(ValueError):
        basic_phot_obj.subshape = (2, 2)
    with pytest.raises(ValueError):
        basic_phot_obj.subshape = 2
    with pytest.raises(ValueError):
        basic_phot_obj.subshape = (-1, 0)
    with pytest.raises(ValueError):
        basic_phot_obj.subshape = (3, 3, 3)


@pytest.mark.filterwarnings('ignore:Both init_guesses and finder are '
                            'different than None')
@pytest.mark.filterwarnings('ignore:No sources were found')
@pytest.mark.skipif(not HAS_SCIPY, reason='scipy is required')
@pytest.mark.skipif(not minversion(astropy, '5.3'),
                    reason='astropy 5.3 is required')
@pytest.mark.parametrize('sigma_psf, sources',
                         [(sigma_psfs[0], sources1),
                          (sigma_psfs[1], sources2)])
def test_psf_photometry_oneiter_uncert(sigma_psf, sources):
    """
    Make an image with a group of two overlapped stars and an
    isolated one, and check that the best-fit fluxes have smaller
    uncertainties when the measured fluxes have smaller uncertainties.
    """
    img_shape = (32, 32)
    # generate image with read-out noise (Gaussian) and
    # background noise (Poisson)
    image = (make_gaussian_prf_sources_image(img_shape, sources)
             + make_noise_image(img_shape, distribution='poisson', mean=6.0,
                                seed=0)
             + make_noise_image(img_shape, distribution='gaussian', mean=0.0,
                                stddev=2.0, seed=0))

    sigma_clip = SigmaClip(sigma=3.0)
    bkgrms = StdBackgroundRMS(sigma_clip)
    std = bkgrms(image)
    phot_objs = make_psf_photometry_objs(std, sigma_psf)

    flux_uncertainties_0 = []
    flux_uncertainties_1 = []

    for uncertainty_scale_factor, flux_uncert in zip(
        [1e-5, 0.1], [flux_uncertainties_0, flux_uncertainties_1]
    ):
        for phot_proc in phot_objs:
            uncertainty = (
                uncertainty_scale_factor * np.std(image) * np.ones_like(image)
            )
            with pytest.warns(AstropyDeprecationWarning):
                result_tab = phot_proc(image, uncertainty=uncertainty)
            flux_uncert.append(np.array(result_tab['flux_unc']))

    for uncert_0, uncert_1 in zip(flux_uncertainties_0, flux_uncertainties_1):
        assert np.all(uncert_0 < uncert_1)
