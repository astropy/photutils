# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import division
import numpy as np
from astropy.table import Table
from astropy.stats import gaussian_sigma_to_fwhm
from astropy.modeling.fitting import LevMarLSQFitter
from numpy.testing import assert_allclose, assert_array_equal
from ..models import IntegratedGaussianPRF
from ...datasets import make_gaussian_sources
from ...datasets import make_noise_image
from ..groupstars import DAOGroup
from ..psfphotometry import StetsonPSFPhotometry
from ...detection import DAOStarFinder
from ...background import MedianBackground
from ...background import StdBackgroundRMS


class TestStetsonPSFPhotometry(object):
    def test_complete_photometry_one(self):
        """
        Tests the whole photometry process.
        """
        sigma_psf = 2.0
        sources = Table()
        sources['flux'] = [700, 700]
        sources['x_mean'] = [12, 18]
        sources['y_mean'] = [15, 15]
        sources['x_stddev'] = [sigma_psf, sigma_psf]
        sources['y_stddev'] = sources['x_stddev']
        sources['theta'] = [0, 0]
        tshape = (32, 32)

        image = (make_gaussian_sources(tshape, sources) +
                 make_noise_image(tshape, type='poisson', mean=4.,
                                  random_state=1)) 

        bkgrms = StdBackgroundRMS(sigma=3.)
        std = bkgrms(image)

        daofind = DAOStarFinder(threshold=5.0*std,
                                fwhm=sigma_psf*gaussian_sigma_to_fwhm)
        daogroup = DAOGroup(1.5*sigma_psf*gaussian_sigma_to_fwhm)
        median_bkg = MedianBackground(sigma=3.)
        psf_model = IntegratedGaussianPRF(sigma=sigma_psf)
        fitter = LevMarLSQFitter()
        nstar_photometry = StetsonPSFPhotometry(find=daofind, group=daogroup,
                                                bkg=median_bkg, psf=psf_model,
                                                fitter=LevMarLSQFitter(),
                                                niters=1, fitshape=(5,5))
        
        result_tab, residual_image = nstar_photometry(image)

        assert_allclose(result_tab['x_fit'], sources['x_mean'], rtol=1e-2)
        assert_allclose(result_tab['y_fit'], sources['y_mean'], rtol=1e-2)
        assert_allclose(result_tab['flux_fit'], sources['flux'], rtol=1e-1)
        assert_array_equal(result_tab['id'], np.arange(2) + 1)
        assert_array_equal(result_tab['group_id'], np.ones(2, dtype=np.int))
        assert_allclose(np.mean(residual_image), 0.0, atol=1e-1)

    def test_complete_photometry_two(self):
        sigma_psf = 2.0
        sources = Table()
        sources['flux'] = [700, 700, 700, 700]
        sources['x_mean'] = [12, 18, 12, 18]
        sources['y_mean'] = [15, 15, 21, 21]
        sources['x_stddev'] = sigma_psf*np.ones(4)
        sources['y_stddev'] = sources['x_stddev']
        sources['theta'] = [0, 0, 0, 0]
        tshape = (32, 32)

        image = (make_gaussian_sources(tshape, sources) +
                 make_noise_image(tshape, type='poisson', mean=2.,
                                  random_state=1))
        
        bkgrms = StdBackgroundRMS(sigma=3.)
        std = bkgrms(image)

        daofind = DAOStarFinder(threshold=5.0*std,
                                fwhm=sigma_psf*gaussian_sigma_to_fwhm)
        daogroup = DAOGroup(1.5*sigma_psf*gaussian_sigma_to_fwhm)
        median_bkg = MedianBackground(sigma=3.)
        psf_model = IntegratedGaussianPRF(sigma=sigma_psf)
        fitter = LevMarLSQFitter()
        nstar_photometry = StetsonPSFPhotometry(find=daofind, group=daogroup,
                                                bkg=median_bkg, psf=psf_model,
                                                fitter=LevMarLSQFitter(),
                                                niters=1, fitshape=(5,5))
        
        result_tab, residual_image = nstar_photometry(image)

        assert_allclose(result_tab['x_fit'], sources['x_mean'], rtol=1e-2)
        assert_allclose(result_tab['y_fit'], sources['y_mean'], rtol=1e-2)
        assert_allclose(result_tab['flux_fit'], sources['flux'], rtol=1e-1)
        assert_array_equal(result_tab['id'], np.arange(4) + 1)
        assert_array_equal(result_tab['group_id'], np.ones(4, dtype=np.int))
        assert_allclose(np.mean(residual_image), 0.0, atol=1e-1)
