# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the epsf module.
"""

import itertools

import numpy as np
import pytest
from astropy.modeling.fitting import TRFLSQFitter
from astropy.nddata import (InverseVariance, NDData, StdDevUncertainty,
                            VarianceUncertainty)
from astropy.stats import SigmaClip
from astropy.table import Table
from astropy.utils.exceptions import AstropyUserWarning
from numpy.testing import assert_allclose, assert_almost_equal
from scipy.spatial import cKDTree

from photutils.datasets import make_model_image
from photutils.psf.epsf import EPSFBuilder, EPSFFitter
from photutils.psf.epsf_stars import EPSFStars, extract_stars
from photutils.psf.functional_models import CircularGaussianPRF


class TestEPSFBuild:
    def setup_class(self):
        """
        Create a simulated image for testing.
        """
        shape = (750, 750)

        # define random star positions
        nstars = 100

        rng = np.random.default_rng(0)
        xx = rng.uniform(low=0, high=shape[1], size=nstars)
        yy = rng.uniform(low=0, high=shape[0], size=nstars)

        # enforce a minimum separation
        min_dist = 25
        coords = [(yy[0], xx[0])]
        for xxi, yyi in zip(xx, yy, strict=True):
            newcoord = [yyi, xxi]
            dist, _ = cKDTree([newcoord]).query(coords, 1)
            if np.min(dist) > min_dist:
                coords.append(newcoord)
        yy, xx = np.transpose(coords)
        zz = rng.uniform(low=0, high=200000, size=len(xx))

        # define a table of model parameters
        self.fwhm = 4.7
        sources = Table()
        sources['amplitude'] = zz
        sources['x_0'] = xx
        sources['y_0'] = yy
        sources['fwhm'] = np.zeros(len(xx)) + self.fwhm
        sources['theta'] = 0.0

        psf_model = CircularGaussianPRF(fwhm=self.fwhm)
        self.data = make_model_image(shape, psf_model, sources)
        self.nddata = NDData(self.data)

        init_stars = Table()
        init_stars['x'] = xx.astype(int)
        init_stars['y'] = yy.astype(int)
        self.init_stars = init_stars

    def test_extract_stars(self):
        size = 25
        match = 'were not extracted because their cutout region extended'
        with pytest.warns(AstropyUserWarning, match=match):
            stars = extract_stars(self.nddata, self.init_stars, size=size)

        assert len(stars) == 81
        assert isinstance(stars, EPSFStars)
        assert isinstance(stars[0], EPSFStars)
        assert stars[0].data.shape == (size, size)

    def test_extract_stars_uncertainties(self):
        rng = np.random.default_rng(0)
        shape = self.nddata.data.shape
        error = np.abs(rng.normal(loc=0, scale=1, size=shape))
        uncertainty1 = StdDevUncertainty(error)
        uncertainty2 = uncertainty1.represent_as(VarianceUncertainty)
        uncertainty3 = uncertainty1.represent_as(InverseVariance)
        ndd1 = NDData(self.nddata.data, uncertainty=uncertainty1)
        ndd2 = NDData(self.nddata.data, uncertainty=uncertainty2)
        ndd3 = NDData(self.nddata.data, uncertainty=uncertainty3)

        size = 25
        match = 'were not extracted because their cutout region extended'
        with pytest.warns(AstropyUserWarning, match=match):
            ndd_inputs = (ndd1, ndd2, ndd3)

            outputs = [extract_stars(ndd_input, self.init_stars, size=size)
                       for ndd_input in ndd_inputs]

            for stars in outputs:
                assert len(stars) == 81
                assert isinstance(stars, EPSFStars)
                assert isinstance(stars[0], EPSFStars)
                assert stars[0].data.shape == (size, size)
                assert stars[0].weights.shape == (size, size)

        assert_allclose(outputs[0].weights, outputs[1].weights)
        assert_allclose(outputs[0].weights, outputs[2].weights)

        match = 'One or more weight values is not finite'
        with pytest.warns(AstropyUserWarning, match=match):
            uncertainty = StdDevUncertainty(np.zeros(shape))
            ndd = NDData(self.nddata.data, uncertainty=uncertainty)
            stars = extract_stars(ndd, self.init_stars[0:3], size=size)

    def test_epsf_build(self):
        """
        This is an end-to-end test of EPSFBuilder on a simulated image.
        """
        size = 25
        oversampling = 4
        match = 'were not extracted because their cutout region extended'
        with pytest.warns(AstropyUserWarning, match=match):
            stars = extract_stars(self.nddata, self.init_stars, size=size)
        epsf_builder = EPSFBuilder(oversampling=oversampling, maxiters=15,
                                   progress_bar=False, norm_radius=25,
                                   recentering_maxiters=15)
        epsf, fitted_stars = epsf_builder(stars)

        ref_size = (size * oversampling) + 1
        assert epsf.data.shape == (ref_size, ref_size)

        y0 = (ref_size - 1) / 2 / oversampling
        y = np.arange(ref_size, dtype=float) / oversampling

        psf_model = CircularGaussianPRF(fwhm=self.fwhm)
        z = epsf.data
        x = psf_model.evaluate(y.reshape(-1, 1), y.reshape(1, -1), 1, y0, y0,
                               self.fwhm)
        assert_allclose(z, x, rtol=1e-2, atol=1e-5)

        resid_star = fitted_stars[0].compute_residual_image(epsf)
        assert_almost_equal(np.sum(resid_star) / fitted_stars[0].flux, 0,
                            decimal=3)

    def test_epsf_fitting_bounds(self):
        size = 25
        oversampling = 4
        match = 'were not extracted because their cutout region extended'
        with pytest.warns(AstropyUserWarning, match=match):
            stars = extract_stars(self.nddata, self.init_stars, size=size)
        epsf_builder = EPSFBuilder(oversampling=oversampling, maxiters=8,
                                   progress_bar=True, norm_radius=25,
                                   recentering_maxiters=5,
                                   fitter=EPSFFitter(fit_boxsize=31),
                                   smoothing_kernel='quadratic')
        # With a boxsize larger than the cutout we expect the fitting to
        # fail for all stars, due to star._fit_error_status
        match1 = 'The ePSF fitting failed for all stars'
        match2 = r'The star at .* cannot be fit because its fitting region '
        with (pytest.raises(ValueError, match=match1),
                pytest.warns(AstropyUserWarning, match=match2)):
            epsf_builder(stars)

    def test_epsf_build_invalid_fitter(self):
        """
        Test that the input fitter is an EPSFFitter instance.
        """
        match = 'fitter must be an EPSFFitter instance'
        with pytest.raises(TypeError, match=match):
            EPSFBuilder(fitter=EPSFFitter, maxiters=3)

        with pytest.raises(TypeError, match=match):
            EPSFBuilder(fitter=TRFLSQFitter(), maxiters=3)

        with pytest.raises(TypeError, match=match):
            EPSFBuilder(fitter=TRFLSQFitter, maxiters=3)


def test_epsfbuilder_inputs():
    # invalid inputs
    match = "'oversampling' must be specified"
    with pytest.raises(ValueError, match=match):
        EPSFBuilder(oversampling=None)
    match = 'oversampling must be > 0'
    with pytest.raises(ValueError, match=match):
        EPSFBuilder(oversampling=-1)
    match = "'maxiters' must be a positive number"
    with pytest.raises(ValueError, match=match):
        EPSFBuilder(maxiters=-1)
    match = 'oversampling must be > 0'
    with pytest.raises(ValueError, match=match):
        EPSFBuilder(oversampling=[-1, 4])

    # valid inputs
    EPSFBuilder(oversampling=6)
    EPSFBuilder(oversampling=[4, 6])

    # invalid inputs
    for sigma_clip in [None, [], 'a']:
        match = 'sigma_clip must be an astropy.stats.SigmaClip instance'
        with pytest.raises(TypeError, match=match):
            EPSFBuilder(sigma_clip=sigma_clip)

    # valid inputs
    EPSFBuilder(sigma_clip=SigmaClip(sigma=2.5, cenfunc='mean', maxiters=2))


@pytest.mark.parametrize('oversamp', [3, 4])
def test_epsf_build_oversampling(oversamp):
    offsets = (np.arange(oversamp) * 1.0 / oversamp - 0.5 + 1.0
               / (2.0 * oversamp))
    xydithers = np.array(list(itertools.product(offsets, offsets)))
    xdithers = np.transpose(xydithers)[0]
    ydithers = np.transpose(xydithers)[1]

    nstars = oversamp**2
    fwhm = 7.0
    sources = Table()
    offset = 50
    size = oversamp * offset + offset
    y, x = np.mgrid[0:oversamp, 0:oversamp] * offset + offset
    sources['amplitude'] = np.full((nstars,), 100.0)
    sources['x_0'] = x.ravel() + xdithers
    sources['y_0'] = y.ravel() + ydithers
    sources['fwhm'] = np.full((nstars,), fwhm)

    psf_model = CircularGaussianPRF(fwhm=fwhm)
    shape = (size, size)
    data = make_model_image(shape, psf_model, sources)
    nddata = NDData(data=data)
    stars_tbl = Table()
    stars_tbl['x'] = sources['x_0']
    stars_tbl['y'] = sources['y_0']
    stars = extract_stars(nddata, stars_tbl, size=25)
    epsf_builder = EPSFBuilder(oversampling=oversamp, maxiters=15,
                               progress_bar=False, recentering_maxiters=20)
    epsf, _ = epsf_builder(stars)

    # input PSF shape
    size = epsf.data.shape[0]
    cen = (size - 1) / 2
    fwhm2 = oversamp * fwhm
    m = CircularGaussianPRF(flux=1, x_0=cen, y_0=cen, fwhm=fwhm2)
    yy, xx = np.mgrid[0:size, 0:size]
    psf = m(xx, yy)

    assert_allclose(epsf.data, psf * epsf.data.sum(), atol=2.5e-4)
