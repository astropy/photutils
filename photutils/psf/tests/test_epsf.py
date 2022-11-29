# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the epsf module.
"""

import itertools

import numpy as np
import pytest
from astropy.modeling.fitting import LevMarLSQFitter
from astropy.nddata import NDData
from astropy.stats import SigmaClip
from astropy.table import Table
from astropy.utils.exceptions import AstropyUserWarning
from numpy.testing import assert_allclose, assert_almost_equal

from photutils.datasets import make_gaussian_prf_sources_image
from photutils.psf.epsf import EPSFBuilder, EPSFFitter
from photutils.psf.epsf_stars import EPSFStars, extract_stars
from photutils.psf.models import EPSFModel, IntegratedGaussianPRF
from photutils.utils._optional_deps import HAS_SCIPY


@pytest.mark.skipif(not HAS_SCIPY, reason='scipy is required')
class TestEPSFBuild:
    def setup_class(self):
        """
        Create a simulated image for testing.
        """
        from scipy.spatial import cKDTree

        shape = (750, 750)

        # define random star positions
        nstars = 100

        rng = np.random.default_rng(0)
        xx = rng.uniform(low=0, high=shape[1], size=nstars)
        yy = rng.uniform(low=0, high=shape[0], size=nstars)

        # enforce a minimum separation
        min_dist = 25
        coords = [(yy[0], xx[0])]
        for xxi, yyi in zip(xx, yy):
            newcoord = [yyi, xxi]
            dist, _ = cKDTree([newcoord]).query(coords, 1)
            if np.min(dist) > min_dist:
                coords.append(newcoord)
        yy, xx = np.transpose(coords)
        zz = rng.uniform(low=0, high=200000, size=len(xx))

        # define a table of model parameters
        self.stddev = 2.0
        sources = Table()
        sources['amplitude'] = zz
        sources['x_0'] = xx
        sources['y_0'] = yy
        sources['sigma'] = np.zeros(len(xx)) + self.stddev
        sources['theta'] = 0.0

        self.data = make_gaussian_prf_sources_image(shape, sources)
        self.nddata = NDData(self.data)

        init_stars = Table()
        init_stars['x'] = xx.astype(int)
        init_stars['y'] = yy.astype(int)
        self.init_stars = init_stars

    def test_extract_stars(self):
        size = 25
        with pytest.warns(AstropyUserWarning, match='were not extracted'):
            stars = extract_stars(self.nddata, self.init_stars, size=size)

        assert len(stars) == 81
        assert isinstance(stars, EPSFStars)
        assert isinstance(stars[0], EPSFStars)
        assert stars[0].data.shape == (size, size)

    def test_epsf_build(self):
        """
        This is an end-to-end test of EPSFBuilder on a simulated image.
        """
        size = 25
        oversampling = 4
        with pytest.warns(AstropyUserWarning, match='were not extracted'):
            stars = extract_stars(self.nddata, self.init_stars, size=size)
        epsf_builder = EPSFBuilder(oversampling=oversampling, maxiters=15,
                                   progress_bar=False, norm_radius=25,
                                   recentering_maxiters=15)
        epsf, fitted_stars = epsf_builder(stars)

        ref_size = (size * oversampling) + 1
        assert epsf.data.shape == (ref_size, ref_size)

        y0 = (ref_size - 1) / 2 / oversampling
        y = np.arange(ref_size, dtype=float) / oversampling

        psf_model = IntegratedGaussianPRF(sigma=self.stddev)
        z = epsf.data
        x = psf_model.evaluate(y.reshape(-1, 1), y.reshape(1, -1), 1, y0, y0,
                               self.stddev)
        assert_allclose(z, x, rtol=1e-2, atol=1e-5)

        resid_star = fitted_stars[0].compute_residual_image(epsf)
        assert_almost_equal(np.sum(resid_star) / fitted_stars[0].flux, 0,
                            decimal=3)

    def test_epsf_fitting_bounds(self):
        size = 25
        oversampling = 4
        with pytest.warns(AstropyUserWarning, match='were not extracted'):
            stars = extract_stars(self.nddata, self.init_stars, size=size)
        epsf_builder = EPSFBuilder(oversampling=oversampling, maxiters=8,
                                   progress_bar=True, norm_radius=25,
                                   recentering_maxiters=5,
                                   fitter=EPSFFitter(fit_boxsize=31),
                                   smoothing_kernel='quadratic')
        # With a boxsize larger than the cutout we expect the fitting to
        # fail for all stars, due to star._fit_error_status
        with pytest.raises(ValueError), pytest.warns(AstropyUserWarning):
            epsf_builder(stars)

    def test_epsf_build_invalid_fitter(self):
        """
        Test that the input fitter is an EPSFFitter instance.
        """
        with pytest.raises(TypeError):
            EPSFBuilder(fitter=EPSFFitter, maxiters=3)

        with pytest.raises(TypeError):
            EPSFBuilder(fitter=LevMarLSQFitter(), maxiters=3)

        with pytest.raises(TypeError):
            EPSFBuilder(fitter=LevMarLSQFitter, maxiters=3)


def test_epsfbuilder_inputs():
    # invalid inputs
    with pytest.raises(ValueError):
        EPSFBuilder(oversampling=None)
    with pytest.raises(ValueError):
        EPSFBuilder(oversampling=-1)
    with pytest.raises(ValueError):
        EPSFBuilder(maxiters=-1)
    with pytest.raises(ValueError):
        EPSFBuilder(oversampling=[-1, 4])

    # valid inputs
    EPSFBuilder(oversampling=6)
    EPSFBuilder(oversampling=[4, 6])

    # invalid inputs
    for sigma_clip in [None, [], 'a']:
        with pytest.raises(ValueError):
            EPSFBuilder(sigma_clip=sigma_clip)

    # valid inputs
    EPSFBuilder(sigma_clip=SigmaClip(sigma=2.5, cenfunc='mean', maxiters=2))


def test_epsfmodel_inputs():
    data = np.array([[], []])
    with pytest.raises(ValueError):
        EPSFModel(data)

    data = np.ones((5, 5), dtype=float)
    data[2, 2] = np.inf
    with pytest.raises(ValueError, match='must be finite'):
        EPSFModel(data)

    data[2, 2] = np.nan
    with pytest.raises(ValueError, match='must be finite'):
        EPSFModel(data, flux=None)

    data[2, 2] = 1
    for oversampling in [-1, [-2, 4], (1, 4, 8), ((1, 2), (3, 4)),
                         np.ones((2, 2, 2)), 2.1, np.nan, (1, np.inf)]:
        with pytest.raises(ValueError):
            EPSFModel(data, oversampling=oversampling)

    origin = (1, 2, 3)
    with pytest.raises(TypeError):
        EPSFModel(data, origin=origin)


@pytest.mark.skipif(not HAS_SCIPY, reason='scipy is required')
@pytest.mark.parametrize('oversamp', [3, 4])
def test_epsf_build_oversampling(oversamp):
    offsets = (np.arange(oversamp) * 1.0 / oversamp - 0.5 + 1.0
               / (2.0 * oversamp))
    xydithers = np.array(list(itertools.product(offsets, offsets)))
    xdithers = np.transpose(xydithers)[0]
    ydithers = np.transpose(xydithers)[1]

    nstars = oversamp**2
    sigma = 3.0
    sources = Table()
    offset = 50
    size = oversamp * offset + offset
    y, x = np.mgrid[0:oversamp, 0:oversamp] * offset + offset
    sources['amplitude'] = np.full((nstars,), 100.0)
    sources['x_0'] = x.ravel() + xdithers
    sources['y_0'] = y.ravel() + ydithers
    sources['sigma'] = np.full((nstars,), sigma)

    data = make_gaussian_prf_sources_image((size, size), sources)
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
    sigma2 = oversamp * sigma
    m = IntegratedGaussianPRF(sigma2, x_0=cen, y_0=cen, flux=1)
    yy, xx = np.mgrid[0:size, 0:size]
    psf = m(xx, yy)

    assert_allclose(epsf.data, psf * epsf.data.sum(), atol=2.5e-4)
