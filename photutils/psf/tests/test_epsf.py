# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the epsf module.
"""

from astropy.modeling.fitting import LevMarLSQFitter
from astropy.nddata import NDData
from astropy.table import Table
import numpy as np
from numpy.testing import assert_allclose, assert_almost_equal
import pytest

from ..epsf import EPSFBuilder, EPSFFitter
from ..epsf_stars import extract_stars, EPSFStar, EPSFStars
from ..models import IntegratedGaussianPRF, EPSFModel
from ...datasets import make_gaussian_prf_sources_image, apply_poisson_noise
from ...centroids import centroid_com

try:
    import scipy  # noqa
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


@pytest.mark.skipif('not HAS_SCIPY')
class TestEPSFBuild:
    def setup_class(self):
        """
        Create a simulated image for testing.
        """

        from scipy.spatial import cKDTree

        shape = (750, 750)

        # define random star positions
        nstars = 100
        from astropy.utils.misc import NumpyRNGContext
        with NumpyRNGContext(12345):  # seed for repeatability
            xx = np.random.uniform(low=0, high=shape[1], size=nstars)
            yy = np.random.uniform(low=0, high=shape[0], size=nstars)

        # enforce a minimum separation
        min_dist = 25
        coords = [(yy[0], xx[0])]
        for xxi, yyi in zip(xx, yy):
            newcoord = [yyi, xxi]
            dist, _ = cKDTree([newcoord]).query(coords, 1)
            if np.min(dist) > min_dist:
                coords.append(newcoord)
        yy, xx = np.transpose(coords)

        with NumpyRNGContext(12345):  # seed for repeatability
            zz = np.random.uniform(low=0, high=200000., size=len(xx))

        # define a table of model parameters
        self.stddev = 2.
        sources = Table()
        sources['amplitude'] = zz
        sources['x_0'] = xx
        sources['y_0'] = yy
        sources['sigma'] = np.zeros(len(xx)) + self.stddev
        sources['theta'] = 0.

        self.data = make_gaussian_prf_sources_image(shape, sources)
        self.nddata = NDData(self.data)

        init_stars = Table()
        init_stars['x'] = xx.astype(int)
        init_stars['y'] = yy.astype(int)
        self.init_stars = init_stars

    def test_extract_stars(self):
        size = 25
        stars = extract_stars(self.nddata, self.init_stars, size=size)

        assert len(stars) == 79
        assert isinstance(stars, EPSFStars)
        assert isinstance(stars[0], EPSFStar)
        assert stars[0].data.shape == (size, size)

    def test_epsf_build(self):
        """
        This is an end-to-end test of EPSFBuilder on a simulated image.
        """

        size = 25
        oversampling = 4.
        stars = extract_stars(self.nddata, self.init_stars, size=size)
        epsf_builder = EPSFBuilder(oversampling=oversampling, maxiters=8,
                                   progress_bar=False, norm_radius=25,
                                   recentering_maxiters=5)
        epsf, fitted_stars = epsf_builder(stars)

        ref_size = (size * oversampling) + 1
        assert epsf.data.shape == (ref_size, ref_size)

        y0 = (ref_size - 1) / 2 / oversampling
        y = np.arange(ref_size, dtype=float) / oversampling

        psf_model = IntegratedGaussianPRF(sigma=self.stddev)
        z = epsf.data
        x = psf_model.evaluate(y.reshape(-1, 1), y.reshape(1, -1), 1, y0, y0, self.stddev)
        assert_allclose(z, x, rtol=1e-2, atol=1e-5)

        resid_star = fitted_stars[0].compute_residual_image(epsf)
        assert_almost_equal(np.sum(resid_star)/fitted_stars[0].flux, 0, decimal=3)

    def test_epsf_fitting_bounds(self):
        size = 25
        oversampling = 4.
        stars = extract_stars(self.nddata, self.init_stars, size=size)
        epsf_builder = EPSFBuilder(oversampling=oversampling, maxiters=8,
                                   progress_bar=True, norm_radius=25,
                                   recentering_maxiters=5,
                                   fitter=EPSFFitter(fit_boxsize=30),
                                   smoothing_kernel='quadratic')
        # With a boxsize larger than the cutout we expect the fitting to
        # fail for all stars, due to star._fit_error_status
        with pytest.raises(ValueError):
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
    with pytest.raises(ValueError):
        EPSFBuilder(oversampling=None)
    with pytest.raises(ValueError):
        EPSFBuilder(oversampling=-1)
    with pytest.raises(ValueError):
        EPSFBuilder(maxiters=-1)
    with pytest.raises(ValueError):
        EPSFBuilder(oversampling=3)
    with pytest.raises(ValueError):
        EPSFBuilder(oversampling=[3, 6])
    with pytest.raises(ValueError):
        EPSFBuilder(oversampling=[-1, 4])


def test_epsfmodel_inputs():
    data = np.array([[], []])
    with pytest.raises(ValueError):
        EPSFModel(data)

    data = np.ones((5, 5), dtype=float)
    data[2, 2] = np.inf
    with pytest.raises(ValueError):
        EPSFModel(data)

    data[2, 2] = np.finfo(np.float64).max * 2
    with pytest.raises(ValueError):
        EPSFModel(data, flux=None)

    data[2, 2] = 1
    for oversampling in [3, np.NaN, 'a', -1, [3, 4], [-2, 4]]:
        with pytest.raises(ValueError):
            EPSFModel(data, oversampling=oversampling)

    for origin in ['a', (1, 2, 3)]:
        with pytest.raises(TypeError):
            EPSFModel(data, origin=origin)


@pytest.mark.skipif('not HAS_SCIPY')
def test_epsf_build_with_noise():
    oversampling = 4
    size = 25
    sigma = 0.5

    # should be "truth" ePSF
    m = IntegratedGaussianPRF(sigma=sigma, x_0=12.5, y_0=12.5, flux=1)
    yy, xx = np.mgrid[0:size*oversampling+1,
                      0:size*oversampling+1]
    xx = xx / oversampling
    yy = yy / oversampling
    truth_epsf = m(xx, yy)

    Nstars = 16  # one star per oversampling=4 point, roughly
    xdim = np.ceil(np.sqrt(Nstars)).astype(int)
    ydim = np.ceil(Nstars / xdim).astype(int)
    xarray = np.arange((size-1)/2+2, (size-1)/2+2 + xdim*size, size)
    yarray = np.arange((size-1)/2+2, (size-1)/2+2 + ydim*size, size)
    xarray, yarray = np.meshgrid(xarray, yarray)
    xarray, yarray = xarray.ravel(), yarray.ravel()

    np.random.seed(seed=76312)
    xpos = np.random.uniform(-0.5, 0.5, Nstars)
    ypos = np.random.uniform(-0.5, 0.5, Nstars)
    amps = np.random.uniform(50, 1000, Nstars)

    sources = Table()
    sources['amplitude'] = amps
    sources['x_0'] = xarray[:Nstars] + xpos
    sources['y_0'] = yarray[:Nstars] + ypos
    sources['sigma'] = [sigma]*Nstars

    stars_tbl = Table()
    stars_tbl['x'] = sources['x_0']
    stars_tbl['y'] = sources['y_0']

    data = make_gaussian_prf_sources_image((size*ydim+4,
                                            size*xdim+4), sources)

    data += 20  # counts/s
    data *= 100  # seconds
    data = apply_poisson_noise(data).astype(float)
    data /= 100
    data -= 20
    nddata = NDData(data=data)

    stars = extract_stars(nddata, stars_tbl, size=size)

    for star in stars:
        star.cutout_center = centroid_com(star.data)

    epsf_builder = EPSFBuilder(oversampling=oversampling, maxiters=5,
                               progress_bar=False, norm_radius=7.5,
                               recentering_func=centroid_com,
                               shift_val=0.5)
    epsf, fitted_stars = epsf_builder(stars)
    assert_allclose(epsf.data, truth_epsf, rtol=1e-1, atol=5e-2)


@pytest.mark.skipif('not HAS_SCIPY')
def test_epsf_offset():
    size = 25
    sigma = 0.5

    Nstars = 40
    xdim = np.ceil(np.sqrt(Nstars)).astype(int)
    ydim = np.ceil(Nstars / xdim).astype(int)
    xarray = np.arange((size-1)/2+2, (size-1)/2+2 + xdim*size, size)
    yarray = np.arange((size-1)/2+2, (size-1)/2+2 + ydim*size, size)
    xarray, yarray = np.meshgrid(xarray, yarray)
    xarray, yarray = xarray.ravel(), yarray.ravel()

    np.random.seed(seed=95758)
    xpos = np.random.uniform(-0.5, 0.5, Nstars)
    ypos = np.random.uniform(-0.5, 0.5, Nstars)
    amps = np.random.uniform(50, 1000, Nstars)

    sources = Table()
    sources['amplitude'] = amps
    sources['x_0'] = xarray[:Nstars] + xpos
    sources['y_0'] = yarray[:Nstars] + ypos
    sources['sigma'] = [sigma]*Nstars

    stars_tbl = Table()
    stars_tbl['x'] = sources['x_0']
    stars_tbl['y'] = sources['y_0']

    data = make_gaussian_prf_sources_image((size*ydim+4,
                                            size*xdim+4), sources)

    data += 20  # counts/s
    data *= 100  # seconds
    data = apply_poisson_noise(data).astype(float)
    data /= 100
    data -= 20
    nddata = NDData(data=data)

    for oversampling, offset in zip([4, 1, 3, 3], [None, None, None, 1/6]):
        # should be "truth" ePSF
        m = IntegratedGaussianPRF(sigma=sigma, x_0=12.5, y_0=12.5, flux=1)
        extra_pixel = 1 if offset is not None else 0
        yy, xx = np.mgrid[0:size*oversampling + extra_pixel,
                          0:size*oversampling + extra_pixel]
        if offset is None:
            if oversampling == 1:
                _offset = 0.5
            else:
                _offset = 0
        else:
            _offset = offset
        xx = xx / oversampling + _offset
        yy = yy / oversampling + _offset
        truth_epsf = m(xx, yy)

        stars = extract_stars(nddata, stars_tbl, size=size)

        for star in stars:
            star.cutout_center = centroid_com(star.data)

        epsf_builder = EPSFBuilder(oversampling=oversampling, maxiters=5,
                                   progress_bar=False, norm_radius=7.5,
                                   recentering_func=centroid_com,
                                   grid_offset=offset)
        epsf, fitted_stars = epsf_builder(stars)
        # Test built ePSF via EPSFBuilder
        assert np.all(epsf.data.shape == truth_epsf.shape)
        assert_allclose(epsf.data, truth_epsf, rtol=1e-1, atol=5e-2)

        # Test ePSF via re-creation through EPSFModel independently
        new_epsf = EPSFModel(epsf.data, oversampling=oversampling,
                             grid_offset=offset)
        epsf_data = new_epsf.evaluate(xx, yy, 1, 0, 0)
        assert np.all(epsf_data.shape == truth_epsf.shape)
        assert_allclose(epsf_data, truth_epsf, rtol=1e-1, atol=5e-2)
