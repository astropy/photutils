# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the models module.
"""

from itertools import product

from astropy.modeling.models import Gaussian2D, Moffat2D
from astropy.nddata import NDData
import numpy as np
from numpy.testing import assert_allclose
import pytest

from ..models import (FittableImageModel, GriddedPSFModel,
                      IntegratedGaussianPRF, PRFAdapter)
from ...segmentation import detect_sources, SourceCatalog
from ...utils._optional_deps import HAS_SCIPY  # noqa


@pytest.mark.skipif('not HAS_SCIPY')
class TestFittableImageModel:
    def setup_class(self):
        self.gm = Gaussian2D(x_stddev=3, y_stddev=3)

    def test_fittable_image_model(self):
        yy, xx = np.mgrid[-2:3, -2:3]
        model_nonorm = FittableImageModel(self.gm(xx, yy))

        assert_allclose(model_nonorm(0, 0), self.gm(0, 0))
        assert_allclose(model_nonorm(1, 1), self.gm(1, 1))
        assert_allclose(model_nonorm(-2, 1), self.gm(-2, 1))

        # subpixel should *not* match, but be reasonably close
        # in this case good to ~0.1% seems to be fine
        assert_allclose(model_nonorm(0.5, 0.5), self.gm(0.5, 0.5), rtol=.001)
        assert_allclose(model_nonorm(-0.5, 1.75), self.gm(-0.5, 1.75),
                        rtol=.001)

        model_norm = FittableImageModel(self.gm(xx, yy), normalize=True)
        assert not np.allclose(model_norm(0, 0), self.gm(0, 0))
        assert_allclose(np.sum(model_norm(xx, yy)), 1)

        model_norm2 = FittableImageModel(self.gm(xx, yy), normalize=True,
                                         normalization_correction=2)
        assert not np.allclose(model_norm2(0, 0), self.gm(0, 0))
        assert_allclose(model_norm(0, 0), model_norm2(0, 0)*2)
        assert_allclose(np.sum(model_norm2(xx, yy)), 0.5)

    def test_fittable_image_model_oversampling(self):
        oversamp = 3  # oversampling factor
        yy, xx = np.mgrid[-3:3.00001:(1/oversamp), -3:3.00001:(1/oversamp)]

        im = self.gm(xx, yy)
        assert im.shape[0] > 7

        model_oversampled = FittableImageModel(im, oversampling=oversamp)
        assert_allclose(model_oversampled(0, 0), self.gm(0, 0))
        assert_allclose(model_oversampled(1, 1), self.gm(1, 1))
        assert_allclose(model_oversampled(-2, 1), self.gm(-2, 1))
        assert_allclose(model_oversampled(0.5, 0.5), self.gm(0.5, 0.5),
                        rtol=.001)
        assert_allclose(model_oversampled(-0.5, 1.75), self.gm(-0.5, 1.75),
                        rtol=.001)

        # without oversampling the same tests should fail except for at
        # the origin
        model_wrongsampled = FittableImageModel(im)
        assert_allclose(model_wrongsampled(0, 0), self.gm(0, 0))
        assert not np.allclose(model_wrongsampled(1, 1), self.gm(1, 1))
        assert not np.allclose(model_wrongsampled(-2, 1), self.gm(-2, 1))
        assert not np.allclose(model_wrongsampled(0.5, 0.5),
                               self.gm(0.5, 0.5), rtol=.001)
        assert not np.allclose(model_wrongsampled(-0.5, 1.75),
                               self.gm(-0.5, 1.75), rtol=.001)

    def test_centering_oversampled(self):
        gm = Gaussian2D(x_stddev=2, y_stddev=3)

        oversamp = 3
        yy, xx = np.mgrid[-3:3.00001:(1 / oversamp), -3:3.00001:(1 / oversamp)]

        model_oversampled = FittableImageModel(gm(xx, yy),
                                               oversampling=oversamp)

        valcen = gm(0, 0)
        val36 = gm(0.66, 0.66)

        assert_allclose(valcen, model_oversampled(0, 0))
        assert_allclose(val36, model_oversampled(0.66, 0.66), rtol=1.e-6)

        model_oversampled.x_0 = 2.5
        model_oversampled.y_0 = -3.5

        assert_allclose(valcen, model_oversampled(2.5, -3.5))
        assert_allclose(val36, model_oversampled(2.5 + 0.66, -3.5 + 0.66),
                        rtol=1.e-6)

    def test_oversampling_inputs(self):
        data = np.arange(30).reshape(5, 6)
        for oversampling in [4, (3, 3), (3, 4)]:
            fim = FittableImageModel(data, oversampling=oversampling)
            if not hasattr(oversampling, '__len__'):
                _oversamp = float(oversampling)
            else:
                _oversamp = tuple(float(o) for o in oversampling)
            assert np.all(fim._oversampling == _oversamp)

        for oversampling in [-1, [-2, 4], (1, 4, 8), ((1, 2), (3, 4)),
                             np.ones((2, 2, 2)), 2.1, np.nan, (1, np.inf)]:
            with pytest.raises(ValueError):
                FittableImageModel(data, oversampling=oversampling)


class TestGriddedPSFModel:
    def setup_class(self):
        psfs = []
        y, x = np.mgrid[0:101, 0:101]
        for i in range(16):
            theta = i * 10. * np.pi / 180.
            g = Gaussian2D(1, 50, 50, 10, 5, theta=theta)
            m = g(x, y)
            psfs.append(m)

        xgrid = [0, 40, 160, 200]
        ygrid = [0, 60, 140, 200]
        grid_xypos = list(product(xgrid, ygrid))

        meta = {}
        meta['grid_xypos'] = grid_xypos
        meta['oversampling'] = 4
        self.nddata = NDData(psfs, meta=meta)
        self.psfmodel = GriddedPSFModel(self.nddata)

    def test_gridded_psf_model(self):
        keys = ['grid_xypos', 'oversampling']
        for key in keys:
            assert key in self.psfmodel.meta
        assert len(self.psfmodel.meta) == 2
        assert len(self.psfmodel.meta['grid_xypos']) == 16
        assert self.psfmodel.oversampling == 4
        assert (self.psfmodel.meta['oversampling'] ==
                self.psfmodel.oversampling)
        assert self.psfmodel.data.shape == (16, 101, 101)

    @pytest.mark.skipif('not HAS_SCIPY')
    def test_gridded_psf_model_basic_eval(self):
        y, x = np.mgrid[0:100, 0:100]
        psf = self.psfmodel.evaluate(x=x, y=y, flux=100, x_0=40, y_0=60)
        assert psf.shape == (100, 100)

    @pytest.mark.skipif('not HAS_SCIPY')
    def test_gridded_psf_model_eval_outside_grid(self):
        y, x = np.mgrid[-50:50, -50:50]
        psf1 = self.psfmodel.evaluate(x=x, y=y, flux=100, x_0=0, y_0=0)
        y, x = np.mgrid[-60:40, -60:40]
        psf2 = self.psfmodel.evaluate(x=x, y=y, flux=100, x_0=-10, y_0=-10)
        assert_allclose(psf1, psf2)

        y, x = np.mgrid[150:250, 150:250]
        psf3 = self.psfmodel.evaluate(x=x, y=y, flux=100, x_0=200, y_0=200)
        y, x = np.mgrid[170:270, 170:270]
        psf4 = self.psfmodel.evaluate(x=x, y=y, flux=100, x_0=220, y_0=220)
        assert_allclose(psf3, psf4)

    @pytest.mark.skipif('not HAS_SCIPY')
    def test_gridded_psf_model_interp(self):
        # test xyref length
        with pytest.raises(ValueError):
            self.psfmodel._bilinear_interp([1, 1], 1, 1, 1)

        # test zref shape
        with pytest.raises(ValueError):
            xyref = [[0, 0], [0, 1], [1, 0], [1, 1]]
            zref = np.ones((3, 4, 4))
            self.psfmodel._bilinear_interp(xyref, zref, 1, 1)

        # test if refxy points form a rectangle
        with pytest.raises(ValueError):
            xyref = [[0, 0], [0, 1], [1, 0], [2, 2]]
            zref = np.ones((4, 4, 4))
            self.psfmodel._bilinear_interp(xyref, zref, 1, 1)

        # test if xi and yi are outside of xyref
        xyref = [[0, 0], [0, 1], [1, 0], [1, 1]]
        zref = np.ones((4, 4, 4))
        with pytest.raises(ValueError):
            self.psfmodel._bilinear_interp(xyref, zref, 100, 1)
        with pytest.raises(ValueError):
            self.psfmodel._bilinear_interp(xyref, zref, 1, 100)

        # test non-scalar xi and yi
        idx = [0, 1, 4, 5]
        xyref = np.array(self.psfmodel.grid_xypos)[idx]
        psfs = self.psfmodel.data[idx, :, :]
        val1 = self.psfmodel._bilinear_interp(xyref, psfs, 10, 20)
        val2 = self.psfmodel._bilinear_interp(xyref, psfs, [10], [20])
        assert_allclose(val1, val2)

    def test_gridded_psf_model_invalid_inputs(self):
        data = np.ones((4, 3, 3))

        # check if NDData
        with pytest.raises(TypeError):
            GriddedPSFModel(data)

        # check PSF data dimension
        with pytest.raises(ValueError):
            GriddedPSFModel(NDData(np.ones((3, 3))))

        # check that grid_xypos is in meta
        meta = {'oversampling': 4}
        nddata = NDData(data, meta=meta)
        with pytest.raises(ValueError):
            GriddedPSFModel(nddata)

        # check grid_xypos length
        meta = {'grid_xypos': [[0, 0], [1, 0], [1, 0]],
                'oversampling': 4}
        nddata = NDData(data, meta=meta)
        with pytest.raises(ValueError):
            GriddedPSFModel(nddata)

        # check if grid_xypos is a regular grid
        meta = {'grid_xypos': [[0, 0], [1, 0], [1, 0], [3, 4]],
                'oversampling': 4}
        nddata = NDData(data, meta=meta)
        with pytest.raises(ValueError):
            GriddedPSFModel(nddata)

        # check that oversampling is in meta
        meta = {'grid_xypos': [[0, 0], [0, 1], [1, 0], [1, 1]]}
        nddata = NDData(data, meta=meta)
        with pytest.raises(ValueError):
            GriddedPSFModel(nddata)

        # check oversampling is a scalar
        meta = {'grid_xypos': [[0, 0], [0, 1], [1, 0], [1, 1]],
                'oversampling': [4, 4]}
        nddata = NDData(data, meta=meta)
        with pytest.raises(ValueError):
            GriddedPSFModel(nddata)

    @pytest.mark.skipif('not HAS_SCIPY')
    def test_gridded_psf_model_eval(self):
        """
        Create a simulated image using GriddedPSFModel and test
        the properties of the generated sources.
        """

        shape = (200, 200)
        data = np.zeros(shape)
        eval_xshape = (np.ceil(self.psfmodel.data.shape[2] /
                               self.psfmodel.oversampling)).astype(int)
        eval_yshape = (np.ceil(self.psfmodel.data.shape[1] /
                               self.psfmodel.oversampling)).astype(int)

        xx = [40, 50, 160, 160]
        yy = [60, 150, 50, 140]
        zz = [100, 100, 100, 100]
        for xxi, yyi, zzi in zip(xx, yy, zz):
            x0 = np.floor(xxi - (eval_xshape - 1) / 2.).astype(int)
            y0 = np.floor(yyi - (eval_yshape - 1) / 2.).astype(int)
            x1 = x0 + eval_xshape
            y1 = y0 + eval_yshape

            if x0 < 0:
                x0 = 0
            if y0 < 0:
                y0 = 0
            if x1 > shape[1]:
                x1 = shape[1]
            if y1 > shape[0]:
                y1 = shape[0]

            y, x = np.mgrid[y0:y1, x0:x1]
            data[y, x] += self.psfmodel.evaluate(x=x, y=y, flux=zzi, x_0=xxi,
                                                 y_0=yyi)

        segm = detect_sources(data, 0., 5)
        cat = SourceCatalog(data, segm)
        orients = cat.orientation.value
        assert_allclose(orients[1], 50., rtol=1.e-5)
        assert_allclose(orients[2], -80., rtol=1.e-5)
        assert 88.3 < orients[0] < 88.4
        assert 64. < orients[3] < 64.2


@pytest.mark.skipif('not HAS_SCIPY')
class TestIntegratedGaussianPRF:
    widths = [0.001, 0.01, 0.1, 1]
    sigmas = [0.5, 1., 2., 10., 12.34]

    @pytest.mark.parametrize('width', widths)
    def test_subpixel_gauss_psf(self, width):
        """
        Test subpixel accuracy of IntegratedGaussianPRF by checking the
        sum of pixels.
        """

        gauss_psf = IntegratedGaussianPRF(width)
        y, x = np.mgrid[-10:11, -10:11]
        assert_allclose(gauss_psf(x, y).sum(), 1)

    @pytest.mark.parametrize('sigma', sigmas)
    def test_gaussian_psf_integral(self, sigma):
        """
        Test if IntegratedGaussianPRF integrates to unity on larger
        scales.
        """

        psf = IntegratedGaussianPRF(sigma=sigma)
        y, x = np.mgrid[-100:101, -100:101]
        assert_allclose(psf(y, x).sum(), 1)


@pytest.mark.skipif('not HAS_SCIPY')
class TestPRFAdapter:
    def normalize_moffat(self, mof):
        # this is the analytic value needed to get a total flux of 1
        mof = mof.copy()
        mof.amplitude = (mof.alpha-1)/(np.pi*mof.gamma**2)
        return mof

    @pytest.mark.parametrize("adapterkwargs", [
        dict(xname='x_0', yname='y_0', fluxname=None, renormalize_psf=False),
        dict(xname=None, yname=None, fluxname=None, renormalize_psf=False),
        dict(xname='x_0', yname='y_0', fluxname='amplitude',
             renormalize_psf=False)])
    def test_create_eval_prfadapter(self, adapterkwargs):
        mof = Moffat2D(gamma=1, alpha=4.8)
        prf = PRFAdapter(mof, **adapterkwargs)

        # test that these work without errors
        prf.x_0 = 0.5
        prf.y_0 = -0.5
        prf.flux = 1.2
        prf(0, 0)

    @pytest.mark.parametrize("adapterkwargs", [
        dict(xname='x_0', yname='y_0', fluxname=None, renormalize_psf=True),
        dict(xname='x_0', yname='y_0', fluxname=None, renormalize_psf=False),
        dict(xname=None, yname=None, fluxname=None, renormalize_psf=False)
    ])
    def test_prfadapter_integrates(self, adapterkwargs):
        from scipy.integrate import dblquad

        mof = Moffat2D(gamma=1.5, alpha=4.8)
        if not adapterkwargs['renormalize_psf']:
            mof = self.normalize_moffat(mof)
        prf1 = PRFAdapter(mof, **adapterkwargs)

        # first check that the PRF over a central grid ends up summing to the
        # integrand over the whole PSF
        xg, yg = np.meshgrid(*([(-1, 0, 1)]*2))
        evalmod = prf1(xg, yg)

        if adapterkwargs['renormalize_psf']:
            mof = self.normalize_moffat(mof)

        integrand, itol = dblquad(mof, -1.5, 1.5, lambda x: -1.5,
                                  lambda x: 1.5)
        assert_allclose(np.sum(evalmod), integrand, atol=itol * 10)

    @pytest.mark.parametrize("adapterkwargs", [
        dict(xname='x_0', yname='y_0', fluxname=None, renormalize_psf=False),
        dict(xname=None, yname=None, fluxname=None, renormalize_psf=False)])
    def test_prfadapter_sizematch(self, adapterkwargs):
        from scipy.integrate import dblquad

        mof1 = self.normalize_moffat(Moffat2D(gamma=1, alpha=4.8))
        prf1 = PRFAdapter(mof1, **adapterkwargs)

        # now try integrating over differently-sampled PRFs
        # and check that they match
        mof2 = self.normalize_moffat(Moffat2D(gamma=2, alpha=4.8))
        prf2 = PRFAdapter(mof2, **adapterkwargs)

        xg1, yg1 = np.meshgrid(*([(-0.5, 0.5)]*2))
        xg2, yg2 = np.meshgrid(*([(-1.5, -0.5, 0.5, 1.5)]*2))

        eval11 = prf1(xg1, yg1)
        eval22 = prf2(xg2, yg2)

        integrand, itol = dblquad(mof1, -2, 2, lambda x: -2, lambda x: 2)
        # it's a bit of a guess that the above itol is appropriate, but
        # it should be close
        assert_allclose(np.sum(eval11), np.sum(eval22), atol=itol*100)
