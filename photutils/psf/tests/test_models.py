# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the models module.
"""

import numpy as np
import pytest
from astropy.modeling.models import Gaussian2D, Moffat2D
from numpy.testing import assert_allclose

from photutils.psf import FittableImageModel, IntegratedGaussianPRF, PRFAdapter
from photutils.utils._optional_deps import HAS_SCIPY


@pytest.fixture(name='gmodel')
def fixture_gmodel():
    return Gaussian2D(x_stddev=3, y_stddev=3)


@pytest.mark.skipif(not HAS_SCIPY, reason='scipy is required')
class TestFittableImageModel:
    """
    Tests for FittableImageModel.
    """

    def test_fittable_image_model(self, gmodel):
        yy, xx = np.mgrid[-2:3, -2:3]
        model_nonorm = FittableImageModel(gmodel(xx, yy))

        assert_allclose(model_nonorm(0, 0), gmodel(0, 0))
        assert_allclose(model_nonorm(1, 1), gmodel(1, 1))
        assert_allclose(model_nonorm(-2, 1), gmodel(-2, 1))

        # subpixel should *not* match, but be reasonably close
        # in this case good to ~0.1% seems to be fine
        assert_allclose(model_nonorm(0.5, 0.5), gmodel(0.5, 0.5), rtol=.001)
        assert_allclose(model_nonorm(-0.5, 1.75), gmodel(-0.5, 1.75),
                        rtol=.001)

        model_norm = FittableImageModel(gmodel(xx, yy), normalize=True)
        assert not np.allclose(model_norm(0, 0), gmodel(0, 0))
        assert_allclose(np.sum(model_norm(xx, yy)), 1)

        model_norm2 = FittableImageModel(gmodel(xx, yy), normalize=True,
                                         normalization_correction=2)
        assert not np.allclose(model_norm2(0, 0), gmodel(0, 0))
        assert_allclose(model_norm(0, 0), model_norm2(0, 0) * 2)
        assert_allclose(np.sum(model_norm2(xx, yy)), 0.5)

    def test_fittable_image_model_oversampling(self, gmodel):
        oversamp = 3  # oversampling factor
        yy, xx = np.mgrid[-3:3.00001:(1 / oversamp), -3:3.00001:(1 / oversamp)]

        im = gmodel(xx, yy)
        assert im.shape[0] > 7

        model_oversampled = FittableImageModel(im, oversampling=oversamp)
        assert_allclose(model_oversampled(0, 0), gmodel(0, 0))
        assert_allclose(model_oversampled(1, 1), gmodel(1, 1))
        assert_allclose(model_oversampled(-2, 1), gmodel(-2, 1))
        assert_allclose(model_oversampled(0.5, 0.5), gmodel(0.5, 0.5),
                        rtol=.001)
        assert_allclose(model_oversampled(-0.5, 1.75), gmodel(-0.5, 1.75),
                        rtol=.001)

        # without oversampling the same tests should fail except for at
        # the origin
        model_wrongsampled = FittableImageModel(im)
        assert_allclose(model_wrongsampled(0, 0), gmodel(0, 0))
        assert not np.allclose(model_wrongsampled(1, 1), gmodel(1, 1))
        assert not np.allclose(model_wrongsampled(-2, 1), gmodel(-2, 1))
        assert not np.allclose(model_wrongsampled(0.5, 0.5), gmodel(0.5, 0.5),
                               rtol=.001)
        assert not np.allclose(model_wrongsampled(-0.5, 1.75),
                               gmodel(-0.5, 1.75), rtol=.001)

    def test_centering_oversampled(self, gmodel):
        oversamp = 3
        yy, xx = np.mgrid[-3:3.00001:(1 / oversamp), -3:3.00001:(1 / oversamp)]

        model_oversampled = FittableImageModel(gmodel(xx, yy),
                                               oversampling=oversamp)

        valcen = gmodel(0, 0)
        val36 = gmodel(0.66, 0.66)

        assert_allclose(valcen, model_oversampled(0, 0))
        assert_allclose(val36, model_oversampled(0.66, 0.66), rtol=1.0e-6)

        model_oversampled.x_0 = 2.5
        model_oversampled.y_0 = -3.5

        assert_allclose(valcen, model_oversampled(2.5, -3.5))
        assert_allclose(val36, model_oversampled(2.5 + 0.66, -3.5 + 0.66),
                        rtol=1.0e-6)

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


@pytest.mark.skipif(not HAS_SCIPY, reason='scipy is required')
class TestIntegratedGaussianPRF:
    """
    Tests for IntegratedGaussianPRF.
    """

    widths = [0.001, 0.01, 0.1, 1]
    sigmas = [0.5, 1.0, 2.0, 10.0, 12.34]

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


@pytest.mark.skipif(not HAS_SCIPY, reason='scipy is required')
class TestPRFAdapter:
    """
    Tests for PRFAdapter.
    """

    def normalize_moffat(self, mof):
        # this is the analytic value needed to get a total flux of 1
        mof = mof.copy()
        mof.amplitude = (mof.alpha - 1) / (np.pi * mof.gamma**2)
        return mof

    @pytest.mark.parametrize('adapterkwargs', [
        {'xname': 'x_0', 'yname': 'y_0', 'fluxname': None,
         'renormalize_psf': False},
        {'xname': None, 'yname': None, 'fluxname': None,
         'renormalize_psf': False},
        {'xname': 'x_0', 'yname': 'y_0', 'fluxname': 'amplitude',
         'renormalize_psf': False}])
    def test_create_eval_prfadapter(self, adapterkwargs):
        mof = Moffat2D(gamma=1, alpha=4.8)
        prf = PRFAdapter(mof, **adapterkwargs)

        # test that these work without errors
        prf.x_0 = 0.5
        prf.y_0 = -0.5
        prf.flux = 1.2
        prf(0, 0)

    @pytest.mark.parametrize('adapterkwargs', [
        {'xname': 'x_0', 'yname': 'y_0', 'fluxname': None,
         'renormalize_psf': True},
        {'xname': 'x_0', 'yname': 'y_0', 'fluxname': None,
         'renormalize_psf': False},
        {'xname': None, 'yname': None, 'fluxname': None,
         'renormalize_psf': False}])
    def test_prfadapter_integrates(self, adapterkwargs):
        from scipy.integrate import dblquad

        mof = Moffat2D(gamma=1.5, alpha=4.8)
        if not adapterkwargs['renormalize_psf']:
            mof = self.normalize_moffat(mof)
        prf1 = PRFAdapter(mof, **adapterkwargs)

        # first check that the PRF over a central grid ends up summing to the
        # integrand over the whole PSF
        xg, yg = np.meshgrid(*([(-1, 0, 1)] * 2))
        evalmod = prf1(xg, yg)

        if adapterkwargs['renormalize_psf']:
            mof = self.normalize_moffat(mof)

        integrand, itol = dblquad(mof, -1.5, 1.5, lambda x: -1.5,
                                  lambda x: 1.5)
        assert_allclose(np.sum(evalmod), integrand, atol=itol * 10)

    @pytest.mark.parametrize('adapterkwargs', [
        {'xname': 'x_0', 'yname': 'y_0', 'fluxname': None,
         'renormalize_psf': False},
        {'xname': None, 'yname': None, 'fluxname': None,
         'renormalize_psf': False}])
    def test_prfadapter_sizematch(self, adapterkwargs):
        from scipy.integrate import dblquad

        mof1 = self.normalize_moffat(Moffat2D(gamma=1, alpha=4.8))
        prf1 = PRFAdapter(mof1, **adapterkwargs)

        # now try integrating over differently-sampled PRFs
        # and check that they match
        mof2 = self.normalize_moffat(Moffat2D(gamma=2, alpha=4.8))
        prf2 = PRFAdapter(mof2, **adapterkwargs)

        xg1, yg1 = np.meshgrid(*([(-0.5, 0.5)] * 2))
        xg2, yg2 = np.meshgrid(*([(-1.5, -0.5, 0.5, 1.5)] * 2))

        eval11 = prf1(xg1, yg1)
        eval22 = prf2(xg2, yg2)

        _, itol = dblquad(mof1, -2, 2, lambda x: -2, lambda x: 2)
        # it's a bit of a guess that the above itol is appropriate, but
        # it should be close
        assert_allclose(np.sum(eval11), np.sum(eval22), atol=itol * 100)
