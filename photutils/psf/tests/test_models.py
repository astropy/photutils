# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the models module.
"""

import astropy.units as u
import numpy as np
import pytest
from astropy.modeling.fitting import LevMarLSQFitter
from astropy.modeling.models import Gaussian2D, Moffat2D
from astropy.stats import gaussian_fwhm_to_sigma
from numpy.testing import assert_allclose

from photutils.psf import (CircularGaussianPRF, CircularGaussianPSF,
                           FittableImageModel, GaussianPRF, GaussianPSF,
                           IntegratedGaussianPRF, PRFAdapter)
from photutils.utils._optional_deps import HAS_SCIPY


@pytest.fixture(name='gmodel')
def fixture_gmodel():
    return Gaussian2D(x_stddev=3, y_stddev=3)


def make_gaussian_models(name):
    flux = 71.4
    x_0 = 24.3
    y_0 = 25.2
    x_fwhm = 10.1
    y_fwhm = 5.82
    theta = 21.7
    flux_i = 50
    x_0_i = 20
    y_0_i = 30
    x_fwhm_i = 15
    y_fwhm_i = 8
    theta_i = 31
    if name == 'GaussianPSF':
        model = GaussianPSF(flux=flux, x_0=x_0, y_0=y_0, x_fwhm=x_fwhm,
                            y_fwhm=y_fwhm, theta=theta)
        model_init = GaussianPSF(flux=flux_i, x_0=x_0_i, y_0=y_0_i,
                                 x_fwhm=x_fwhm_i, y_fwhm=y_fwhm_i,
                                 theta=theta_i)
    elif name == 'GaussianPRF':
        model = GaussianPRF(flux=flux, x_0=x_0, y_0=y_0, x_fwhm=x_fwhm,
                            y_fwhm=y_fwhm, theta=theta)
        model_init = GaussianPRF(flux=flux_i, x_0=x_0_i, y_0=y_0_i,
                                 x_fwhm=x_fwhm_i, y_fwhm=y_fwhm_i,
                                 theta=theta_i)
    elif name == 'CircularGaussianPSF':
        model = CircularGaussianPSF(flux=flux, x_0=x_0, y_0=y_0, fwhm=x_fwhm)
        model_init = CircularGaussianPSF(flux=flux_i, x_0=x_0_i, y_0=y_0_i,
                                         fwhm=x_fwhm_i)
    elif name == 'CircularGaussianPRF':
        model = CircularGaussianPRF(flux=flux, x_0=x_0, y_0=y_0, fwhm=x_fwhm)
        model_init = CircularGaussianPRF(flux=flux_i, x_0=x_0_i, y_0=y_0_i,
                                         fwhm=x_fwhm_i)
    elif name == 'IntegratedGaussianPRF':
        model = IntegratedGaussianPRF(flux=flux, x_0=x_0, y_0=y_0,
                                      sigma=x_fwhm / 2.35)
        model_init = IntegratedGaussianPRF(flux=flux_i, x_0=x_0_i, y_0=y_0_i,
                                           sigma=x_fwhm_i / 2.35)

    return model, model_init


def gaussian_tests(name, use_units):
    model, model_init = make_gaussian_models(name)
    fixed_types = ('fwhm', 'sigma', 'theta')
    for param in model.param_names:
        for fixed_type in fixed_types:
            if fixed_type in param:
                tparam = getattr(model_init, param)
                tparam.fixed = False

    yy, xx = np.mgrid[0:51, 0:51]

    if use_units:
        unit = u.m
        xx <<= unit
        yy <<= unit

        unit_params = ('x_0', 'y_0', 'x_fwhm', 'y_fwhm', 'fwhm', 'sigma')
        for param in model.param_names:
            if param in unit_params:
                tparam = getattr(model, param)
                tparam <<= unit
                setattr(model, param, tparam)

    data = model(xx, yy)
    if use_units:
        data = data.value
    assert_allclose(data.sum(), model.flux.value)
    try:
        assert_allclose(model.x_sigma, model.x_fwhm * gaussian_fwhm_to_sigma)
        assert_allclose(model.y_sigma, model.y_fwhm * gaussian_fwhm_to_sigma)
    except AttributeError:
        assert_allclose(model.sigma, model.fwhm * gaussian_fwhm_to_sigma)

    try:
        xsigma = model.x_sigma
        ysigma = model.y_sigma
        if isinstance(xsigma, u.Quantity):
            xsigma = xsigma.value
            ysigma = ysigma.value
        assert_allclose(model.amplitude * (2 * np.pi * xsigma * ysigma),
                        model.flux)
    except AttributeError:
        sigma = model.sigma
        if isinstance(sigma, u.Quantity):
            sigma = sigma.value
        assert_allclose(model.amplitude * (2 * np.pi * sigma**2), model.flux)

    fitter = LevMarLSQFitter()
    fit_model = fitter(model_init, xx, yy, data)
    assert_allclose(fit_model.x_0.value, model.x_0.value, rtol=1e-5)
    assert_allclose(fit_model.y_0.value, model.y_0.value, rtol=1e-5)
    try:
        assert_allclose(fit_model.x_fwhm.value, model.x_fwhm.value)
        assert_allclose(fit_model.y_fwhm.value, model.y_fwhm.value)
        assert_allclose(fit_model.theta.value, model.theta.value)
    except AttributeError:
        if name == 'IntegratedGaussianPRF':
            assert_allclose(fit_model.sigma.value, model.sigma.value)
        else:
            assert_allclose(fit_model.fwhm.value, model.fwhm.value)

    # test the model derivatives
    fit_model2 = fitter(model_init, xx, yy, data, estimate_jacobian=True)
    assert_allclose(fit_model2.x_0, fit_model.x_0)
    assert_allclose(fit_model2.y_0, fit_model.y_0)
    try:
        assert_allclose(fit_model2.x_fwhm, fit_model.x_fwhm)
        assert_allclose(fit_model2.y_fwhm, fit_model.y_fwhm)
        assert_allclose(fit_model2.theta, fit_model.theta)
    except AttributeError:
        assert_allclose(fit_model2.fwhm, fit_model.fwhm)

    if use_units and 'Circular' not in name:
        model.y_0 = model.y_0.value * u.s
        yy = yy.value * u.s
        match = 'Units .* inputs should match'
        with pytest.raises(u.UnitsError, match=match):
            fitter(model_init, xx, yy, data)


@pytest.mark.parametrize('name', ['GaussianPSF', 'CircularGaussianPSF'])
@pytest.mark.parametrize('use_units', [False, True])
def test_gaussian_psfs(name, use_units):
    gaussian_tests(name, use_units)


@pytest.mark.skipif(not HAS_SCIPY, reason='scipy is required')
@pytest.mark.parametrize('name', ['GaussianPRF', 'CircularGaussianPRF',
                                  'IntegratedGaussianPRF'])
@pytest.mark.parametrize('use_units', [False, True])
def test_gaussian_prfs(name, use_units):
    gaussian_tests(name, use_units)


@pytest.mark.skipif(not HAS_SCIPY, reason='scipy is required')
def test_gaussian_prf_sums():
    """
    Test that subpixel accuracy of Gaussian PRFs by checking the sum of
    pixels.
    """
    model1 = GaussianPRF(x_0=0, y_0=0, x_fwhm=0.001, y_fwhm=0.001)
    model2 = CircularGaussianPRF(x_0=0, y_0=0, fwhm=0.001)
    model3 = IntegratedGaussianPRF(x_0=0, y_0=0, sigma=0.001)
    yy, xx = np.mgrid[-10:11, -10:11]
    for model in (model1, model2, model3):
        assert_allclose(model(xx, yy).sum(), 1.0)


def test_gaussian_bounding_boxes():
    model1 = GaussianPSF(x_0=0, y_0=0, x_fwhm=2, y_fwhm=3)
    model2 = GaussianPRF(x_0=0, y_0=0, x_fwhm=2, y_fwhm=3)
    xbbox = (-4.6712699, 4.6712699)
    ybbox = (-7.0069049, 7.0069048)
    for model in (model1, model2):
        assert_allclose(model.bounding_box, (xbbox, ybbox))

    model3 = CircularGaussianPSF(x_0=0, y_0=0, fwhm=2)
    model4 = CircularGaussianPRF(x_0=0, y_0=0, fwhm=2)
    for model in (model3, model4):
        assert_allclose(model.bounding_box, (xbbox, xbbox))

    model5 = IntegratedGaussianPRF(x_0=0, y_0=0, sigma=2)
    assert_allclose(model5.bounding_box, ((-11, 11), (-11, 11)))


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

        match = 'oversampling must be > 0'
        for oversampling in [-1, [-2, 4]]:
            with pytest.raises(ValueError, match=match):
                FittableImageModel(data, oversampling=oversampling)

        match = 'oversampling must have 1 or 2 elements'
        oversampling = (1, 4, 8)
        with pytest.raises(ValueError, match=match):
            FittableImageModel(data, oversampling=oversampling)

        match = 'oversampling must be 1D'
        for oversampling in [((1, 2), (3, 4)), np.ones((2, 2, 2))]:
            with pytest.raises(ValueError, match=match):
                FittableImageModel(data, oversampling=oversampling)

        match = 'oversampling must have integer values'
        with pytest.raises(ValueError, match=match):
            FittableImageModel(data, oversampling=2.1)

        match = 'oversampling must be a finite value'
        for oversampling in [np.nan, (1, np.inf)]:
            with pytest.raises(ValueError, match=match):
                FittableImageModel(data, oversampling=oversampling)


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
