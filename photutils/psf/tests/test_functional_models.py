# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the functional_models module.
"""

import astropy.units as u
import numpy as np
import pytest
from astropy.modeling.fitting import TRFLSQFitter
from astropy.stats import gaussian_fwhm_to_sigma
from numpy.testing import assert_allclose

from photutils.psf import (AiryDiskPSF, CircularGaussianPRF,
                           CircularGaussianPSF, CircularGaussianSigmaPRF,
                           GaussianPRF, GaussianPSF, MoffatPSF)


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
    elif name == 'CircularGaussianSigmaPRF':
        model = CircularGaussianSigmaPRF(flux=flux, x_0=x_0, y_0=y_0,
                                         sigma=x_fwhm / 2.35)
        model_init = CircularGaussianSigmaPRF(flux=flux_i, x_0=x_0_i,
                                              y_0=y_0_i, sigma=x_fwhm_i / 2.35)
    else:
        msg = 'invalid model name'
        raise ValueError(msg)

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

    fitter = TRFLSQFitter()
    fit_model = fitter(model_init, xx, yy, data)
    assert_allclose(fit_model.x_0.value, model.x_0.value, rtol=1e-5)
    assert_allclose(fit_model.y_0.value, model.y_0.value, rtol=1e-5)
    try:
        assert_allclose(fit_model.x_fwhm.value, model.x_fwhm.value)
        assert_allclose(fit_model.y_fwhm.value, model.y_fwhm.value)
        assert_allclose(fit_model.theta.value, model.theta.value)
    except AttributeError:
        if name == 'CircularGaussianSigmaPRF':
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


@pytest.mark.parametrize('name', ['GaussianPRF', 'CircularGaussianPRF',
                                  'CircularGaussianSigmaPRF'])
@pytest.mark.parametrize('use_units', [False, True])
def test_gaussian_prfs(name, use_units):
    gaussian_tests(name, use_units)


def test_gaussian_prf_sums():
    """
    Test that subpixel accuracy of Gaussian PRFs by checking the sum of
    pixels.
    """
    model1 = GaussianPRF(x_0=0, y_0=0, x_fwhm=0.001, y_fwhm=0.001)
    model2 = CircularGaussianPRF(x_0=0, y_0=0, fwhm=0.001)
    model3 = CircularGaussianSigmaPRF(x_0=0, y_0=0, sigma=0.001)
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

    model5 = CircularGaussianSigmaPRF(x_0=0, y_0=0, sigma=2)
    assert_allclose(model5.bounding_box, ((-11, 11), (-11, 11)))


@pytest.mark.parametrize('use_units', [False, True])
def test_moffat_psf_model(use_units):
    model = MoffatPSF(flux=71.4, x_0=24.3, y_0=25.2, alpha=8.1, beta=7.2)
    model_init = MoffatPSF(flux=50, x_0=20, y_0=30, alpha=5, beta=4)
    model_init.alpha.fixed = False
    model_init.beta.fixed = False

    yy, xx = np.mgrid[0:51, 0:51]

    if use_units:
        unit = u.cm
        xx <<= unit
        yy <<= unit
        model.x_0 <<= unit
        model.y_0 <<= unit
        model.alpha <<= unit

    data = model(xx, yy)
    assert_allclose(data.sum(), model.flux.value, rtol=5e-6)
    fwhm = 2 * model.alpha * np.sqrt(2**(1 / model.beta) - 1)
    assert_allclose(model.fwhm, fwhm)

    fitter = TRFLSQFitter()
    fit_model = fitter(model_init, xx, yy, data)
    assert_allclose(fit_model.x_0.value, model.x_0.value)
    assert_allclose(fit_model.y_0.value, model.y_0.value)
    assert_allclose(fit_model.alpha.value, model.alpha.value)
    assert_allclose(fit_model.beta.value, model.beta.value)

    # test bounding box
    model = MoffatPSF(x_0=0, y_0=0, alpha=1.0, beta=2.0)
    bbox = 12.871885058111655
    assert_allclose(model.bounding_box, ((-bbox, bbox), (-bbox, bbox)))


@pytest.mark.parametrize('use_units', [False, True])
def test_airydisk_psf_model(use_units):
    model = AiryDiskPSF(flux=71.4, x_0=24.3, y_0=25.2, radius=2.1)
    model_init = AiryDiskPSF(flux=50, x_0=23, y_0=27, radius=2.5)
    model_init.radius.fixed = False

    yy, xx = np.mgrid[0:51, 0:51]

    if use_units:
        unit = u.cm
        xx <<= unit
        yy <<= unit
        model.x_0 <<= unit
        model.y_0 <<= unit
        model.radius <<= unit

    data = model(xx, yy)
    assert_allclose(data.sum(), model.flux.value, rtol=0.015)
    fwhm = 0.8436659602162364 * model.radius
    assert_allclose(model.fwhm, fwhm)

    fitter = TRFLSQFitter()
    fit_model = fitter(model_init, xx, yy, data)
    assert_allclose(fit_model.x_0.value, model.x_0.value)
    assert_allclose(fit_model.y_0.value, model.y_0.value)
    assert_allclose(fit_model.radius.value, model.radius.value)

    # test bounding box
    model = AiryDiskPSF(x_0=0, y_0=0, radius=5)
    bbox = 42.18329801081182
    assert_allclose(model.bounding_box, ((-bbox, bbox), (-bbox, bbox)))
