# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the utils module.
"""

import astropy.units as u
import numpy as np
import pytest
from astropy.modeling.models import Gaussian1D, Gaussian2D
from astropy.table import QTable
from astropy.utils.exceptions import AstropyUserWarning
from numpy.testing import assert_allclose

from photutils.psf import CircularGaussianPRF, make_psf_model_image
from photutils.psf.utils import (_get_psf_model_params,
                                 _interpolate_missing_data,
                                 _validate_psf_model, fit_2dgaussian, fit_fwhm)


@pytest.fixture(name='test_data')
def fixture_test_data():
    psf_model = CircularGaussianPRF()
    model_shape = (9, 9)
    n_sources = 10
    shape = (101, 101)
    data, true_params = make_psf_model_image(shape, psf_model, n_sources,
                                             model_shape=model_shape,
                                             flux=(500, 700), fwhm=(2.7, 2.7),
                                             min_separation=10, seed=0)
    return data, true_params


@pytest.mark.parametrize('fix_fwhm', [False, True])
def test_fit_2dgaussian_single(fix_fwhm):
    yy, xx = np.mgrid[:51, :51]
    fwhm = 3.123
    model = CircularGaussianPRF(x_0=22.17, y_0=28.87, fwhm=fwhm)
    data = model(xx, yy)

    fit = fit_2dgaussian(data, fwhm=3, fix_fwhm=fix_fwhm)
    fit_tbl = fit.results
    assert isinstance(fit_tbl, QTable)
    assert len(fit_tbl) == 1
    if fix_fwhm:
        assert 'fwhm_fit' not in fit_tbl.colnames
    else:
        assert 'fwhm_fit' in fit_tbl.colnames
        assert_allclose(fit_tbl['fwhm_fit'], fwhm)


@pytest.mark.parametrize(('fix_fwhm', 'with_units'),
                         [(False, True), (True, False)])
def test_fit_2dgaussian_multiple(test_data, fix_fwhm, with_units):
    data, sources = test_data

    unit = u.nJy
    if with_units:
        data = data * unit

    xypos = list(zip(sources['x_0'], sources['y_0'], strict=True))
    fit = fit_2dgaussian(data, xypos=xypos, fit_shape=(5, 5),
                         fix_fwhm=fix_fwhm)
    fit_tbl = fit.results
    assert isinstance(fit_tbl, QTable)
    assert len(fit_tbl) == len(sources)
    if fix_fwhm:
        assert 'fwhm_fit' not in fit_tbl.colnames
    else:
        assert 'fwhm_fit' in fit_tbl.colnames
        assert_allclose(fit_tbl['fwhm_fit'], sources['fwhm'])

    if with_units:
        for column in fit_tbl.colnames:
            if 'flux' in column:
                assert fit_tbl['flux_fit'].unit == unit


def test_fit_fwhm_single():
    yy, xx = np.mgrid[:51, :51]
    fwhm0 = 3.123
    model = CircularGaussianPRF(x_0=22.17, y_0=28.87, fwhm=fwhm0)
    data = model(xx, yy)

    fwhm = fit_fwhm(data, fwhm=3)
    assert isinstance(fwhm, np.ndarray)
    assert len(fwhm) == 1
    assert_allclose(fwhm, fwhm0)

    # test warning message
    match = 'may not have converged. Please carefully check your results'
    with pytest.warns(AstropyUserWarning, match=match):
        fwhm = fit_fwhm(np.zeros(data.shape) + 1)
    assert len(fwhm) == 1


@pytest.mark.parametrize('with_units', [False, True])
def test_fit_fwhm_multiple(test_data, with_units):
    data, sources = test_data

    unit = u.nJy
    if with_units:
        data = data * unit

    xypos = list(zip(sources['x_0'], sources['y_0'], strict=True))
    fwhms = fit_fwhm(data, xypos=xypos, fit_shape=(5, 5))
    assert isinstance(fwhms, np.ndarray)
    assert len(fwhms) == len(sources)
    assert_allclose(fwhms, sources['fwhm'])


def test_interpolate_missing_data():
    data = np.arange(100).reshape(10, 10)
    mask = np.zeros_like(data, dtype=bool)
    mask[5, 5] = True

    data_int = _interpolate_missing_data(data, mask, method='nearest')
    assert 54 <= data_int[5, 5] <= 56

    data_int = _interpolate_missing_data(data, mask, method='cubic')
    assert 54 <= data_int[5, 5] <= 56

    match = "'data' must be a 2D array."
    with pytest.raises(ValueError, match=match):
        _interpolate_missing_data(np.arange(10), mask)

    match = "'mask' and 'data' must have the same shape."
    with pytest.raises(ValueError, match=match):
        _interpolate_missing_data(data, mask[1:, :])

    match = 'Unsupported interpolation method'
    with pytest.raises(ValueError, match=match):
        _interpolate_missing_data(data, mask, method='invalid')


def test_validate_psf_model():
    model = np.arange(10)

    match = 'psf_model must be an Astropy Model subclass'
    with pytest.raises(TypeError, match=match):
        _validate_psf_model(model)

    match = 'psf_model must be two-dimensional'
    model = Gaussian1D()
    with pytest.raises(ValueError, match=match):
        _validate_psf_model(model)

    match = 'psf_model must be two-dimensional'
    model = Gaussian1D()
    with pytest.raises(ValueError, match=match):
        _validate_psf_model(model)


def test_get_psf_model_params():
    model = CircularGaussianPRF(fwhm=1.0)
    params = _get_psf_model_params(model)
    assert len(params) == 3
    assert params == ('x_0', 'y_0', 'flux')

    match = 'Invalid PSF model - could not find PSF parameter names'
    model = Gaussian2D()
    with pytest.raises(ValueError, match=match):
        _get_psf_model_params(model)

    set_params = ('x_mean', 'y_mean', 'amplitude')
    model.x_name = set_params[0]
    model.y_name = set_params[1]
    model.flux_name = set_params[2]
    params = _get_psf_model_params(model)
    assert len(params) == 3
    assert params == set_params
