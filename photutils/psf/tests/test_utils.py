# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the utils module.
"""

import numpy as np
import pytest
from astropy.modeling.models import Gaussian1D, Gaussian2D

from photutils.psf import CircularGaussianPSF
from photutils.psf.utils import (_get_psf_model_params,
                                 _interpolate_missing_data,
                                 _validate_psf_model)
from photutils.utils._optional_deps import HAS_SCIPY


@pytest.mark.skipif(not HAS_SCIPY, reason='scipy is required')
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
    model = CircularGaussianPSF(fwhm=1.0)
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
