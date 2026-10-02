# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the copy methods of the PSF image models.
"""

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
from astropy.modeling.fitting import TRFLSQFitter
from astropy.modeling.models import Const2D
from numpy.testing import assert_allclose, assert_equal

from photutils.psf import ImagePSF

# The mutable containers that astropy keeps on a model instance and that
# a model copy may share with the original model. They are filled when
# the model is created and are only read afterward.
ASTROPY_SHARED_ATTRIBUTES = frozenset({'_param_metrics',
                                       '_input_units_strict',
                                       '_input_units_allow_dimensionless'})


@pytest.fixture(name='image_psf')
def fixture_image_psf():
    yy, xx = np.mgrid[0:25, 0:25]
    data = np.exp(-((xx - 12.0)**2 + (yy - 12.0)**2) / 8.0)
    return ImagePSF(data, flux=1.0, x_0=0.0, y_0=0.0, origin=(12.0, 12.0))


@pytest.fixture(name='model_case', params=['ImagePSF', 'GriddedPSFModel'])
def fixture_model_case(request):
    """
    Fixture that returns a PSF image model and the range of the x and
    y positions of the sources to fit with it.
    """
    if request.param == 'ImagePSF':
        return request.getfixturevalue('image_psf'), (8.0, 40.0)
    return request.getfixturevalue('psfmodel'), (20.0, 180.0)


def test_copy_isolates_fit_state(model_case):
    """
    Test that copies share the image data but not astropy's parameter
    array, constraints, and constraints cache, which the fitters update
    in place.
    """
    model, _ = model_case
    copy1 = model.copy()
    copy2 = model.copy()
    assert copy1.data is model.data
    for name in ('_parameters', '_mconstraints', '_constraints_cache'):
        assert getattr(copy1, name) is not getattr(model, name), name
        assert getattr(copy1, name) is not getattr(copy2, name), name
    assert copy1._mconstraints == model._mconstraints
    for name in model.param_names:
        assert getattr(copy1, name) is not getattr(model, name)
        # A parameter passes its model to the parameter validator
        assert getattr(copy1, name).model is copy1
        assert getattr(model, name).model is model

    # Any other mutable container that astropy keeps on the instance
    # and that the copies share must be known to be safe to share. This
    # fails if astropy adds new mutable state to the model instances.
    astropy_keys = set(Const2D().__dict__)
    shared = {key for key, val in model.__dict__.items()
              if key in astropy_keys
              and isinstance(val, (dict, list, set, np.ndarray))
              and copy1.__dict__[key] is val}
    assert shared <= ASTROPY_SHARED_ATTRIBUTES, shared

    copy1.parameters = copy1.parameters + 10.0
    copy1.x_0.bounds = (1.0, 3.0)
    copy1.flux.fixed = True
    copy1.tied  # noqa: B018, fills the constraints cache
    assert_equal(copy2.parameters, model.parameters)
    assert copy2.x_0.bounds == model.x_0.bounds
    assert not copy2.flux.fixed
    assert not model.flux.fixed


@pytest.mark.usefixtures('short_switch_interval')
def test_concurrent_fits_of_copies(model_case):
    """
    Test that copies fitted at the same time in several threads give
    the same results as fitting them one after the other.
    """
    model, position_range = model_case
    rng = np.random.default_rng(0)
    n_fits = 24
    positions = rng.uniform(*position_range, (n_fits, 2))
    fluxes = rng.uniform(50.0, 500.0, n_fits)
    yy, xx = np.mgrid[0:7, 0:7]

    def fit(index):
        x0, y0 = positions[index]
        truth = model.copy()
        truth.flux = fluxes[index]
        truth.x_0 = x0 + 0.3
        truth.y_0 = y0 - 0.2
        x = xx + int(x0) - 3
        y = yy + int(y0) - 3
        data = truth(x, y)
        guess = model.copy()
        guess.flux = fluxes[index] * 0.8
        guess.x_0 = x0
        guess.y_0 = y0
        guess.x_0.bounds = (x0 - 1.0, x0 + 1.0)
        guess.y_0.bounds = (y0 - 1.0, y0 + 1.0)
        fitter = TRFLSQFitter()
        fitted = fitter(guess, x, y, data, maxiter=50)
        return np.array([fitted.flux.value, fitted.x_0.value,
                         fitted.y_0.value])

    serial = np.array([fit(i) for i in range(n_fits)])
    with ThreadPoolExecutor(max_workers=8) as executor:
        threaded = np.array(list(executor.map(fit, range(n_fits))))
    assert_allclose(threaded, serial, rtol=1e-12)
    assert_allclose(serial[:, 0], fluxes, rtol=1e-6)
