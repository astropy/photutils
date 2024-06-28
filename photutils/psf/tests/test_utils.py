# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the utils module.
"""

import numpy as np
import pytest
from astropy.convolution.utils import discretize_model
from astropy.modeling.fitting import LevMarLSQFitter
from astropy.modeling.models import Const2D, Gaussian2D, Moffat2D
from astropy.nddata import NDData
from astropy.table import Table
from numpy.testing import assert_allclose, assert_equal

from photutils import datasets
from photutils.detection import find_peaks
from photutils.psf import (EPSFBuilder, IntegratedGaussianPRF, extract_stars,
                           grid_from_epsfs, make_psf_model,
                           make_psf_model_image)
from photutils.psf.utils import (_integrate_model, _interpolate_missing_data,
                                 _InverseShift)
from photutils.utils._optional_deps import HAS_SCIPY

PSF_SIZE = 11
GAUSSIAN_WIDTH = 1.0
IMAGE_SIZE = 101

# Position and FLUXES of test sources
SOURCES = Table([[50.0, 23, 12, 86], [50.0, 83, 80, 84],
                 [np.pi * 10, 3.654, 20.0, 80 / np.sqrt(3)]],
                names=['x_0', 'y_0', 'flux_0'])

# Create test psf
psf_model = Gaussian2D(1.0 / (2 * np.pi * GAUSSIAN_WIDTH ** 2), PSF_SIZE // 2,
                       PSF_SIZE // 2, GAUSSIAN_WIDTH, GAUSSIAN_WIDTH)
test_psf = discretize_model(psf_model, (0, PSF_SIZE), (0, PSF_SIZE),
                            mode='oversample')

# Set up grid for test image
image = np.zeros((IMAGE_SIZE, IMAGE_SIZE))

# Add sources to test image
for x, y, flux in SOURCES:
    model = Gaussian2D(flux / (2 * np.pi * GAUSSIAN_WIDTH ** 2),
                       x, y, GAUSSIAN_WIDTH, GAUSSIAN_WIDTH)
    image += discretize_model(model, (0, IMAGE_SIZE), (0, IMAGE_SIZE),
                              mode='oversample')


def test_InverseShift():
    model = _InverseShift(10)
    assert model(1) == -9.0
    assert model(-10) == -20.0
    assert model.fit_deriv(10)[0] == -1.0


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


@pytest.mark.skipif(not HAS_SCIPY, reason='scipy is required')
def test_integrate_model():
    model = Gaussian2D(1, 5, 5, 1, 1) * Const2D(0.0)
    integral = _integrate_model(model, x_name='x_mean_0', y_name='y_mean_0')
    assert integral == 0.0

    integral = _integrate_model(model, x_name='x_mean_0', y_name='y_mean_0',
                                use_dblquad=True)
    assert integral == 0.0

    match = 'dx and dy must be > 0'
    with pytest.raises(ValueError, match=match):
        _integrate_model(model, x_name='x_mean_0', y_name='y_mean_0',
                         dx=-10, dy=10)
    with pytest.raises(ValueError, match=match):
        _integrate_model(model, x_name='x_mean_0', y_name='y_mean_0',
                         dx=10, dy=-10)

    match = 'subsample must be >= 1'
    with pytest.raises(ValueError, match=match):
        _integrate_model(model, x_name='x_mean_0', y_name='y_mean_0',
                         subsample=-1)

    match = 'model x and y positions must be finite'
    with pytest.raises(ValueError, match=match):
        model = Gaussian2D(1, np.inf, 5, 1, 1)
        _integrate_model(model, x_name='x_mean', y_name='y_mean')


@pytest.fixture(scope='module')
def moffat_source():
    model = Moffat2D(alpha=4.8)

    # this is the analytic value needed to get a total flux of 1
    model.amplitude = (model.alpha - 1.0) / (np.pi * model.gamma**2)

    # make sure it really is normalized
    # assert (1.0 - integrate.dblquad(model, -10, 10, -10, 10)[0]) < 1e-6

    xx, yy = np.meshgrid(*([np.linspace(-2, 2, 100)] * 2))

    return model, (xx, yy, model(xx, yy))


@pytest.mark.skipif(not HAS_SCIPY, reason='scipy is required')
def test_moffat_fitting(moffat_source):
    """
    Test fitting with a Moffat2D model.
    """
    model, (xx, yy, data) = moffat_source

    # initial Moffat2D model close to the original
    guess_moffat = Moffat2D(x_0=0.1, y_0=-0.05, gamma=1.05,
                            amplitude=model.amplitude * 1.06, alpha=4.75)

    fitter = LevMarLSQFitter()
    fit = fitter(guess_moffat, xx, yy, data)
    assert_allclose(fit.parameters, model.parameters, rtol=0.01, atol=0.0005)


# we set the tolerances in flux to be 2-3% because the guessed model
# parameters are known to be wrong
@pytest.mark.parametrize('kwargs, tols',
                         [({'x_name': 'x_0', 'y_name': 'y_0',
                            'flux_name': None, 'normalize': True},
                           (1e-3, 0.02)),
                          ({'x_name': None, 'y_name': None, 'flux_name': None,
                            'normalize': True}, (1e-3, 0.02)),
                          ({'x_name': None, 'y_name': None, 'flux_name': None,
                            'normalize': False}, (1e-3, 0.03)),
                          ({'x_name': 'x_0', 'y_name': 'y_0',
                            'flux_name': 'amplitude', 'normalize': False},
                           (1e-3, None))])
@pytest.mark.skipif(not HAS_SCIPY, reason='scipy is required')
def test_make_psf_model(moffat_source, kwargs, tols):
    model, (xx, yy, data) = moffat_source

    # a close-but-wrong "guessed Moffat"
    guess_moffat = Moffat2D(x_0=0.1, y_0=-0.05, gamma=1.01,
                            amplitude=model.amplitude * 1.01, alpha=4.79)
    if kwargs['normalize']:
        # definitely very wrong, so this ensures the re-normalization
        # works
        guess_moffat.amplitude = 5.0

    if kwargs['x_name'] is None:
        guess_moffat.x_0 = 0
    if kwargs['y_name'] is None:
        guess_moffat.y_0 = 0

    psf_model = make_psf_model(guess_moffat, **kwargs)
    fitter = LevMarLSQFitter()
    fit_model = fitter(psf_model, xx, yy, data)
    xytol, fluxtol = tols

    if xytol is not None:
        assert np.abs(getattr(fit_model, fit_model.x_name)) < xytol
        assert np.abs(getattr(fit_model, fit_model.y_name)) < xytol
    if fluxtol is not None:
        assert np.abs(1.0 - getattr(fit_model, fit_model.flux_name)) < fluxtol

    # ensure the model parameters did not change
    assert fit_model[2].gamma == guess_moffat.gamma
    assert fit_model[2].alpha == guess_moffat.alpha
    if kwargs['flux_name'] is None:
        assert fit_model[2].amplitude == guess_moffat.amplitude


@pytest.mark.skipif(not HAS_SCIPY, reason='scipy is required')
def test_make_psf_model_compound():
    model = (Const2D(0.0) + Const2D(1.0) + Gaussian2D(1, 5, 5, 1, 1)
             * Const2D(1.0) * Const2D(1.0))
    psf_model = make_psf_model(model, x_name='x_mean_2', y_name='y_mean_2',
                               normalize=True)
    assert psf_model.x_name == 'x_mean_4'
    assert psf_model.y_name == 'y_mean_4'
    assert psf_model.flux_name == 'amplitude_7'


def test_make_psf_model_inputs():
    model = Gaussian2D(1, 5, 5, 1, 1)
    match = 'parameter name not found in the input model'
    with pytest.raises(ValueError, match=match):
        make_psf_model(model, x_name='x_mean_0', y_name='y_mean')
    with pytest.raises(ValueError, match=match):
        make_psf_model(model, x_name='x_mean', y_name='y_mean_10')


@pytest.mark.skipif(not HAS_SCIPY, reason='scipy is required')
def test_make_psf_model_integral():
    model = Gaussian2D(1, 5, 5, 1, 1) * Const2D(0.0)
    match = 'Cannot normalize the model because the integrated flux is zero'
    with pytest.raises(ValueError, match=match):
        make_psf_model(model, x_name='x_mean_0', y_name='y_mean_0',
                       normalize=True)


def test_make_psf_model_offset():
    """
    Test to ensure the offset is in the correct direction.
    """
    moffat = Moffat2D(x_0=0, y_0=0, alpha=4.8)
    psfmod1 = make_psf_model(moffat.copy(), x_name='x_0', y_name='y_0',
                             normalize=False)
    psfmod2 = make_psf_model(moffat.copy(), normalize=False)
    moffat.x_0 = 10
    psfmod1.x_0_2 = 10
    psfmod2.offset_0 = 10

    assert moffat(10, 0) == psfmod1(10, 0) == psfmod2(10, 0) == 1.0


@pytest.mark.remote_data
@pytest.mark.skipif(not HAS_SCIPY, reason='scipy is required')
class TestGridFromEPSFs:
    """
    Tests for `photutils.psf.utils.grid_from_epsfs`.
    """

    def setup_class(self, cutout_size=25):
        # make a set of 4 EPSF models

        self.cutout_size = cutout_size

        # make simulated image
        hdu = datasets.load_simulated_hst_star_image()
        data = hdu.data

        # break up the image into four quadrants
        q1 = data[0:500, 0:500]
        q2 = data[0:500, 500:1000]
        q3 = data[500:1000, 0:500]
        q4 = data[500:1000, 500:1000]

        # select some starts from each quadrant to use to build the epsf
        quad_stars = {'q1': {'data': q1, 'fiducial': (0., 0.), 'epsf': None},
                      'q2': {'data': q2, 'fiducial': (1000., 1000.),
                             'epsf': None},
                      'q3': {'data': q3, 'fiducial': (1000., 0.),
                             'epsf': None},
                      'q4': {'data': q4, 'fiducial': (0., 1000.),
                             'epsf': None}}

        for q in ['q1', 'q2', 'q3', 'q4']:
            quad_data = quad_stars[q]['data']
            peaks_tbl = find_peaks(quad_data, threshold=500.)

            # filter out sources near edge
            size = cutout_size
            hsize = (size - 1) / 2
            x = peaks_tbl['x_peak']
            y = peaks_tbl['y_peak']
            mask = ((x > hsize) & (x < (quad_data.shape[1] - 1 - hsize))
                    & (y > hsize) & (y < (quad_data.shape[0] - 1 - hsize)))

            stars_tbl = Table()
            stars_tbl['x'] = peaks_tbl['x_peak'][mask]
            stars_tbl['y'] = peaks_tbl['y_peak'][mask]

            stars = extract_stars(NDData(quad_data), stars_tbl,
                                  size=cutout_size)

            epsf_builder = EPSFBuilder(oversampling=4, maxiters=3,
                                       progress_bar=False)
            epsf, fitted_stars = epsf_builder(stars)

            # set x_0, y_0 to fiducial point
            epsf.y_0 = quad_stars[q]['fiducial'][0]
            epsf.x_0 = quad_stars[q]['fiducial'][1]

            quad_stars[q]['epsf'] = epsf

        self.epsfs = [quad_stars[x]['epsf'] for x in quad_stars]
        self.grid_xypos = [quad_stars[x]['fiducial'] for x in quad_stars]

    def test_basic_test_grid_from_epsfs(self):
        psf_grid = grid_from_epsfs(self.epsfs)

        assert np.all(psf_grid.oversampling == self.epsfs[0].oversampling)
        assert psf_grid.data.shape == (4, psf_grid.oversampling[0] * 25 + 1,
                                       psf_grid.oversampling[1] * 25 + 1)

    def test_grid_xypos(self):
        """
        Test both options for setting PSF locations.
        """
        # default option x_0 and y_0s on input EPSFs
        psf_grid = grid_from_epsfs(self.epsfs)

        assert psf_grid.meta['grid_xypos'] == [(0.0, 0.0), (1000.0, 1000.0),
                                               (0.0, 1000.0), (1000.0, 0.0)]

        # or pass in a list
        grid_xypos = [(250.0, 250.0), (750.0, 750.0),
                      (250.0, 750.0), (750.0, 250.0)]

        psf_grid = grid_from_epsfs(self.epsfs, grid_xypos=grid_xypos)
        assert psf_grid.meta['grid_xypos'] == grid_xypos

    def test_meta(self):
        """
        Test the option for setting 'meta'.
        """
        keys = ['grid_xypos', 'oversampling', 'fill_value']

        # when 'meta' isn't provided, there should be just three keys
        psf_grid = grid_from_epsfs(self.epsfs)
        for key in keys:
            assert key in psf_grid.meta

        # when meta is provided, those new keys should exist and anything
        # in the list above should be overwritten
        meta = {'grid_xypos': 0.0, 'oversampling': 0.0,
                'fill_value': -999, 'extra_key': 'extra'}
        psf_grid = grid_from_epsfs(self.epsfs, meta=meta)
        for key in keys + ['extra_key']:
            assert key in psf_grid.meta
        assert psf_grid.meta['grid_xypos'].sort() == self.grid_xypos.sort()
        assert_equal(psf_grid.meta['oversampling'], [4, 4])
        assert psf_grid.meta['fill_value'] == 0.0


@pytest.mark.skipif(not HAS_SCIPY, reason='scipy is required')
def test_make_psf_model_image():
    shape = (401, 451)
    n_sources = 100
    model = IntegratedGaussianPRF(sigma=2.7 / 2.35)
    data, params = make_psf_model_image(shape, model, n_sources)
    assert data.shape == shape
    assert isinstance(params, Table)
    assert len(params) == n_sources

    model_shape = (13, 13)
    data2, params2 = make_psf_model_image(shape, model, n_sources,
                                          model_shape=model_shape)
    assert_equal(data, data2)
    assert len(params2) == n_sources

    flux = (100, 200)
    sigma = (1, 2)
    alpha = (0, 1)
    n_sources = 10
    data, params = make_psf_model_image(shape, model, n_sources,
                                        seed=0, flux=flux, sigma=sigma,
                                        alpha=alpha)
    assert len(params) == n_sources
    colnames = ('id', 'x_0', 'y_0', 'flux', 'sigma')
    for colname in colnames:
        assert colname in params.colnames
    assert 'alpha' not in params.colnames
    assert np.min(params['flux']) >= flux[0]
    assert np.max(params['flux']) <= flux[1]
    assert np.min(params['sigma']) >= sigma[0]
    assert np.max(params['sigma']) <= sigma[1]


@pytest.mark.skipif(not HAS_SCIPY, reason='scipy is required')
def test_make_psf_model_image_custom():
    shape = (401, 451)
    n_sources = 100
    model = Gaussian2D()
    psf_model = make_psf_model(model, x_name='x_mean', y_name='y_mean')
    data, params = make_psf_model_image(shape, psf_model, n_sources,
                                        model_shape=(11, 11))
    assert data.shape == shape
    assert isinstance(params, Table)
    assert len(params) == n_sources


@pytest.mark.skipif(not HAS_SCIPY, reason='scipy is required')
def test_make_psf_model_image_inputs():
    shape = (50, 50)
    match = 'psf_model must be an Astropy Model subclass'
    with pytest.raises(TypeError, match=match):
        make_psf_model_image(shape, None, 2)

    match = 'psf_model must be two-dimensional'
    with pytest.raises(ValueError, match=match):
        model = IntegratedGaussianPRF(sigma=2.7 / 2.35)
        model.n_inputs = 3
        make_psf_model_image(shape, model, 2)

    match = 'model_shape must be specified if the model does not have'
    with pytest.raises(ValueError, match=match):
        model = IntegratedGaussianPRF(sigma=2.7 / 2.35)
        model.bounding_box = None
        make_psf_model_image(shape, model, 2)

    match = 'Invalid PSF model - could not find PSF parameter names'
    with pytest.raises(ValueError, match=match):
        model = Gaussian2D()
        make_psf_model_image(shape, model, 2)
