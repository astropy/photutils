# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the model_helpers module.
"""

import astropy.units as u
import numpy as np
import pytest
from astropy.modeling.fitting import TRFLSQFitter
from astropy.modeling.models import Const2D, Gaussian2D, Moffat2D
from astropy.nddata import NDData
from astropy.table import Table
from astropy.units import Quantity
from numpy.testing import assert_allclose, assert_equal
from scipy.interpolate import RectBivariateSpline
from scipy.special import erf

from photutils import datasets
from photutils.detection import find_peaks
from photutils.psf import (EPSFBuilder, ImagePSF, extract_stars,
                           grid_from_epsfs, make_epsf_from_psf, make_psf_model)
from photutils.psf.model_helpers import _integrate_model, _InverseShift
from photutils.utils.exceptions import PhotutilsDeprecationWarning


def test_inverse_shift():
    model = _InverseShift(10)
    assert model(1) == -9.0
    assert model(-10) == -20.0
    assert model.fit_deriv(10, 1)[0] == -1.0


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
    model = Gaussian2D(1, np.inf, 5, 1, 1)
    with pytest.raises(ValueError, match=match):
        _integrate_model(model, x_name='x_mean', y_name='y_mean')


@pytest.fixture(name='moffat_source', scope='module')
def fixture_moffat_source():
    model = Moffat2D(alpha=4.8)

    # This is the analytic value needed to get a total flux of 1
    model.amplitude = (model.alpha - 1.0) / (np.pi * model.gamma**2)

    xx, yy = np.meshgrid(*([np.linspace(-2, 2, 100)] * 2))

    return model, (xx, yy, model(xx, yy))


def test_moffat_fitting(moffat_source):
    """
    Test fitting with a Moffat2D model.
    """
    model, (xx, yy, data) = moffat_source

    # Initial Moffat2D model close to the original
    guess_moffat = Moffat2D(x_0=0.1, y_0=-0.05, gamma=1.05,
                            amplitude=model.amplitude * 1.06, alpha=4.75)

    fitter = TRFLSQFitter()
    fit = fitter(guess_moffat, xx, yy, data)
    assert_allclose(fit.parameters, model.parameters, rtol=0.01, atol=0.0005)


# We set the tolerances in flux to be 2-3% because the guessed model
# parameters are known to be wrong
@pytest.mark.parametrize(('kwargs', 'tols'),
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
def test_make_psf_model(moffat_source, kwargs, tols):
    model, (xx, yy, data) = moffat_source

    # A close-but-wrong "guessed Moffat"
    guess_moffat = Moffat2D(x_0=0.1, y_0=-0.05, gamma=1.01,
                            amplitude=model.amplitude * 1.01, alpha=4.79)
    if kwargs['normalize']:
        # Definitely very wrong, so this ensures the renormalization
        # works
        guess_moffat.amplitude = 5.0

    if kwargs['x_name'] is None:
        guess_moffat.x_0 = 0
    if kwargs['y_name'] is None:
        guess_moffat.y_0 = 0

    psf_model = make_psf_model(guess_moffat, **kwargs)
    fitter = TRFLSQFitter()
    fit_model = fitter(psf_model, xx, yy, data)
    xytol, fluxtol = tols

    if xytol is not None:
        assert np.abs(getattr(fit_model, fit_model.x_name)) < xytol
        assert np.abs(getattr(fit_model, fit_model.y_name)) < xytol
    if fluxtol is not None:
        assert np.abs(1.0 - getattr(fit_model, fit_model.flux_name)) < fluxtol

    # Ensure the model parameters did not change
    assert fit_model[2].gamma == guess_moffat.gamma
    assert fit_model[2].alpha == guess_moffat.alpha
    if kwargs['flux_name'] is None:
        assert fit_model[2].amplitude == guess_moffat.amplitude


def test_make_psf_model_units():
    model = Moffat2D(amplitude=1.0 * u.Jy, x_0=25, y_0=25, alpha=4.8,
                     gamma=3.1)
    model.amplitude = (model.amplitude.unit * (model.alpha - 1.0)
                       / (np.pi * model.gamma**2))  # normalize to flux=1

    psf_model = make_psf_model(model, x_name='x_0', y_name='y_0',
                               normalize=True)
    yy, xx = np.mgrid[:51, :51]
    data1 = model(xx, yy)
    data2 = psf_model(xx, yy)
    assert_allclose(data1, data2)


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


def test_make_psf_model_invalid_flux_name():
    """
    Test that an invalid flux_name raises a clear ValueError.
    """
    match = 'parameter name not found in the input model'
    with pytest.raises(ValueError, match=match):
        make_psf_model(Moffat2D(), x_name='x_0', y_name='y_0',
                       flux_name='invalid')

    model = Gaussian2D(1, 5, 5, 1, 1) * Const2D(1.0)
    with pytest.raises(ValueError, match=match):
        make_psf_model(model, x_name='x_mean_0', y_name='y_mean_0',
                       flux_name='invalid')


def test_make_psf_model_integral():
    model = Gaussian2D(1, 5, 5, 1, 1) * Const2D(0.0)
    match = 'Cannot normalize the model because the integrated flux is zero'
    with pytest.raises(ValueError, match=match):
        make_psf_model(model, x_name='x_mean_0', y_name='y_mean_0',
                       normalize=True)


def test_make_psf_model_normalize_dx_dy():
    """
    Regression test that make_psf_model normalization works with
    dx != dy.
    """
    gauss = Gaussian2D(1, 0, 0, 1, 1)
    psf = make_psf_model(gauss, x_name='x_mean', y_name='y_mean',
                         dx=11, dy=21, subsample=10)
    yy, xx = np.mgrid[-10:11, -10:11]
    total = psf(xx, yy).sum()
    assert_allclose(total, 1.0, atol=1e-3)

    # The square default grid must still integrate a unit Gaussian
    # to 2 * pi
    model = Gaussian2D(1, 0, 0, 1, 1)
    integral = _integrate_model(model, x_name='x_mean', y_name='y_mean')
    assert_allclose(integral, 2.0 * np.pi, rtol=1e-6)


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
class TestGridFromEPSFs:
    """
    Tests for `photutils.psf.utils.grid_from_epsfs`.
    """

    def setup_class(self, *, cutout_size=25):
        # Make a set of 4 EPSF models

        self.cutout_size = cutout_size

        # Make simulated image
        hdu = datasets.load_simulated_hst_star_image()
        data = hdu.data

        # Break up the image into four quadrants
        q1 = data[0:500, 0:500]
        q2 = data[0:500, 500:1000]
        q3 = data[500:1000, 0:500]
        q4 = data[500:1000, 500:1000]

        # Select some starts from each quadrant to use to build the epsf
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

            # Filter out sources near edge
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
            epsf, _ = epsf_builder(stars)

            # Set x_0, y_0 to fiducial point
            epsf.y_0 = quad_stars[q]['fiducial'][0]
            epsf.x_0 = quad_stars[q]['fiducial'][1]

            quad_stars[q]['epsf'] = epsf

        self.epsfs = [val['epsf'] for val in quad_stars.values()]
        self.grid_xypos = [val['fiducial'] for val in quad_stars.values()]

    def test_basic_test_grid_from_epsfs(self):
        with pytest.warns(PhotutilsDeprecationWarning):
            psf_grid = grid_from_epsfs(self.epsfs)

        assert np.all(psf_grid.oversampling == self.epsfs[0].oversampling)
        assert psf_grid.data.shape == (4, psf_grid.oversampling[0] * 25 + 1,
                                       psf_grid.oversampling[1] * 25 + 1)

    def test_grid_xypos(self):
        """
        Test both options for setting PSF locations.
        """
        # Default option x_0 and y_0s on input EPSFs
        with pytest.warns(PhotutilsDeprecationWarning):
            psf_grid = grid_from_epsfs(self.epsfs)

        # meta stores the positions sorted by y and then by x
        assert_equal(psf_grid.meta['grid_xypos'],
                     [(0.0, 0.0), (1000.0, 0.0),
                      (0.0, 1000.0), (1000.0, 1000.0)])
        assert_equal(psf_grid.meta['grid_xypos'], psf_grid.grid_xypos)

        # Pass in a list
        grid_xypos = [(250.0, 250.0), (750.0, 750.0),
                      (250.0, 750.0), (750.0, 250.0)]

        with pytest.warns(PhotutilsDeprecationWarning):
            psf_grid = grid_from_epsfs(self.epsfs, grid_xypos=grid_xypos)
        assert_equal(psf_grid.meta['grid_xypos'],
                     [(250.0, 250.0), (750.0, 250.0),
                      (250.0, 750.0), (750.0, 750.0)])
        assert_equal(psf_grid.meta['grid_xypos'], psf_grid.grid_xypos)

    def test_meta(self):
        """
        Test the option for setting 'meta'.
        """
        keys = ['grid_xypos', 'oversampling', 'fill_value']

        # When 'meta' isn't provided, there should be just three keys
        with pytest.warns(PhotutilsDeprecationWarning):
            psf_grid = grid_from_epsfs(self.epsfs)
        for key in keys:
            assert key in psf_grid.meta

        # When meta is provided, those new keys should exist and
        # anything in the list above should be overwritten
        meta = {'grid_xypos': 0.0, 'oversampling': 0.0,
                'fill_value': -999, 'extra_key': 'extra'}
        with pytest.warns(PhotutilsDeprecationWarning):
            psf_grid = grid_from_epsfs(self.epsfs, meta=meta)
        for key in [*keys, 'extra_key']:
            assert key in psf_grid.meta
        assert_equal(psf_grid.meta['grid_xypos'],
                     [(0.0, 0.0), (1000.0, 0.0),
                      (0.0, 1000.0), (1000.0, 1000.0)])
        assert_equal(psf_grid.meta['oversampling'], [4, 4])
        assert psf_grid.meta['fill_value'] == 0.0


def _sampled_gaussian(sigma, oversampling, size):
    """
    Make a Gaussian PSF sampled at the points of an oversampled grid
    and the matching pixel-integrated ePSF.
    """
    oversampling = np.broadcast_to(oversampling, 2)
    profiles = []
    for factor in oversampling:
        n_points = size * factor + (size * factor + 1) % 2
        offsets = (np.arange(n_points) - n_points // 2) / factor
        sampled = np.exp(-offsets**2 / (2 * sigma**2))
        scale = np.sqrt(2) * sigma
        integrated = 0.5 * (erf((offsets + 0.5) / scale)
                            - erf((offsets - 0.5) / scale))
        profiles.append((sampled * factor / sampled.sum(), integrated))
    psf = np.outer(profiles[0][0], profiles[1][0])
    epsf = np.outer(profiles[0][1], profiles[1][1])
    return psf, epsf


class TestMakeEPSFFromPSF:
    @pytest.mark.parametrize(('oversampling', 'atol', 'sum_rtol'),
                             [(1, 2e-2, 1e-3), (2, 3e-3, 1e-6),
                              (4, 3e-4, 1e-6), (5, 3e-4, 1e-6),
                              ((3, 4), 3e-4, 1e-6)])
    def test_gaussian(self, oversampling, atol, sum_rtol):
        """
        Test that the result matches the analytic pixel-integrated
        Gaussian and preserves the normalization.

        The accuracy is set by how well the spline through the samples
        represents the PSF, so it improves with the oversampling.
        """
        psf, expected = _sampled_gaussian(0.8, oversampling, 15)
        result = make_epsf_from_psf(psf, oversampling=oversampling)
        assert result.shape == psf.shape
        assert_allclose(result.sum(), psf.sum(), rtol=sum_rtol)
        assert_allclose(result, expected, atol=atol * expected.max())

    def test_spline_integral(self):
        """
        Test the result against the integral of the bicubic spline,
        including the truncated windows at the image edges.
        """
        rng = np.random.default_rng(0)
        data = rng.random((9, 12))
        result = make_epsf_from_psf(data, oversampling=(2, 3))
        yy = np.arange(9.0)
        xx = np.arange(12.0)
        spline = RectBivariateSpline(yy, xx, data, kx=3, ky=3, s=0)
        for idx_y, idx_x in [(0, 0), (4, 6), (8, 11), (1, 10)]:
            expected = spline.integral(max(idx_y - 1.0, 0),
                                       min(idx_y + 1.0, 8),
                                       max(idx_x - 1.5, 0),
                                       min(idx_x + 1.5, 11)) / 6
            assert_allclose(result[idx_y, idx_x], expected, rtol=1e-10)

    @pytest.mark.parametrize('oversampling', [4, 5])
    def test_flux_conservation(self, oversampling):
        """
        Test that a model of an undersampled PSF conserves flux at any
        subpixel position only after the pixel integration.
        """
        psf, _ = _sampled_gaussian(0.42, oversampling, 15)
        epsf = make_epsf_from_psf(psf, oversampling=oversampling)
        yy, xx = np.mgrid[-10:11, -10:11]
        offsets = [(0, 0), (0.5, 0), (0.5, 0.5), (0.37, -0.13)]
        sums = {}
        for name, data in (('psf', psf), ('epsf', epsf)):
            model = ImagePSF(data, oversampling=oversampling)
            sums[name] = []
            for x_0, y_0 in offsets:
                model.x_0 = x_0
                model.y_0 = y_0
                sums[name].append(model(xx, yy).sum())
        assert_allclose(sums['epsf'], 1.0, atol=2e-5)
        assert np.ptp(sums['psf']) > 0.2

    def test_stack(self):
        """
        Test that a 3D stack is integrated image by image.
        """
        psf1, _ = _sampled_gaussian(0.6, 3, 9)
        psf2, _ = _sampled_gaussian(1.1, 3, 9)
        result = make_epsf_from_psf(np.array([psf1, psf2]),
                                    oversampling=3)
        assert result.shape == (2, *psf1.shape)
        for image, psf in zip(result, (psf1, psf2), strict=True):
            expected = make_epsf_from_psf(psf, oversampling=3)
            assert_allclose(image, expected, rtol=1e-12, atol=1e-15)

    @pytest.mark.parametrize('oversampling', [2, 4, (4, 2)])
    def test_midpoints_gaussian(self, oversampling):
        """
        Test that ``midpoints=True`` puts the center of a PSF that is
        centered on an even-sized image on the central grid point of
        the output.
        """
        sigma = 0.8
        scale = np.sqrt(2) * sigma
        psf_profiles = []
        epsf_profiles = []
        for factor in np.broadcast_to(oversampling, 2):
            n_points = 16 * factor
            offsets = (np.arange(n_points) - (n_points - 1) / 2) / factor
            sampled = np.exp(-offsets**2 / (2 * sigma**2))
            psf_profiles.append(sampled * factor / sampled.sum())
            midpoints = offsets[:-1] + 0.5 / factor
            epsf_profiles.append(0.5 * (erf((midpoints + 0.5) / scale)
                                        - erf((midpoints - 0.5) / scale)))
        psf = np.outer(*psf_profiles)
        expected = np.outer(*epsf_profiles)

        result = make_epsf_from_psf(psf, oversampling=oversampling,
                                    midpoints=True)
        assert result.shape == (psf.shape[0] - 1, psf.shape[1] - 1)
        assert result.shape[0] % 2 == 1
        assert result.shape[1] % 2 == 1
        assert result.argmax() == result.size // 2
        assert_allclose(result, result[::-1, ::-1], rtol=1e-10, atol=1e-15)
        assert_allclose(result.sum(), psf.sum(), rtol=1e-6)
        atol = 3e-3 if np.min(oversampling) == 2 else 3e-4
        assert_allclose(result, expected, atol=atol * expected.max())

    def test_midpoints_spline_integral(self):
        """
        Test the ``midpoints=True`` result against the integral of the
        bicubic spline over the pixels centered between the grid
        points, including the truncated windows at the image edges.
        """
        rng = np.random.default_rng(0)
        data = rng.random((9, 12))
        result = make_epsf_from_psf(data, oversampling=(2, 3),
                                    midpoints=True)
        assert result.shape == (8, 11)
        yy = np.arange(9.0)
        xx = np.arange(12.0)
        spline = RectBivariateSpline(yy, xx, data, kx=3, ky=3, s=0)
        for idx_y, idx_x in [(0, 0), (4, 6), (7, 10), (1, 9)]:
            expected = spline.integral(max(idx_y - 0.5, 0),
                                       min(idx_y + 1.5, 8),
                                       max(idx_x - 1.0, 0),
                                       min(idx_x + 2.0, 11)) / 6
            assert_allclose(result[idx_y, idx_x], expected, rtol=1e-10)

    def test_midpoints_stack(self):
        """
        Test ``midpoints=True`` for a 3D stack.
        """
        rng = np.random.default_rng(0)
        data = rng.random((3, 10, 8))
        result = make_epsf_from_psf(data, oversampling=2, midpoints=True)
        assert result.shape == (3, 9, 7)
        for image, psf in zip(result, data, strict=True):
            expected = make_epsf_from_psf(psf, oversampling=2,
                                          midpoints=True)
            assert_allclose(image, expected, rtol=1e-12, atol=1e-15)

    def test_midpoints_model(self):
        """
        Test that models made from the ePSFs on the two grids agree
        and that both conserve flux.
        """
        oversampling = 4
        offsets = (np.arange(80) - 39.5) / oversampling
        sampled = np.exp(-offsets**2 / (2 * 0.6**2))
        sampled *= oversampling / sampled.sum()
        psf = np.outer(sampled, sampled)
        yy, xx = np.mgrid[-7:8, -7:8]
        images = []
        for midpoints in (False, True):
            epsf = make_epsf_from_psf(psf, oversampling=oversampling,
                                      midpoints=midpoints)
            model = ImagePSF(epsf, oversampling=oversampling, x_0=0.3,
                             y_0=-0.2)
            images.append(model(xx, yy))
            assert_allclose(images[-1].sum(), 1.0, atol=2e-5)
        assert_allclose(images[0], images[1], atol=1e-3 * images[0].max())

    def test_input_unchanged(self):
        psf, _ = _sampled_gaussian(0.8, 2, 9)
        psf_orig = psf.copy()
        make_epsf_from_psf(psf, oversampling=2)
        assert_equal(psf, psf_orig)

    def test_units_dropped(self):
        """
        Test that the units of a Quantity input are dropped.
        """
        psf, _ = _sampled_gaussian(0.8, 2, 9)
        result = make_epsf_from_psf(Quantity(psf, 'Jy'), oversampling=2)
        assert not isinstance(result, Quantity)
        assert_equal(result, make_epsf_from_psf(psf, oversampling=2))

    def test_masked_input(self):
        """
        Test that a masked array is allowed only if no values are
        masked.
        """
        psf, _ = _sampled_gaussian(0.8, 2, 9)
        expected = make_epsf_from_psf(psf, oversampling=2)
        data = np.ma.MaskedArray(psf, mask=np.zeros(psf.shape, dtype=bool))
        assert_equal(make_epsf_from_psf(data, oversampling=2), expected)

        data.mask[4, 4] = True
        match = 'data must not have masked values'
        with pytest.raises(ValueError, match=match):
            make_epsf_from_psf(data, oversampling=2)

    def test_invalid_inputs(self):
        match = 'data must be a 2D or 3D array'
        with pytest.raises(ValueError, match=match):
            make_epsf_from_psf(np.ones(10), oversampling=2)

        match = 'must both be at least 4'
        with pytest.raises(ValueError, match=match):
            make_epsf_from_psf(np.ones((3, 10)), oversampling=2)

        data = np.ones((10, 10))
        data[4, 4] = np.nan
        match = 'All elements of data must be finite'
        with pytest.raises(ValueError, match=match):
            make_epsf_from_psf(data, oversampling=2)

        match = 'oversampling must be > 0'
        with pytest.raises(ValueError, match=match):
            make_epsf_from_psf(np.ones((10, 10)), oversampling=0)
