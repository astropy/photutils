# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the image_models module.
"""

import copy
import pickle  # nosec B403
from functools import cached_property

import astropy.units as u
import numpy as np
import pytest
from astropy.modeling.fitting import TRFLSQFitter
from astropy.table import QTable
from astropy.utils.exceptions import AstropyDeprecationWarning
from numpy.testing import assert_allclose, assert_equal
from scipy.interpolate import RectBivariateSpline

from photutils.datasets import make_model_image
from photutils.psf import CircularGaussianPSF, ImagePSF


@pytest.fixture(name='gaussian_psf')
def fixture_gaussian_psf():
    return CircularGaussianPSF(fwhm=2.1)


@pytest.fixture(name='image_psf')
def fixture_image_psf(gaussian_psf):
    yy, xx = np.mgrid[-10:11, -10:11]
    psf_data = gaussian_psf(xx, yy)
    psf_data /= np.sum(psf_data)
    return ImagePSF(psf_data)


class TestImagePSF:

    def test_imagepsf(self, gaussian_psf):
        yy, xx = np.mgrid[-10:11, -10:11]
        psf_data = gaussian_psf(xx, yy)
        psf_data /= np.sum(psf_data)
        model = ImagePSF(psf_data)

        assert_allclose(model(xx, yy), gaussian_psf(xx, yy), atol=1e-6)

        # Subpixel should not match, but be reasonably close
        for x, y in [(0.5, 0.5), (-0.5, 1.75)]:
            assert_allclose(model(x, y), gaussian_psf(x, y), atol=4e-3)

    @pytest.mark.parametrize('oversampling', [1, 2, (2, 3)])
    @pytest.mark.parametrize('origin', [None, (11.0, 13.0)])
    def test_fit_deriv(self, oversampling, origin):
        gaussian_psf = CircularGaussianPSF(flux=1, x_0=12, y_0=12, fwhm=3.5)
        yy, xx = np.mgrid[0:25, 0:25].astype(float)
        psf_data = gaussian_psf(xx, yy)

        flux, x_0, y_0 = 2.0, 12.3, 11.7
        model = ImagePSF(psf_data, flux=flux, x_0=x_0, y_0=y_0,
                         oversampling=oversampling, origin=origin)

        # Evaluation points covering the valid model domain, with a
        # margin so that the central differences below do not cross
        # the fill_value boundary
        ny, nx = psf_data.shape
        margin = 0.5
        x_lo = x_0 - model.origin[0] / model.oversampling[1] + margin
        x_hi = (x_0 + (nx - 1 - model.origin[0]) / model.oversampling[1]
                - margin)
        y_lo = y_0 - model.origin[1] / model.oversampling[0] + margin
        y_hi = (y_0 + (ny - 1 - model.origin[1]) / model.oversampling[0]
                - margin)
        x, y = np.meshgrid(np.linspace(x_lo, x_hi, 15),
                           np.linspace(y_lo, y_hi, 15))
        x = x.ravel()
        y = y.ravel()

        d_flux, d_x_0, d_y_0 = model.fit_deriv(x, y, flux, x_0, y_0)

        eps = 1e-6

        def ev(f, a, b):
            return model.evaluate(x, y, f, a, b)

        num_flux = (ev(flux + eps, x_0, y_0)
                    - ev(flux - eps, x_0, y_0)) / (2 * eps)
        num_x_0 = (ev(flux, x_0 + eps, y_0)
                   - ev(flux, x_0 - eps, y_0)) / (2 * eps)
        num_y_0 = (ev(flux, x_0, y_0 + eps)
                   - ev(flux, x_0, y_0 - eps)) / (2 * eps)

        assert_allclose(d_flux, num_flux, atol=1e-8)
        assert_allclose(d_x_0, num_x_0, atol=1e-7)
        assert_allclose(d_y_0, num_y_0, atol=1e-7)

    def test_fit_deriv_out_of_bounds(self):
        gaussian_psf = CircularGaussianPSF(flux=1, x_0=5, y_0=5, fwhm=2.0)
        yy, xx = np.mgrid[0:11, 0:11].astype(float)
        psf_data = gaussian_psf(xx, yy)
        model = ImagePSF(psf_data, flux=1.0, x_0=5.0, y_0=5.0)

        # Include positions that map outside the input pixel grid
        x = np.array([5.0, -50.0, 100.0])
        y = np.array([5.0, 5.0, 5.0])
        derivs = model.fit_deriv(x, y, 1.0, 5.0, 5.0)

        # All derivatives must be zero outside the input pixel grid
        for deriv in derivs:
            assert_equal(deriv[1:], 0.0)
        # The flux derivative is nonzero at the in-bounds peak position
        assert derivs[0][0] > 0.0

    def test_fit_deriv_no_fill_value(self):
        """
        Test the derivatives with fill_value=None, where a point outside
        the image takes the spline value at the nearest point on the
        image edge. The model then does not change with the position
        along an axis on which the point is outside the image.
        """
        yy, xx = np.mgrid[0:25, 0:25].astype(float)
        psf_data = (np.exp(-((xx - 12.0)**2 + (yy - 12.0)**2) / 200.0)
                    * (1.0 + 0.05 * xx + 0.03 * yy))
        flux, x_0, y_0 = 2.0, 0.3, -0.2
        model = ImagePSF(psf_data, origin=(12.0, 12.0), fill_value=None)

        # The points are outside the image along x, along y, along
        # both, along neither, and along x on the other side
        x = np.array([14.0, 3.0, 20.0, 1.0, -15.0])
        y = np.array([0.0, 15.0, 20.0, 2.0, 3.0])
        d_flux, d_x_0, d_y_0 = model.fit_deriv(x, y, flux, x_0, y_0)
        assert np.all(d_flux > 0.0)
        assert np.all(d_x_0[[0, 2, 4]] == 0.0)
        assert np.all(d_y_0[[1, 2]] == 0.0)
        assert np.all(d_x_0[[1, 3]] != 0.0)
        assert np.all(d_y_0[[0, 3, 4]] != 0.0)

        eps = 1e-6

        def ev(a, b):
            return model.evaluate(x, y, flux, a, b)

        num_x_0 = (ev(x_0 + eps, y_0) - ev(x_0 - eps, y_0)) / (2 * eps)
        num_y_0 = (ev(x_0, y_0 + eps) - ev(x_0, y_0 - eps)) / (2 * eps)
        assert_allclose(d_x_0, num_x_0, atol=1e-7)
        assert_allclose(d_y_0, num_y_0, atol=1e-7)

    def test_fit_deriv_scalar(self):
        gaussian_psf = CircularGaussianPSF(flux=1, x_0=5, y_0=5, fwhm=2.0)
        yy, xx = np.mgrid[0:11, 0:11].astype(float)
        psf_data = gaussian_psf(xx, yy)
        model = ImagePSF(psf_data, flux=1.0, x_0=5.0, y_0=5.0)

        # Scalar inputs are promoted to 1D arrays, matching evaluate
        derivs = model.fit_deriv(4.5, 5.5, 1.0, 5.0, 5.0)
        expected = model.fit_deriv(np.array([4.5]), np.array([5.5]),
                                   1.0, 5.0, 5.0)
        for deriv, exp in zip(derivs, expected, strict=True):
            assert deriv.shape == (1,)
            assert_allclose(deriv, exp)

        # Scalar out-of-bounds inputs give zero derivatives
        derivs = model.fit_deriv(-50.0, 5.0, 1.0, 5.0, 5.0)
        for deriv in derivs:
            assert_equal(deriv, 0.0)

    def test_fit_deriv_fitting(self):
        """
        Test that fitting with the analytic Jacobian recovers the true
        parameters and matches the finite-difference approximation.
        """
        gaussian_psf = CircularGaussianPSF(flux=1, x_0=12, y_0=12, fwhm=3.5)
        yy, xx = np.mgrid[0:25, 0:25].astype(float)
        psf_data = gaussian_psf(xx, yy)

        truth = ImagePSF(psf_data, flux=250.0, x_0=11.6, y_0=12.4)
        rng = np.random.default_rng(0)
        data = truth(xx, yy) + rng.normal(0.0, 0.02, xx.shape)

        assert ImagePSF.fit_deriv is not None
        fit_params = []
        for estimate_jacobian in (False, True):
            init = ImagePSF(psf_data, flux=200.0, x_0=12.0, y_0=12.0)
            fitter = TRFLSQFitter()
            fit = fitter(init, xx.ravel(), yy.ravel(), data.ravel(),
                         estimate_jacobian=estimate_jacobian)
            fit_params.append(fit.parameters)

        assert_allclose(fit_params[0], fit_params[1], rtol=1e-5)
        assert_allclose(fit_params[0], (250.0, 11.6, 12.4), rtol=1e-2)

    def test_imagepsf_oversampling(self, gaussian_psf):
        oversamp = 3
        yy, xx = np.mgrid[-3:3.00001:(1 / oversamp), -3:3.00001:(1 / oversamp)]
        psf_data = gaussian_psf(xx, yy)

        model = ImagePSF(psf_data, oversampling=oversamp)
        for x, y in [(0, 0), (1, 1), (-2, 1)]:
            assert_allclose(model(x, y), gaussian_psf(x, y))
        for x, y in [(0.5, 0.5), (-0.5, 1.75)]:  # subpixel values
            assert_allclose(model(x, y), gaussian_psf(x, y), rtol=0.001)
        for x, y in [(0.33, 0.33), (0.66, 0.66)]:
            assert_allclose(model(x, y), gaussian_psf(x, y), rtol=2.0e-5)

        x_0 = 2.5
        y_0 = -3.5
        model.x_0 = x_0
        model.y_0 = y_0
        for x, y in [(0, 0), (0.66, 0.66)]:
            assert_allclose(model(x, y), gaussian_psf(x + x_0, y + y_0),
                            atol=3.0e-6)

        # Without oversampling the same tests should fail except for at
        # the origin
        model = ImagePSF(psf_data)
        assert_allclose(model(0, 0), gaussian_psf(0, 0))
        for x, y in [(1, 1), (-2, 1)]:  # integer values
            assert not np.allclose(model(x, y), gaussian_psf(x, y))
        for x, y in [(0.5, 0.5), (-0.5, 1.75)]:
            assert not np.allclose(model(x, y), gaussian_psf(x, y), rtol=0.001)

    def test_origin(self):
        yy, xx = np.mgrid[:5, :5]
        gaussian_psf = CircularGaussianPSF(x_0=2, y_0=2, fwhm=2.1)
        psf_data = gaussian_psf(xx, yy)
        origin = (0, 0)
        model = ImagePSF(psf_data, x_0=2, y_0=2, origin=origin)
        assert_equal(model.origin, origin)
        for x, y in [(0, 0), (1, 1), (-2, 1)]:
            assert_allclose(model(x + 2, y + 2), gaussian_psf(x, y), atol=5e-6)

    def test_bounding_box(self):
        psf_data = np.arange(30, dtype=float).reshape(5, 6)
        psf_data /= np.sum(psf_data)
        model = ImagePSF(psf_data, flux=1, x_0=0, y_0=0)
        assert_equal(model.bounding_box.bounding_box(), ((-2.5, 2.5),
                                                         (-3.0, 3.0)))

        model = ImagePSF(psf_data, flux=1, x_0=0, y_0=0, oversampling=2)
        assert_equal(model.bounding_box.bounding_box(), ((-1.25, 1.25),
                                                         (-1.5, 1.5)))

    def test_data_inputs(self):
        match = 'Input data must be a 2D numpy array'
        with pytest.raises(TypeError, match=match):
            ImagePSF(42)

        with pytest.raises(ValueError, match=match):
            ImagePSF(np.ones(10))

        with pytest.raises(ValueError, match=match):
            ImagePSF(np.ones((10, 10, 10)))

        match = 'The length of the x and y axes must both be at least 4'
        with pytest.raises(ValueError, match=match):
            ImagePSF(np.ones((3, 4)))

        data = np.ones((10, 10))
        data[0, 0] = np.nan
        match = 'All elements of input data must be finite'
        with pytest.raises(ValueError, match=match):
            ImagePSF(data)

    def test_oversampling_inputs(self):
        data = np.arange(30).reshape(5, 6)

        for oversampling in [4, (3, 3), (3, 4)]:
            model = ImagePSF(data, oversampling=oversampling)
            if np.ndim(oversampling) == 0:
                assert_equal(model.oversampling, (oversampling, oversampling))
            else:
                assert_equal(model.oversampling, oversampling)

        match = 'oversampling must be > 0'
        for oversampling in [-1, [-2, 4]]:
            with pytest.raises(ValueError, match=match):
                ImagePSF(data, oversampling=oversampling)

        match = 'oversampling must have 1 or 2 elements'
        oversampling = (1, 4, 8)
        with pytest.raises(ValueError, match=match):
            ImagePSF(data, oversampling=oversampling)

        match = 'oversampling must be 1D'
        for oversampling in [((1, 2), (3, 4)), np.ones((2, 2, 2))]:
            with pytest.raises(ValueError, match=match):
                ImagePSF(data, oversampling=oversampling)

        match = 'oversampling must have integer values'
        with pytest.raises(ValueError, match=match):
            ImagePSF(data, oversampling=2.1)

        match = 'oversampling must be a finite value'
        for oversampling in [np.nan, (1, np.inf)]:
            with pytest.raises(ValueError, match=match):
                ImagePSF(data, oversampling=oversampling)

    def test_shape(self, image_psf):
        assert image_psf.shape == image_psf.data.shape
        assert image_psf.shape == (21, 21)

    def test_evaluate_scalar_coords(self, image_psf):
        """
        Test that evaluate accepts scalar coordinates when called
        directly.
        """
        value = image_psf.evaluate(0.5, 0.5, 1.0, 0.0, 0.0)
        assert np.isfinite(value)

    def test_data_setter(self):
        yy, xx = np.mgrid[0:25, 0:25]
        data1 = CircularGaussianPSF(x_0=12, y_0=12, fwhm=3.0)(xx, yy)
        data2 = CircularGaussianPSF(x_0=12, y_0=12, fwhm=8.0)(xx, yy)

        model = ImagePSF(data1, x_0=12, y_0=12)
        assert_allclose(model(12.0, 12.0), data1[12, 12])

        # The cached interpolator must be discarded when data is set
        model.data = data2
        assert_allclose(model(12.0, 12.0), data2[12, 12])

    @pytest.mark.parametrize('ndim', [1, 2])
    def test_evaluate_matches_spline(self, ndim):
        """
        Test that evaluate and fit_deriv, computed by the compiled
        kernel, match the direct spline and derivative-spline
        evaluation, including the fill_value region outside the image.
        """
        model = ImagePSF(self._gaussian_image(), flux=3.0, x_0=0.0,
                         y_0=0.0, origin=(12.0, 12.0), oversampling=(2, 3))
        x_0, y_0, flux = 30.7, 41.2, 2.5
        rng = np.random.default_rng(0)
        if ndim == 1:
            x = rng.uniform(x_0 - 8, x_0 + 8, 300)
            y = rng.uniform(y_0 - 8, y_0 + 8, 300)
        else:
            y, x = np.mgrid[-7:8, -7:8] + np.array([[[y_0]], [[x_0]]])
            x = x + 0.37
            y = y - 0.21
        xi = model.oversampling[1] * (x - x_0) + model.origin[0]
        yi = model.oversampling[0] * (y - y_0) + model.origin[1]
        spline = model.interpolator
        outside = ((xi < 0) | (xi > model.data.shape[1] - 1)
                   | (yi < 0) | (yi > model.data.shape[0] - 1))
        assert outside.any()  # the fill_value branch is exercised

        expected = flux * spline(xi, yi, grid=False)
        expected[outside] = 0.0
        evaluated = model.evaluate(x, y, flux, x_0, y_0)
        assert evaluated.shape == x.shape
        assert_allclose(evaluated, expected, rtol=1e-12, atol=1e-14)

        d_flux, d_x_0, d_y_0 = model.fit_deriv(x, y, flux, x_0, y_0)
        exp_flux = spline(xi, yi, grid=False)
        exp_x = (-flux * model.oversampling[1]
                 * spline.partial_derivative(1, 0)(xi, yi, grid=False))
        exp_y = (-flux * model.oversampling[0]
                 * spline.partial_derivative(0, 1)(xi, yi, grid=False))
        for arr in (exp_flux, exp_x, exp_y):
            arr[outside] = 0.0
        assert_allclose(d_flux, exp_flux, rtol=1e-12, atol=1e-14)
        assert_allclose(d_x_0, exp_x, rtol=1e-12, atol=1e-13)
        assert_allclose(d_y_0, exp_y, rtol=1e-12, atol=1e-13)

    @pytest.mark.parametrize('fill_value', [0.0, None])
    @pytest.mark.parametrize('shapes', [((5,), ()), ((1, 5), (5, 1)),
                                        ((3, 5), (5,))])
    def test_broadcast_inputs(self, fill_value, shapes):
        """
        Test that evaluate and fit_deriv broadcast x and y inputs of
        different shapes against each other.
        """
        model = ImagePSF(self._gaussian_image(), origin=(12.0, 12.0),
                         fill_value=fill_value)
        rng = np.random.default_rng(0)
        # Some of the points are outside the image
        x = rng.uniform(-14.0, 14.0, shapes[0])
        y = rng.uniform(-14.0, 14.0, shapes[1])
        xb, yb = (np.ascontiguousarray(arr)
                  for arr in np.broadcast_arrays(x, y))
        params = (2.0, 0.5, -0.25)

        result = model.evaluate(x, y, *params)
        assert result.shape == xb.shape
        assert_equal(result, model.evaluate(xb, yb, *params))
        for got, expected in zip(model.fit_deriv(x, y, *params),
                                 model.fit_deriv(xb, yb, *params),
                                 strict=True):
            assert got.shape == xb.shape
            assert_equal(got, expected)

    @pytest.mark.parametrize('fill_value', [0.0, None])
    def test_flux_units(self, fill_value):
        """
        Test that a flux with units gives model values and position
        derivatives with the same units.
        """
        data = self._gaussian_image()
        model = ImagePSF(data, flux=500.0 * u.Jy, origin=(12.0, 12.0),
                         fill_value=fill_value)
        plain = ImagePSF(data, flux=500.0, origin=(12.0, 12.0),
                         fill_value=fill_value)
        # Some of the points are outside the image
        x = np.linspace(-14.0, 14.0, 30)
        y = np.linspace(-10.0, 10.0, 30)[::-1]

        value = model(x, y)
        assert value.unit == u.Jy
        assert_equal(value.value, plain(x, y))

        derivs = model.fit_deriv(x, y, 500.0 * u.Jy, 0.5, -0.25)
        expected = plain.fit_deriv(x, y, 500.0, 0.5, -0.25)
        assert_equal(derivs[0], expected[0])
        for deriv, exp in zip(derivs[1:], expected[1:], strict=True):
            assert deriv.unit == u.Jy
            assert_equal(deriv.value, exp)

    def test_flux_units_model_image(self):
        """
        Test that a model image can be made from fluxes with units.
        """
        model = ImagePSF(self._gaussian_image())
        params = QTable({'x_0': [20.0, 30.5], 'y_0': [20.0, 15.25],
                         'flux': [5.0, 3.0] * u.Jy})
        image = make_model_image((40, 45), model, params,
                                 model_shape=(9, 9))
        plain = QTable({'x_0': params['x_0'], 'y_0': params['y_0'],
                        'flux': params['flux'].value})
        expected = make_model_image((40, 45), model, plain,
                                    model_shape=(9, 9))
        assert image.unit == u.Jy
        assert_equal(image.value, expected)

    def test_flux_array(self):
        """
        Test that a flux array is broadcast against the coordinates.
        """
        model = ImagePSF(self._gaussian_image(), origin=(12.0, 12.0),
                         fill_value=None)
        x = np.array([0.5])
        y = np.array([-0.25])
        fluxes = np.array([1.0, 2.0, 3.0])
        single = model.evaluate(x, y, 1.0, 0.0, 0.0)
        assert_allclose(model.evaluate(x, y, fluxes, 0.0, 0.0),
                        fluxes * single)
        derivs = model.fit_deriv(x, y, fluxes, 0.0, 0.0)
        expected = model.fit_deriv(x, y, 1.0, 0.0, 0.0)
        assert_allclose(derivs[1], fluxes * expected[1])
        assert_allclose(derivs[2], fluxes * expected[2])

    def test_custom_interpolator(self):
        """
        Test that a subclass that overrides the interpolator is
        deprecated, and that it uses the interpolator and its
        partial_derivative method instead of the compiled kernel,
        giving the same results.
        """
        data = self._gaussian_image()

        class WrappedSpline:
            # A custom interpolator with the RectBivariateSpline call
            # and partial_derivative interface
            def __init__(self, spline):
                self.spline = spline

            def __call__(self, xi, yi, grid=False):
                return self.spline(xi, yi, grid=grid)

            def partial_derivative(self, dx, dy):
                return self.spline.partial_derivative(dx, dy)

        class CustomImagePSF(ImagePSF):
            @cached_property
            def interpolator(self):
                x = np.arange(self.data.shape[1])
                y = np.arange(self.data.shape[0])
                return WrappedSpline(RectBivariateSpline(x, y, self.data.T,
                                                         kx=3, ky=3, s=0))

        model = ImagePSF(data, origin=(12.0, 12.0))
        match = 'Overriding the ImagePSF.interpolator attribute'
        with pytest.warns(AstropyDeprecationWarning, match=match):
            custom = CustomImagePSF(data, origin=(12.0, 12.0))

        x = np.linspace(-9.0, 9.0, 50)
        y = np.linspace(-8.0, 8.0, 50)[::-1]
        assert_allclose(custom.evaluate(x, y, 2.0, 0.5, -0.25),
                        model.evaluate(x, y, 2.0, 0.5, -0.25), rtol=1e-12)
        for got, expected in zip(custom.fit_deriv(x, y, 2.0, 0.5, -0.25),
                                 model.fit_deriv(x, y, 2.0, 0.5, -0.25),
                                 strict=True):
            assert_allclose(got, expected, rtol=1e-12, atol=1e-13)

        # The interpolators are built ahead of fitting only for a
        # custom interpolator. The copy method builds the spline of
        # the other models.
        custom._precompute_interpolators()
        assert '_deriv_interpolators' in custom.__dict__
        assert '_spline' not in custom.__dict__
        model = ImagePSF(data, origin=(12.0, 12.0))
        model._precompute_interpolators()
        assert '_deriv_interpolators' not in model.__dict__
        assert '_spline' not in model.__dict__
        model.copy()
        assert '_spline' in model.__dict__

    @pytest.mark.parametrize('evaluate_first', [False, True])
    @pytest.mark.parametrize('degree', [2, 3])
    def test_assigned_interpolator(self, evaluate_first, degree):
        """
        Test that assigning an interpolator to a model is deprecated,
        and that the model and its copies call the assigned
        interpolator, whether it is assigned before or after the model
        is first evaluated, until the data are set.
        """
        data = self._gaussian_image()
        x = np.linspace(-9.0, 9.0, 50)
        y = np.linspace(-8.0, 8.0, 50)[::-1]
        params = (2.0, 0.5, -0.25)
        model = ImagePSF(data, origin=(12.0, 12.0))
        reference = model.evaluate(x, y, *params)
        if not evaluate_first:
            model = ImagePSF(data, origin=(12.0, 12.0))

        idx = np.arange(25)
        spline = RectBivariateSpline(idx, idx, 3.0 * data.T, kx=degree,
                                     ky=degree, s=0)
        match = 'Assigning a custom interpolator'
        with pytest.warns(AstropyDeprecationWarning, match=match):
            model.interpolator = spline
        xi = x - 0.5 + 12.0
        yi = y + 0.25 + 12.0
        expected = 2.0 * spline(xi, yi, grid=False)
        d_x = spline.partial_derivative(1, 0)(xi, yi, grid=False)
        for psf in (model, model.copy()):
            assert psf.interpolator is spline
            assert_allclose(psf.evaluate(x, y, *params), expected,
                            rtol=1e-12)
            derivs = psf.fit_deriv(x, y, *params)
            assert_allclose(derivs[0], expected / 2.0, rtol=1e-12)
            assert_allclose(derivs[1], -2.0 * d_x, rtol=1e-12, atol=1e-13)

        # Setting the data discards the assigned interpolator
        model.data = data.copy()
        assert_allclose(model.evaluate(x, y, *params), reference, rtol=1e-12)

    def test_data_spline_public_interface(self, monkeypatch, public_spline):
        """
        Test that the model evaluates the spline that it builds from
        the data with the compiled kernel using only the public
        interface of the scipy spline.
        """
        data = self._gaussian_image()
        idx = np.arange(25)
        spline = RectBivariateSpline(idx, idx, data.T, kx=3, ky=3, s=0)
        monkeypatch.setattr('photutils.psf.image_models.RectBivariateSpline',
                            public_spline)
        model = ImagePSF(data, origin=(12.0, 12.0))

        flux, x_0, y_0 = 2.0, 0.5, -0.25
        x = np.linspace(-9.0, 9.0, 50)
        y = np.linspace(-8.0, 8.0, 50)[::-1]
        xi = x - x_0 + 12.0
        yi = y - y_0 + 12.0
        assert_allclose(model.evaluate(x, y, flux, x_0, y_0),
                        flux * spline(xi, yi, grid=False), rtol=1e-12,
                        atol=1e-14)
        d_flux, d_x_0, _ = model.fit_deriv(x, y, flux, x_0, y_0)
        assert_allclose(d_flux, spline(xi, yi, grid=False), rtol=1e-12,
                        atol=1e-14)
        assert_allclose(d_x_0, -flux * spline.partial_derivative(1, 0)(
            xi, yi, grid=False), rtol=1e-12, atol=1e-13)

    def test_data_setter_clears_spline(self):
        """
        Test that setting new data discards the cached spline
        coefficients used by the compiled kernel.
        """
        yy, xx = np.mgrid[0:25, 0:25]
        data1 = CircularGaussianPSF(x_0=12, y_0=12, fwhm=3.0)(xx, yy)
        data2 = CircularGaussianPSF(x_0=12, y_0=12, fwhm=8.0)(xx, yy)
        model = ImagePSF(data1, x_0=12, y_0=12)
        model.evaluate(np.array([12.0]), np.array([12.0]), 1.0, 12.0, 12.0)
        assert '_spline' in model.__dict__
        model.data = data2
        assert '_spline' not in model.__dict__
        d_flux = model.fit_deriv(np.array([12.0]), np.array([12.0]), 1.0,
                                 12.0, 12.0)[0]
        assert_allclose(d_flux, data2[12, 12])

    @pytest.mark.parametrize('pickled', [False, True])
    def test_deepcopy_pickle_keep_spline(self, pickled):
        """
        Test that a deep copy and an unpickled copy of an evaluated
        model keep the cached spline and give the same values.
        """
        model = ImagePSF(self._gaussian_image(), origin=(12.0, 12.0))
        x = np.linspace(-9.0, 9.0, 50)
        y = np.linspace(-8.0, 8.0, 50)[::-1]
        params = (2.0, 0.5, -0.25)
        values = model.evaluate(x, y, *params)
        derivs = model.fit_deriv(x, y, *params)

        if pickled:
            new_model = pickle.loads(  # noqa: S301
                pickle.dumps(model))  # nosec B301
        else:
            new_model = copy.deepcopy(model)
        assert '_spline' in new_model.__dict__
        assert_equal(new_model.evaluate(x, y, *params), values)
        assert_equal(new_model.fit_deriv(x, y, *params), derivs)

    def test_data_setter_validation(self):
        model = ImagePSF(np.ones((10, 10)))

        match = 'Input data must be a 2D numpy array'
        with pytest.raises(TypeError, match=match):
            model.data = 42
        with pytest.raises(ValueError, match=match):
            model.data = np.ones(10)

        match = 'The length of the x and y axes must both be at least 4'
        with pytest.raises(ValueError, match=match):
            model.data = np.ones((3, 4))

    def test_data_setter_copy_independence(self, gaussian_psf):
        yy, xx = np.mgrid[0:25, 0:25]
        data1 = gaussian_psf(xx, yy)
        data2 = CircularGaussianPSF(x_0=12, y_0=12, fwhm=8.0)(xx, yy)

        model = ImagePSF(data1, x_0=12, y_0=12)
        value = model(12.0, 12.0)  # populate the interpolator cache

        model_copy = model.copy()
        model_copy.data = data2
        assert_allclose(model(12.0, 12.0), value)
        assert_allclose(model_copy(12.0, 12.0), data2[12, 12])

    def test_oversampling_setter(self):
        model = ImagePSF(np.ones((10, 10)))
        model.oversampling = 4
        assert_equal(model.oversampling, (4, 4))

        match = 'oversampling must be > 0'
        with pytest.raises(ValueError, match=match):
            model.oversampling = -3
        assert_equal(model.oversampling, (4, 4))

    def test_origin_inputs(self):
        match = 'origin must be 1D and have 2-elements'
        with pytest.raises(ValueError, match=match):
            ImagePSF(np.ones((10, 10)), origin=(1, 2, 3))
        with pytest.raises(ValueError, match=match):
            ImagePSF(np.ones((10, 10)), origin=np.ones((2, 2)))

        match = 'All elements of origin must be finite'
        with pytest.raises(ValueError, match=match):
            ImagePSF(np.ones((10, 10)), origin=(np.nan, 1))

    @pytest.mark.parametrize('deepcopy', [False, True])
    def test_copy(self, deepcopy):
        data = np.arange(30).reshape(5, 6)
        model = ImagePSF(data, flux=1, x_0=0, y_0=0)
        model_copy = model.deepcopy() if deepcopy else model.copy()

        assert_equal(model.data, model_copy.data)
        assert_equal(model.flux, model_copy.flux)
        assert_equal(model.x_0, model_copy.x_0)
        assert_equal(model.y_0, model_copy.y_0)
        assert_equal(model.oversampling, model_copy.oversampling)
        assert_equal(model.origin, model_copy.origin)

        model_copy.data[0, 0] = 42
        if deepcopy:
            assert model.data[0, 0] != model_copy.data[0, 0]
        else:
            assert model.data[0, 0] == model_copy.data[0, 0]

        model_copy.flux = 2
        assert model.flux != model_copy.flux

        model_copy.x_0.fixed = True
        model_copy.y_0.fixed = True
        model_copy2 = model_copy.copy()
        assert model_copy2.x_0.fixed
        assert model_copy2.fixed == model_copy.fixed

    @pytest.mark.parametrize('unpickled', [False, True])
    def test_copies_share_spline(self, spline_builds, unpickled):
        """
        Test that the copies of a model that was never evaluated share
        one spline instead of each building their own, including for
        a model that was unpickled (e.g., in a worker process), which
        never has a cached spline.
        """
        model = ImagePSF(self._gaussian_image(), origin=(12.0, 12.0))
        if unpickled:
            model = pickle.loads(  # noqa: S301
                pickle.dumps(model))  # nosec B301
        x = np.linspace(-9.0, 9.0, 50)
        y = np.linspace(-8.0, 8.0, 50)[::-1]
        params = (2.0, 0.5, -0.25)

        copies = [model.copy() for _ in range(3)]
        for model_copy in copies:
            model_copy.evaluate(x, y, *params)
            model_copy.fit_deriv(x, y, *params)
        assert spline_builds.count == 1
        assert_equal(copies[2].evaluate(x, y, *params),
                     model.evaluate(x, y, *params))
        assert spline_builds.count == 1

    @staticmethod
    def _gaussian_image():
        yy, xx = np.mgrid[0:25, 0:25]
        return np.exp(-((xx - 12.0)**2 + (yy - 12.0)**2) / 8.0)

    def test_repr(self, image_psf):
        model_repr = repr(image_psf)
        expected = ('<ImagePSF(flux=1., x_0=0., y_0=0., origin=[10.0, 10.0], '
                    'oversampling=[1, 1], fill_value=0.0)>')
        assert model_repr == expected
        for param in image_psf.param_names:
            assert param in model_repr

    def test_str(self, image_psf):
        model_str = str(image_psf)
        keys = ('PSF shape', 'Origin', 'Oversampling', 'Fill Value')
        for key in keys:
            assert key in model_str
        for param in image_psf.param_names:
            assert param in model_str


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('oversampling', [1, 2])
def test_read_only_inputs(gaussian_psf, dtype, oversampling):
    """
    Regression test that read-only (non-writeable) input arrays are
    accepted, are not modified, and give results identical to writeable
    arrays.
    """
    yy, xx = np.mgrid[-10:11, -10:11]
    psf_data = gaussian_psf(xx, yy).astype(dtype)
    y = np.linspace(-3, 3, 13)
    x = np.linspace(-2.5, 3.5, 13)
    arrays = (psf_data, x, y)
    originals = [arr.copy() for arr in arrays]

    def compute():
        model = ImagePSF(psf_data, flux=10, x_0=0.3, y_0=-0.2,
                         oversampling=oversampling)
        return (model(x, y), *model.fit_deriv(x, y, 10, 0.3, -0.2),
                model.copy()(x, y))

    expected = compute()
    for arr in arrays:
        arr.setflags(write=False)
    result = compute()

    for res, exp in zip(result, expected, strict=True):
        assert_equal(res, exp)
    for arr, original in zip(arrays, originals, strict=True):
        assert_equal(arr, original)
