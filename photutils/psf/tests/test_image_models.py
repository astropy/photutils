# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the image_models module.
"""

import numpy as np
import pytest
from astropy.modeling.fitting import TRFLSQFitter
from numpy.testing import assert_allclose, assert_equal

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
        d_flux, d_x_0, d_y_0 = model.fit_deriv(x, y, 1.0, 5.0, 5.0)

        # Derivatives must be zero outside the input pixel grid
        assert d_flux[1] == 0.0
        assert d_x_0[1] == 0.0
        assert d_y_0[2] == 0.0
        derivs = model.fit_deriv(x, y, 1.0, 5.0, 5.0)

        # All derivatives must be zero outside the input pixel grid
        for deriv in derivs:
            assert_equal(deriv[1:], 0.0)
        # The flux derivative is nonzero at the in-bounds peak position
        assert derivs[0][0] > 0.0

        # With fill_value=None, out-of-bounds derivatives are
        # extrapolated from the spline fit instead of being zeroed
        model = ImagePSF(psf_data, flux=1.0, x_0=5.0, y_0=5.0,
                         fill_value=None)
        derivs = model.fit_deriv(np.array([12.0]), np.array([5.0]),
                                 1.0, 5.0, 5.0)
        assert np.all(np.isfinite([deriv[0] for deriv in derivs]))
        assert derivs[1][0] != 0.0

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
