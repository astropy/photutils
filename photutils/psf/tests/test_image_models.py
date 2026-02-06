# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the image_models module.
"""

import numpy as np
import pytest
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

        # subpixel should not match, but be reasonably close
        for x, y in [(0.5, 0.5), (-0.5, 1.75)]:
            assert_allclose(model(x, y), gaussian_psf(x, y), atol=4e-3)

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

        # without oversampling the same tests should fail except for at
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
