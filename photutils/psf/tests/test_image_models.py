# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the image_models module.
"""

import numpy as np
import pytest
from astropy.modeling.models import Gaussian2D
from numpy.testing import assert_allclose, assert_equal

from photutils.psf import CircularGaussianPSF, FittableImageModel, ImagePSF


@pytest.fixture(name='gmodel_old')
def fixture_gmodel_old():
    # remove when FittableImageModel is removed
    return Gaussian2D(x_stddev=3, y_stddev=3)


@pytest.fixture(name='gaussian_psf')
def fixture_gaussian_psf():
    return CircularGaussianPSF(fwhm=2.1)


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

    def test_bounding_box(self, gaussian_psf):
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


class TestFittableImageModel:
    """
    Tests for FittableImageModel.
    """

    def test_fittable_image_model(self, gmodel_old):
        yy, xx = np.mgrid[-2:3, -2:3]
        model_nonorm = FittableImageModel(gmodel_old(xx, yy))

        assert_allclose(model_nonorm(0, 0), gmodel_old(0, 0))
        assert_allclose(model_nonorm(1, 1), gmodel_old(1, 1))
        assert_allclose(model_nonorm(-2, 1), gmodel_old(-2, 1))

        # subpixel should *not* match, but be reasonably close
        # in this case good to ~0.1% seems to be fine
        assert_allclose(model_nonorm(0.5, 0.5), gmodel_old(0.5, 0.5),
                        rtol=.001)
        assert_allclose(model_nonorm(-0.5, 1.75), gmodel_old(-0.5, 1.75),
                        rtol=.001)

        model_norm = FittableImageModel(gmodel_old(xx, yy), normalize=True)
        assert not np.allclose(model_norm(0, 0), gmodel_old(0, 0))
        assert_allclose(np.sum(model_norm(xx, yy)), 1)

        model_norm2 = FittableImageModel(gmodel_old(xx, yy), normalize=True,
                                         normalization_correction=2)
        assert not np.allclose(model_norm2(0, 0), gmodel_old(0, 0))
        assert_allclose(model_norm(0, 0), model_norm2(0, 0) * 2)
        assert_allclose(np.sum(model_norm2(xx, yy)), 0.5)

    def test_fittable_image_model_oversampling(self, gmodel_old):
        oversamp = 3  # oversampling factor
        yy, xx = np.mgrid[-3:3.00001:(1 / oversamp), -3:3.00001:(1 / oversamp)]

        im = gmodel_old(xx, yy)
        assert im.shape[0] > 7

        model_oversampled = FittableImageModel(im, oversampling=oversamp)
        assert_allclose(model_oversampled(0, 0), gmodel_old(0, 0))
        assert_allclose(model_oversampled(1, 1), gmodel_old(1, 1))
        assert_allclose(model_oversampled(-2, 1), gmodel_old(-2, 1))
        assert_allclose(model_oversampled(0.5, 0.5), gmodel_old(0.5, 0.5),
                        rtol=.001)
        assert_allclose(model_oversampled(-0.5, 1.75), gmodel_old(-0.5, 1.75),
                        rtol=.001)

        # without oversampling the same tests should fail except for at
        # the origin
        model_wrongsampled = FittableImageModel(im)
        assert_allclose(model_wrongsampled(0, 0), gmodel_old(0, 0))
        assert not np.allclose(model_wrongsampled(1, 1), gmodel_old(1, 1))
        assert not np.allclose(model_wrongsampled(-2, 1), gmodel_old(-2, 1))
        assert not np.allclose(model_wrongsampled(0.5, 0.5),
                               gmodel_old(0.5, 0.5), rtol=.001)
        assert not np.allclose(model_wrongsampled(-0.5, 1.75),
                               gmodel_old(-0.5, 1.75), rtol=.001)

    def test_centering_oversampled(self, gmodel_old):
        oversamp = 3
        yy, xx = np.mgrid[-3:3.00001:(1 / oversamp), -3:3.00001:(1 / oversamp)]

        model_oversampled = FittableImageModel(gmodel_old(xx, yy),
                                               oversampling=oversamp)

        valcen = gmodel_old(0, 0)
        val36 = gmodel_old(0.66, 0.66)

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
