# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the image_models module.
"""

import numpy as np
import pytest
from astropy.modeling.models import Gaussian2D
from numpy.testing import assert_allclose

from photutils.psf import FittableImageModel
from photutils.utils._optional_deps import HAS_SCIPY


@pytest.fixture(name='gmodel')
def fixture_gmodel():
    return Gaussian2D(x_stddev=3, y_stddev=3)


@pytest.mark.skipif(not HAS_SCIPY, reason='scipy is required')
class TestFittableImageModel:
    """
    Tests for FittableImageModel.
    """

    def test_fittable_image_model(self, gmodel):
        yy, xx = np.mgrid[-2:3, -2:3]
        model_nonorm = FittableImageModel(gmodel(xx, yy))

        assert_allclose(model_nonorm(0, 0), gmodel(0, 0))
        assert_allclose(model_nonorm(1, 1), gmodel(1, 1))
        assert_allclose(model_nonorm(-2, 1), gmodel(-2, 1))

        # subpixel should *not* match, but be reasonably close
        # in this case good to ~0.1% seems to be fine
        assert_allclose(model_nonorm(0.5, 0.5), gmodel(0.5, 0.5), rtol=.001)
        assert_allclose(model_nonorm(-0.5, 1.75), gmodel(-0.5, 1.75),
                        rtol=.001)

        model_norm = FittableImageModel(gmodel(xx, yy), normalize=True)
        assert not np.allclose(model_norm(0, 0), gmodel(0, 0))
        assert_allclose(np.sum(model_norm(xx, yy)), 1)

        model_norm2 = FittableImageModel(gmodel(xx, yy), normalize=True,
                                         normalization_correction=2)
        assert not np.allclose(model_norm2(0, 0), gmodel(0, 0))
        assert_allclose(model_norm(0, 0), model_norm2(0, 0) * 2)
        assert_allclose(np.sum(model_norm2(xx, yy)), 0.5)

    def test_fittable_image_model_oversampling(self, gmodel):
        oversamp = 3  # oversampling factor
        yy, xx = np.mgrid[-3:3.00001:(1 / oversamp), -3:3.00001:(1 / oversamp)]

        im = gmodel(xx, yy)
        assert im.shape[0] > 7

        model_oversampled = FittableImageModel(im, oversampling=oversamp)
        assert_allclose(model_oversampled(0, 0), gmodel(0, 0))
        assert_allclose(model_oversampled(1, 1), gmodel(1, 1))
        assert_allclose(model_oversampled(-2, 1), gmodel(-2, 1))
        assert_allclose(model_oversampled(0.5, 0.5), gmodel(0.5, 0.5),
                        rtol=.001)
        assert_allclose(model_oversampled(-0.5, 1.75), gmodel(-0.5, 1.75),
                        rtol=.001)

        # without oversampling the same tests should fail except for at
        # the origin
        model_wrongsampled = FittableImageModel(im)
        assert_allclose(model_wrongsampled(0, 0), gmodel(0, 0))
        assert not np.allclose(model_wrongsampled(1, 1), gmodel(1, 1))
        assert not np.allclose(model_wrongsampled(-2, 1), gmodel(-2, 1))
        assert not np.allclose(model_wrongsampled(0.5, 0.5), gmodel(0.5, 0.5),
                               rtol=.001)
        assert not np.allclose(model_wrongsampled(-0.5, 1.75),
                               gmodel(-0.5, 1.75), rtol=.001)

    def test_centering_oversampled(self, gmodel):
        oversamp = 3
        yy, xx = np.mgrid[-3:3.00001:(1 / oversamp), -3:3.00001:(1 / oversamp)]

        model_oversampled = FittableImageModel(gmodel(xx, yy),
                                               oversampling=oversamp)

        valcen = gmodel(0, 0)
        val36 = gmodel(0.66, 0.66)

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
