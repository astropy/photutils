# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import division

import numpy as np
from astropy.modeling.models import Gaussian2D
import pytest

from .. import FittableImageModel

try:
    import scipy
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


@pytest.mark.skipif('not HAS_SCIPY')
def test_image_model():
    gm = Gaussian2D(x_stddev=3, y_stddev=3)
    xg, yg = np.mgrid[-2:3, -2:3]

    imod_nonorm = FittableImageModel(gm(xg, yg))
    assert np.allclose(imod_nonorm(0, 0), gm(0, 0))
    assert np.allclose(imod_nonorm(1, 1), gm(1, 1))
    assert np.allclose(imod_nonorm(-2, 1), gm(-2, 1))

    # now sub-pixel should *not* match, but be reasonably close
    assert not np.allclose(imod_nonorm(0.5, 0.5), gm(0.5, 0.5))
    # in this case good to ~0.1% seems to be fine
    assert np.allclose(imod_nonorm(0.5, 0.5), gm(0.5, 0.5), rtol=.001)
    assert np.allclose(imod_nonorm(-0.5, 1.75), gm(-0.5, 1.75), rtol=.001)

    imod_norm = FittableImageModel(gm(xg, yg), normalize=True)
    assert not np.allclose(imod_norm(0, 0), gm(0, 0))
    assert np.allclose(np.sum(imod_norm(xg, yg)), 1)

    imod_norm2 = FittableImageModel(gm(xg, yg), normalize=True,
                                    normalization_correction=2)
    assert not np.allclose(imod_norm2(0, 0), gm(0, 0))
    assert np.allclose(imod_norm(0, 0), imod_norm2(0, 0)*2)
    assert np.allclose(np.sum(imod_norm2(xg, yg)), 0.5)


@pytest.mark.skipif('not HAS_SCIPY')
def test_image_model_oversampling():
    gm = Gaussian2D(x_stddev=3, y_stddev=3)

    osa = 3  #oversampling factor
    xg, yg = np.mgrid[-3:3.00001:(1/osa), -3:3.00001:(1/osa)]

    im = gm(xg, yg)
    assert im.shape[0] > 7  # should be obvious, but at least ensures the test is right
    imod_oversampled = FittableImageModel(im, oversampling=osa)

    assert np.allclose(imod_oversampled(0, 0), gm(0, 0))
    assert np.allclose(imod_oversampled(1, 1), gm(1, 1))
    assert np.allclose(imod_oversampled(-2, 1), gm(-2, 1))
    assert np.allclose(imod_oversampled(0.5, 0.5), gm(0.5, 0.5), rtol=.001)
    assert np.allclose(imod_oversampled(-0.5, 1.75), gm(-0.5, 1.75), rtol=.001)

    imod_wrongsampled = FittableImageModel(im)

    # now make sure that all *fails* without the oversampling
    assert np.allclose(imod_wrongsampled(0, 0), gm(0, 0))  # except for at the origin
    assert not np.allclose(imod_wrongsampled(1, 1), gm(1, 1))
    assert not np.allclose(imod_wrongsampled(-2, 1), gm(-2, 1))
    assert not np.allclose(imod_wrongsampled(0.5, 0.5), gm(0.5, 0.5), rtol=.001)
    assert not np.allclose(imod_wrongsampled(-0.5, 1.75), gm(-0.5, 1.75), rtol=.001)


@pytest.mark.skipif('not HAS_SCIPY')
def test_centering_oversampled():
    gm = Gaussian2D(x_stddev=2, y_stddev=3)

    osa = 3  #oversampling factor
    xg, yg = np.mgrid[-3:3.00001:(1/osa), -3:3.00001:(1/osa)]

    imod_oversampled = FittableImageModel(gm(xg, yg), oversampling=osa)

    valcen = gm(0, 0)
    val36 = gm(0.66, 0.66)

    assert np.allclose(valcen, imod_oversampled(0, 0))
    assert np.allclose(val36, imod_oversampled(0.66, 0.66))

    imod_oversampled.x_0 = 2.5
    imod_oversampled.y_0 = -3.5

    assert np.allclose(valcen, imod_oversampled(2.5, -3.5))
    assert np.allclose(val36, imod_oversampled(2.5 + 0.66, -3.5 + 0.66))
