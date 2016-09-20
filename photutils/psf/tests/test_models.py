# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import division

import numpy as np
from astropy.modeling.models import Gaussian2D
from astropy.tests.helper import pytest

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

    imod_norm = FittableImageModel(gm(xg, yg), normalize=True)
    assert not np.allclose(imod_norm(0, 0), gm(0, 0))
    assert np.allclose(np.sum(imod_norm(xg, yg)), 1)

    imod_norm2 = FittableImageModel(gm(xg, yg), normalize=True,
                                    normalization_correction=2)
    assert not np.allclose(imod_norm2(0, 0), gm(0, 0))
    assert np.allclose(imod_norm(0, 0), imod_norm2(0, 0)*2)
    assert np.allclose(np.sum(imod_norm2(xg, yg)), 0.5)
