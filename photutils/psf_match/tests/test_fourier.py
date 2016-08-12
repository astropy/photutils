# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from astropy.tests.helper import pytest
import numpy as np
from numpy.testing import assert_allclose
from astropy.modeling.models import Gaussian2D
from ..fourier import resize_psf, create_matching_kernel
from ..windows import TopHatWindow

try:
    import scipy    # noqa
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


@pytest.mark.skipif('not HAS_SCIPY')
def test_resize_psf():
    psf1 = np.ones((5, 5))
    psf2 = resize_psf(psf1, 0.1, 0.05)
    assert psf2.shape == (10, 10)


def test_create_matching_kernel():
    """Test with noiseless 2D Gaussians."""
    y, x = np.mgrid[0:101, 0:101]
    gm1 = Gaussian2D(100, 50, 50, 3, 3)
    gm2 = Gaussian2D(100, 50, 50, 4, 4)
    gm3 = Gaussian2D(100, 50, 50, 5, 5)
    g1 = gm1(x, y)
    g2 = gm2(x, y)
    g3 = gm3(x, y)
    g1 /= g1.sum()
    g2 /= g2.sum()
    g3 /= g3.sum()

    window = TopHatWindow(32./101)
    k = create_matching_kernel(g1, g3, window=window)
    assert_allclose(k, g3, atol=1.e-2)


def test_create_matching_kernel_shapes():
    """Test with wrong PSF shapes."""
    with pytest.raises(ValueError):
        psf1 = np.ones((5, 5))
        psf2 = np.ones((3, 3))
        create_matching_kernel(psf1, psf2)
