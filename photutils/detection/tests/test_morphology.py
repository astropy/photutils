# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from astropy.tests.helper import pytest
import numpy as np
from numpy.testing import assert_allclose
import itertools
from photutils.detection import morphology
from astropy.modeling import models
try:
    import scipy
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
try:
    import skimage
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False


XCS = [25.0, 25.7, 26.1]
YCS = [25.0, 26.2, 26.9]
XSTDDEVS = [2.0, 3.2]
YSTDDEVS = [2.0, 5.7]
THETAS = np.array([30., 45.]) * np.pi / 180.


@pytest.mark.parametrize(('xc_ref', 'yc_ref', 'x_stddev', 'y_stddev', 'theta'),
                         list(itertools.product(XCS, YCS, XSTDDEVS, YSTDDEVS,
                                                THETAS)))
@pytest.mark.skipif('not HAS_SKIMAGE')
def test_centroids(xc_ref, yc_ref, x_stddev, y_stddev, theta):
    model = models.Gaussian2D(2.4, xc_ref, yc_ref, x_stddev=x_stddev,
                              y_stddev=y_stddev, theta=theta)
    y, x = np.mgrid[0:50, 0:50]
    data = model(x, y)
    xc, yc = morphology.centroid_com(data)
    assert_allclose([xc_ref, yc_ref], [xc, yc], rtol=0, atol=1.e-3)
    xc2, yc2 = morphology.centroid_1dg(data)
    assert_allclose([xc_ref, yc_ref], [xc2, yc2], rtol=0, atol=1.e-3)
    xc3, yc3 = morphology.centroid_2dg(data)
    assert_allclose([xc_ref, yc_ref], [xc3, yc3], rtol=0, atol=1.e-3)
