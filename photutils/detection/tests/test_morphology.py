# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from astropy.tests.helper import pytest
import numpy as np
from numpy.testing import assert_allclose
import itertools
from ..morphology import (centroid_com, centroid_1dg, centroid_2dg,
                          gaussian1d_moments, shape_params)
from astropy.modeling import models
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
    xc, yc = centroid_com(data)
    assert_allclose([xc_ref, yc_ref], [xc, yc], rtol=0, atol=1.e-3)
    xc2, yc2 = centroid_1dg(data)
    assert_allclose([xc_ref, yc_ref], [xc2, yc2], rtol=0, atol=1.e-3)
    xc3, yc3 = centroid_2dg(data)
    assert_allclose([xc_ref, yc_ref], [xc3, yc3], rtol=0, atol=1.e-3)


@pytest.mark.skipif('not HAS_SKIMAGE')
def test_centroid_com_mask():
    """Test centroid_com with and without an image_mask."""
    data = np.ones((2, 2)).astype(np.float)
    mask = [[False, False], [True, True]]
    centroid = centroid_com(data, image_mask=None)
    centroid_mask = centroid_com(data, image_mask=mask)
    assert_allclose([0.5, 0.5], centroid, rtol=0, atol=1.e-6)
    assert_allclose([0.5, 0.0], centroid_mask, rtol=0, atol=1.e-6)


@pytest.mark.skipif('not HAS_SKIMAGE')
def test_centroid_com_mask_shape():
    """
    Test if ValueError raises if image_mask shape doesn't match image
    shape.
    """
    with pytest.raises(ValueError):
        mask = np.zeros((2, 2), dtype=bool)
        centroid_com(np.zeros((4, 4)), image_mask=mask)


@pytest.mark.skipif('not HAS_SKIMAGE')
def test_shape_params_mask():
    data = np.ones((2, 2)).astype(np.float)
    mask = [[False, False], [True, True]]
    params = shape_params(data, image_mask=None)
    params_mask = shape_params(data, image_mask=mask)
    result = [params['xcen'], params['ycen'], params['eccen']]
    result_mask = [params_mask['xcen'], params_mask['ycen'],
                   params_mask['eccen']]
    assert_allclose([0.5, 0.5, 0.0], result, rtol=0, atol=1.e-6)
    assert_allclose([0.5, 0.0, 1.0], result_mask, rtol=0, atol=1.e-6)


def test_gaussian1d_moments():
    x = np.arange(100)
    ref = (75, 50, 5)
    g = models.Gaussian1D(*ref)
    data = g(x)
    result = gaussian1d_moments(data)
    assert_allclose(ref, result, rtol=0, atol=1.e-6)
