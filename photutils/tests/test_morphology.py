# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from astropy.tests.helper import pytest
import numpy as np
from numpy.testing import assert_allclose
import itertools
from ..morphology import (centroid_com, centroid_1dg, centroid_2dg,
                          gaussian1d_moments, data_properties)
from astropy.modeling import models
from astropy.convolution.kernels import Gaussian2DKernel
try:
    import skimage
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False


XCS = [25.7]
YCS = [26.2]
XSTDDEVS = [3.2, 4.0]
YSTDDEVS = [5.7, 4.1]
THETAS = np.array([30., 45.]) * np.pi / 180.
DATA = np.zeros((3, 3))
DATA[0:2, 1] = 1.
DATA[1, 0:2] = 1.
DATA[1, 1] = 2.


@pytest.mark.parametrize(
    ('xc_ref', 'yc_ref', 'x_stddev', 'y_stddev', 'theta'),
    list(itertools.product(XCS, YCS, XSTDDEVS, YSTDDEVS, THETAS)))
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


@pytest.mark.parametrize(
    ('xc_ref', 'yc_ref', 'x_stddev', 'y_stddev', 'theta'),
    list(itertools.product(XCS, YCS, XSTDDEVS, YSTDDEVS, THETAS)))
@pytest.mark.skipif('not HAS_SKIMAGE')
def test_centroids_witherror(xc_ref, yc_ref, x_stddev, y_stddev, theta):
    model = models.Gaussian2D(2.4, xc_ref, yc_ref, x_stddev=x_stddev,
                              y_stddev=y_stddev, theta=theta)
    y, x = np.mgrid[0:50, 0:50]
    data = model(x, y)
    error = np.sqrt(data)
    xc2, yc2 = centroid_1dg(data, error=error)
    assert_allclose([xc_ref, yc_ref], [xc2, yc2], rtol=0, atol=1.e-3)
    xc3, yc3 = centroid_2dg(data, error=error)
    assert_allclose([xc_ref, yc_ref], [xc3, yc3], rtol=0, atol=1.e-3)


@pytest.mark.skipif('not HAS_SKIMAGE')
def test_centroids_withmask():
    size = 9
    xc_ref, yc_ref = (size - 1) / 2, (size - 1) / 2
    data = Gaussian2DKernel(1., x_size=size, y_size=size).array
    mask = np.zeros_like(data, dtype=bool)
    data[0, 0] = 1.
    mask[0, 0] = True
    xc, yc = centroid_com(data, mask=mask)
    assert_allclose([xc, yc], [xc_ref, yc_ref], rtol=0, atol=1.e-3)
    xc2, yc2 = centroid_1dg(data, mask=mask)
    assert_allclose([xc2, yc2], [xc_ref, yc_ref], rtol=0, atol=1.e-3)
    xc3, yc3 = centroid_2dg(data, mask=mask)
    assert_allclose([xc3, yc3], [xc_ref, yc_ref], rtol=0, atol=1.e-3)


@pytest.mark.skipif('not HAS_SKIMAGE')
def test_centroids_mask():
    """Test centroid_com with and without an image_mask."""
    data = np.ones((2, 2)).astype(np.float)
    mask = [[False, False], [True, True]]
    centroid = centroid_com(data, mask=None)
    centroid_mask = centroid_com(data, mask=mask)
    assert_allclose([0.5, 0.5], centroid, rtol=0, atol=1.e-6)
    assert_allclose([0.5, 0.0], centroid_mask, rtol=0, atol=1.e-6)


@pytest.mark.skipif('not HAS_SKIMAGE')
def test_centroid_com_mask_shape():
    """
    Test if ValueError raises if mask shape doesn't match data
    shape.
    """
    with pytest.raises(ValueError):
        mask = np.zeros((2, 2), dtype=bool)
        centroid_com(np.zeros((4, 4)), mask=mask)


@pytest.mark.skipif('not HAS_SKIMAGE')
def test_data_properties():
    data = np.ones((2, 2)).astype(np.float)
    mask = np.array([[False, False], [True, True]])
    props = data_properties(data, mask=None)
    props2 = data_properties(data, mask=mask)
    properties = ['xcentroid', 'ycentroid', 'area']
    result = [props[i].value for i in properties]
    result2 = [props2[i].value for i in properties]
    assert_allclose([0.5, 0.5, 4.0], result, rtol=0, atol=1.e-6)
    assert_allclose([0.5, 0.0, 2.0], result2, rtol=0, atol=1.e-6)


def test_gaussian1d_moments():
    x = np.arange(100)
    ref = (75, 50, 5)
    g = models.Gaussian1D(*ref)
    data = g(x)
    result = gaussian1d_moments(data)
    assert_allclose(ref, result, rtol=0, atol=1.e-6)
