# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from astropy.tests.helper import pytest
import numpy as np
from numpy.testing import assert_allclose
import itertools
from ..morphology import (centroid_com, centroid_1dg, centroid_2dg,
                          gaussian1d_moments, data_properties,
                          fit_2dgaussian, cutout_footprint)
from astropy.modeling.models import Gaussian1D, Gaussian2D
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
    model = Gaussian2D(2.4, xc_ref, yc_ref, x_stddev=x_stddev,
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
    model = Gaussian2D(2.4, xc_ref, yc_ref, x_stddev=x_stddev,
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
    xc_ref, yc_ref = 24.7, 25.2
    model = Gaussian2D(2.4, xc_ref, yc_ref, x_stddev=5.0, y_stddev=5.0)
    y, x = np.mgrid[0:50, 0:50]
    data = model(x, y)
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
    desired = (75, 50, 5)
    g = Gaussian1D(*desired)
    data = g(x)
    result = gaussian1d_moments(data)
    assert_allclose(result, desired, rtol=0, atol=1.e-6)


def test_fit2dgaussian_dof():
    data = np.ones((2, 2))
    result = fit_2dgaussian(data)
    assert result is None


class TestCutoutFootprint(object):
    def test_dataonly(self):
        data = np.ones((5, 5))
        position = (2, 2)
        result1 = cutout_footprint(data, position, 3)
        result2 = cutout_footprint(data, position, footprint=np.ones((3, 3)))
        assert_allclose(result1[:-2], result2[:-2])
        assert result1[-2] is None
        assert result2[-2] is None
        assert result1[-1] == result2[-1]

    def test_mask_error(self):
        data = error = np.ones((5, 5))
        mask = np.zeros_like(data, dtype=bool)
        position = (2, 2)
        box_size1 = 3
        box_size2 = (3, 3)
        footprint = np.ones((3, 3))
        result1 = cutout_footprint(data, position, box_size1, mask=mask,
                                   error=error)
        result2 = cutout_footprint(data, position, box_size2, mask=mask,
                                   error=error)
        result3 = cutout_footprint(data, position, box_size1,
                                   footprint=footprint, mask=mask,
                                   error=error)
        assert_allclose(result1[:-1], result2[:-1])
        assert_allclose(result1[:-1], result3[:-1])
        assert result1[-1] == result2[-1]

    def test_position_len(self):
        with pytest.raises(ValueError):
            cutout_footprint(np.ones((3, 3)), [1])

    def test_nofootprint(self):
        with pytest.raises(ValueError):
            cutout_footprint(np.ones((3, 3)), (1, 1), box_size=None,
                             footprint=None)

    def test_wrongboxsize(self):
        with pytest.raises(ValueError):
            cutout_footprint(np.ones((3, 3)), (1, 1), box_size=(1, 2, 3))
