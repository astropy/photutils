# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the core module.
"""

from contextlib import nullcontext

import astropy.units as u
import numpy as np
import pytest
from astropy.modeling.models import Gaussian2D
from astropy.utils.exceptions import AstropyUserWarning
from numpy.testing import assert_allclose

from photutils.centroids.core import (centroid_com, centroid_quadratic,
                                      centroid_sources)
from photutils.centroids.gaussian import centroid_1dg, centroid_2dg
from photutils.datasets import make_4gaussians_image, make_noise_image


@pytest.fixture(name='test_simple_data')
def fixture_test_simple_data():
    xcen = 25.7
    ycen = 26.2
    data = np.zeros((3, 3))
    data[0:2, 1] = 1.0
    data[1, 0:2] = 1.0
    data[1, 1] = 2.0
    return data, xcen, ycen


@pytest.fixture(name='test_data')
def fixture_test_data():
    ysize = 50
    xsize = 47
    yy, xx = np.mgrid[0:ysize, 0:xsize]
    data = np.zeros((ysize, xsize))
    xpos = (1, 25, 25, 35, 46)
    ypos = (1, 25, 12, 35, 49)
    for xc, yc in zip(xpos, ypos, strict=True):
        model = Gaussian2D(10.0, xc, yc, x_stddev=2, y_stddev=2,
                           theta=0)
        data += model(xx, yy)
    return data, xpos, ypos


# NOTE: the fitting routines in astropy use scipy.optimize
@pytest.mark.parametrize('x_std', [3.2, 4.0])
@pytest.mark.parametrize('y_std', [5.7, 4.1])
@pytest.mark.parametrize('theta', np.deg2rad([30.0, 45.0]))
@pytest.mark.parametrize('units', [True, False])
def test_centroid_comquad(test_simple_data, x_std, y_std, theta, units):
    data, xcen, ycen = test_simple_data

    if units:
        data = data * u.nJy

    model = Gaussian2D(2.4, xcen, ycen, x_stddev=x_std, y_stddev=y_std,
                       theta=theta)
    y, x = np.mgrid[0:50, 0:47]
    data = model(x, y)
    xc, yc = centroid_com(data)
    assert_allclose((xc, yc), (xcen, ycen), rtol=0, atol=1.0e-3)

    xc, yc = centroid_quadratic(data)
    assert_allclose((xc, yc), (xcen, ycen), rtol=0, atol=0.015)

    # test with mask
    mask = np.zeros(data.shape, dtype=bool)
    data[10, 10] = 1.0e5
    mask[10, 10] = True
    xc, yc = centroid_com(data, mask=mask)
    assert_allclose((xc, yc), (xcen, ycen), rtol=0, atol=1.0e-3)

    xc, yc = centroid_quadratic(data, mask=mask)
    assert_allclose((xc, yc), (xcen, ycen), rtol=0, atol=0.015)


@pytest.mark.parametrize('ndim', [1, 2, 3, 4, 5])
def test_centroid_com_zero_sum(ndim):
    data = np.zeros([10] * ndim)
    cen = centroid_com(data)
    assert cen.shape == (ndim,)
    for cen_ in cen:
        assert np.isnan(cen_)


@pytest.mark.parametrize('use_mask', [True, False])
def test_centroid_comquad_nan_withmask(use_mask):
    xc_ref = 24.7
    yc_ref = 25.2
    model = Gaussian2D(2.4, xc_ref, yc_ref, x_stddev=5.0, y_stddev=5.0)
    y, x = np.mgrid[0:50, 0:50]
    data = model(x, y)
    data[20, :] = np.nan
    if use_mask:
        mask = np.zeros(data.shape, dtype=bool)
        mask[20, :] = True
        nwarn = 0
        ctx = nullcontext()
    else:
        mask = None
        nwarn = 1
        match = 'Input data contains non-finite values'
        ctx = pytest.warns(AstropyUserWarning, match=match)

    with ctx as warnlist:
        xc, yc = centroid_com(data, mask=mask)
        assert_allclose(xc, xc_ref, rtol=0, atol=1.0e-3)
        assert yc > yc_ref
        if nwarn == 1:
            assert len(warnlist) == nwarn

    with ctx as warnlist:
        xc, yc = centroid_quadratic(data, mask=mask)
        assert_allclose(xc, xc_ref, rtol=0, atol=0.15)
        if nwarn == 1:
            assert len(warnlist) == nwarn


def test_centroid_com_allmask():
    xc_ref = 24.7
    yc_ref = 25.2
    model = Gaussian2D(2.4, xc_ref, yc_ref, x_stddev=5.0, y_stddev=5.0)
    y, x = np.mgrid[0:50, 0:50]
    data = model(x, y)
    mask = np.ones(data.shape, dtype=bool)
    xc, yc = centroid_com(data, mask=mask)
    assert np.isnan(xc)
    assert np.isnan(yc)

    data = np.zeros((25, 25))
    xc, yc = centroid_com(data, mask=None)
    assert np.isnan(xc)
    assert np.isnan(yc)


def test_centroid_com_invalid_inputs():
    data = np.zeros((4, 4))
    mask = np.zeros((2, 2), dtype=bool)
    match = 'data and mask must have the same shape'
    with pytest.raises(ValueError, match=match):
        centroid_com(data, mask=mask)


def test_centroid_quadratic_xypeak():
    data = np.zeros((11, 11))
    data[5, 5] = 100
    data[7, 7] = 110
    data[9, 9] = 120

    xycen1 = centroid_quadratic(data, fit_boxsize=3)
    assert_allclose(xycen1, (9, 9))

    xycen2 = centroid_quadratic(data, xpeak=5, ypeak=5, fit_boxsize=3)
    assert_allclose(xycen2, (5, 5))

    xycen3 = centroid_quadratic(data, xpeak=5, ypeak=5, fit_boxsize=3,
                                search_boxsize=5)
    assert_allclose(xycen3, (7, 7))

    match = 'xpeak is outside of the input data'
    with pytest.raises(ValueError, match=match):
        centroid_quadratic(data, xpeak=15, ypeak=5)
    match = 'ypeak is outside of the input data'
    with pytest.raises(ValueError, match=match):
        centroid_quadratic(data, xpeak=5, ypeak=15)
    match = 'xpeak is outside of the input data'
    with pytest.raises(ValueError, match=match):
        centroid_quadratic(data, xpeak=15, ypeak=15)


def test_centroid_quadratic_nan():
    gmodel = Gaussian2D(42.1, 47.8, 52.4, 4.7, 4.7, 0)
    yy, xx = np.mgrid[0:100, 0:100]
    data = gmodel(xx, yy)
    error = make_noise_image(data.shape, mean=0., stddev=2.4, seed=123)
    data += error

    data[50, 50] = np.nan
    mask = ~np.isfinite(data)
    xycen = centroid_quadratic(data, xpeak=47, ypeak=52, mask=mask)
    assert_allclose(xycen, [47.58324, 51.827182])


def test_centroid_quadratic_npts():
    data = np.zeros((3, 3))
    data[1, 1] = 1
    mask = np.zeros(data.shape, dtype=bool)
    mask[0, :] = True
    mask[2, :] = True
    match = 'at least 6 unmasked data points'
    with pytest.warns(AstropyUserWarning, match=match):
        centroid_quadratic(data, mask=mask)


def test_centroid_quadratic_invalid_inputs():
    data = np.zeros((4, 4, 4))
    match = 'data must be a 2D array'
    with pytest.raises(ValueError, match=match):
        centroid_quadratic(data)

    data = np.zeros((4, 4))
    mask = np.zeros((2, 2), dtype=bool)
    match = 'xpeak and ypeak must both be input or "None"'
    with pytest.raises(ValueError, match=match):
        centroid_quadratic(data, xpeak=3, ypeak=None)
    with pytest.raises(ValueError, match=match):
        centroid_quadratic(data, xpeak=None, ypeak=3)

    match = 'fit_boxsize must have 1 or 2 elements'
    with pytest.raises(ValueError, match=match):
        centroid_quadratic(data, fit_boxsize=(2, 2, 2))

    match = 'fit_boxsize must have an odd value for both axes'
    with pytest.raises(ValueError, match=match):
        centroid_quadratic(data, fit_boxsize=(-2, 2))
    with pytest.raises(ValueError, match=match):
        centroid_quadratic(data, fit_boxsize=(2, 2))

    match = 'data and mask must have the same shape'
    with pytest.raises(ValueError, match=match):
        centroid_quadratic(data, mask=mask)


def test_centroid_quadratic_edge():
    data = np.zeros((11, 11))
    data[1, 1] = 100
    data[9, 9] = 100

    xycen = centroid_quadratic(data, xpeak=1, ypeak=1, fit_boxsize=5)
    assert_allclose(xycen, (0.923077, 0.923077))

    xycen = centroid_quadratic(data, xpeak=9, ypeak=9, fit_boxsize=5)
    assert_allclose(xycen, (9.076923, 9.076923))

    data = np.zeros((5, 5))
    data[0, 0] = 100
    match = 'maximum value is at the edge'
    with pytest.warns(AstropyUserWarning, match=match):
        xycen = centroid_quadratic(data)
    assert_allclose(xycen, (0, 0))


class TestCentroidSources:

    @staticmethod
    def test_centroid_sources():
        theta = np.pi / 6.0
        model = Gaussian2D(2.4, 25.7, 26.2, x_stddev=3.2, y_stddev=5.7,
                           theta=theta)
        y, x = np.mgrid[0:50, 0:47]
        data = model(x, y)
        error = np.ones(data.shape, dtype=float)
        mask = np.zeros(data.shape, dtype=bool)
        mask[10, 10] = True
        xpos = [25.0]
        ypos = [26.0]
        xc, yc = centroid_sources(data, xpos, ypos, box_size=21, mask=mask)
        assert_allclose(xc, (25.67,), atol=1e-1)
        assert_allclose(yc, (26.18,), atol=1e-1)

        xc, yc = centroid_sources(data, xpos, ypos, error=error, box_size=11,
                                  centroid_func=centroid_1dg)
        assert_allclose(xc, (25.67,), atol=1e-1)
        assert_allclose(yc, (26.41,), atol=1e-1)

        match = 'xpos must be a 1D array'
        with pytest.raises(ValueError, match=match):
            centroid_sources(data, [[25]], 26, box_size=11)
        match = 'ypos must be a 1D array'
        with pytest.raises(ValueError, match=match):
            centroid_sources(data, 25, [[26]], box_size=11)
        match = 'box_size must have 1 or 2 elements'
        with pytest.raises(ValueError, match=match):
            centroid_sources(data, 25, 26, box_size=(1, 2, 3))
        match = 'box_size or footprint must be defined'
        with pytest.raises(ValueError, match=match):
            centroid_sources(data, 25, 26, box_size=None, footprint=None)
        match = 'footprint must be a 2D array'
        with pytest.raises(ValueError, match=match):
            centroid_sources(data, 25, 26, footprint=np.ones((3, 3, 3)))

        def test_func(data):
            return data

        match = 'The input "centroid_func" must have a "mask" keyword'
        with pytest.raises(ValueError, match=match):
            centroid_sources(data, [25], 26, centroid_func=test_func)

    @pytest.mark.parametrize('centroid_func', [centroid_com,
                                               centroid_quadratic,
                                               centroid_1dg, centroid_2dg])
    def test_xypos(self, test_data, centroid_func):
        data = test_data[0]
        match = 'xpos, ypos values contains points outside of input data'
        with pytest.raises(ValueError, match=match):
            centroid_sources(data, 47, 50, box_size=5,
                             centroid_func=centroid_func)

    def test_gaussian_fits_npts(self, test_data):
        data, xpos, ypos = test_data
        xcen, ycen = centroid_sources(data, xpos, ypos, box_size=3,
                                      centroid_func=centroid_1dg)

        xres = np.copy(xpos).astype(float)
        yres = np.copy(ypos).astype(float)
        xres[-1] = 46.689208
        yres[-1] = 49.689208
        assert_allclose(xcen, xres)
        assert_allclose(ycen, yres)

        xcen, ycen = centroid_sources(data, xpos, ypos, box_size=3,
                                      centroid_func=centroid_2dg)
        xres[-1] = np.nan
        yres[-1] = np.nan
        assert_allclose(xcen, xres)
        assert_allclose(ycen, yres)

        xcen, ycen = centroid_sources(data, xpos, ypos, box_size=5,
                                      centroid_func=centroid_1dg)
        assert_allclose(xcen, xpos)
        assert_allclose(ycen, ypos)

        xcen, ycen = centroid_sources(data, xpos, ypos, box_size=3,
                                      centroid_func=centroid_quadratic)
        assert_allclose(xcen, xres)
        assert_allclose(ycen, yres)

    @pytest.mark.filterwarnings(r'ignore:.*no quadratic fit was performed')
    def test_centroid_quadratic_kwargs(self):
        data = np.zeros((11, 11))
        data[5, 5] = 100
        data[7, 7] = 110
        data[9, 9] = 120

        xycen1 = centroid_sources(data, xpos=5, ypos=5, box_size=9,
                                  centroid_func=centroid_quadratic,
                                  fit_boxsize=3)
        assert_allclose(xycen1, ([9], [9]))

        xycen2 = centroid_sources(data, xpos=7, ypos=7, box_size=5,
                                  centroid_func=centroid_quadratic,
                                  fit_boxsize=3)
        assert_allclose(xycen2, ([9], [9]))

        xycen3 = centroid_sources(data, xpos=7, ypos=7, box_size=5,
                                  centroid_func=centroid_quadratic,
                                  xpeak=7, ypeak=7, fit_boxsize=3)
        assert_allclose(xycen3, ([7], [7]))

        xycen4 = centroid_sources(data, xpos=5, ypos=5, box_size=5,
                                  centroid_func=centroid_quadratic,
                                  xpeak=5, ypeak=5, fit_boxsize=3)
        assert_allclose(xycen4, ([5], [5]))

        xycen5 = centroid_sources(data, xpos=5, ypos=5, box_size=5,
                                  centroid_func=centroid_quadratic,
                                  fit_boxsize=5)
        assert_allclose(xycen5, ([7], [7]))

    def test_centroid_quadratic_mask(self):
        """
        Regression test to check that when a mask is input the original
        data is not alterned.
        """
        xc_ref = 24.7
        yc_ref = 25.2
        model = Gaussian2D(2.4, xc_ref, yc_ref, x_stddev=5.0, y_stddev=5.0)
        y, x = np.mgrid[0:51, 0:51]
        data = model(x, y)
        mask = data < 1
        xycen = centroid_quadratic(data, mask=mask)
        assert ~np.any(np.isnan(data))
        assert_allclose(xycen, (xc_ref, yc_ref), atol=0.01)

    def test_mask(self, test_data):
        data = test_data[0]
        xcen1, ycen1 = centroid_sources(data, 25, 23, box_size=(55, 55))

        mask = np.zeros(data.shape, dtype=bool)
        mask[0, 0] = True
        mask[24, 24] = True
        mask[11, 24] = True
        xcen2, ycen2 = centroid_sources(data, 25, 23, box_size=(55, 55),
                                        mask=mask)
        assert not np.allclose(xcen1, xcen2)
        assert not np.allclose(ycen1, ycen2)

    def test_error_none(self, test_data):
        data = test_data[0]
        xycen1 = centroid_sources(data, xpos=25, ypos=25, error=None,
                                  centroid_func=centroid_1dg)
        xycen2 = centroid_sources(data, xpos=25, ypos=25, error=None,
                                  centroid_func=centroid_2dg)
        assert_allclose(xycen1, ([25], [25]), atol=1.0e-3)
        assert_allclose(xycen2, ([25], [25]), atol=1.0e-3)

    def test_xypeaks_none(self, test_data):
        data = test_data[0]
        xycen1 = centroid_sources(data, xpos=25, ypos=25, error=None,
                                  xpeak=None, ypeak=25,
                                  centroid_func=centroid_quadratic)
        xycen2 = centroid_sources(data, xpos=25, ypos=25, error=None,
                                  xpeak=25, ypeak=None,
                                  centroid_func=centroid_quadratic)
        xycen3 = centroid_sources(data, xpos=25, ypos=25, error=None,
                                  xpeak=None, ypeak=None,
                                  centroid_func=centroid_quadratic)
        assert_allclose(xycen1, ([25], [25]), atol=1.0e-3)
        assert_allclose(xycen2, ([25], [25]), atol=1.0e-3)
        assert_allclose(xycen3, ([25], [25]), atol=1.0e-3)


def test_cutout_mask():
    """
    Test that the cutout is not completely masked (see #1514).
    """
    data = make_4gaussians_image()
    x_init = (25, 91, 151, 160)
    y_init = (40, 61, 24, 71)
    footprint = np.zeros((3, 3))
    match = 'is completely masked'
    with pytest.raises(ValueError, match=match):
        _ = centroid_sources(data, x_init, y_init, footprint=footprint,
                             centroid_func=centroid_com)

    footprint = np.zeros(data.shape, dtype=bool)
    with pytest.raises(ValueError, match=match):
        _ = centroid_sources(data, x_init, y_init, footprint=footprint,
                             centroid_func=centroid_com)

    mask = np.ones(data.shape, dtype=bool)
    with pytest.raises(ValueError, match=match):
        _ = centroid_sources(data, x_init, y_init, box_size=11, mask=mask)
