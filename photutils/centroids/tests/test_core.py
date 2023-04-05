# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the core module.
"""

import itertools
from contextlib import nullcontext

import numpy as np
import pytest
from astropy.modeling.models import Gaussian2D
from astropy.utils.exceptions import AstropyUserWarning
from numpy.testing import assert_allclose

from photutils.centroids.core import (centroid_com, centroid_quadratic,
                                      centroid_sources)
from photutils.centroids.gaussian import centroid_1dg, centroid_2dg
from photutils.datasets import make_4gaussians_image, make_noise_image
from photutils.utils._optional_deps import HAS_SCIPY

XCEN = 25.7
YCEN = 26.2
XSTDS = [3.2, 4.0]
YSTDS = [5.7, 4.1]
THETAS = np.array([30.0, 45.0]) * np.pi / 180.0

DATA = np.zeros((3, 3))
DATA[0:2, 1] = 1.0
DATA[1, 0:2] = 1.0
DATA[1, 1] = 2.0

CENTROID_FUNCS = (centroid_com, centroid_quadratic, centroid_1dg,
                  centroid_2dg)


# NOTE: the fitting routines in astropy use scipy.optimize
@pytest.mark.skipif(not HAS_SCIPY, reason='scipy is required')
@pytest.mark.parametrize(('x_std', 'y_std', 'theta'),
                         list(itertools.product(XSTDS, YSTDS, THETAS)))
def test_centroid_comquad(x_std, y_std, theta):
    model = Gaussian2D(2.4, XCEN, YCEN, x_stddev=x_std, y_stddev=y_std,
                       theta=theta)
    y, x = np.mgrid[0:50, 0:47]
    data = model(x, y)
    xc, yc = centroid_com(data)
    assert_allclose((xc, yc), (XCEN, YCEN), rtol=0, atol=1.0e-3)

    xc, yc = centroid_quadratic(data)
    assert_allclose((xc, yc), (XCEN, YCEN), rtol=0, atol=0.015)

    # test with mask
    mask = np.zeros(data.shape, dtype=bool)
    data[10, 10] = 1.0e5
    mask[10, 10] = True
    xc, yc = centroid_com(data, mask=mask)
    assert_allclose((xc, yc), (XCEN, YCEN), rtol=0, atol=1.0e-3)

    xc, yc = centroid_quadratic(data, mask=mask)
    assert_allclose((xc, yc), (XCEN, YCEN), rtol=0, atol=0.015)


@pytest.mark.skipif(not HAS_SCIPY, reason='scipy is required')
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
        ctx = pytest.warns(AstropyUserWarning,
                           match='Input data contains non-finite values')

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
    with pytest.raises(ValueError):
        centroid_com(data, mask=mask)


@pytest.mark.skipif(not HAS_SCIPY, reason='scipy is required')
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

    with pytest.raises(ValueError):
        centroid_quadratic(data, xpeak=15, ypeak=5)
    with pytest.raises(ValueError):
        centroid_quadratic(data, xpeak=5, ypeak=15)
    with pytest.raises(ValueError):
        centroid_quadratic(data, xpeak=15, ypeak=15)


@pytest.mark.skipif(not HAS_SCIPY, reason='scipy is required')
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


@pytest.mark.skipif(not HAS_SCIPY, reason='scipy is required')
def test_centroid_quadratic_npts():
    data = np.zeros((3, 3))
    data[1, 1] = 1
    mask = np.zeros(data.shape, dtype=bool)
    mask[0, :] = True
    mask[2, :] = True
    with pytest.warns(AstropyUserWarning,
                      match='at least 6 unmasked data points'):
        centroid_quadratic(data, mask=mask)


@pytest.mark.skipif(not HAS_SCIPY, reason='scipy is required')
def test_centroid_quadratic_invalid_inputs():
    data = np.zeros((4, 4))
    mask = np.zeros((2, 2), dtype=bool)
    with pytest.raises(ValueError):
        centroid_quadratic(data, xpeak=3, ypeak=None)
    with pytest.raises(ValueError):
        centroid_quadratic(data, xpeak=None, ypeak=3)
    with pytest.raises(ValueError):
        centroid_quadratic(data, fit_boxsize=(2, 2, 2))
    with pytest.raises(ValueError):
        centroid_quadratic(data, fit_boxsize=(-2, 2))
    with pytest.raises(ValueError):
        centroid_quadratic(data, fit_boxsize=(2, 2))
    with pytest.raises(ValueError):
        centroid_quadratic(data, mask=mask)


@pytest.mark.skipif(not HAS_SCIPY, reason='scipy is required')
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
    with pytest.warns(AstropyUserWarning,
                      match='maximum value is at the edge'):
        xycen = centroid_quadratic(data)
    assert_allclose(xycen, (0, 0))


@pytest.mark.skipif(not HAS_SCIPY, reason='scipy is required')
class TestCentroidSources:
    def setup_class(self):
        ysize = 50
        xsize = 47
        yy, xx = np.mgrid[0:ysize, 0:xsize]
        data = np.zeros((ysize, xsize))
        xcen = (1, 25, 25, 35, 46)
        ycen = (1, 25, 12, 35, 49)
        for xc, yc in zip(xcen, ycen):
            model = Gaussian2D(10.0, xc, yc, x_stddev=2, y_stddev=2,
                               theta=0)
            data += model(xx, yy)
        self.xpos = xcen
        self.ypos = ycen
        self.data = data

    @staticmethod
    def test_centroid_sources():
        theta = np.pi / 6.0
        model = Gaussian2D(2.4, XCEN, YCEN, x_stddev=3.2, y_stddev=5.7,
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

        with pytest.raises(ValueError):
            centroid_sources(data, 25, [[26]], box_size=11)
        with pytest.raises(ValueError):
            centroid_sources(data, [[25]], 26, box_size=11)
        with pytest.raises(ValueError):
            centroid_sources(data, 25, 26, box_size=(1, 2, 3))
        with pytest.raises(ValueError):
            centroid_sources(data, 25, 26, box_size=None, footprint=None)
        with pytest.raises(ValueError):
            centroid_sources(data, 25, 26, footprint=np.ones((3, 3, 3)))

        def test_func(data):
            return data

        with pytest.raises(ValueError):
            centroid_sources(data, [25], 26, centroid_func=test_func)

    @pytest.mark.parametrize('centroid_func', CENTROID_FUNCS)
    def test_xypos(self, centroid_func):
        with pytest.raises(ValueError):
            centroid_sources(self.data, 47, 50, box_size=5,
                             centroid_func=centroid_func)

    def test_gaussian_fits_npts(self):
        xcen, ycen = centroid_sources(self.data, self.xpos, self.ypos,
                                      box_size=3, centroid_func=centroid_1dg)
        assert_allclose(xcen, np.full(5, np.nan))
        assert_allclose(ycen, np.full(5, np.nan))

        xcen, ycen = centroid_sources(self.data, self.xpos, self.ypos,
                                      box_size=3, centroid_func=centroid_2dg)
        xres = np.copy(self.xpos).astype(float)
        yres = np.copy(self.ypos).astype(float)
        xres[-1] = np.nan
        yres[-1] = np.nan
        assert_allclose(xcen, xres)
        assert_allclose(ycen, yres)

        xcen, ycen = centroid_sources(self.data, self.xpos, self.ypos,
                                      box_size=5, centroid_func=centroid_1dg)
        assert_allclose(xcen, xres)
        assert_allclose(ycen, yres)

        xcen, ycen = centroid_sources(self.data, self.xpos, self.ypos,
                                      box_size=3,
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

    def test_mask(self):
        xcen1, ycen1 = centroid_sources(self.data, 25, 23, box_size=(55, 55))

        mask = np.zeros(self.data.shape, dtype=bool)
        mask[0, 0] = True
        mask[24, 24] = True
        mask[11, 24] = True
        xcen2, ycen2 = centroid_sources(self.data, 25, 23,
                                        box_size=(55, 55), mask=mask)
        assert not np.allclose(xcen1, xcen2)
        assert not np.allclose(ycen1, ycen2)

    def test_error_none(self):
        xycen1 = centroid_sources(self.data, xpos=25, ypos=25, error=None,
                                  centroid_func=centroid_1dg)
        xycen2 = centroid_sources(self.data, xpos=25, ypos=25, error=None,
                                  centroid_func=centroid_2dg)
        assert_allclose(xycen1, ([25], [25]), atol=1.0e-3)
        assert_allclose(xycen2, ([25], [25]), atol=1.0e-3)

    def test_xypeaks_none(self):
        xycen1 = centroid_sources(self.data, xpos=25, ypos=25, error=None,
                                  xpeak=None, ypeak=25,
                                  centroid_func=centroid_quadratic)
        xycen2 = centroid_sources(self.data, xpos=25, ypos=25, error=None,
                                  xpeak=25, ypeak=None,
                                  centroid_func=centroid_quadratic)
        xycen3 = centroid_sources(self.data, xpos=25, ypos=25, error=None,
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
    with pytest.raises(ValueError):
        footprint = np.zeros((3, 3))
        _ = centroid_sources(data, x_init, y_init, footprint=footprint,
                             centroid_func=centroid_com)

    with pytest.raises(ValueError):
        footprint = np.zeros(data.shape, dtype=bool)
        _ = centroid_sources(data, x_init, y_init, footprint=footprint,
                             centroid_func=centroid_com)

    with pytest.raises(ValueError):
        mask = np.ones(data.shape, dtype=bool)
        _ = centroid_sources(data, x_init, y_init, box_size=11, mask=mask)
