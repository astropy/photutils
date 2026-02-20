# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the core module.
"""

from contextlib import nullcontext
from unittest.mock import patch

import astropy.units as u
import numpy as np
import pytest
from astropy.modeling.models import Gaussian2D
from astropy.utils.exceptions import (AstropyDeprecationWarning,
                                      AstropyUserWarning)
from numpy.testing import assert_allclose

from photutils.centroids.core import (CentroidQuadratic, _process_data_mask,
                                      centroid_com, centroid_quadratic,
                                      centroid_sources)
from photutils.centroids.gaussian import centroid_1dg, centroid_2dg
from photutils.datasets import make_4gaussians_image, make_noise_image


@pytest.fixture(name='test_data')
def fixture_test_data():
    """
    Create test data with multiple Gaussian sources.
    """
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


@pytest.fixture(name='nan_data')
def fixture_nan_data():
    """
    Create Gaussian test data with NaN values in row 20.
    """
    xc_ref = 24.7
    yc_ref = 25.2
    model = Gaussian2D(2.4, xc_ref, yc_ref, x_stddev=5.0, y_stddev=5.0)
    y, x = np.mgrid[0:50, 0:50]
    data = model(x, y)
    data[20, :] = np.nan
    return data, xc_ref, yc_ref


@pytest.mark.parametrize('x_std', [3.2, 4.0])
@pytest.mark.parametrize('y_std', [5.7, 4.1])
@pytest.mark.parametrize('theta', np.deg2rad([30.0, 45.0]))
@pytest.mark.parametrize('units', [True, False])
def test_centroid_com(x_std, y_std, theta, units):
    """
    Test centroid_com with Gaussian data.
    """
    xcen = 25.7
    ycen = 26.2
    model = Gaussian2D(2.4, xcen, ycen, x_stddev=x_std, y_stddev=y_std,
                       theta=theta)
    y, x = np.mgrid[0:50, 0:47]
    data = model(x, y)

    if units:
        data = data * u.nJy

    xc, yc = centroid_com(data)
    assert_allclose((xc, yc), (xcen, ycen), rtol=0, atol=1.0e-3)

    # Test with mask
    x0 = 11
    y0 = 15
    data[y0, x0] = 1.0e5 * u.nJy if units else 1.0e5
    mask = np.zeros(data.shape, dtype=bool)
    mask[y0, x0] = True
    xc, yc = centroid_com(data, mask=mask)
    assert_allclose((xc, yc), (xcen, ycen), rtol=0, atol=1.0e-3)


@pytest.mark.parametrize('use_mask', [True, False])
def test_centroid_com_nan_withmask(nan_data, use_mask):
    """
    Test centroid_com with NaN values and optional mask.
    """
    data, xc_ref, yc_ref = nan_data
    if use_mask:
        mask = np.zeros(data.shape, dtype=bool)
        mask[20, :] = True
        ctx = nullcontext()
    else:
        mask = None
        match = 'Input data contains non-finite values'
        ctx = pytest.warns(AstropyUserWarning, match=match)

    with ctx as warnlist:
        xc, yc = centroid_com(data, mask=mask)
        assert_allclose(xc, xc_ref, rtol=0, atol=1.0e-3)
        assert yc > yc_ref
        if not use_mask:
            assert len(warnlist) == 1


def test_centroid_com_allmask():
    """
    Test centroid_com when all data are masked or zero.
    """
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
    """
    Test centroid_com with invalid inputs.
    """
    data = np.zeros((4, 4))
    mask = np.zeros((2, 2), dtype=bool)
    match = 'data and mask must have the same shape'
    with pytest.raises(ValueError, match=match):
        centroid_com(data, mask=mask)


@pytest.mark.parametrize('ndim', [1, 2, 3, 4, 5])
def test_centroid_com_zero_sum(ndim):
    """
    Test centroid_com when the sum of the data is zero, which should
    return NaN.
    """
    data = np.zeros([10] * ndim)
    cen = centroid_com(data)
    assert cen.shape == (ndim,)
    for cen_ in cen:
        assert np.isnan(cen_)


def test_centroid_com_masked_array():
    """
    Test centroid_com with a MaskedArray input.
    """
    data = np.ma.array([[1.0, 1.0, 1.0],
                        [1.0, 100.0, 1.0],
                        [10.0, 1.0, 1.0]],
                       mask=[[0, 0, 0],
                             [0, 1, 0],
                             [0, 0, 0]])

    # If mask is respected, peak (1, 1) is ignored and centroid will be
    # pulled towards (0, 0).
    xc1, yc1 = centroid_com(data)

    # Compare with explicit mask
    xc2, yc2 = centroid_com(data.data, mask=data.mask)
    assert xc1 == xc2
    assert yc1 == yc2

    # Combined mask test
    mask_arg = np.zeros(data.shape, dtype=bool)
    mask_arg[0, 0] = True
    # Now both (1, 1) and (0, 0) are masked.
    xc3, yc3 = centroid_com(data, mask=mask_arg)

    full_mask = data.mask | mask_arg
    xc4, yc4 = centroid_com(data.data, mask=full_mask)
    assert xc3 == xc4
    assert yc3 == yc4


def test_process_data_mask_no_mutation():
    """
    Test that _process_data_mask does not mutate the input mask when the
    data is a MaskedArray.
    """
    masked_data = np.ma.array([[1.0, 2.0], [3.0, 4.0]],
                              mask=[[False, True], [False, False]])
    input_mask = np.zeros((2, 2), dtype=bool)
    input_mask_orig = input_mask.copy()

    _process_data_mask(masked_data, input_mask)
    assert np.array_equal(input_mask, input_mask_orig)


def test_process_data_mask_masked_array_returns_ndarray():
    """
    Test that _process_data_mask returns a plain ndarray (not a
    MaskedArray) when given MaskedArray input so that callers do not
    have to handle MaskedArray semantics.
    """
    masked_data = np.ma.array([[1.0, 2.0], [3.0, 4.0]],
                              mask=[[False, True], [False, False]])
    result = _process_data_mask(masked_data, None)
    assert not isinstance(result, np.ma.MaskedArray)
    assert isinstance(result, np.ndarray)


@pytest.mark.parametrize('x_std', [3.2, 4.0])
@pytest.mark.parametrize('y_std', [5.7, 4.1])
@pytest.mark.parametrize('theta', np.deg2rad([30.0, 45.0]))
@pytest.mark.parametrize('units', [True, False])
def test_centroid_quadratic(x_std, y_std, theta, units):
    """
    Test centroid_quadratic with Gaussian data.
    """
    xcen = 25.7
    ycen = 26.2
    model = Gaussian2D(2.4, xcen, ycen, x_stddev=x_std, y_stddev=y_std,
                       theta=theta)
    y, x = np.mgrid[0:50, 0:47]
    data = model(x, y)
    if units:
        data = data * u.nJy

    xc, yc = centroid_quadratic(data)
    assert_allclose((xc, yc), (xcen, ycen), rtol=0, atol=0.015)

    # Test with mask
    x0 = 11
    y0 = 15
    data[y0, x0] = 1.0e5 * u.nJy if units else 1.0e5
    mask = np.zeros(data.shape, dtype=bool)
    mask[y0, x0] = True
    data[y0, x0] = 1.0e5 * u.nJy if units else 1.0e5
    xc, yc = centroid_quadratic(data, mask=mask)
    assert_allclose((xc, yc), (xcen, ycen), rtol=0, atol=0.015)


def test_centroid_quadratic_xypeak():
    """
    Test centroid_quadratic with xpeak and ypeak inputs.
    """
    data = np.zeros((11, 11))
    data[5, 5] = 100
    data[7, 7] = 110
    data[9, 9] = 120

    xycen1 = centroid_quadratic(data, fit_boxsize=3)
    assert_allclose(xycen1, (9, 9))

    with pytest.warns(AstropyDeprecationWarning):
        xycen2 = centroid_quadratic(data, xpeak=5, ypeak=5, fit_boxsize=3)
    assert_allclose(xycen2, (5, 5))

    with pytest.warns(AstropyDeprecationWarning):
        xycen3 = centroid_quadratic(data, xpeak=5, ypeak=5, fit_boxsize=3,
                                    search_boxsize=5)
    assert_allclose(xycen3, (7, 7))

    match = 'xpeak is outside the input data'
    with (pytest.warns(AstropyDeprecationWarning),
          pytest.raises(ValueError, match=match)):
        centroid_quadratic(data, xpeak=15, ypeak=5)
    with (pytest.warns(AstropyDeprecationWarning),
          pytest.raises(ValueError, match=match)):
        centroid_quadratic(data, xpeak=15, ypeak=15)

    match = 'ypeak is outside the input data'
    with (pytest.warns(AstropyDeprecationWarning),
          pytest.raises(ValueError, match=match)):
        centroid_quadratic(data, xpeak=5, ypeak=15)


def test_centroid_quadratic_nan():
    """
    Test centroid_quadratic with NaN values.
    """
    gmodel = Gaussian2D(42.1, 47.8, 52.4, 4.7, 4.7, 0)
    yy, xx = np.mgrid[0:100, 0:100]
    data = gmodel(xx, yy)
    error = make_noise_image(data.shape, mean=0., stddev=2.4, seed=123)
    data += error

    data[50, 50] = np.nan
    mask = ~np.isfinite(data)
    xycen = centroid_quadratic(data, mask=mask)
    assert_allclose(xycen, [47.58324, 51.827182])


@pytest.mark.parametrize('use_mask', [True, False])
def test_centroid_quadratic_nan_withmask(nan_data, use_mask):
    """
    Test centroid_quadratic with NaN values and optional mask.
    """
    data, xc_ref, yc_ref = nan_data
    if use_mask:
        mask = np.zeros(data.shape, dtype=bool)
        mask[20, :] = True
        ctx = nullcontext()
    else:
        mask = None
        match = 'Input data contains non-finite values'
        ctx = pytest.warns(AstropyUserWarning, match=match)

    with ctx as warnlist:
        xc, yc = centroid_quadratic(data, mask=mask)
        assert_allclose(xc, xc_ref, rtol=0, atol=0.15)
        assert_allclose(yc, yc_ref, rtol=0, atol=0.15)
        if not use_mask:
            assert len(warnlist) == 1


def test_centroid_quadratic_npts():
    """
    Test centroid_quadratic with insufficient unmasked data points.
    """
    data = np.zeros((3, 3))
    data[1, 1] = 1
    mask = np.zeros(data.shape, dtype=bool)
    mask[0, :] = True
    mask[2, :] = True
    match = 'at least 6 unmasked data points'
    with pytest.warns(AstropyUserWarning, match=match):
        centroid_quadratic(data, mask=mask)


def test_centroid_quadratic_invalid_inputs():
    """
    Test centroid_quadratic with invalid inputs.
    """
    data = np.zeros((4, 4, 4))
    match = 'data must be a 2D array'
    with pytest.raises(ValueError, match=match):
        centroid_quadratic(data)

    data = np.zeros((4, 4))
    mask = np.zeros((2, 2), dtype=bool)
    match = 'xpeak and ypeak must both be input or "None"'
    with (pytest.warns(AstropyDeprecationWarning),
          pytest.raises(ValueError, match=match)):
        centroid_quadratic(data, xpeak=3, ypeak=None)
    with (pytest.warns(AstropyDeprecationWarning),
          pytest.raises(ValueError, match=match)):
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
    """
    Test centroid_quadratic when the maximum is at the edge.
    """
    data = np.zeros((11, 11))
    data[1, 1] = 100
    data[9, 9] = 100

    with pytest.warns(AstropyDeprecationWarning):
        xycen = centroid_quadratic(data, xpeak=1, ypeak=1, fit_boxsize=5)
    assert_allclose(xycen, (0.923077, 0.923077))

    with pytest.warns(AstropyDeprecationWarning):
        xycen = centroid_quadratic(data, xpeak=9, ypeak=9, fit_boxsize=5)
    assert_allclose(xycen, (9.076923, 9.076923))

    data = np.zeros((5, 5))
    data[0, 0] = 100
    match = 'maximum value is at the edge'
    with pytest.warns(AstropyUserWarning, match=match):
        xycen = centroid_quadratic(data)
    assert_allclose(xycen, (0, 0))


def test_centroid_quadratic_fit_failed():
    """
    Test centroid_quadratic when the quadratic fit fails.

    This tests the LinAlgError exception handling. Since lstsq is
    very robust (uses SVD internally), we use mocking to trigger this
    condition.
    """
    data = np.zeros((11, 11))
    data[5, 5] = 10.0
    data[4, 5] = 8.0
    data[6, 5] = 8.0
    data[5, 4] = 8.0
    data[5, 6] = 8.0

    with patch('numpy.linalg.lstsq', side_effect=np.linalg.LinAlgError):
        match = 'quadratic fit failed'
        with (pytest.warns(AstropyDeprecationWarning),
              pytest.warns(AstropyUserWarning, match=match)):
            xycen = centroid_quadratic(data, xpeak=5, ypeak=5, fit_boxsize=5)
        assert np.isnan(xycen[0])
        assert np.isnan(xycen[1])


def test_centroid_quadratic_no_maximum():
    """
    Test centroid_quadratic when the quadratic fit does not have a
    maximum.

    This tests the case where the fitted polynomial has a saddle point
    or minimum instead of a maximum (det <= 0 or positive curvature).
    """
    # Create data with a saddle-like pattern that will result in a
    # quadratic fit without a proper maximum
    data = np.zeros((11, 11))
    y, x = np.mgrid[0:11, 0:11]

    # Create a saddle: z = x^2 - y^2 (positive curvature in x, negative
    # in y)
    data = (x - 5.0)**2 - (y - 5.0)**2 + 10
    # Add a peak so the fit box centers there
    data[5, 5] = 20.0

    match = 'quadratic fit does not have a maximum'
    with (pytest.warns(AstropyDeprecationWarning),
          pytest.warns(AstropyUserWarning, match=match)):
        xycen = centroid_quadratic(data, xpeak=5, ypeak=5, fit_boxsize=5)
    assert np.isnan(xycen[0])
    assert np.isnan(xycen[1])


def test_centroid_quadratic_max_outside_image():
    """
    Test centroid_quadratic when the polynomial maximum falls outside
    the image.

    This tests the case where the quadratic fit has a valid maximum but
    it lies outside the image boundaries.
    """
    # Create data where values increase toward the origin (0, 0) but
    # with a local peak at (3, 3). This causes the quadratic fit to
    # extrapolate the maximum outside the image boundaries.
    data = np.zeros((7, 7))
    y, x = np.mgrid[0:7, 0:7]
    data = 10 - x.astype(float) - y.astype(float)
    data = np.maximum(data, 0.1)
    data[3, 3] = 6.0  # local peak to center the fit

    match = 'quadratic polynomial maximum value falls outside'
    with (pytest.warns(AstropyDeprecationWarning),
          pytest.warns(AstropyUserWarning, match=match)):
        xycen = centroid_quadratic(data, xpeak=3, ypeak=3, fit_boxsize=5)
    assert np.isnan(xycen[0])
    assert np.isnan(xycen[1])


class TestCentroidSources:
    """
    Test the centroid_sources function.
    """

    @staticmethod
    def test_centroid_sources():
        """
        Test centroid_sources with Gaussian data.
        """
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
        """
        Test centroid_sources with xpos/ypos outside data range.
        """
        data = test_data[0]
        match = 'xpos, ypos values contain points outside the input data'
        with pytest.raises(ValueError, match=match):
            centroid_sources(data, 47, 50, box_size=5,
                             centroid_func=centroid_func)

    def test_gaussian_fits_npts(self, test_data):
        """
        Test centroid_sources with Gaussian fits with insufficient
        points.
        """
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

    def test_centroid_quadratic_mask(self):
        """
        Test centroid_sources with centroid_quadratic and a mask.

        The original data should not be altered when a mask is input.
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
        """
        Test centroid_sources with mask input.
        """
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
        """
        Test centroid_sources with error=None for Gaussian centroids.
        """
        data = test_data[0]
        xycen1 = centroid_sources(data, xpos=25, ypos=25, error=None,
                                  centroid_func=centroid_1dg)
        xycen2 = centroid_sources(data, xpos=25, ypos=25, error=None,
                                  centroid_func=centroid_2dg)
        assert_allclose(xycen1, ([25], [25]), atol=1.0e-3)
        assert_allclose(xycen2, ([25], [25]), atol=1.0e-3)

    @pytest.mark.filterwarnings(r'ignore:.*was deprecated')
    def test_xypeaks_none(self, test_data):
        """
        Test centroid_sources with xpeak and ypeak as None for
        centroid_quadratic.
        """
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

    def test_centroid_quadratic_kwargs(self):
        """
        Test centroid_sources with centroid_quadratic and various
        keyword arguments.
        """
        data = np.zeros((11, 11))
        data[5, 5] = 100
        data[7, 7] = 110
        data[9, 9] = 120

        with pytest.warns(AstropyDeprecationWarning):
            xycen3 = centroid_sources(data, xpos=7, ypos=7, box_size=5,
                                      centroid_func=centroid_quadratic,
                                      xpeak=7, ypeak=7, fit_boxsize=3)
        assert_allclose(xycen3, ([7], [7]))


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


class TestCentroidQuadraticClass:
    """
    Test the CentroidQuadratic class.
    """

    @pytest.mark.parametrize('x_std', [3.2, 4.0])
    @pytest.mark.parametrize('y_std', [5.7, 4.1])
    @pytest.mark.parametrize('theta', np.deg2rad([30.0, 45.0]))
    def test_basic(self, x_std, y_std, theta):
        """
        Test basic CentroidQuadratic functionality.
        """
        xcen = 25.7
        ycen = 26.2
        model = Gaussian2D(2.4, xcen, ycen, x_stddev=x_std, y_stddev=y_std,
                           theta=theta)
        y, x = np.mgrid[0:50, 0:47]
        data = model(x, y)

        # Test with default parameters
        centroid_func = CentroidQuadratic()
        xc, yc = centroid_func(data)
        assert_allclose((xc, yc), (xcen, ycen), rtol=0, atol=0.015)

    def test_mask(self):
        """
        Test CentroidQuadratic with mask input.
        """
        xcen = 25.7
        ycen = 26.2
        model = Gaussian2D(2.4, xcen, ycen, x_stddev=3.2, y_stddev=5.7,
                           theta=0)
        y, x = np.mgrid[0:50, 0:47]
        data = model(x, y)

        # Add an outlier
        x0 = 11
        y0 = 15
        data[y0, x0] = 1.0e5
        mask = np.zeros(data.shape, dtype=bool)
        mask[y0, x0] = True

        centroid_func = CentroidQuadratic()
        xc, yc = centroid_func(data, mask=mask)
        assert_allclose((xc, yc), (xcen, ycen), rtol=0, atol=0.015)

    def test_fit_boxsize(self):
        """
        Test CentroidQuadratic with custom fit_boxsize.
        """
        data = np.zeros((11, 11))
        data[5, 5] = 100
        data[7, 7] = 110
        data[9, 9] = 120

        centroid_func = CentroidQuadratic(fit_boxsize=3)
        xycen = centroid_func(data)
        assert_allclose(xycen, (9, 9))

    def test_with_centroid_sources(self):
        """
        Test CentroidQuadratic with centroid_sources function.
        """
        data = np.zeros((11, 11))
        data[5, 5] = 100
        data[7, 7] = 110
        data[9, 9] = 120

        # Test with custom fit_boxsize
        centroid_func = CentroidQuadratic(fit_boxsize=3)
        xycen = centroid_sources(data, xpos=5, ypos=5, box_size=7,
                                 centroid_func=centroid_func)
        assert_allclose(xycen, ([7], [7]))

    def test_repr(self):
        """
        Test CentroidQuadratic __repr__ method.
        """
        centroid_func = CentroidQuadratic()
        cls_repr = repr(centroid_func)
        assert cls_repr == 'CentroidQuadratic(fit_boxsize=5)'

        centroid_func = CentroidQuadratic(fit_boxsize=3)
        cls_repr = repr(centroid_func)
        assert cls_repr == 'CentroidQuadratic(fit_boxsize=3)'

        centroid_func = CentroidQuadratic(fit_boxsize=(3, 5))
        cls_repr = repr(centroid_func)
        assert cls_repr == 'CentroidQuadratic(fit_boxsize=(3, 5))'

    def test_str(self):
        """
        Test CentroidQuadratic __str__ method.
        """
        centroid_func = CentroidQuadratic()
        cls_str = str(centroid_func)
        cls_name = 'photutils.centroids.core.CentroidQuadratic'
        expected = f'<{cls_name}>\nfit_boxsize: 5'
        assert cls_str == expected

        centroid_func = CentroidQuadratic(fit_boxsize=3)
        cls_str = str(centroid_func)
        expected = f'<{cls_name}>\nfit_boxsize: 3'
        assert cls_str == expected
