# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the model module.
"""

import warnings
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
from astropy.io import fits
from astropy.modeling.models import Gaussian2D
from astropy.utils.data import get_pkg_data_filename
from numpy.testing import assert_array_equal

from photutils.datasets.load import _get_path
from photutils.isophote._ellipse_model import build_ellipse_model_c
from photutils.isophote.ellipse import Ellipse
from photutils.isophote.geometry import EllipseGeometry
from photutils.isophote.isophote import IsophoteList
from photutils.isophote.model import build_ellipse_model
from photutils.isophote.tests.make_test_data import make_test_image


@pytest.mark.remote_data
def test_model():
    path = _get_path('isophote/M105-S001-RGB.fits',
                     location='photutils-datasets', cache=True)
    hdu = fits.open(path)
    data = hdu[0].data[0]
    hdu.close()

    g = EllipseGeometry(530.0, 511, 10.0, 0.1, np.deg2rad(10.0))
    ellipse = Ellipse(data, geometry=g, threshold=1.0e5)

    # NOTE: this sometimes emits warnings (e.g., py38, ubuntu), but
    # sometimes not. Here we simply ignore any RuntimeWarning, whether
    # there is one or not.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        isophote_list = ellipse.fit_image()
    model = build_ellipse_model(data.shape, isophote_list,
                                fill=np.mean(data[10:100, 10:100]))

    assert data.shape == model.shape

    residual = data - model
    assert np.abs(np.mean(residual)) <= 5.0


@pytest.mark.parametrize('sma_interval', [0.05, 0.1])
def test_model_simulated_data(sma_interval):
    data = make_test_image(nx=200, ny=200, i0=10.0, sma=5.0, eps=0.5,
                           pa=np.pi / 3.0, noise=0.05, seed=0)

    g = EllipseGeometry(100.0, 100.0, 5.0, 0.5, np.pi / 3.0)
    ellipse = Ellipse(data, geometry=g, threshold=1.0e5)

    # Catch warnings that may arise from empty slices. This started
    # to happen on windows with scipy 1.15.0.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        isophote_list = ellipse.fit_image()

    model = build_ellipse_model(data.shape, isophote_list,
                                fill=np.mean(data[0:50, 0:50]),
                                sma_interval=sma_interval)

    assert data.shape == model.shape

    residual = data - model
    assert np.abs(np.mean(residual)) <= 0.01
    assert np.abs(np.median(residual)) <= 0.01


def test_model_minimum_radius():
    # This test requires a "defective" image that drives the
    # model building algorithm into a corner, where it fails.
    # With the algorithm fixed, it bypasses the failure and
    # succeeds in building the model image.
    filepath = get_pkg_data_filename('data/minimum_radius_test.fits')
    with fits.open(filepath) as hdu:
        data = hdu[0].data
        g = EllipseGeometry(50.0, 45, 530.0, 0.1, np.deg2rad(10.0))
        g.find_center(data)
        ellipse = Ellipse(data, geometry=g)

        match1 = 'Degrees of freedom'
        match2 = 'Mean of empty slice'
        match3 = 'invalid value encountered'
        ctx1 = pytest.warns(RuntimeWarning, match=match1)
        ctx2 = pytest.warns(RuntimeWarning, match=match2)
        ctx3 = pytest.warns(RuntimeWarning, match=match3)
        with ctx1, ctx2, ctx3:
            isophote_list = ellipse.fit_image(sma0=40, minsma=0,
                                              maxsma=350.0, step=0.4, n_clip=3)

        model = build_ellipse_model(data.shape, isophote_list,
                                    fill=np.mean(data[0:50, 0:50]))

        # It's enough that the algorithm reached this point. The
        # actual accuracy of the modelling is being tested elsewhere.
        assert data.shape == model.shape


def test_model_inputs():
    match = 'isolist must not be empty'
    with pytest.raises(ValueError, match=match):
        build_ellipse_model((10, 10), IsophoteList([]))


def test_model_harmonics():
    """
    Test that high harmonics are included in build_ellipse_model.
    """
    x0 = y0 = 50
    xsig = 10
    ysig = 5
    eps = ysig / xsig
    theta = np.deg2rad(41)
    m = Gaussian2D(100, x0, y0, xsig, ysig, theta)
    yy, xx = np.mgrid[:101, :101]
    data = m(xx, yy)
    yy -= y0
    xx -= x0
    dt = np.arctan2(yy, xx) - np.deg2rad(10)
    harm = (0.1 * np.sin(3 * dt)
            + 0.1 * np.cos(3 * dt)
            + 0.6 * np.sin(4 * dt)
            - 0.5 * np.cos(4 * dt))
    harm -= np.min(harm)
    data += 5 * harm

    geometry = EllipseGeometry(x0=x0, y0=y0, sma=30, eps=eps, pa=theta)
    ellipse = Ellipse(data, geometry=geometry)
    isolist = ellipse.fit_image(fix_center=True, fix_eps=True)

    model_image = build_ellipse_model(data.shape, isolist, high_harmonics=True)
    residual = data - model_image

    mask = model_image > 0
    assert np.std(residual[mask]) < 0.4


def test_model_integration():
    """
    Test that model integration does not stop as soon as the angle reaches
    the edge of the image.
    """
    data = make_test_image(nx=80, ny=110, i0=100.0, sma=60.0, eps=0.5,
                           pa=np.pi / 3.0, noise=0.05, seed=0)
    g = EllipseGeometry(40, 55, 5.0, 0.5, np.pi / 3.0)
    ellipse = Ellipse(data, geometry=g, threshold=1.0e5)
    isophote_list = ellipse.fit_image()
    model = build_ellipse_model(data.shape, isophote_list,
                                fill=np.nanmean(data[105:, :5]),
                                sma_interval=0.05)
    assert np.nanmean(np.abs(model[100:, 60:] - data[100:, 60:])) < 2


def test_build_ellipse_model_c_threadsafe():
    """
    Test that build_ellipse_model_c is thread safe by running it in
    multiple threads and checking that the results are consistent.
    """
    n = 64
    sma = np.linspace(0.5, 20.0, n)
    intens = np.full(n, 1.0)
    eps = np.full(n, 0.3)
    pa = np.full(n, 0.5)
    x0 = np.full(n, 50.0)
    y0 = np.full(n, 50.0)

    def fn():
        return build_ellipse_model_c(100, 100, sma, intens, eps, pa, x0, y0)

    expected = fn()
    n_threads = 8
    n_calls_per_thread = 4

    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        futures = [executor.submit(fn)
                   for _ in range(n_threads * n_calls_per_thread)]
        for future in futures:
            result = future.result()
            assert_array_equal(result[0], expected[0])
            assert_array_equal(result[1], expected[1])


def test_build_ellipse_model_c_readonly_arrays():
    """
    Test that read-only (non-writeable) input arrays are accepted by
    build_ellipse_model_c and give results identical to writeable
    arrays.
    """
    n = 16
    sma = np.linspace(0.5, 20.0, n)
    intens = np.full(n, 1.0)
    eps = np.full(n, 0.3)
    pa = np.full(n, 0.5)
    x0 = np.full(n, 50.0)
    y0 = np.full(n, 50.0)
    arrays = (sma, intens, eps, pa, x0, y0)
    expected = build_ellipse_model_c(100, 100, *arrays)

    arrays_ro = []
    for arr in arrays:
        arr_ro = arr.copy()
        arr_ro.setflags(write=False)
        arrays_ro.append(arr_ro)
    result = build_ellipse_model_c(100, 100, *arrays_ro)

    assert_array_equal(result[0], expected[0])
    assert_array_equal(result[1], expected[1])


def test_build_ellipse_model_c_noncontiguous_arrays():
    """
    Test that non-contiguous (strided) input arrays are accepted by
    build_ellipse_model_c and give results identical to contiguous
    arrays.

    The input arguments are declared as non-contiguous ``double[:]``
    typed memoryviews. This test guards against a regression if they
    were ever changed to C-contiguous ``double[::1]`` memoryviews, which
    would reject strided arrays.
    """
    n = 16
    sma = np.linspace(0.5, 20.0, n)
    intens = np.full(n, 1.0)
    eps = np.full(n, 0.3)
    pa = np.full(n, 0.5)
    x0 = np.full(n, 50.0)
    y0 = np.full(n, 50.0)
    arrays = (sma, intens, eps, pa, x0, y0)
    expected = build_ellipse_model_c(100, 100, *arrays)

    # Build strided (non-contiguous) views by interleaving and slicing
    arrays_strided = []
    for arr in arrays:
        strided = np.repeat(arr, 2)[::2]
        assert not strided.flags['C_CONTIGUOUS']
        arrays_strided.append(strided)
    result = build_ellipse_model_c(100, 100, *arrays_strided)

    assert_array_equal(result[0], expected[0])
    assert_array_equal(result[1], expected[1])
