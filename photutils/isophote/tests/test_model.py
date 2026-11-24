# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the model module.
"""

import warnings
from concurrent.futures import ThreadPoolExecutor

import astropy.units as u
import numpy as np
import pytest
from astropy.io import fits
from astropy.modeling.models import Gaussian2D
from astropy.utils.data import get_pkg_data_filename
from astropy.utils.exceptions import AstropyUserWarning
from numpy.testing import assert_allclose, assert_array_equal

from photutils.datasets.load import _get_path
from photutils.isophote._ellipse_model import build_ellipse_model_c
from photutils.isophote.ellipse import Ellipse
from photutils.isophote.geometry import EllipseGeometry
from photutils.isophote.isophote import IsophoteList
from photutils.isophote.model import build_ellipse_model
from photutils.isophote.tests.make_test_data import make_test_image
from photutils.utils.exceptions import PhotutilsDeprecationWarning


@pytest.mark.remote_data
def test_model():
    path = _get_path('isophote/M105-S001-RGB.fits',
                     location='photutils-datasets', cache=True)
    hdu = fits.open(path)
    data = hdu[0].data[0]
    hdu.close()

    g = EllipseGeometry(530.0, 511, 10.0, 0.1, np.deg2rad(10.0) * u.rad)
    ellipse = Ellipse(data, geometry=g)

    # NOTE: this sometimes emits warnings (e.g., py38, ubuntu), but
    # sometimes not. Here we simply ignore any RuntimeWarning, whether
    # there is one or not.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        isophote_list = ellipse.fit_image()
    model = isophote_list.make_model_image(
        data.shape, fill_value=np.mean(data[10:100, 10:100]))

    assert data.shape == model.shape

    residual = data - model
    assert np.abs(np.mean(residual)) <= 5.0


@pytest.fixture(name='simulated_isolist', scope='module')
def fixture_simulated_isolist():
    """
    Fit isophotes to a simulated galaxy image.

    The fit does not depend on the model-building parameters, so it is
    performed only once and shared across tests.
    """
    data = make_test_image(nx=200, ny=200, i0=10.0, sma=5.0, eps=0.5,
                           pa=np.pi / 3.0, noise=0.05, seed=0)

    g = EllipseGeometry(100.0, 100.0, 5.0, 0.5, (np.pi / 3.0) * u.rad)
    ellipse = Ellipse(data, geometry=g)

    # Catch warnings that may arise from empty slices. This started
    # to happen on windows with scipy 1.15.0.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        isophote_list = ellipse.fit_image()

    return data, isophote_list


@pytest.mark.parametrize('sma_interval', [0.05, 0.1])
def test_model_simulated_data(simulated_isolist, sma_interval):
    data, isophote_list = simulated_isolist
    model = isophote_list.make_model_image(
        data.shape, fill_value=np.mean(data[0:50, 0:50]),
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
        g = EllipseGeometry(50.0, 45, 530.0, 0.1, np.deg2rad(10.0) * u.rad)
        g.find_center(data)
        ellipse = Ellipse(data, geometry=g)

        # Sigma clipping of samples with no valid data points does not
        # emit numpy RuntimeWarnings.
        isophote_list = ellipse.fit_image(sma_start=40, min_sma=0,
                                          max_sma=350.0, step=0.4,
                                          n_clip=3)

        model = isophote_list.make_model_image(
            data.shape, fill_value=np.mean(data[0:50, 0:50]))

        # It's enough that the algorithm reached this point. The
        # actual accuracy of the modelling is being tested elsewhere.
        assert data.shape == model.shape


def test_make_model_image_matches_function():
    """
    Test that the deprecated function gives the same image as the
    method.
    """
    data = make_test_image(seed=0)
    isolist = Ellipse(data).fit_image()
    match = "The 'build_ellipse_model' function was deprecated"
    for kwargs in ({}, {'high_harmonics': True}, {'sma_interval': 0.37}):
        model = isolist.make_model_image(data.shape, fill_value=1.0,
                                         **kwargs)
        with pytest.warns(PhotutilsDeprecationWarning, match=match):
            old = build_ellipse_model(data.shape, isolist, fill=1.0,
                                      **kwargs)
        assert_array_equal(model, old)


def test_make_model_image_keyword_only():
    data = make_test_image(seed=0)
    isolist = Ellipse(data).fit_image(max_sma=20.0)
    with pytest.raises(TypeError):
        isolist.make_model_image(data.shape, 1.0)


def test_make_model_image_nan_gradient():
    """
    Test that the gradient is not needed without high harmonics.
    """
    data = make_test_image(seed=0)
    isolist = Ellipse(data).fit_image(max_sma=40.0)
    expected = isolist.make_model_image(data.shape)
    for isophote in isolist:
        isophote.gradient = np.nan
    model = isolist.make_model_image(data.shape)
    assert_array_equal(model, expected)

    match = "'gradient'"
    with pytest.raises(ValueError, match=match):
        isolist.make_model_image(data.shape, high_harmonics=True)


def test_make_model_image_table_list_high_harmonics():
    """
    Test that the high harmonics require the gradients, which a table
    without a gradient column does not have.
    """
    data = make_test_image(seed=0)
    isolist = Ellipse(data).fit_image(max_sma=40.0)
    names = ['sma', 'x0', 'y0', 'ellipticity', 'orientation', 'intensity',
             'a3', 'b3', 'a4', 'b4']
    loaded = IsophoteList.from_table(isolist.to_table(columns=names))
    assert_allclose(loaded.make_model_image(data.shape),
                    isolist.make_model_image(data.shape), rtol=1e-10)
    match = "'gradient'"
    with pytest.raises(ValueError, match=match):
        loaded.make_model_image(data.shape, high_harmonics=True)


def test_model_inputs():
    match = 'isolist must not be empty'
    with pytest.raises(ValueError, match=match):
        IsophoteList([]).make_model_image((10, 10))

    data = make_test_image(seed=0)
    isolist = Ellipse(data).fit_image(max_sma=20.0)
    match = 'isolist must contain at least 4 isophotes'
    with pytest.raises(ValueError, match=match):
        isolist[-3:].make_model_image(data.shape)
    model = isolist[-4:].make_model_image(data.shape)
    assert np.all(np.isfinite(model))
    assert np.max(model) > 0

    match = 'must be sorted by strictly increasing semimajor axis'
    with pytest.raises(ValueError, match=match):
        isolist[::-1].make_model_image(data.shape)
    with pytest.raises(ValueError, match=match):
        (isolist + isolist[-1:]).make_model_image(data.shape)


def test_model_outer_isophote():
    """
    Test that the model extends to the outermost isophote.
    """
    data = make_test_image(seed=0)
    isolist = Ellipse(data).fit_image(max_sma=40.0)
    model = isolist.make_model_image(data.shape, fill_value=np.nan,
                                     sma_interval=10.0)

    # The fitted isophotes have position angles near 0 or pi, so the
    # major axis is along the x axis.
    sma = isolist[-1].sma
    assert np.isfinite(model[256, 256 + int(sma)])
    assert np.isfinite(model[256, 256 - int(sma)])


def test_model_pa_wrap():
    """
    Test the model for a galaxy whose position angle is near 0.

    The fitted position angles of neighboring isophotes then jump
    between values near 0 and near pi. The interpolated position angle
    must not sweep through the angles in between.
    """
    data = make_test_image(seed=0)
    isolist = Ellipse(data).fit_image(max_sma=60.0)
    pa = isolist.orientation[isolist.sma > 0].to_value(u.rad)
    assert np.any(pa < 0.1)
    assert np.any(pa > np.pi - 0.1)

    model = isolist.make_model_image(data.shape)
    yy, xx = np.mgrid[0:data.shape[0], 0:data.shape[1]]
    radius = np.hypot(xx - 256, yy - 256)
    annulus = (radius >= 10) & (radius < 30)
    residual = np.abs(model - data)[annulus] / data[annulus]
    assert np.max(residual) < 0.01


def test_model_fill():
    """
    Test that only the pixels with no contribution from any isophote
    are set to the fill value.
    """
    data = make_test_image(nx=200, ny=200, seed=0)
    isolist = Ellipse(data).fit_image(max_sma=30.0)

    # Model pixels with a zero intensity must not be filled.
    for isophote in isolist:
        isophote.intensity = 0.0
    model = isolist.make_model_image(data.shape, fill_value=np.nan)
    assert model[100, 100] == 0.0
    assert np.isnan(model[0, 0])
    assert np.all(model[np.isfinite(model)] == 0.0)


def test_model_harmonics():
    """
    Test that high harmonics are included in make_model_image.
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

    geometry = EllipseGeometry(x0=x0, y0=y0, sma=30, ellipticity=eps,
                               orientation=theta * u.rad)
    ellipse = Ellipse(data, geometry=geometry)
    isolist = ellipse.fit_image(fix_center=True, fix_ellipticity=True)

    model_image = isolist.make_model_image(data.shape, high_harmonics=True)
    residual = data - model_image

    # The orientation of the innermost isophotes, within about 2 pixels
    # of the center, is poorly defined and depends on rounding, so the
    # central pixels are excluded.
    mask = (model_image > 0) & (np.hypot(xx, yy) > 3)
    assert np.std(residual[mask]) < 0.4


def test_model_harmonics_gradient_sign():
    """
    Test that the model harmonics do not depend on the sign of the
    local gradient.

    The isophote harmonics are normalized by the absolute value of the
    gradient, so the model must scale them back with the absolute value
    too.
    """
    data = make_test_image(nx=200, ny=200, seed=0)
    isolist = Ellipse(data).fit_image(max_sma=40.0)
    model = isolist.make_model_image(data.shape, high_harmonics=True)

    for isophote in isolist:
        isophote.gradient = -isophote.gradient
    model_flipped = isolist.make_model_image(data.shape, high_harmonics=True)
    assert_allclose(model_flipped, model)


def test_model_missing_harmonics():
    """
    Test that harmonics that could not be computed are treated as zero
    instead of making the model nonfinite.
    """
    data = make_test_image(nx=200, ny=200, seed=0)
    isolist = Ellipse(data).fit_image(max_sma=40.0)
    isolist[10].a3 = None
    isolist[10].b4 = np.nan
    model = isolist.make_model_image(data.shape, high_harmonics=True)
    assert np.all(np.isfinite(model))

    isolist[10].a3 = 0.0
    isolist[10].b4 = 0.0
    model_zero = isolist.make_model_image(data.shape, high_harmonics=True)
    assert_allclose(model, model_zero)


def test_model_integration():
    """
    Test that model integration does not stop as soon as the angle reaches
    the edge of the image.
    """
    data = make_test_image(nx=80, ny=110, i0=100.0, sma=60.0, eps=0.5,
                           pa=np.pi / 3.0, noise=0.05, seed=0)
    g = EllipseGeometry(40, 55, 5.0, 0.5, (np.pi / 3.0) * u.rad)
    ellipse = Ellipse(data, geometry=g)
    isophote_list = ellipse.fit_image()
    model = isophote_list.make_model_image(
        data.shape, fill_value=np.nanmean(data[105:, :5]),
        sma_interval=0.05)
    assert np.nanmean(np.abs(model[100:, 60:] - data[100:, 60:])) < 2


def test_build_ellipse_model_c_threadsafe():
    """
    Test that build_ellipse_model_c is thread-safe by running it in
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


@pytest.mark.parametrize('value', [np.nan, np.inf, -np.inf, 1.0e30])
def test_build_ellipse_model_c_invalid_center(value):
    """
    Test that build_ellipse_model_c skips an isophote whose center
    coordinates are nonfinite or too large to be pixel indices.
    """
    n = 16
    sma = np.linspace(0.5, 20.0, n)
    intens = np.full(n, 1.0)
    eps = np.full(n, 0.3)
    pa = np.full(n, 0.5)
    x0 = np.full(n, 50.0)
    y0 = np.full(n, 50.0)

    # The reference isophote lies outside of the image.
    x0[8] = -1000.0
    expected = build_ellipse_model_c(100, 100, sma, intens, eps, pa, x0, y0)

    x0[8] = value
    result = build_ellipse_model_c(100, 100, sma, intens, eps, pa, x0, y0)
    assert_array_equal(result[0], expected[0])
    assert_array_equal(result[1], expected[1])

    result = build_ellipse_model_c(100, 100, sma, intens, eps, pa, y0, x0)
    assert np.all(np.isfinite(result[0]))
    assert np.all(np.isfinite(result[1]))


def test_build_ellipse_model_c_readonly_arrays():
    """
    Test that read-only (nonwritable) input arrays are accepted by
    build_ellipse_model_c and give results identical to writable
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
    Test that noncontiguous (strided) input arrays are accepted by
    build_ellipse_model_c and give results identical to contiguous
    arrays.

    The input arguments are declared as noncontiguous ``double[:]``
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

    # Build strided (noncontiguous) views by interleaving and slicing.
    arrays_strided = []
    for arr in arrays:
        strided = np.repeat(arr, 2)[::2]
        assert not strided.flags['C_CONTIGUOUS']
        arrays_strided.append(strided)
    result = build_ellipse_model_c(100, 100, *arrays_strided)

    assert_array_equal(result[0], expected[0])
    assert_array_equal(result[1], expected[1])


@pytest.mark.parametrize('index', range(5))
def test_build_ellipse_model_c_length_mismatch(index):
    """
    Test that a ValueError is raised if any of the main input arrays
    does not have the same length as finely_spaced_sma.
    """
    n = 16
    sma = np.linspace(0.5, 20.0, n)
    arrays = [np.full(n, value) for value in (1.0, 0.3, 0.5, 50.0, 50.0)]
    arrays[index] = arrays[index][:-1]
    match = 'All input arrays must be same length'
    with pytest.raises(ValueError, match=match):
        build_ellipse_model_c(100, 100, sma, *arrays)


@pytest.mark.parametrize('index', range(4))
def test_build_ellipse_model_c_harmonic_length_mismatch(index):
    """
    Test that a ValueError is raised if any of the harmonic arrays
    does not have the same length as finely_spaced_sma.
    """
    n = 16
    sma = np.linspace(0.5, 20.0, n)
    arrays = [np.full(n, value) for value in (1.0, 0.3, 0.5, 50.0, 50.0)]
    names = ('a3_array', 'b3_array', 'a4_array', 'b4_array')
    harmonics = {name: np.full(n, 0.01) for name in names}
    harmonics[names[index]] = harmonics[names[index]][:-1]
    match = 'All input arrays must be same length'
    with pytest.raises(ValueError, match=match):
        build_ellipse_model_c(100, 100, sma, *arrays, **harmonics)


def test_build_ellipse_model_c_partial_harmonics():
    """
    Test that a ValueError is raised if only some of the harmonic
    arrays are input.
    """
    n = 16
    sma = np.linspace(0.5, 20.0, n)
    arrays = [np.full(n, value) for value in (1.0, 0.3, 0.5, 50.0, 50.0)]
    match = 'Must supply all harmonic arrays if any is not None'
    with pytest.raises(ValueError, match=match):
        build_ellipse_model_c(100, 100, sma, *arrays,
                              a3_array=np.full(n, 0.01))


def test_model_nonfinite_central_pixel():
    """
    Test that isophotes with a nonfinite intensity, such as the central
    pixel of a galaxy with a masked center, are ignored.
    """
    data = make_test_image(nx=200, ny=200, seed=0)
    isolist = Ellipse(data).fit_image(max_sma=30.0)
    expected = isolist.make_model_image(data.shape)

    mask = np.zeros(data.shape, dtype=bool)
    mask[100, 100] = True
    ellipse = Ellipse(np.ma.masked_array(data, mask=mask))
    match = 'Too small sample to warrant a fit'
    with pytest.warns(AstropyUserWarning, match=match):
        isolist = ellipse.fit_image(max_sma=30.0)
    assert isolist[0].sma == 0
    assert np.isnan(isolist[0].intensity)

    model = isolist.make_model_image(data.shape)
    yy, xx = np.mgrid[0:data.shape[0], 0:data.shape[1]]
    radius = np.hypot(xx - 100, yy - 100)
    annulus = (radius > 3) & (radius < 20)
    assert np.all(np.isfinite(model))
    assert_allclose(model[annulus], expected[annulus], rtol=0.01)

    match = 'at least 4 isophotes with a finite intensity'
    with pytest.raises(ValueError, match=match):
        isolist[:4].make_model_image(data.shape)
