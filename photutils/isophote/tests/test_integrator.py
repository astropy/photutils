# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the integrator module.
"""

import sys
import warnings

import astropy.units as u
import numpy as np
import pytest
from astropy.io import fits
from astropy.modeling.models import Gaussian2D
from numpy.testing import assert_allclose, assert_array_equal

import photutils.isophote
from photutils.datasets.load import _get_path
from photutils.isophote import integrator
from photutils.isophote.ellipse import Ellipse
from photutils.isophote.geometry import EllipseGeometry
from photutils.isophote.integrator import _INTEGRATORS
from photutils.isophote.sample import EllipseSample
from photutils.isophote.tests.make_test_data import make_test_image
from photutils.utils.exceptions import PhotutilsDeprecationWarning


@pytest.mark.remote_data
class TestData:
    def setup_class(self):
        path = _get_path('isophote/synth_highsnr.fits',
                         location='photutils-datasets', cache=True)
        hdu = fits.open(path)
        self.data = hdu[0].data
        hdu.close()

    def make_sample(self, *, masked=False, sma=40.0, integrmode='bilinear'):
        if masked:
            data = np.ma.masked_values(self.data, 200.0, atol=10.0, rtol=0.0)
        else:
            data = self.data
        sample = EllipseSample(data, sma, integration_mode=integrmode)
        s = sample.extract()

        assert len(s) == 3
        assert len(s[0]) == len(s[1])
        assert len(s[0]) == len(s[2])

        return s, sample


@pytest.mark.remote_data
class TestUnmasked(TestData):
    def test_bilinear(self):
        s, sample = self.make_sample()

        assert len(s[0]) == 225
        # Intensities
        assert_allclose(np.mean(s[2]), 200.76, atol=0.01)
        assert_allclose(np.std(s[2]), 21.55, atol=0.01)
        # Radii
        assert_allclose(np.max(s[1]), 40.0, atol=0.01)
        assert_allclose(np.min(s[1]), 32.0, atol=0.01)

        assert sample.n_total == 225
        assert sample.n_valid == 225

    def test_bilinear_small(self):
        # Small radius forces subpixel sampling.
        s, sample = self.make_sample(sma=10.0)

        # Intensities
        assert_allclose(np.mean(s[2]), 1045.8, atol=0.1)
        assert_allclose(np.std(s[2]), 143.5, atol=0.1)
        # Radii
        assert_allclose(np.max(s[1]), 10.0, atol=0.1)
        assert_allclose(np.min(s[1]), 8.0, atol=0.1)

        assert sample.n_total == 57
        assert sample.n_valid == 57

    def test_nearest_neighbor(self):
        s, sample = self.make_sample(integrmode='nearest_neighbor')

        assert len(s[0]) == 225
        # Intensities
        assert_allclose(np.mean(s[2]), 200.6, atol=0.1)
        assert_allclose(np.std(s[2]), 21.6, atol=0.1)
        # Radii
        assert_allclose(np.max(s[1]), 40.0, atol=0.01)
        assert_allclose(np.min(s[1]), 32.0, atol=0.01)

        assert sample.n_total == 225
        assert sample.n_valid == 225

    def test_mean(self):
        s, sample = self.make_sample(integrmode='mean')

        assert len(s[0]) == 64
        # Intensities
        assert_allclose(np.mean(s[2]), 199.9, atol=0.1)
        assert_allclose(np.std(s[2]), 21.3, atol=0.1)
        # Radii
        assert_allclose(np.max(s[1]), 40.0, atol=0.01)
        assert_allclose(np.min(s[1]), 32.0, atol=0.01)

        assert_allclose(sample.sector_area, 12.63, atol=0.01)
        assert sample.n_total == 64
        assert sample.n_valid == 64

    def test_mean_small(self):
        s, sample = self.make_sample(sma=5.0, integrmode='mean')

        assert len(s[0]) == 29
        # Intensities
        assert_allclose(np.mean(s[2]), 2340.8, atol=0.1)
        assert_allclose(np.std(s[2]), 284.6, atol=0.1)
        # Radii
        assert_allclose(np.max(s[1]), 5.0, atol=0.01)
        assert_allclose(np.min(s[1]), 4.0, atol=0.01)

        assert_allclose(sample.sector_area, 2.0, atol=0.1)
        assert sample.n_total == 29
        assert sample.n_valid == 29

    def test_median(self):
        s, sample = self.make_sample(integrmode='median')

        assert len(s[0]) == 64
        # Intensities
        assert_allclose(np.mean(s[2]), 199.5, atol=0.1)
        assert_allclose(np.std(s[2]), 21.2, atol=0.1)
        # Radii
        assert_allclose(np.max(s[1]), 40.0, atol=0.01)
        assert_allclose(np.min(s[1]), 32.01, atol=0.01)

        assert_allclose(sample.sector_area, 12.63, atol=0.01)
        assert sample.n_total == 64
        assert sample.n_valid == 64


@pytest.mark.remote_data
class TestMasked(TestData):
    def test_bilinear(self):
        s, sample = self.make_sample(masked=True, integrmode='bilinear')

        assert len(s[0]) == 163
        # Intensities
        assert_allclose(np.mean(s[2]), 201.30, atol=0.01)
        assert_allclose(np.std(s[2]), 24.89, atol=0.01)
        # Radii
        assert_allclose(np.max(s[1]), 40.0, atol=0.01)
        assert_allclose(np.min(s[1]), 32.0, atol=0.01)

        assert sample.n_total == 225
        assert sample.n_valid == 163

    def test_mean(self):
        s, sample = self.make_sample(masked=True, integrmode='mean')

        assert len(s[0]) == 51
        # Intensities
        assert_allclose(np.mean(s[2]), 199.9, atol=0.1)
        assert_allclose(np.std(s[2]), 24.12, atol=0.1)
        # Radii
        assert_allclose(np.max(s[1]), 40.0, atol=0.01)
        assert_allclose(np.min(s[1]), 32.0, atol=0.01)

        assert_allclose(sample.sector_area, 12.63, atol=0.01)
        assert sample.n_total == 64
        assert sample.n_valid == 51


@pytest.mark.parametrize('integrmode',
                         ['nearest_neighbor', 'bilinear', 'mean', 'median'])
def test_masked_and_nonfinite(integrmode):
    """
    Test that masked pixels and nonfinite pixels are excluded in the
    same way.
    """
    data = make_test_image(seed=0)
    mask = np.zeros(data.shape, dtype=bool)
    mask[230:260, 280:300] = True

    sample = EllipseSample(data, 40.0, integration_mode=integrmode)
    sample.update()
    sample_masked = EllipseSample(np.ma.masked_array(data, mask=mask), 40.0,
                                  integration_mode=integrmode)
    sample_masked.update()
    data_nan = data.copy()
    data_nan[mask] = np.nan
    sample_nan = EllipseSample(data_nan, 40.0, integration_mode=integrmode)
    sample_nan.update()

    assert sample_masked.n_valid < sample.n_valid
    assert sample_masked.n_total == sample.n_total
    assert_array_equal(sample_masked.extract(), sample_nan.extract())


@pytest.mark.parametrize('integrmode',
                         ['nearest_neighbor', 'bilinear', 'mean', 'median'])
def test_masked_infinite_values(integrmode):
    """
    Test that infinite values of both signs in masked pixels do not
    change the sample.
    """
    data = make_test_image(nx=128, ny=128, seed=0)
    mask = np.zeros(data.shape, dtype=bool)
    mask[60:70, 80:90] = True
    data_inf = data.copy()
    data_inf[mask] = np.inf
    data_inf[62:66, 82:86] = -np.inf

    kwargs = {'x0': 64.0, 'y0': 64.0, 'integration_mode': integrmode}
    sample = EllipseSample(np.ma.masked_array(data, mask=mask), 20.0,
                           **kwargs)
    sample_inf = EllipseSample(np.ma.masked_array(data_inf, mask=mask), 20.0,
                               **kwargs)

    assert sample.extract().shape[1] > 0
    assert_array_equal(sample_inf.extract(), sample.extract())


@pytest.mark.parametrize('integrmode', ['nearest_neighbor', 'bilinear'])
@pytest.mark.parametrize('x0', [1.0e30, -1.0e30])
def test_very_large_path_coordinates(integrmode, x0):
    """
    Test that path points with coordinates too large to be pixel
    indices are skipped like the points outside of the image.
    """
    data = make_test_image(nx=128, ny=128, seed=0)
    sample = EllipseSample(data, 10.0, x0=x0, y0=64.0,
                           integration_mode=integrmode)
    values = sample.extract()
    assert values.shape == (3, 0)
    assert sample.n_total > 0
    assert sample.n_valid == 0


@pytest.mark.parametrize('integrmode', ['mean', 'median'])
def test_area_sectors_outside_image(integrmode):
    """
    Test that the area integrators skip the sectors that are not
    entirely inside the image.
    """
    data = make_test_image(nx=128, ny=128, seed=0)
    kwargs = {'x0': 100.0, 'y0': 64.0, 'integration_mode': integrmode}
    sample = EllipseSample(data, 40.0, **kwargs)
    angles, _, values = sample.extract()
    assert 0 < sample.n_valid < sample.n_total

    # The sectors that are kept have the same values as in a wider
    # image that contains the whole ellipse.
    data_wide = np.pad(data, ((0, 0), (0, 64)))
    sample_wide = EllipseSample(data_wide, 40.0, **kwargs)
    angles_wide, _, values_wide = sample_wide.extract()
    assert sample_wide.n_total == sample.n_total
    assert sample_wide.n_valid == sample_wide.n_total
    keep = np.isin(angles_wide, angles)
    assert_array_equal(values_wide[keep], values)


@pytest.mark.parametrize(('values', 'expected'),
                         [([3.0, 1.0, 2.0], 2.0),
                          ([4.0, 1.0, 3.0, 2.0], 2.5),
                          ([10.0, 1.0, 2.0, 3.0, 4.0, 100.0], 3.5)])
def test_median_integrator_value(values, expected):
    """
    Test that the median of an even number of pixel values is the
    average of the two middle values, as in the STSDAS ellipse task.
    """
    geometry = EllipseGeometry(10.0, 10.0, 5.0, 0.2, 0.0 * u.rad)
    integrator = _INTEGRATORS['median'](np.zeros((20, 20)), geometry, [], [],
                                        [])
    assert integrator.compute_sample_value(np.array(values)) == expected


def _sample_point(integrmode, data, x, y):
    """
    Sample a single image point using the given integrator.
    """
    geometry = EllipseGeometry(x, y, 1.0, 0.0, 0.0 * u.rad)
    intensities = []
    integrator = _INTEGRATORS[integrmode](data, geometry, [], [], intensities)
    integrator.integrate(0.0, 0.0)
    return intensities


@pytest.mark.parametrize(('x', 'y', 'expected'),
                         [(0.5, 0.5, [5.0]),
                          (0.0, 0.0, [0.0]),
                          (0.99, 0.99, [9.9]),
                          (-0.01, 0.5, []),
                          (-0.5, 0.5, []),
                          (-0.99, 0.5, []),
                          (0.5, -0.5, [])])
def test_bilinear_edges(x, y, expected):
    """
    Test that the bilinear integrator skips points that lie before
    the first pixel center instead of extrapolating to them.
    """
    data = np.array([[0.0, 10.0], [0.0, 10.0]])
    assert_allclose(_sample_point('bilinear', data, x, y), expected)


@pytest.mark.parametrize(('x', 'y', 'expected'),
                         [(0.0, 1.0, [0.0]),
                          (0.3, 1.0, [0.0]),
                          (0.7, 1.0, [10.0]),
                          (1.7, 1.0, [20.0]),
                          (2.7, 1.0, [30.0]),
                          (3.3, 1.0, [30.0]),
                          (-0.3, 1.0, [0.0]),
                          (-0.7, 1.0, []),
                          (3.7, 1.0, []),
                          (1.0, -0.7, []),
                          (1.0, 3.7, [])])
def test_nearest_neighbor_pixel(x, y, expected):
    """
    Test that the nearest-neighbor integrator samples the pixel whose
    center is nearest to the sample point.

    Pixel centers are at integer coordinates.
    """
    data = np.tile(np.arange(4.0) * 10.0, (4, 1))
    assert_allclose(_sample_point('nearest_neighbor', data, x, y), expected)


@pytest.mark.parametrize('integrmode', ['nearest_neighbor', 'bilinear'])
def test_fit_center(integrmode):
    """
    Test that the fitted center of a symmetric source is not offset.
    """
    yy, xx = np.mgrid[0:101, 0:101]
    data = Gaussian2D(100.0, 50.0, 50.0, 12.0, 8.0, theta=0.5)(xx, yy)
    geometry = EllipseGeometry(50.0, 50.0, 10.0, 0.3, 0.5 * u.rad)
    ellipse = Ellipse(data, geometry=geometry)
    isolist = ellipse.fit_image(integration_mode=integrmode, sma_start=10.0,
                                max_sma=25.0)
    assert_allclose(np.median(isolist.x0), 50.0, atol=0.05)
    assert_allclose(np.median(isolist.y0), 50.0, atol=0.05)


@pytest.mark.parametrize('integrmode', ['mean', 'median'])
def test_area_integrator_full_sectors(integrmode):
    """
    Test that the area integrators sample whole sectors, including a
    sector that extends past a polar angle of 2 pi.

    The image values are the polar angles of the pixels, so each sector
    value is close to the polar angle of the sector.
    """
    yy, xx = np.mgrid[0:121, 0:121]
    data = np.arctan2(yy - 60.0, xx - 60.0)
    sample = EllipseSample(data, 40.0, x0=60.0, y0=60.0, ellipticity=0.2,
                           integration_mode=integrmode)
    angles, _, values = sample.extract()
    angles = np.angle(np.exp(1j * angles))  # wrap to (-pi, pi]

    # Exclude the sectors at the discontinuity of the image values.
    good = np.abs(angles) < 3.0
    assert np.any(angles[good] < 0.0)
    assert_allclose(values[good], angles[good], atol=0.012)


@pytest.mark.parametrize('name', ['BILINEAR', 'NEAREST_NEIGHBOR', 'MEAN',
                                  'MEDIAN', 'INTEGRATORS'])
def test_integrator_constants_deprecated(name):
    assert name not in integrator.__all__
    assert name not in vars(photutils.isophote)
    with pytest.warns(PhotutilsDeprecationWarning, match=name):
        value = getattr(photutils.isophote, name)
    with pytest.warns(PhotutilsDeprecationWarning, match=name):
        assert getattr(integrator, name) == value


def _record_deprecation_warnings(func):
    """
    Call ``func`` with all warnings shown and return the recorded
    deprecation warnings.
    """
    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter('always')
        func()
    return [record for record in records
            if issubclass(record.category, PhotutilsDeprecationWarning)]


def _from_package_import():
    from photutils.isophote import MEAN
    return MEAN


def _from_module_import():
    from photutils.isophote.integrator import MEAN
    return MEAN


def _attribute_access():
    return photutils.isophote.MEAN


@pytest.mark.parametrize('func', [_from_package_import, _from_module_import,
                                  _attribute_access])
def test_integrator_constant_warns_once(func):
    """
    Test that each way of using a deprecated constant emits exactly one
    warning at the line of the caller, even when repeated warnings are
    shown.
    """
    records = _record_deprecation_warnings(func)
    assert len(records) == 1
    assert 'MEAN was deprecated' in str(records[0].message)
    assert records[0].filename == __file__


def test_integrator_constant_silent_uses():
    def uses():
        namespace = {}
        exec(  # noqa: S102
            'from photutils.isophote import *', namespace)  # nosec B102
        dir(photutils.isophote)
        dir(integrator)
        assert not hasattr(photutils.isophote, 'nonexistent')
        assert not hasattr(integrator, 'nonexistent')

    assert _record_deprecation_warnings(uses) == []


def _no_frame_inspection(monkeypatch):
    monkeypatch.delattr(sys, '_getframe')


def _shallow_call_stack(monkeypatch):
    def get_frame(depth=0):  # noqa: ARG001
        msg = 'call stack is not deep enough'
        raise ValueError(msg)

    monkeypatch.setattr(sys, '_getframe', get_frame)


@pytest.mark.parametrize('patch', [_no_frame_inspection,
                                   _shallow_call_stack])
def test_integrator_constant_without_import_probe(monkeypatch, patch):
    """
    Test that a deprecated constant still warns if the caller cannot be
    inspected, e.g., on a Python implementation without
    ``sys._getframe``.
    """
    patch(monkeypatch)
    with pytest.warns(PhotutilsDeprecationWarning, match='MEAN'):
        assert photutils.isophote.MEAN == 'mean'


def test_integrator_getattr_unknown_name():
    for module in (photutils.isophote, integrator):
        assert not hasattr(module, 'NOT_A_NAME')
        with pytest.raises(AttributeError, match='NOT_A_NAME'):
            module.NOT_A_NAME  # noqa: B018


def test_isophote_wildcard_import():
    namespace = {}
    exec(  # noqa: S102
        'from photutils.isophote import *', namespace)  # nosec B102
    assert 'Ellipse' in namespace
    assert 'BILINEAR' not in namespace
