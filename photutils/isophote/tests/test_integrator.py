# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the integrator module.
"""

import numpy as np
import pytest
from astropy.io import fits
from astropy.modeling.models import Gaussian2D
from numpy.testing import assert_allclose, assert_array_equal

from photutils.datasets.load import _get_path
from photutils.isophote.ellipse import Ellipse
from photutils.isophote.geometry import EllipseGeometry
from photutils.isophote.integrator import (BILINEAR, INTEGRATORS, MEAN, MEDIAN,
                                           NEAREST_NEIGHBOR)
from photutils.isophote.sample import EllipseSample
from photutils.isophote.tests.make_test_data import make_test_image


@pytest.mark.remote_data
class TestData:
    def setup_class(self):
        path = _get_path('isophote/synth_highsnr.fits',
                         location='photutils-datasets', cache=True)
        hdu = fits.open(path)
        self.data = hdu[0].data
        hdu.close()

    def make_sample(self, *, masked=False, sma=40.0, integrmode=BILINEAR):
        if masked:
            data = np.ma.masked_values(self.data, 200.0, atol=10.0, rtol=0.0)
        else:
            data = self.data
        sample = EllipseSample(data, sma, integrmode=integrmode)
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

        assert sample.total_points == 225
        assert sample.actual_points == 225

    def test_bilinear_small(self):
        # Small radius forces subpixel sampling.
        s, sample = self.make_sample(sma=10.0)

        # Intensities
        assert_allclose(np.mean(s[2]), 1045.4, atol=0.1)
        assert_allclose(np.std(s[2]), 143.0, atol=0.1)
        # Radii
        assert_allclose(np.max(s[1]), 10.0, atol=0.1)
        assert_allclose(np.min(s[1]), 8.0, atol=0.1)

        assert sample.total_points == 57
        assert sample.actual_points == 57

    def test_nearest_neighbor(self):
        s, sample = self.make_sample(integrmode=NEAREST_NEIGHBOR)

        assert len(s[0]) == 225
        # Intensities
        assert_allclose(np.mean(s[2]), 200.7, atol=0.1)
        assert_allclose(np.std(s[2]), 21.6, atol=0.1)
        # Radii
        assert_allclose(np.max(s[1]), 40.0, atol=0.01)
        assert_allclose(np.min(s[1]), 32.0, atol=0.01)

        assert sample.total_points == 225
        assert sample.actual_points == 225

    def test_mean(self):
        s, sample = self.make_sample(integrmode=MEAN)

        assert len(s[0]) == 64
        # Intensities
        assert_allclose(np.mean(s[2]), 199.9, atol=0.1)
        assert_allclose(np.std(s[2]), 21.3, atol=0.1)
        # Radii
        assert_allclose(np.max(s[1]), 40.0, atol=0.01)
        assert_allclose(np.min(s[1]), 32.0, atol=0.01)

        assert_allclose(sample.sector_area, 12.4, atol=0.1)
        assert sample.total_points == 64
        assert sample.actual_points == 64

    def test_mean_small(self):
        s, sample = self.make_sample(sma=5.0, integrmode=MEAN)

        assert len(s[0]) == 29
        # Intensities
        assert_allclose(np.mean(s[2]), 2339.0, atol=0.1)
        assert_allclose(np.std(s[2]), 284.7, atol=0.1)
        # Radii
        assert_allclose(np.max(s[1]), 5.0, atol=0.01)
        assert_allclose(np.min(s[1]), 4.0, atol=0.01)

        assert_allclose(sample.sector_area, 2.0, atol=0.1)
        assert sample.total_points == 29
        assert sample.actual_points == 29

    def test_median(self):
        s, sample = self.make_sample(integrmode=MEDIAN)

        assert len(s[0]) == 64
        # Intensities
        assert_allclose(np.mean(s[2]), 199.9, atol=0.1)
        assert_allclose(np.std(s[2]), 21.3, atol=0.1)
        # Radii
        assert_allclose(np.max(s[1]), 40.0, atol=0.01)
        assert_allclose(np.min(s[1]), 32.01, atol=0.01)

        assert_allclose(sample.sector_area, 12.4, atol=0.1)
        assert sample.total_points == 64
        assert sample.actual_points == 64


@pytest.mark.remote_data
class TestMasked(TestData):
    def test_bilinear(self):
        s, sample = self.make_sample(masked=True, integrmode=BILINEAR)

        assert len(s[0]) == 157
        # Intensities
        assert_allclose(np.mean(s[2]), 201.52, atol=0.01)
        assert_allclose(np.std(s[2]), 25.21, atol=0.01)
        # Radii
        assert_allclose(np.max(s[1]), 40.0, atol=0.01)
        assert_allclose(np.min(s[1]), 32.0, atol=0.01)

        assert sample.total_points == 225
        assert sample.actual_points == 157

    def test_mean(self):
        s, sample = self.make_sample(masked=True, integrmode=MEAN)

        assert len(s[0]) == 51
        # Intensities
        assert_allclose(np.mean(s[2]), 199.9, atol=0.1)
        assert_allclose(np.std(s[2]), 24.12, atol=0.1)
        # Radii
        assert_allclose(np.max(s[1]), 40.0, atol=0.01)
        assert_allclose(np.min(s[1]), 32.0, atol=0.01)

        assert_allclose(sample.sector_area, 12.4, atol=0.1)
        assert sample.total_points == 64
        assert sample.actual_points == 51


@pytest.mark.parametrize('integrmode',
                         [NEAREST_NEIGHBOR, BILINEAR, MEAN, MEDIAN])
def test_masked_and_nonfinite(integrmode):
    """
    Test that masked pixels and nonfinite pixels are excluded in the
    same way.
    """
    data = make_test_image(seed=0)
    mask = np.zeros(data.shape, dtype=bool)
    mask[230:260, 280:300] = True

    sample = EllipseSample(data, 40.0, integrmode=integrmode)
    sample.update()
    sample_masked = EllipseSample(np.ma.masked_array(data, mask=mask), 40.0,
                                  integrmode=integrmode)
    sample_masked.update()
    data_nan = data.copy()
    data_nan[mask] = np.nan
    sample_nan = EllipseSample(data_nan, 40.0, integrmode=integrmode)
    sample_nan.update()

    assert sample_masked.actual_points < sample.actual_points
    assert sample_masked.total_points == sample.total_points
    assert_array_equal(sample_masked.values, sample_nan.values)


@pytest.mark.parametrize('integrmode',
                         [NEAREST_NEIGHBOR, BILINEAR, MEAN, MEDIAN])
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

    kwargs = {'x0': 64.0, 'y0': 64.0, 'integrmode': integrmode}
    sample = EllipseSample(np.ma.masked_array(data, mask=mask), 20.0,
                           **kwargs)
    sample_inf = EllipseSample(np.ma.masked_array(data_inf, mask=mask), 20.0,
                               **kwargs)

    assert sample.extract().shape[1] > 0
    assert_array_equal(sample_inf.extract(), sample.extract())


@pytest.mark.parametrize('integrmode', [NEAREST_NEIGHBOR, BILINEAR])
@pytest.mark.parametrize('x0', [np.nan, np.inf, 1.0e30])
def test_nonfinite_path(integrmode, x0):
    """
    Test that path points with nonfinite or very large coordinates
    are skipped like the points outside of the image.
    """
    data = make_test_image(nx=128, ny=128, seed=0)
    sample = EllipseSample(data, 10.0, x0=x0, y0=64.0,
                           integrmode=integrmode)
    values = sample.extract()
    assert values.shape == (3, 0)
    assert sample.total_points > 0
    assert sample.actual_points == 0


def _sample_point(integrmode, data, x, y):
    """
    Sample a single image point using the given integrator.
    """
    geometry = EllipseGeometry(x, y, 1.0, 0.0, 0.0)
    intensities = []
    integrator = INTEGRATORS[integrmode](data, geometry, [], [], intensities)
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
    assert_allclose(_sample_point(BILINEAR, data, x, y), expected)


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
    assert_allclose(_sample_point(NEAREST_NEIGHBOR, data, x, y), expected)


@pytest.mark.parametrize('integrmode', [NEAREST_NEIGHBOR, BILINEAR])
def test_fit_center(integrmode):
    """
    Test that the fitted center of a symmetric source is not offset.
    """
    yy, xx = np.mgrid[0:101, 0:101]
    data = Gaussian2D(100.0, 50.0, 50.0, 12.0, 8.0, theta=0.5)(xx, yy)
    geometry = EllipseGeometry(50.0, 50.0, 10.0, 0.3, 0.5)
    ellipse = Ellipse(data, geometry=geometry)
    isolist = ellipse.fit_image(integrmode=integrmode, sma0=10.0,
                                maxsma=25.0)
    assert_allclose(np.median(isolist.x0), 50.0, atol=0.05)
    assert_allclose(np.median(isolist.y0), 50.0, atol=0.05)
