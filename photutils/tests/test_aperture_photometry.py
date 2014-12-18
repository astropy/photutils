# Licensed under a 3-clause BSD style license - see LICENSE.rst

# The tests in this file test the accuracy of the photometric results.
# Here we test directly with aperture objects since we are checking the
# algorithms in aperture_photometry, not in the wrappers.
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import astropy.units as u
from astropy.io import fits
from astropy.nddata import NDData
from astropy.tests.helper import remote_data

from ..aperture_core import *

try:
    import matplotlib
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


APERTURE_CL = [CircularAperture,
               CircularAnnulus,
               EllipticalAperture,
               EllipticalAnnulus,
               RectangularAperture,
               RectangularAnnulus]


TEST_APERTURES = list(zip(APERTURE_CL, ((3.,), (3., 5.),
                                        (3., 5., 1.), (3., 5., 4., 1.),
                                        (5, 8, np.pi / 4), (8, 12, 8, np.pi / 8))))

@pytest.mark.parametrize(('aperture_class', 'params'), TEST_APERTURES)
def test_outside_array(aperture_class, params):
    data = np.ones((10, 10), dtype=np.float)
    aperture = aperture_class((-60, 60), *params)
    fluxtable = aperture_photometry(data, aperture)
    # aperture is fully outside array:
    assert np.isnan(fluxtable['aperture_sum'])


@pytest.mark.parametrize(('aperture_class', 'params'), TEST_APERTURES)
def test_inside_array_simple(aperture_class, params):
    data = np.ones((40, 40), dtype=np.float)
    aperture = aperture_class((20., 20.), *params)
    table1 = aperture_photometry(data, aperture, method='center', subpixels=10)
    table2 = aperture_photometry(data, aperture, method='subpixel', subpixels=10)
    table3 = aperture_photometry(data, aperture, method='exact', subpixels=10)
    true_flux = aperture.area()

    if not isinstance(aperture, (RectangularAperture, RectangularAnnulus)):
        assert_allclose(table3['aperture_sum'], true_flux)
        assert_allclose(table2['aperture_sum'], table3['aperture_sum'], atol=0.1)
    assert table1['aperture_sum'] < table3['aperture_sum']


@pytest.mark.skipif('not HAS_MATPLOTLIB')
@pytest.mark.parametrize(('aperture_class', 'params'), TEST_APERTURES)
def test_aperture_plots(aperture_class, params):
    # This test should run without any errors, and there is no return value

    # TODO for 0.2: check the content of the plot

    aperture = aperture_class((20., 20.), *params)
    aperture.plot()


class BaseTestAperturePhotometry(object):

    def test_scalar_error_no_effective_gain(self):

        # Scalar error, no effective_gain.
        error = 1.
        if not hasattr(self, 'mask'):
            mask = None
        else:
            mask = self.mask

        table1 = aperture_photometry(self.data,
                                     self.aperture, method='center',
                                     mask=mask, error=error)
        table2 = aperture_photometry(self.data,
                                     self.aperture,
                                     method='subpixel', subpixels=12,
                                     mask=mask, error=error)
        table3 = aperture_photometry(self.data,
                                     self.aperture, method='exact',
                                     mask=mask, error=error)

        if not isinstance(self.aperture, (RectangularAperture, RectangularAnnulus)):
            assert_allclose(table3['aperture_sum'], self.true_flux)
            assert_allclose(table2['aperture_sum'], table3['aperture_sum'],
                            atol=0.1)
        assert np.all(table1['aperture_sum'] < table3['aperture_sum'])

        true_error = np.sqrt(self.area)
        if not isinstance(self.aperture, (RectangularAperture, RectangularAnnulus)):
            assert_allclose(table3['aperture_sum_err'], true_error)
            assert_allclose(table2['aperture_sum'], table3['aperture_sum'],
                            atol=0.1)
        assert np.all(table1['aperture_sum_err'] < table3['aperture_sum_err'])

    def test_scalar_error_scalar_effective_gain(self):

        # Scalar error, scalar effective_gain.
        error = 1.
        effective_gain = 1.
        if not hasattr(self, 'mask'):
            mask = None
        else:
            mask = self.mask

        table1 = aperture_photometry(self.data, self.aperture,
                                     method='center', mask=mask, error=error,
                                     effective_gain=effective_gain)
        table2 = aperture_photometry(self.data, self.aperture,
                                     method='subpixel', subpixels=12,
                                     mask=mask, error=error,
                                     effective_gain=effective_gain)
        table3 = aperture_photometry(self.data, self.aperture, method='exact',
                                     mask=mask, error=error,
                                     effective_gain=effective_gain)

        if not isinstance(self.aperture,
                          (RectangularAperture, RectangularAnnulus)):
            assert_allclose(table3['aperture_sum'], self.true_flux)
            assert_allclose(table2['aperture_sum'], table3['aperture_sum'],
                            atol=0.1)
        assert np.all(table1['aperture_sum'] < table3['aperture_sum'])

        true_error = np.sqrt(self.area + self.true_flux)
        if not isinstance(self.aperture,
                          (RectangularAperture, RectangularAnnulus)):
            assert_allclose(table3['aperture_sum_err'], true_error)
            assert_allclose(table2['aperture_sum'], table3['aperture_sum'],
                            atol=0.1)
        assert np.all(table1['aperture_sum_err'] < table3['aperture_sum_err'])

    def test_quantity_scalar_error_scalar_effective_gain(self):

        # Scalar error, scalar effective_gain.
        error = u.Quantity(1.)
        effective_gain = u.Quantity(1.)
        if not hasattr(self, 'mask'):
            mask = None
        else:
            mask = self.mask

        table1 = aperture_photometry(self.data, self.aperture,
                                     method='center', mask=mask, error=error,
                                     effective_gain=effective_gain)
        assert np.all(table1['aperture_sum'] < self.true_flux)

        true_error = np.sqrt(self.area + self.true_flux)
        assert np.all(table1['aperture_sum_err'] < true_error)

    def test_scalar_error_array_effective_gain(self):

        # Scalar error, Array effective_gain.
        error = 1.
        effective_gain = np.ones(self.data.shape, dtype=np.float)
        if not hasattr(self, 'mask'):
            mask = None
            true_error = np.sqrt(self.area + self.true_flux)
        else:
            mask = self.mask
            # 1 masked pixel
            true_error = np.sqrt(self.area - 1 + self.true_flux)

        table1 = aperture_photometry(self.data, self.aperture,
                                     method='center', mask=mask, error=error,
                                     effective_gain=effective_gain)
        table2 = aperture_photometry(self.data, self.aperture,
                                     method='subpixel', subpixels=12,
                                     mask=mask, error=error,
                                     effective_gain=effective_gain)
        table3 = aperture_photometry(self.data, self.aperture,
                                     method='exact', mask=mask, error=error,
                                     effective_gain=effective_gain)

        if not isinstance(self.aperture,
                          (RectangularAperture, RectangularAnnulus)):
            assert_allclose(table3['aperture_sum'], self.true_flux)
            assert_allclose(table2['aperture_sum'], table3['aperture_sum'],
                            atol=0.1)
        assert np.all(table1['aperture_sum'] < table3['aperture_sum'])

        if not isinstance(self.aperture,
                          (RectangularAperture, RectangularAnnulus)):
            assert_allclose(table3['aperture_sum_err'], true_error)
            assert_allclose(table2['aperture_sum'], table3['aperture_sum'],
                            atol=0.1)
        assert np.all(table1['aperture_sum_err'] < table3['aperture_sum_err'])

    def test_array_error_no_effective_gain(self):

        # Array error, no effective_gain.
        error = np.ones(self.data.shape, dtype=np.float)
        if not hasattr(self, 'mask'):
            mask = None
            true_error = np.sqrt(self.area)
        else:
            mask = self.mask
            # 1 masked pixel
            true_error = np.sqrt(self.area - 1)

        table1 = aperture_photometry(self.data,
                                     self.aperture, method='center',
                                     mask=mask, error=error)
        table2 = aperture_photometry(self.data,
                                     self.aperture,
                                     method='subpixel', subpixels=12,
                                     mask=mask, error=error)
        table3 = aperture_photometry(self.data,
                                     self.aperture, method='exact',
                                     mask=mask, error=error)

        if not isinstance(self.aperture, (RectangularAperture, RectangularAnnulus)):
            assert_allclose(table3['aperture_sum'], self.true_flux)
            assert_allclose(table2['aperture_sum'], table3['aperture_sum'],
                            atol=0.1)
        assert np.all(table1['aperture_sum'] < table3['aperture_sum'])

        if not isinstance(self.aperture, (RectangularAperture, RectangularAnnulus)):
            assert_allclose(table3['aperture_sum_err'], true_error)
            assert_allclose(table2['aperture_sum_err'], table3['aperture_sum_err'],
                            atol=0.1)
        assert np.all(table1['aperture_sum_err'] < table3['aperture_sum_err'])

    def test_array_error_array_effective_gain(self):

        error = np.ones(self.data.shape, dtype=np.float)
        effective_gain = np.ones(self.data.shape, dtype=np.float)
        if not hasattr(self, 'mask'):
            mask = None
            true_error = np.sqrt(self.area + self.true_flux)
        else:
            mask = self.mask
            # 1 masked pixel
            true_error = np.sqrt(self.area - 1 + self.true_flux)

        table1 = aperture_photometry(self.data, self.aperture,
                                     method='center', mask=mask, error=error,
                                     effective_gain=effective_gain)
        table2 = aperture_photometry(self.data, self.aperture,
                                     method='subpixel', subpixels=12,
                                     mask=mask, error=error,
                                     effective_gain=effective_gain)
        table3 = aperture_photometry(self.data, self.aperture, method='exact',
                                     mask=mask, error=error,
                                     effective_gain=effective_gain)

        if not isinstance(self.aperture, (RectangularAperture, RectangularAnnulus)):
            assert_allclose(table3['aperture_sum'], self.true_flux)
            assert_allclose(table2['aperture_sum'], table3['aperture_sum'],
                            atol=0.1)
        assert np.all(table1['aperture_sum'] < table3['aperture_sum'])

        if not isinstance(self.aperture, (RectangularAperture, RectangularAnnulus)):
            assert_allclose(table3['aperture_sum_err'], true_error)
            assert_allclose(table2['aperture_sum_err'], table3['aperture_sum_err'],
                            atol=0.1)
        assert np.all(table1['aperture_sum_err'] < table3['aperture_sum_err'])


class TestCircular(BaseTestAperturePhotometry):

    def setup_class(self):
        self.data = np.ones((40, 40), dtype=np.float)
        position = (20., 20.)
        r = 10.
        self.aperture = CircularAperture(position, r)
        self.area = np.pi * r * r
        self.true_flux = self.area


class TestCircularArray(BaseTestAperturePhotometry):

    def setup_class(self):
        self.data = np.ones((40, 40), dtype=np.float)
        position = ((20., 20.), (25., 25.))
        r = 10.
        self.aperture = CircularAperture(position, r)
        self.area = np.pi * r * r
        self.area = np.array((self.area, ) * 2)
        self.true_flux = self.area


class TestCircularAnnulus(BaseTestAperturePhotometry):

    def setup_class(self):
        self.data = np.ones((40, 40), dtype=np.float)
        position = (20., 20.)
        r_in = 8.
        r_out = 10.
        self.aperture = CircularAnnulus(position, r_in, r_out)
        self.area = np.pi * (r_out * r_out - r_in * r_in)
        self.true_flux = self.area


class TestCircularAnnulusArray(BaseTestAperturePhotometry):

    def setup_class(self):
        self.data = np.ones((40, 40), dtype=np.float)
        position = ((20., 20.), (25., 25.))
        r_in = 8.
        r_out = 10.
        self.aperture = CircularAnnulus(position, r_in, r_out)
        self.area = np.pi * (r_out * r_out - r_in * r_in)
        self.area = np.array((self.area, ) * 2)
        self.true_flux = self.area


class TestElliptical(BaseTestAperturePhotometry):

    def setup_class(self):
        self.data = np.ones((40, 40), dtype=np.float)
        position = (20., 20.)
        a = 10.
        b = 5.
        theta = -np.pi / 4.
        self.aperture = EllipticalAperture(position, a, b, theta)
        self.area = np.pi * a * b
        self.true_flux = self.area


class TestEllipticalAnnulus(BaseTestAperturePhotometry):

    def setup_class(self):
        self.data = np.ones((40, 40), dtype=np.float)
        position = (20., 20.)
        a_in = 5.
        a_out = 8.
        b_out = 6.
        theta = -np.pi / 4.
        self.aperture = EllipticalAnnulus(position, a_in, a_out, b_out, theta)
        self.area = np.pi * (a_out * b_out) - np.pi * (a_in * b_out * a_in / a_out)
        self.true_flux = self.area


class TestRectangularAperture(BaseTestAperturePhotometry):

    def setup_class(self):
        self.data = np.ones((40, 40), dtype=np.float)
        position = (20., 20.)
        h = 5.
        w = 8.
        theta = np.pi / 4.
        self.aperture = RectangularAperture(position, w, h, theta)
        self.area = h * w
        self.true_flux = self.area


class TestRectangularAnnulus(BaseTestAperturePhotometry):

    def setup_class(self):
        self.data = np.ones((40, 40), dtype=np.float)
        position = (20., 20.)
        h_out = 8.
        w_in = 8.
        w_out = 12.
        h_in = w_in * h_out / w_out
        theta = np.pi / 8.
        self.aperture = RectangularAnnulus(position, w_in, w_out, h_out, theta)
        self.area = h_out * w_out - h_in * w_in
        self.true_flux = self.area


class TestMaskedSkipCircular(BaseTestAperturePhotometry):

    def setup_class(self):
        self.data = np.ones((40, 40), dtype=np.float)
        self.mask = np.zeros((40, 40), dtype=bool)
        self.mask[20, 20] = True
        position = (20., 20.)
        r = 10.
        self.aperture = CircularAperture(position, r)
        self.area = np.pi * r * r
        self.true_flux = self.area - 1


class BaseTestDifferentData(object):

    def test_basic_circular_aperture_photometry(self):

        aperture = CircularAperture(self.position, self.radius)
        table = aperture_photometry(self.data, aperture,
                                    method='exact')

        assert_allclose(table['aperture_sum'], self.true_flux)
        assert table['aperture_sum'].unit, self.fluxunit

        assert np.all(table['xcenter'] ==
                      np.transpose(self.position)[0])
        assert np.all(table['ycenter'] ==
                      np.transpose(self.position)[1])


class TestInputPrimaryHDU(BaseTestDifferentData):

    def setup_class(self):
        data = np.ones((40, 40), dtype=np.float)
        self.data = fits.ImageHDU(data=data)
        self.data.header['BUNIT'] = 'adu'
        self.radius = 3
        self.position = (20, 20)
        self.true_flux = np.pi * self.radius * self.radius
        self.fluxunit = u.adu


class TestInputHDUList(BaseTestDifferentData):

    def setup_class(self):
        data0 = np.ones((40, 40), dtype=np.float)
        data1 = np.empty((40, 40), dtype=np.float)
        data1.fill(2)
        self.data = fits.HDUList([fits.ImageHDU(data=data0),
                                  fits.ImageHDU(data=data1)])
        self.radius = 3
        self.position = (20, 20)
        # It should stop at the first extension
        self.true_flux = np.pi * self.radius * self.radius


class TestInputNDData(BaseTestDifferentData):

    def setup_class(self):
        data = np.ones((40, 40), dtype=np.float)
        self.data = NDData(data, unit=u.adu)
        self.radius = 3
        self.position = [(20, 20), (30, 30)]
        self.true_flux = np.pi * self.radius * self.radius
        self.fluxunit = u.adu


@remote_data
def test_wcs_based_photometry():
    from astropy.table import Table
    from astropy.coordinates import SkyCoord
    from ..datasets import get_path

    pathcat = get_path('spitzer_example_catalog.xml', location='remote')
    pathhdu = get_path('spitzer_example_image.fits', location='remote')
    hdu = fits.open(pathhdu)
    catalog = Table.read(pathcat)
    fluxes_catalog = catalog['f4_5']
    pos_skycoord = SkyCoord(catalog['l'], catalog['b'], frame='galactic')
    pos_skycoord_s = SkyCoord(catalog['l'][0] * catalog['l'].unit,
                              catalog['b'][0] * catalog['b'].unit,
                              frame='galactic')

    photometry_skycoord = aperture_photometry(
        hdu, SkyCircularAperture(pos_skycoord, 4 * u.arcsec))
    photometry_skycoord_s = aperture_photometry(
        hdu, SkyCircularAperture(pos_skycoord_s, 4 * u.arcsec))

    assert_allclose(photometry_skycoord['aperture_sum'][0],
                    photometry_skycoord_s['aperture_sum'])

    photometry_skycoord_pix = aperture_photometry(
        hdu, SkyCircularAperture(pos_skycoord, 4 / 1.2 * u.pixel))

    assert_allclose(photometry_skycoord['aperture_sum'],
                    photometry_skycoord_pix['aperture_sum'])

    photometry_skycoord_rec = aperture_photometry(
        hdu, SkyRectangularAperture(pos_skycoord,
                                    8 * u.arcsec, 8 * u.arcsec,
                                    0 * u.deg))
    photometry_skycoord_rec_s = aperture_photometry(
        hdu, SkyRectangularAperture(pos_skycoord_s,
                                    8 * u.arcsec, 8 * u.arcsec,
                                    0 * u.deg))

    assert_allclose(photometry_skycoord_rec['aperture_sum'][0],
                    photometry_skycoord_rec_s['aperture_sum'])

    assert np.all(photometry_skycoord_rec['aperture_sum'] >
                  photometry_skycoord['aperture_sum'])

    # TODO compare with fluxes_catalog


def test_basic_circular_aperture_photometry_unit():

    data1 = np.ones((40, 40), dtype=np.float)
    data2 = u.Quantity(data1, unit=u.adu)
    data3 = u.Quantity(data1, unit=u.Jy)

    radius = 3
    position = (20, 20)
    true_flux = np.pi * radius * radius
    unit = u.adu

    table1 = aperture_photometry(data1, CircularAperture(position, radius),
                                 unit=unit)
    table2 = aperture_photometry(data2, CircularAperture(position, radius),
                                 unit=unit)
    with pytest.raises(u.UnitsError) as err:
        aperture_photometry(data3, CircularAperture(position, radius), unit=unit)
    assert ("UnitsError: Unit of input data (Jy) and unit given by unit "
            "argument (adu) are not identical." in str(err))

    assert_allclose(table1['aperture_sum'], true_flux)
    assert_allclose(table2['aperture_sum'], true_flux)
    assert table1['aperture_sum'].unit == unit
    assert table2['aperture_sum'].unit == data2.unit == unit


def test_aperture_photometry_with_error_units():
    """Test aperture_photometry when error has units (see #176)."""
    data1 = np.ones((40, 40), dtype=np.float)
    data2 = u.Quantity(data1, unit=u.adu)
    error = u.Quantity(data1, unit=u.adu)
    radius = 3
    true_flux = np.pi * radius * radius
    unit = u.adu
    position = (20, 20)
    table1 = aperture_photometry(data2, CircularAperture(position, radius),
                                 error=error)
    assert_allclose(table1['aperture_sum'], true_flux)
    assert_allclose(table1['aperture_sum_err'], np.sqrt(true_flux))
    assert table1['aperture_sum'].unit == unit
    assert table1['aperture_sum_err'].unit == unit


def test_aperture_photometry_inputs_with_mask():
    """
    Test that aperture_photometry does not modify the input
    data or error array when a mask is input.
    """
    data = np.ones((5, 5))
    aperture = CircularAperture((2, 2), 2.)
    mask = np.zeros_like(data, dtype=bool)
    data[2, 2] = 100.   # bad pixel
    mask[2, 2] = True
    error = np.sqrt(data)
    data_in = data.copy()
    error_in = error.copy()
    t1 = aperture_photometry(data, aperture, error=error, mask=mask)
    assert_array_equal(data, data_in)
    assert_array_equal(error, error_in)
    assert_allclose(t1['aperture_sum'][0], 11.5663706144)
    t2 = aperture_photometry(data, aperture)
    assert_allclose(t2['aperture_sum'][0], 111.566370614)



TEST_ELLIPSE_EXACT_APERTURES = [(3.469906, 3.923861394, 3.),
                                (0.3834415188257778, 0.3834415188257778, 0.3)]

@pytest.mark.parametrize('x,y,r', TEST_ELLIPSE_EXACT_APERTURES)
def test_ellipse_exact_grid(x, y, r):
    """
    Test elliptical exact aperture photometry on a grid of pixel positions.

    This is a regression test for the bug discovered in this issue:
    https://github.com/astropy/photutils/issues/198
    """
    data = np.ones((10, 10))

    aperture = EllipticalAperture((x, y), r, r, 0.)
    t = aperture_photometry(data, aperture, method='exact')
    actual= t['aperture_sum'][0] / (np.pi * r ** 2)
    assert_allclose(actual, 1)

