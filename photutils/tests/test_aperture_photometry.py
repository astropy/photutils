# Licensed under a 3-clause BSD style license - see LICENSE.rst

# The tests in this file test the accuracy of the photometric results.
# Here we test directly with aperture objects since we are checking the
# algorithms in aperture_photometry, not in the wrappers.
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import pytest
import numpy as np
from numpy.testing import assert_allclose
import astropy.units as u
from astropy.io import fits
from astropy.nddata import NDData
from astropy.tests.helper import remote_data

from ..aperture_core import *

APERTURE_CL = [CircularAperture,
               CircularAnnulus,
               EllipticalAperture,
               EllipticalAnnulus,
               RectangularAperture]


@pytest.mark.parametrize(('aperture_class', 'params'),
                         zip(APERTURE_CL, ((3.,), (3., 5.), (3., 5., 1.),
                                         (3., 5., 4., 1.), (5, 8, np.pi / 4))))
def test_outside_array(aperture_class, params):
    data = np.ones((10, 10), dtype=np.float)
    aperture = aperture_class((-60, 60), *params)
    fluxtable = aperture_photometry(data, aperture)[0]
    # aperture is fully outside array:
    assert np.isnan(fluxtable['aperture_sum'])


@pytest.mark.parametrize(('aperture_class', 'params'),
                         zip(APERTURE_CL, ((3.,), (3., 5.),
                                           (3., 5., 1.),
                                           (3., 5., 4., 1.),
                                           (5, 8, np.pi / 4))))
def test_inside_array_simple(aperture_class, params):
    data = np.ones((40, 40), dtype=np.float)
    aperture = aperture_class((20., 20.), *params)
    table1 = aperture_photometry(data, aperture, method='center', subpixels=10)[0]
    table2 = aperture_photometry(data, aperture, method='subpixel', subpixels=10)[0]
    table3 = aperture_photometry(data, aperture, method='exact', subpixels=10)[0]
    true_flux = aperture.area()

    if not isinstance(aperture, RectangularAperture):
        assert_allclose(table3['aperture_sum'], true_flux)
    assert table1['aperture_sum'] < table3['aperture_sum']
    assert_allclose(table2['aperture_sum'], table3['aperture_sum'], atol=0.1)


class BaseTestAperturePhotometry(object):

    def test_scalar_error_no_gain(self):

        # Scalar error, no gain.
        error = 1.
        if not hasattr(self, 'mask'):
            mask = None
        else:
            mask = self.mask

        if not hasattr(self, 'mask_method'):
            mask_method = 'skip'
        else:
            mask_method = self.mask_method

        table1 = aperture_photometry(self.data,
                                     self.aperture, method='center',
                                     mask=mask, mask_method=mask_method,
                                     error=error)[0]
        table2 = aperture_photometry(self.data,
                                     self.aperture,
                                     method='subpixel', subpixels=12,
                                     mask=mask, mask_method=mask_method,
                                     error=error)[0]
        table3 = aperture_photometry(self.data,
                                     self.aperture, method='exact',
                                     mask=mask, mask_method=mask_method,
                                     error=error)[0]

        if not isinstance(self.aperture, RectangularAperture):
            assert_allclose(table3['aperture_sum'], self.true_flux)
        assert table1['aperture_sum'] < table3['aperture_sum']
        assert_allclose(table2['aperture_sum'], table3['aperture_sum'],
                        atol=0.1)

        true_error = np.sqrt(self.area)
        if not isinstance(self.aperture, RectangularAperture):
            assert_allclose(table3['aperture_sum_err'], true_error)
        assert table1['aperture_sum_err'] < table3['aperture_sum_err']
        assert_allclose(table2['aperture_sum_err'], table3['aperture_sum_err'],
                        atol=0.1)

    def test_scalar_error_scalar_gain(self):

        # Scalar error, scalar gain.
        error = 1.
        gain = 1.
        if not hasattr(self, 'mask'):
            mask = None
        else:
            mask = self.mask

        if not hasattr(self, 'mask_method'):
            mask_method = 'skip'
        else:
            mask_method = self.mask_method

        table1 = aperture_photometry(self.data, 
                                     self.aperture, method='center',
                                     mask=mask, mask_method=mask_method,
                                     error=error, gain=gain)[0]
        table2 = aperture_photometry(self.data, 
                                     self.aperture,
                                     method='subpixel', subpixels=12,
                                     mask=mask, mask_method=mask_method,
                                     error=error, gain=gain)[0]
        table3 = aperture_photometry(self.data,
                                     self.aperture, method='exact',
                                     mask=mask, mask_method=mask_method,
                                     error=error, gain=gain)[0]

        if not isinstance(self.aperture, RectangularAperture):
            assert_allclose(table3['aperture_sum'], self.true_flux)
        assert table1['aperture_sum'] < table3['aperture_sum']
        assert_allclose(table2['aperture_sum'], table3['aperture_sum'],
                        atol=0.1)

        true_error = np.sqrt(self.area + self.true_flux)
        if not isinstance(self.aperture, RectangularAperture):
            assert_allclose(table3['aperture_sum_err'], true_error)
        assert table1['aperture_sum_err'] < table3['aperture_sum_err']
        assert_allclose(table2['aperture_sum_err'], table3['aperture_sum_err'],
                        atol=0.1)

    def test_quantity_scalar_error_scalar_gain(self):

        # Scalar error, scalar gain.
        error = u.Quantity(1.)
        gain = u.Quantity(1.)
        if not hasattr(self, 'mask'):
            mask = None
        else:
            mask = self.mask

        if not hasattr(self, 'mask_method'):
            mask_method = 'skip'
        else:
            mask_method = self.mask_method

        table1 = aperture_photometry(self.data,
                                     self.aperture, method='center',
                                     mask=mask, mask_method=mask_method,
                                     error=error, gain=gain)[0]
        assert table1['aperture_sum'] < self.true_flux

        true_error = np.sqrt(self.area + self.true_flux)
        assert table1['aperture_sum_err'] < true_error

    def test_scalar_error_array_gain(self):

        # Scalar error, Array gain.
        error = 1.
        gain = np.ones(self.data.shape, dtype=np.float)
        if not hasattr(self, 'mask'):
            mask = None
        else:
            mask = self.mask

        if not hasattr(self, 'mask_method'):
            mask_method = 'skip'
        else:
            mask_method = self.mask_method

        table1 = aperture_photometry(self.data,
                                     self.aperture, method='center',
                                     mask=mask, mask_method=mask_method,
                                     error=error, gain=gain)[0]
        table2 = aperture_photometry(self.data,
                                     self.aperture,
                                     method='subpixel', subpixels=12,
                                     mask=mask, mask_method=mask_method,
                                     error=error, gain=gain)[0]
        table3 = aperture_photometry(self.data,
                                     self.aperture, method='exact',
                                     mask=mask, mask_method=mask_method,
                                     error=error, gain=gain)[0]

        if not isinstance(self.aperture, RectangularAperture):
            assert_allclose(table3['aperture_sum'], self.true_flux)
        assert table1['aperture_sum'] < table3['aperture_sum']
        assert_allclose(table2['aperture_sum'], table3['aperture_sum'],
                        atol=0.1)

        true_error = np.sqrt(self.area + self.true_flux)
        if not isinstance(self.aperture, RectangularAperture):
            assert_allclose(table3['aperture_sum_err'], true_error)
        assert table1['aperture_sum_err'] < table3['aperture_sum_err']
        assert_allclose(table2['aperture_sum_err'], table3['aperture_sum_err'],
                        atol=0.1)

    def test_array_error_no_gain(self):

        # Array error, no gain.
        error = np.ones(self.data.shape, dtype=np.float)
        if not hasattr(self, 'mask'):
            mask = None
        else:
            mask = self.mask

        if not hasattr(self, 'mask_method'):
            mask_method = 'skip'
        else:
            mask_method = self.mask_method

        table1 = aperture_photometry(self.data,
                                     self.aperture, method='center',
                                     mask=mask, mask_method=mask_method,
                                     error=error)[0]
        table2 = aperture_photometry(self.data,
                                     self.aperture,
                                     method='subpixel', subpixels=12,
                                     mask=mask, mask_method=mask_method,
                                     error=error)[0]
        table3 = aperture_photometry(self.data,
                                     self.aperture, method='exact',
                                     mask=mask, mask_method=mask_method,
                                     error=error)[0]

        if not isinstance(self.aperture, RectangularAperture):
            assert_allclose(table3['aperture_sum'], self.true_flux)
        assert table1['aperture_sum'] < table3['aperture_sum']
        assert_allclose(table2['aperture_sum'], table3['aperture_sum'],
                        atol=0.1)

        true_error = np.sqrt(self.area)
        if not isinstance(self.aperture, RectangularAperture):
            assert_allclose(table3['aperture_sum_err'], true_error)
        assert table1['aperture_sum_err'] < table3['aperture_sum_err']
        assert_allclose(table2['aperture_sum_err'], table3['aperture_sum_err'],
                        atol=0.1)

    def test_array_error_array_gain(self):

        error = np.ones(self.data.shape, dtype=np.float)
        gain = np.ones(self.data.shape, dtype=np.float)
        if not hasattr(self, 'mask'):
            mask = None
        else:
            mask = self.mask

        if not hasattr(self, 'mask_method'):
            mask_method = 'skip'
        else:
            mask_method = self.mask_method

        table1 = aperture_photometry(self.data,
                                     self.aperture, method='center',
                                     mask=mask, mask_method=mask_method,
                                     error=error, gain=gain)[0]
        table2 = aperture_photometry(self.data,
                                     self.aperture,
                                     method='subpixel', subpixels=12,
                                     mask=mask, mask_method=mask_method,
                                     error=error, gain=gain)[0]
        table3 = aperture_photometry(self.data,
                                     self.aperture, method='exact',
                                     mask=mask, mask_method=mask_method,
                                     error=error, gain=gain)[0]

        if not isinstance(self.aperture, RectangularAperture):
            assert_allclose(table3['aperture_sum'], self.true_flux)
        assert table1['aperture_sum'] < table3['aperture_sum']
        assert_allclose(table2['aperture_sum'], table3['aperture_sum'],
                        atol=0.1)

        true_error = np.sqrt(self.area + self.true_flux)
        if not isinstance(self.aperture, RectangularAperture):
            assert_allclose(table3['aperture_sum_err'], true_error)
        assert table1['aperture_sum_err'] < table3['aperture_sum_err']
        assert_allclose(table2['aperture_sum_err'], table3['aperture_sum_err'],
                        atol=0.1)


class TestCircular(BaseTestAperturePhotometry):

    def setup_class(self):
        self.data = np.ones((40, 40), dtype=np.float)
        position = (20., 20.)
        r = 10.
        self.aperture = CircularAperture(position, r)
        self.area = np.pi * r * r
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
        self.aperture = RectangularAperture(position, h, w, theta)
        self.area = h * w
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
        self.mask_method = 'skip'


class TestMaskedInterpolationCircular(BaseTestAperturePhotometry):

    def setup_class(self):
        self.data = np.ones((40, 40), dtype=np.float)
        self.mask = np.zeros((40, 40), dtype=bool)
        self.mask[20, 20] = True
        position = (20., 20.)
        r = 10.
        self.aperture = CircularAperture(position, r)
        self.area = np.pi * r * r
        self.true_flux = self.area
        self.mask_method = 'interpolation'


class BaseTestDifferentData(object):

    def test_basic_circular_aperture_photometry(self):

        aperture = CircularAperture(self.position, self.radius)
        table = aperture_photometry(self.data, aperture,
                                    method='exact')[0]

        assert_allclose(table['aperture_sum'], self.true_flux)
        assert table['aperture_sum'].unit, self.fluxunit

        assert np.all(table['input_center'] == self.position)


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

    photometry_skycoord = aperture_photometry(hdu, SkyCircularAperture(pos_skycoord, 4 * u.pixel))

    photometry_skycoord_s = aperture_photometry(hdu, SkyCircularAperture(pos_skycoord_s, 4 * u.pixel))

    assert_allclose(photometry_skycoord[0]['aperture_sum'][0],
                    photometry_skycoord_s[0]['aperture_sum'])

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
                                 unit=unit)[0]
    table2 = aperture_photometry(data2, CircularAperture(position, radius),
                                 unit=unit)[0]
    with pytest.raises(u.UnitsError) as err:
        aperture_photometry(data3, CircularAperture(position, radius), unit=unit)
    assert ("UnitsError: Unit of input data (Jy) and unit given by unit "
            "argument (adu) are not identical." in str(err))

    assert_allclose(table1['aperture_sum'], true_flux)
    assert_allclose(table2['aperture_sum'], true_flux)
    assert table1['aperture_sum'].unit == unit
    assert table2['aperture_sum'].unit == data2.unit == unit
