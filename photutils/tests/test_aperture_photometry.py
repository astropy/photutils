# Licensed under a 3-clause BSD style license - see LICENSE.rst

# The tests in this file test the accuracy of the photometric results.
# Here we test directly with aperture objects since we are checking the
# algorithms in aperture_photometry, not in the wrappers.
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import pytest
import numpy as np
from numpy.testing import assert_allclose


from ..aperture import CircularAperture,\
                       CircularAnnulus, \
                       EllipticalAperture, \
                       EllipticalAnnulus, \
                       RectangularAperture,\
                       aperture_photometry


APERTURES = ['circular',
             'circular_annulus',
             'elliptical',
             'elliptical_annulus']

APERTURE_CL = [CircularAperture,
               CircularAnnulus,
               EllipticalAperture,
               EllipticalAnnulus]


@pytest.mark.parametrize(('aperture', 'radius'),
                         zip(APERTURES, ((3.,), (3., 5.), (3., 5., 1.),
                                         (3., 5., 4., 1.))))
def test_outside_array(aperture, radius):
    data = np.ones((10, 10), dtype=np.float)
    fluxtable = aperture_photometry(data, (-60, 60), ((aperture,) + radius))[0]
    # aperture is fully outside array:
    assert np.isnan(fluxtable['aperture_sum'])


@pytest.mark.parametrize(('aperture_cl', 'aperture', 'radius'),
                         zip(APERTURE_CL, APERTURES, ((3.,), (3., 5.), (3., 5., 1.),
                                                      (3., 5., 4., 1.))))
def test_inside_array_simple(aperture_cl, aperture, radius):
    data = np.ones((40, 40), dtype=np.float)
    table1 = aperture_photometry(data, (20., 20.), ((aperture,) + radius),
                                 method='center', subpixels=10)[0]
    table2 = aperture_photometry(data, (20., 20.), ((aperture,) + radius),
                                 method='subpixel', subpixels=10)[0]
    table3 = aperture_photometry(data, (20., 20.), ((aperture,) + radius),
                                 method='exact', subpixels=10)[0]
    true_flux = aperture_cl((20., 20.), *radius).area()

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

        table1 = aperture_photometry(self.data, self.position,
                                     self.aperture, method='center',
                                     mask=mask, mask_method=mask_method,
                                     error=error)[0]
        table2 = aperture_photometry(self.data, self.position,
                                     self.aperture,
                                     method='subpixel', subpixels=12,
                                     mask=mask, mask_method=mask_method,
                                     error=error)[0]
        table3 = aperture_photometry(self.data, self.position,
                                     self.aperture, method='exact',
                                     mask=mask, mask_method=mask_method,
                                     error=error)[0]

        assert_allclose(table3['aperture_sum'], self.true_flux)
        assert table1['aperture_sum'] < table3['aperture_sum']
        assert_allclose(table2['aperture_sum'], table3['aperture_sum'],
                        atol=0.1)

        if hasattr(self, 'true_variance'):
            true_error = np.sqrt(self.true_variance - self.true_flux)
        else:
            true_error = np.sqrt(self.area)

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

        table1 = aperture_photometry(self.data, self.position,
                                     self.aperture, method='center',
                                     mask=mask, mask_method=mask_method,
                                     error=error)[0]
        table2 = aperture_photometry(self.data, self.position,
                                     self.aperture,
                                     method='subpixel', subpixels=12,
                                     mask=mask, mask_method=mask_method,
                                     error=error)[0]
        table3 = aperture_photometry(self.data, self.position,
                                     self.aperture, method='exact',
                                     mask=mask, mask_method=mask_method,
                                     error=error)[0]

        assert_allclose(table3['aperture_sum'], self.true_flux)
        assert table1['aperture_sum'] < table3['aperture_sum']
        assert_allclose(table2['aperture_sum'], table3['aperture_sum'],
                        atol=0.1)

        if hasattr(self, 'true_variance'):
            true_error = np.sqrt(self.true_variance - self.true_flux)
        else:
            true_error = np.sqrt(self.area)

        assert_allclose(table3['aperture_sum_err'], true_error)
        assert table1['aperture_sum_err'] < table3['aperture_sum_err']
        assert_allclose(table2['aperture_sum_err'], table3['aperture_sum_err'],
                        atol=0.1)


class TestCircular(BaseTestAperturePhotometry):

    def setup_class(self):
        self.data = np.ones((40, 40), dtype=np.float)
        self.position = (20., 20.)
        r = 10.
        self.aperture = ('circular', r)
        self.area = np.pi * r * r
        self.true_flux = self.area


class TestCircularAnnulus(BaseTestAperturePhotometry):

    def setup_class(self):
        self.data = np.ones((40, 40), dtype=np.float)
        self.position = (20., 20.)
        r_in = 8.
        r_out = 10.
        self.aperture = ('circular_annulus', r_in, r_out)
        self.area = np.pi * (r_out * r_out - r_in * r_in)
        self.true_flux = self.area


class TestElliptical(BaseTestAperturePhotometry):

    def setup_class(self):
        self.data = np.ones((40, 40), dtype=np.float)
        self.position = (20., 20.)
        a = 10.
        b = 5.
        theta = -np.pi / 4.
        self.aperture = ('elliptical', a, b, theta)
        self.area = np.pi * a * b
        self.true_flux = self.area


class TestEllipticalAnnulus(BaseTestAperturePhotometry):

    def setup_class(self):
        self.data = np.ones((40, 40), dtype=np.float)
        self.position = (20., 20.)
        a_in = 5.
        a_out = 8.
        b_out = 6.
        theta = -np.pi / 4.
        self.aperture = ('elliptical_annulus', a_in, a_out, b_out, theta)
        self.area = np.pi * (a_out * b_out) - np.pi * (a_in * b_out * a_in / a_out)
        self.true_flux = self.area


def test_rectangular_aperture():
    data = np.ones((40, 40), dtype=np.float)
    position = (20., 20.)

    aperture = ('rectangular', 1., 2., np.pi / 4)

    table1 = aperture_photometry(data, position, aperture, method='center')[0]
    table2 = aperture_photometry(data, position, aperture, method='subpixel',
                                 subpixels=8)[0]

    true_flux = 1. * 2.

    assert table1['aperture_sum'] < true_flux
    assert_allclose(table2['aperture_sum'], true_flux, atol=0.1)


class TestMaskedSkipCircular(BaseTestAperturePhotometry):

    def setup_class(self):
        self.data = np.ones((40, 40), dtype=np.float)
        self.mask = np.zeros((40, 40), dtype=bool)
        self.mask[20, 20] = True
        self.position = (20., 20.)
        r = 10.
        self.aperture = ('circular', r)
        self.area = np.pi * r * r
        self.true_flux = self.area - 1
        self.mask_method = 'skip'


class TestMaskedInterpolationCircular(BaseTestAperturePhotometry):

    def setup_class(self):
        self.data = np.ones((40, 40), dtype=np.float)
        self.mask = np.zeros((40, 40), dtype=bool)
        self.mask[20, 20] = True
        self.position = (20., 20.)
        r = 10.
        self.aperture = ('circular', r)
        self.area = np.pi * r * r
        self.true_flux = self.area
        self.mask_method = 'interpolation'
