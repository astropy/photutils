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
    fluxtable = aperture_photometry(data, (-60, 60), ((aperture,) + radius))
    assert np.isnan(fluxtable['aperture_sum'])   # aperture is fully outside array


@pytest.mark.parametrize(('aperture_cl', 'aperture', 'radius'),
                         zip(APERTURE_CL, APERTURES, ((3.,), (3., 5.), (3., 5., 1.),
                                                      (3., 5., 4., 1.))))
def test_inside_array_simple(aperture_cl, aperture, radius):
    data = np.ones((40, 40), dtype=np.float)
    table1 = aperture_photometry(data, (20., 20.), ((aperture,) + radius),
                                 method='center', subpixels=10)
    table2 = aperture_photometry(data, (20., 20.), ((aperture,) + radius),
                                 method='subpixel', subpixels=10)
    table3 = aperture_photometry(data, (20., 20.), ((aperture,) + radius),
                                 method='exact', subpixels=10)
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

        fluxtable = aperture_photometry(self.data, self.position,
                                        self.aperture,
                                        mask=mask, error=error)
        assert_allclose(fluxtable['aperture_sum'], self.true_flux)

        true_error = error * np.sqrt(self.area)
        assert_allclose(fluxtable['aperture_sum_err'], true_error)

    def test_array_error_no_gain(self):

        # Array error, no gain.
        error = np.ones(self.data.shape, dtype=np.float)
        if not hasattr(self, 'mask'):
            mask = None
        else:
            mask = self.mask

        fluxtable = aperture_photometry(self.data, self.position,
                                        self.aperture,
                                        mask=mask, error=error)
        assert_allclose(fluxtable['aperture_sum'], self.true_flux)

        if hasattr(self, 'true_variance'):
            true_error = np.sqrt(self.true_variance - self.true_flux)
        else:
            true_error = np.sqrt(self.area)

        assert_allclose(fluxtable['aperture_sum_err'], true_error)


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
