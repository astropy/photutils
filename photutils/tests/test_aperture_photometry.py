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


APERTURES = [CircularAperture,
             CircularAnnulus,
             EllipticalAperture,
             EllipticalAnnulus]


@pytest.mark.parametrize(('aperture', 'radius'),
                         zip(APERTURES, ((3.,), (3., 5.), (3., 5., 1.),
                                         (3., 5., 4., 1.))))
def test_outside_array(aperture, radius):
    data = np.ones((10, 10), dtype=np.float)
    table = aperture_photometry(data, aperture((-60, 60), *radius))
    assert np.isnan(table['aperture_sum'])   # aperture is fully outside array


@pytest.mark.parametrize(('aperture', 'radius'),
                         zip(APERTURES, ((3.,), (3., 5.), (3., 5., 1.),
                                         (3., 5., 4., 1.))))
def test_inside_array_simple(aperture, radius):
    data = np.ones((40, 40), dtype=np.float)
    table1 = aperture_photometry(data, aperture((20., 20.), *radius),
                                 method='center', subpixels=10)
    table2 = aperture_photometry(data, aperture((20., 20.), *radius),
                                 method='subpixel', subpixels=10)
    table3 = aperture_photometry(data, aperture((20., 20.), *radius),
                                 method='exact', subpixels=10)
    true_flux = aperture((20., 20.), *radius).area()

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

        table = aperture_photometry(self.data, self.aperture,
                                    mask=mask, error=error)
        assert_allclose(table['aperture_sum'], self.true_flux)

        true_error = error * np.sqrt(self.area)
        assert_allclose(table['aperture_sum_err'], true_error)

    def test_scalar_error_scalar_gain(self):

        # Scalar error, scalar gain.
        error = 1.
        gain = 1.
        if not hasattr(self, 'mask'):
            mask = None
        else:
            mask = self.mask

        table = aperture_photometry(self.data, self.aperture,
                                    mask=mask, error=error, gain=gain)
        assert_allclose(table['aperture_sum'], self.true_flux)

        true_error = np.sqrt(error ** 2 * self.area + table['aperture_sum'])
        assert_allclose(table['aperture_sum_err'], true_error)

    def test_scalar_error_array_gain(self):

        # Scalar error, Array gain.
        error = 1.
        gain = np.ones(self.data.shape, dtype=np.float)
        if not hasattr(self, 'mask'):
            mask = None
        else:
            mask = self.mask
        table = aperture_photometry(self.data, self.aperture,
                                    mask=mask, error=error, gain=gain)
        assert_allclose(table['aperture_sum'], self.true_flux)

        if hasattr(self, 'true_variance'):
            true_error = np.sqrt(error ** 2 * self.true_variance -
                                 (error ** 2 - 1) * table['aperture_sum'])
        else:
            true_error = np.sqrt((error ** 2 * self.area) + table['aperture_sum'])
        assert_allclose(table['aperture_sum_err'], true_error)

    def test_array_error_no_gain(self):

        # Array error, no gain.
        error = np.ones(self.data.shape, dtype=np.float)
        if not hasattr(self, 'mask'):
            mask = None
        else:
            mask = self.mask

        table = aperture_photometry(self.data, self.aperture,
                                    mask=mask, error=error)
        assert_allclose(table['aperture_sum'], self.true_flux)

        if hasattr(self, 'true_variance'):
            true_error = np.sqrt(self.true_variance - self.true_flux)
        else:
            true_error = np.sqrt(self.area)

        assert_allclose(table['aperture_sum_err'], true_error)

    def test_array_error_scalar_gain(self):

        # Array error, scalar gain.
        error = np.ones(self.data.shape, dtype=np.float)
        gain = 1.
        if not hasattr(self, 'mask'):
            mask = None
        else:
            mask = self.mask
        table = aperture_photometry(self.data, self.aperture,
                                    mask=mask, error=error, gain=gain)
        assert_allclose(table['aperture_sum'], self.true_flux)

        if hasattr(self, 'true_variance'):
            true_error = np.sqrt(self.true_variance)
        else:
            true_error = np.sqrt((self.area) + table['aperture_sum'])

        assert_allclose(table['aperture_sum_err'], true_error)

    def test_array_error_array_gain(self):

        # Array error, Array gain.
        error = np.ones(self.data.shape, dtype=np.float)
        gain = np.ones(self.data.shape, dtype=np.float)
        if not hasattr(self, 'mask'):
            mask = None
        else:
            mask = self.mask

        table = aperture_photometry(self.data, self.aperture,
                                    mask=mask, error=error, gain=gain)
        assert_allclose(table['aperture_sum'], self.true_flux)

        if hasattr(self, 'true_variance'):
            true_error = np.sqrt(self.true_variance)
        else:
            true_error = np.sqrt((self.area) + table['aperture_sum'])

        assert_allclose(table['aperture_sum_err'], true_error)


class TestCircular(BaseTestAperturePhotometry):

    def setup_class(self):
        self.data = np.ones((40, 40), dtype=np.float)
        self.xc = 20.
        self.yc = 20.
        r = 10.
        self.aperture = CircularAperture((self.xc, self.yc), r)
        self.area = np.pi * r * r
        self.true_flux = self.area


class TestCircularAnnulus(BaseTestAperturePhotometry):

    def setup_class(self):
        self.data = np.ones((40, 40), dtype=np.float)
        self.xc = 20.
        self.yc = 20.
        r_in = 8.
        r_out = 10.
        self.aperture = CircularAnnulus((self.xc, self.yc), r_in, r_out)
        self.area = np.pi * (r_out * r_out - r_in * r_in)
        self.true_flux = self.area


class TestElliptical(BaseTestAperturePhotometry):

    def setup_class(self):
        self.data = np.ones((40, 40), dtype=np.float)
        self.xc = 20.
        self.yc = 20.
        a = 10.
        b = 5.
        theta = -np.pi / 4.
        self.aperture = EllipticalAperture((self.xc, self.yc), a, b, theta)
        self.area = np.pi * a * b
        self.true_flux = self.area


class TestEllipticalAnnulus(BaseTestAperturePhotometry):

    def setup_class(self):
        self.data = np.ones((40, 40), dtype=np.float)
        self.xc = 20.
        self.yc = 20.
        a_in = 5.
        a_out = 8.
        b_out = 6.
        theta = -np.pi / 4.
        self.aperture = EllipticalAnnulus((self.xc, self.yc),
                                          a_in, a_out, b_out, theta)
        self.area = np.pi * (a_out * b_out) - np.pi * (a_in * b_out * a_in / a_out)
        self.true_flux = self.area


def test_rectangular_aperture():
    data = np.ones((40, 40), dtype=np.float)
    x = 20.
    y = 20.
    aperture = RectangularAperture((x, y), 1., 2., np.pi / 4)
    table1 = aperture_photometry(data, aperture, method='center')
    table2 = aperture_photometry(data, aperture, method='subpixel', subpixels=8)

    with pytest.raises(NotImplementedError):
        aperture_photometry(data, aperture, method='exact')

    true_flux = aperture.area()

    assert table1['aperture_sum'] < true_flux
    assert_allclose(table2['aperture_sum'], true_flux, atol=0.1)


class TestMaskedCircular(BaseTestAperturePhotometry):

    def setup_class(self):
        self.data = np.ones((40, 40), dtype=np.float)
        self.mask = np.zeros((40, 40), dtype=bool)
        self.mask[20, 20] = True
        self.xc = 20.
        self.yc = 20.
        r = 10.
        self.aperture = CircularAperture((self.xc, self.yc), r)
        self.area = np.pi * r * r
        self.true_flux = self.area - 1
        self.true_variance = self.area - 1 + self.true_flux


class TestMaskedMirrored(BaseTestAperturePhotometry):

    def setup_class(self):
        self.data = np.ones((40, 40), dtype=np.float)
        self.mask = np.zeros((40, 40), dtype=bool)
        self.mask[22, 20] = True
        self.xc = 20.
        self.yc = 20.
        r = 10.
        self.aperture = CircularAperture((self.xc, self.yc), r)
        self.area = np.pi * r * r
        self.true_flux = self.area
