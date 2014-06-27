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
    assert np.isnan(table['aperture_sum'])   # aperture is fully outside array


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
