# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Validation tests for the ``method='exact'`` mask mode of pixel
apertures.

When the data array is all ones, the photometric sum returned by an
aperture in exact mode must equal the analytic geometric area of the
aperture shape (up to floating-point error). These tests verify this
invariant for circular, elliptical, and rectangular apertures (including
their annulus counterparts), across a variety of shapes, rotation
angles, and sub-pixel positions.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from photutils.aperture import (CircularAnnulus, CircularAperture,
                                EllipticalAnnulus, EllipticalAperture,
                                RectangularAnnulus, RectangularAperture,
                                aperture_photometry)


def _phot_sum(aperture, data=None):
    if data is None:
        shape = (101, 101)
        data = np.ones(shape, dtype=float)
    table = aperture_photometry(data, aperture, method='exact')
    return float(table['aperture_sum'][0])


@pytest.mark.parametrize('center', [(50.0, 50.0), (50.3, 50.7),
                                    (49.5, 50.5), (12.123, 87.987)])
@pytest.mark.parametrize('r', [1.0, 3.5, 7.25])
def test_circular_aperture_exact_area(center, r):
    aper = CircularAperture(center, r=r)
    assert_allclose(_phot_sum(aper), aper.area, rtol=1e-6, atol=1e-10)


@pytest.mark.parametrize('center', [(50.0, 50.0), (50.4, 50.6)])
@pytest.mark.parametrize(('r_in', 'r_out'), [(1.0, 3.0), (2.5, 4.75),
                                             (5.0, 8.0)])
def test_circular_annulus_exact_area(center, r_in, r_out):
    aper = CircularAnnulus(center, r_in=r_in, r_out=r_out)
    assert_allclose(_phot_sum(aper), aper.area, rtol=1e-6, atol=1e-10)


@pytest.mark.parametrize('center', [(50.0, 50.0), (50.3, 49.8)])
@pytest.mark.parametrize(('a', 'b'), [(5.0, 3.0), (3.5, 1.25),
                                      (8.0, 8.0)])
@pytest.mark.parametrize('theta_deg', [0.0, 17.5, 45.0, 90.0, 137.0])
def test_elliptical_aperture_exact_area(center, a, b, theta_deg):
    aper = EllipticalAperture(center, a=a, b=b,
                              theta=np.deg2rad(theta_deg))
    assert_allclose(_phot_sum(aper), aper.area, rtol=1e-6, atol=1e-10)


@pytest.mark.parametrize(('a_in', 'a_out', 'b_out'),
                         [(2.0, 5.0, 3.0), (1.0, 4.0, 4.0)])
@pytest.mark.parametrize('theta_deg', [0.0, 30.0, 65.0])
def test_elliptical_annulus_exact_area(a_in, a_out, b_out, theta_deg):
    aper = EllipticalAnnulus((50.0, 50.0), a_in=a_in, a_out=a_out,
                             b_out=b_out, theta=np.deg2rad(theta_deg))
    assert_allclose(_phot_sum(aper), aper.area, rtol=1e-6, atol=1e-10)


@pytest.mark.parametrize('center', [(50.0, 50.0), (50.5, 50.5),
                                    (50.123, 49.876)])
@pytest.mark.parametrize(('w', 'h'), [(4.0, 2.0), (5.5, 5.5),
                                      (10.0, 1.0)])
@pytest.mark.parametrize('theta_deg', [0.0, 22.5, 45.0, 90.0, 117.0])
def test_rectangular_aperture_exact_area(center, w, h, theta_deg):
    aper = RectangularAperture(center, w=w, h=h,
                               theta=np.deg2rad(theta_deg))
    assert_allclose(_phot_sum(aper), aper.area, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize(('w_in', 'w_out', 'h_out'),
                         [(2.0, 6.0, 4.0), (1.5, 5.5, 5.5)])
@pytest.mark.parametrize('theta_deg', [0.0, 33.0, 90.0])
def test_rectangular_annulus_exact_area(w_in, w_out, h_out, theta_deg):
    aper = RectangularAnnulus((50.0, 50.0), w_in=w_in, w_out=w_out,
                              h_out=h_out, theta=np.deg2rad(theta_deg))
    assert_allclose(_phot_sum(aper), aper.area, rtol=1e-12, atol=1e-12)
