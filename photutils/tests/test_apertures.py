import random

import numpy as np

from astropy.tests.compat import assert_allclose

from ..aperture import CircularAperture, \
                       CircularAnnulus, \
                       EllipticalAperture, \
                       EllipticalAnnulus
                               

NITER = 1000
TOL = 1.e-10


def sample_grid(r):
    xmin = random.uniform(-10., -r)
    xmax = random.uniform(+r, +10.)
    ymin = random.uniform(-10., -r)
    ymax = random.uniform(+r, +10.)
    nx = int(round(random.uniform(1, 100)))
    ny = int(round(random.uniform(1, 100)))
    area = (xmax - xmin) * (ymax - ymin) / float(nx) / float(ny)
    return xmin, xmax, ymin, ymax, nx, ny, area


def test_accuracy_circular_exact():
    random.seed('test_accuracy_circular_exact')
    for i in range(NITER):
        r = random.uniform(0., 10.)
        ap = CircularAperture(r)
        xmin, xmax, ymin, ymax, nx, ny, area = sample_grid(r)
        frac = ap.encloses(xmin, xmax, ymin, ymax, nx, ny, method='exact')
        assert_allclose(np.sum(frac) * area, ap.area(), rtol=TOL)


def test_accuracy_circular_annulus_exact():
    random.seed('test_accuracy_circular_annulus_exact')
    for i in range(NITER):
        r1 = random.uniform(0., 10.)
        r2 = random.uniform(r1, 10.)
        ap = CircularAnnulus(r1, r2)
        xmin, xmax, ymin, ymax, nx, ny, area = sample_grid(r2)
        frac = ap.encloses(xmin, xmax, ymin, ymax, nx, ny, method='exact')
        assert_allclose(np.sum(frac) * area, ap.area(), rtol=TOL)


def test_accuracy_elliptical_exact():
    random.seed('test_accuracy_elliptical_exact')
    for i in range(NITER):
        a = random.uniform(0., 10.)
        b = random.uniform(0., a)
        theta = random.uniform(0., 2. * np.pi)
        ap = EllipticalAperture(a, b, theta)
        xmin, xmax, ymin, ymax, nx, ny, area = sample_grid(a)
        frac = ap.encloses(xmin, xmax, ymin, ymax, nx, ny, method='exact')
        print a, b, theta
        print xmin, xmax, ymin, ymax, nx, ny, area
        assert_allclose(np.sum(frac) * area, ap.area(), rtol=TOL)


def test_accuracy_elliptical_annulus_exact():
    random.seed('test_accuracy_elliptical_annulus_exact')
    for i in range(NITER):
        a_in = random.uniform(0., 10.)
        a_out = random.uniform(a_in, 10.)
        b_out = random.uniform(0., a_out)
        theta = random.uniform(0., 2. * np.pi)
        ap = EllipticalAnnulus(a_in, a_out, b_out, theta)
        xmin, xmax, ymin, ymax, nx, ny, area = sample_grid(a_out)
        frac = ap.encloses(xmin, xmax, ymin, ymax, nx, ny, method='exact')
        assert_allclose(np.sum(frac) * area, ap.area(), rtol=TOL)
