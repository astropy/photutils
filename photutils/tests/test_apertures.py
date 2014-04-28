import random

import numpy as np

from numpy.testing import assert_allclose

from ..aperture import CircularAperture, \
                       CircularAnnulus, \
                       EllipticalAperture, \
                       EllipticalAnnulus, \
                       RectangularAperture


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
    for _ in range(NITER):
        r = random.uniform(0., 10.)
        ap = CircularAperture(r)
        xmin, xmax, ymin, ymax, nx, ny, area = sample_grid(r)
        frac = ap.encloses(xmin, xmax, ymin, ymax, nx, ny, method='exact')
        assert_allclose(np.sum(frac) * area, ap.area(), rtol=TOL)


def test_accuracy_circular_annulus_exact():
    random.seed('test_accuracy_circular_annulus_exact')
    for _ in range(NITER):
        r1 = random.uniform(0., 10.)
        r2 = random.uniform(r1, 10.)
        ap = CircularAnnulus(r1, r2)
        xmin, xmax, ymin, ymax, nx, ny, area = sample_grid(r2)
        frac = ap.encloses(xmin, xmax, ymin, ymax, nx, ny, method='exact')
        assert_allclose(np.sum(frac) * area, ap.area(), rtol=TOL)


def test_accuracy_elliptical_exact():
    random.seed('test_accuracy_elliptical_exact')
    for _ in range(NITER):
        a = random.uniform(0., 10.)
        b = random.uniform(0., a)
        theta = random.uniform(0., 2. * np.pi)
        ap = EllipticalAperture(a, b, theta)
        xmin, xmax, ymin, ymax, nx, ny, area = sample_grid(a)
        frac = ap.encloses(xmin, xmax, ymin, ymax, nx, ny, method='exact')
        assert_allclose(np.sum(frac) * area, ap.area(), rtol=TOL)


def test_accuracy_elliptical_annulus_exact():
    random.seed('test_accuracy_elliptical_annulus_exact')
    for _ in range(NITER):
        a_in = random.uniform(0., 10.)
        a_out = random.uniform(a_in, 10.)
        b_out = random.uniform(0., a_out)
        theta = random.uniform(0., 2. * np.pi)
        ap = EllipticalAnnulus(a_in, a_out, b_out, theta)
        xmin, xmax, ymin, ymax, nx, ny, area = sample_grid(a_out)
        frac = ap.encloses(xmin, xmax, ymin, ymax, nx, ny, method='exact')
        assert_allclose(np.sum(frac) * area, ap.area(), rtol=TOL)

def test_rectangular():
    #test a few specific cases, mainly to ensure the sign is right
    ap = RectangularAperture(w=.2, h=1, theta=0)
    vertical = ap.encloses(-1, 1, -1, 1, 100, 100, method='center')

    ap2 = RectangularAperture(w=.2, h=1, theta=np.pi/4)  # 45 deg
    tilted = ap2.encloses(-1, 1, -1, 1, 100, 100, method='center')
    tiltedss = ap2.encloses(-1, 1, -1, 1, 100, 100, method='subpixel')

    #make sure subpixel is doing something
    assert np.any(tilted != tiltedss)

    #check the tilted and vertical versions behave correctly
    assert vertical[70,50] > 0
    assert vertical[60,35] < 1
    assert tilted[70,50] < 1
    assert tilted[60,35] > 0
