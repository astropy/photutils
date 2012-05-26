# Licensed under a 3-clause BSD style license - see LICENSE.rst

# The functions defined here allow one to determine the exact area of
# overlap of a rectangle and a circle (written by Thomas Robitaille).

from __future__ import division
import numpy as np
cimport numpy as np

cdef extern from "math.h":

    double asin(double x)
    double sin(double x)
    double sqrt(double x)

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

cimport cython


def distance(double x1, double y1, double x2, double y2):
    return sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def area_arc(double x1, double y1, double x2, double y2, double R):
    '''
    Area of a circle arc with radius R between points (x1, y1) and (x2, y2)

    References
    ----------
    http://mathworld.wolfram.com/CircularSegment.html
    '''
    cdef double a, theta
    a = distance(x1, y1, x2, y2)
    theta = 2. * asin(0.5 * a / R)
    return 0.5 * R * R * (theta - sin(theta))


def area_triangle(double x1, double y1, double x2, double y2, double x3, double y3):
    '''
    Area of a triangle defined by three vertices
    '''
    return 0.5 * abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))


def circular_overlap(np.ndarray[DTYPE_t, ndim=1] xmin,
                     np.ndarray[DTYPE_t, ndim=1] xmax,
                     np.ndarray[DTYPE_t, ndim=1] ymin,
                     np.ndarray[DTYPE_t, ndim=1] ymax,
                     double R):
    '''
    Given a grid with walls set by x, y, find the area of overlap in each
    '''
    cdef int n = xmin.shape[0]
    cdef np.ndarray[DTYPE_t, ndim=1] frac = np.zeros([n], dtype=DTYPE)
    cdef unsigned int i, j

    for i in range(n):
        # We don't technically need to check this here, but it's faster to pre-check
        if xmin[i] > R or xmax[i] < - R or ymin[i] > R or ymax[i] < -R:
            pass
        else:
            frac[i] = circular_overlap_single(xmin[i], ymin[i], xmax[i], ymax[i], R)

    return frac


def circular_overlap_single(double xmin, double ymin, double xmax, double ymax, double r):
    '''
    Area of overlap of a rectangle and a circle
    '''
    if 0. <= xmin:
        if 0. <= ymin:
            return circular_overlap_core(xmin, ymin, xmax, ymax, r)
        elif 0. >= ymax:
            return circular_overlap_core(-ymax, xmin, -ymin, xmax, r)
        else:
            return circular_overlap_single(xmin, ymin, xmax, 0., r) \
                 + circular_overlap_single(xmin, 0., xmax, ymax, r)
    elif 0. >= xmax:
        if 0. <= ymin:
            return circular_overlap_core(-xmax, ymin, -xmin, ymax, r)
        elif 0. >= ymax:
            return circular_overlap_core(-xmax, -ymax, -xmin, -ymin, r)
        else:
            return circular_overlap_single(xmin, ymin, xmax, 0., r) \
                 + circular_overlap_single(xmin, 0., xmax, ymax, r)
    else:
        if 0. <= ymin:
            return circular_overlap_single(xmin, ymin, 0., ymax, r) \
                 + circular_overlap_single(0., ymin, xmax, ymax, r)
        if 0. >= ymax:
            return circular_overlap_single(xmin, ymin, 0., ymax, r) \
                 + circular_overlap_single(0., ymin, xmax, ymax, r)
        else:
            return circular_overlap_single(xmin, ymin, 0., 0., r) \
                 + circular_overlap_single(0., ymin, xmax, 0., r) \
                 + circular_overlap_single(xmin, 0., 0., ymax, r) \
                 + circular_overlap_single(0., 0., xmax, ymax, r)


def circular_overlap_core(double xmin, double ymin, double xmax, double ymax, double R):
    '''
    Assumes that the center of the circle is <= xmin, ymin (can always modify input to conform to this)
    '''

    cdef double area, d1, d2, x1, x2, y1, y2

    if xmin * xmin + ymin * ymin > R * R:
        area = 0.
    elif xmax * xmax + ymax * ymax < R * R:
        area = (xmax - xmin) * (ymax - ymin)
    else:
        area = 0.
        d1 = sqrt(xmax * xmax + ymin * ymin)
        d2 = sqrt(xmin * xmin + ymax * ymax)
        if d1 < R and d2 < R:
            x1, y1 = sqrt(R * R - ymax * ymax), ymax
            x2, y2 = xmax, sqrt(R * R - xmax * xmax)
            area = (xmax - xmin) * (ymax - ymin) - area_triangle(x1, y1, x2, y2, xmax, ymax) + area_arc(x1, y1, x2, y2, R)
        elif d1 < R:
            x1, y1 = xmin, sqrt(R * R - xmin * xmin)
            x2, y2 = xmax, sqrt(R * R - xmax * xmax)
            area = area_arc(x1, y1, x2, y2, R) + area_triangle(x1, y1, x1, ymin, xmax, ymin) + area_triangle(x1, y1, x2, ymin, x2, y2)
        elif d2 < R:
            x1, y1 = sqrt(R * R - ymin * ymin), ymin
            x2, y2 = sqrt(R * R - ymax * ymax), ymax
            area = area_arc(x1, y1, x2, y2, R) + area_triangle(x1, y1, xmin, y1, xmin, ymax) + area_triangle(x1, y1, xmin, y2, x2, y2)
        else:
            x1, y1 = sqrt(R * R - ymin * ymin), ymin
            x2, y2 = xmin, sqrt(R * R - xmin * xmin)
            area = area_arc(x1, y1, x2, y2, R) + area_triangle(x1, y1, x2, y2, xmin, ymin)

    return area
