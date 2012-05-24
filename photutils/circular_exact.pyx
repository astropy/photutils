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

DTYPE = np.float
ctypedef np.float_t DTYPE_t

cimport cython


def distance(float x1, float y1, float x2, float y2):
    return sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def distance_sq(float x1, float y1, float x2, float y2):
    return (x2 - x1) ** 2 + (y2 - y1) ** 2


def area_arc(float x1, float y1, float x2, float y2, float R):
    '''
    Area of a circle arc with radius R between points (x1, y1) and (x2, y2)

    References
    ----------
    http://mathworld.wolfram.com/CircularSegment.html
    '''
    cdef float a, theta
    a = distance(x1, y1, x2, y2)
    theta = 2. * asin(0.5 * a / R)
    return 0.5 * R * R * (theta - sin(theta))


def area_triangle(float x1, float y1, float x2, float y2, float x3, float y3):
    '''
    Area of a triangle defined by three vertices
    '''
    return 0.5 * abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))


def circular_overlap_grid(np.ndarray[DTYPE_t, ndim=1] x, np.ndarray[DTYPE_t, ndim=1] y, float x0, float y0, float R):
    '''
    Given a grid with walls set by x, y, find the area of overlap in each
    '''
    cdef int nx = x.shape[0]
    cdef int ny = y.shape[0]
    cdef np.ndarray[DTYPE_t, ndim=2] frac = np.zeros([ny - 1, nx - 1], dtype=DTYPE)
    cdef unsigned int i, j

    for i in range(nx - 1):
        # We don't technically need to check this here, but it's faster to pre-check
        if x[i] > x0 + R or x[i + 1] < x0 - R:
            pass
        else:
            for j in range(ny - 1):
                if y[j] > y0 + R or y[j + 1] < y0 - R:
                    pass
                else:
                    frac[j, i] = circular_overlap_single(x[i], y[j], x[i + 1], y[j + 1], x0, y0, R)

    return frac


def circular_overlap_single(float xmin, float ymin, float xmax, float ymax, float x0, float y0, float r):
    '''
    Area of overlap of a rectangle and a circle
    '''
    if x0 <= xmin:
        if y0 <= ymin:
            return circular_overlap_core(xmin, ymin, xmax, ymax, x0, y0, r)
        elif y0 >= ymax:
            return circular_overlap_core(-ymax, xmin, -ymin, xmax, -y0, x0, r)
        else:
            return circular_overlap_single(xmin, ymin, xmax, y0, x0, y0, r) \
                 + circular_overlap_single(xmin, y0, xmax, ymax, x0, y0, r)
    elif x0 >= xmax:
        if y0 <= ymin:
            return circular_overlap_core(-xmax, ymin, -xmin, ymax, -x0, y0, r)
        elif y0 >= ymax:
            return circular_overlap_core(-xmax, -ymax, -xmin, -ymin, -x0, -y0, r)
        else:
            return circular_overlap_single(xmin, ymin, xmax, y0, x0, y0, r) \
                 + circular_overlap_single(xmin, y0, xmax, ymax, x0, y0, r)
    else:
        if y0 <= ymin:
            return circular_overlap_single(xmin, ymin, x0, ymax, x0, y0, r) \
                 + circular_overlap_single(x0, ymin, xmax, ymax, x0, y0, r)
        if y0 >= ymax:
            return circular_overlap_single(xmin, ymin, x0, ymax, x0, y0, r) \
                 + circular_overlap_single(x0, ymin, xmax, ymax, x0, y0, r)
        else:
            return circular_overlap_single(xmin, ymin, x0, y0, x0, y0, r) \
                 + circular_overlap_single(x0, ymin, xmax, y0, x0, y0, r) \
                 + circular_overlap_single(xmin, y0, x0, ymax, x0, y0, r) \
                 + circular_overlap_single(x0, y0, xmax, ymax, x0, y0, r)


def circular_overlap_core(float xmin, float ymin, float xmax, float ymax, float x0, float y0, float R):
    '''
    Assumes that the center of the circle is <= xmin, ymin (can always modify input to conform to this)
    '''

    cdef float area, d1, d2, x1, x2, y1, y2

    if distance(xmin, ymin, x0, y0) > R:
        area = 0.
    elif distance(xmax, ymax, x0, y0) < R:
        area = (xmax - xmin) * (ymax - ymin)
    else:
        area = 0.
        d1 = distance(xmax, ymin, x0, y0)
        d2 = distance(xmin, ymax, x0, y0)
        if d1 < R and d2 < R:
            x1, y1 = x0 + sqrt(R * R - (ymax - y0) ** 2), ymax
            x2, y2 = xmax, y0 + sqrt(R * R - (xmax - x0) ** 2)
            area = (xmax - xmin) * (ymax - ymin) - area_triangle(x1, y1, x2, y2, xmax, ymax) + area_arc(x1, y1, x2, y2, R)
        elif d1 < R:
            x1, y1 = xmin, y0 + sqrt(R * R - (xmin - x0) ** 2)
            x2, y2 = xmax, y0 + sqrt(R * R - (xmax - x0) ** 2)
            area = area_arc(x1, y1, x2, y2, R) + area_triangle(x1, y1, x1, ymin, xmax, ymin) + area_triangle(x1, y1, x2, ymin, x2, y2)
        elif d2 < R:
            x1, y1 = x0 + sqrt(R * R - (ymin - y0) ** 2), ymin
            x2, y2 = x0 + sqrt(R * R - (ymax - y0) ** 2), ymax
            area = area_arc(x1, y1, x2, y2, R) + area_triangle(x1, y1, xmin, y1, xmin, ymax) + area_triangle(x1, y1, xmin, y2, x2, y2)
        else:
            x1, y1 = x0 + sqrt(R * R - (ymin - y0) ** 2), ymin
            x2, y2 = xmin, y0 + sqrt(R * R - (xmin - x0) ** 2)
            area = area_arc(x1, y1, x2, y2, R) + area_triangle(x1, y1, x2, y2, xmin, ymin)

    return area
