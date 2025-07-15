# Licensed under a 3-clause BSD style license - see LICENSE.rst
# cython: language_level=3
"""
Faster evaluation of ellipses from model.py.
"""

import cython
import numpy as np

cimport numpy as cnp

__all__ = ['build_ellipse_model_c']

cnp.import_array()

cdef extern from "math.h":

    double cos(double x)
    double sin(double x)
    double sqrt(double x)


DTYPE = np.float64
ctypedef cnp.float64_t DTYPE_t

cdef inline double get_intens_no_harmonics(double intens0, double phi, double a3, double b3, double a4, double b4):
    return intens0

cdef inline double get_intens_harmonics(double intens0, double phi, double a3, double b3, double a4, double b4):
    return (
        intens0
        + a3 * sin(3.0 * phi)
        + b3 * cos(3.0 * phi)
        + a4 * sin(4.0 * phi)
        + b4 * cos(4.0 * phi)
    )


def build_ellipse_model_c(
    unsigned int n_rows,
    unsigned int n_cols,
    cnp.ndarray[DTYPE_t, ndim=1] finely_spaced_sma,
    cnp.ndarray[DTYPE_t, ndim=1] intens_array,
    cnp.ndarray[DTYPE_t, ndim=1] eps_array,
    cnp.ndarray[DTYPE_t, ndim=1] pa_array,
    cnp.ndarray[DTYPE_t, ndim=1] x0_array,
    cnp.ndarray[DTYPE_t, ndim=1] y0_array,
    cnp.ndarray[DTYPE_t, ndim=1] a3_array = None,
    cnp.ndarray[DTYPE_t, ndim=1] b3_array = None,
    cnp.ndarray[DTYPE_t, ndim=1] a4_array = None,
    cnp.ndarray[DTYPE_t, ndim=1] b4_array = None,
    double phi_min = 0.,
    double phi_max = 2.0*np.pi,
):
    cdef cython.Py_ssize_t len_sma
    len_sma = len(finely_spaced_sma)
    for array in (intens_array, eps_array, pa_array, x0_array, y0_array):
        if len(array) != len_sma:
            raise ValueError(f"All input arrays must be same length={len_sma}")
    harmonic_arrays = (a3_array, b3_array, a4_array, b4_array)
    harmonics_is_none = [array is None for array in harmonic_arrays]

    cdef double a3, b3, a4, b4

    if all(harmonics_is_none):
        intens_func = get_intens_no_harmonics
        a3 = 0
        b3 = 0
        a4 = 0
        b4 = 0
        do_harmonics = False
    else:
        if any(harmonics_is_none):
            raise ValueError("Must supply all harmonic arrays if any is not None")
        for array in harmonic_arrays:
            if len(array) != len_sma:
                raise ValueError(f"All input arrays must be same length={len_sma}")
        intens_func = get_intens_harmonics
        do_harmonics = True

    cdef double phi, fx, fy, x, y
    cdef double one_m_fx, one_m_fy, one_m_fx_t_one_m_fy, one_m_fy_t_fx, one_m_fx_t_fy, fy_t_fx
    cdef double r, sma, q, pa, x0, y0, intens0, intens
    cdef int i, j, i_max, j_max
    cdef cython.Py_ssize_t index

    # Define output array
    cdef double[:, :] result = np.zeros([n_rows, n_cols], dtype=DTYPE)
    cdef double[:, :] weight = np.zeros([n_rows, n_cols], dtype=DTYPE)

    i_max = n_cols - 1
    j_max = n_rows - 1

    for index in range(1, len_sma):
        with cython.boundscheck(False):
            sma = finely_spaced_sma[index]
            q = 1.0 - eps_array[index]
            pa = pa_array[index]
            x0 = x0_array[index]
            y0 = y0_array[index]
            intens0 = intens_array[index]
        phi = phi_min
        r = sma
        if do_harmonics:
            with cython.boundscheck(False):
                a3 = a3_array[index]
                b3 = b3_array[index]
                a4 = a4_array[index]
                b4 = b4_array[index]

        while phi <= phi_max:
            # get image coordinates of (r, phi) pixel
            x = r * cos(phi + pa) + x0
            y = r * sin(phi + pa) + y0
            i = int(x)
            j = int(y)

            if (0 < i < i_max) and (0 < j < j_max):
                # get fractional deviations relative to target array
                fx = x - float(i)
                fy = y - float(j)

                intens = intens_func(intens0, phi, a3, b3, a4, b4)

                one_m_fx = (1.0 - fx)
                one_m_fy = (1.0 - fy)
                one_m_fx_t_one_m_fy = one_m_fx*one_m_fy
                one_m_fy_t_fx = one_m_fy*fx
                one_m_fx_t_fy = one_m_fx*fy
                fy_t_fx = fy*fx

                with cython.boundscheck(False):
                    # add up the isophote contribution to the overlapping pixels
                    result[j, i] += intens*one_m_fx_t_one_m_fy
                    result[j, i + 1] += intens*one_m_fy_t_fx
                    result[j + 1, i] += intens*one_m_fx_t_fy
                    result[j + 1, i + 1] += intens*fy_t_fx

                    # add up the fractional area contribution to the
                    # overlapping pixels
                    weight[j, i] += one_m_fx_t_one_m_fy
                    weight[j, i + 1] += one_m_fy_t_fx
                    weight[j + 1, i] += one_m_fx_t_fy
                    weight[j + 1, i + 1] += fy_t_fx

                # step towards next pixel on ellipse
                phi = max((phi + 0.75 / r), phi_min)
                r = sma * q / sqrt((q * cos(phi))**2 + sin(phi)**2)
                # max(r, 0.5) could return nan - this is safer
                if not (r >= 0.5):
                    r = 0.5
            # if outside image boundaries, ignore.
            else:
                break

    return np.asarray(result), np.asarray(weight)
