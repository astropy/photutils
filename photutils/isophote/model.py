# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
This module profiles tools for building a model elliptical galaxy image
from a list of isophotes.
"""


import numpy as np
import ctypes as ct
import os

__all__ = ['build_ellipse_model']


def build_ellipse_model(shape, isolist, nthreads=1, fill=0., high_harmonics=False):
    """
    Build a model elliptical galaxy image from a list of isophotes.
    For each ellipse in the input isophote list the algorithm fills the
    output image array with the corresponding isophotal intensity.
    Pixels in the output array are in general only partially covered by
    the isophote "pixel".  The algorithm takes care of this partial
    pixel coverage by keeping track of how much intensity was added to
    each pixel by storing the partial area information in an auxiliary
    array.  The information in this array is then used to normalize the
    pixel intensities.
    Parameters
    ----------
    shape : 2-tuple
        The (ny, nx) shape of the array used to generate the input
        ``isolist``.
    isolist : `~photutils.isophote.IsophoteList` instance
        The isophote list created by the `~photutils.isophote.Ellipse`
        class.
    nthreads: float, optiomal
    	Number of threads to perform work. Default is 1 (serial code).
    fill : float, optional
        The constant value to fill empty pixels. If an output pixel has
        no contribution from any isophote, it will be assigned this
        value.  The default is 0.
    high_harmonics : bool, optional
        Whether to add the higher-order harmonics (i.e., ``a3``, ``b3``,
        ``a4``, and ``b4``; see `~photutils.isophote.Isophote` for
        details) to the result.
    Returns
    -------
    result : 2D `~numpy.ndarray`
        The image with the model galaxy.
    """
    from scipy.interpolate import LSQUnivariateSpline

    # the target grid is spaced in 0.1 pixel intervals so as
    # to ensure no gaps will result on the output array.
    finely_spaced_sma = np.arange(isolist[0].sma, isolist[-1].sma, 0.1)

    # interpolate ellipse parameters

    # End points must be discarded, but how many?
    # This seems to work so far
    nodes = isolist.sma[2:-2]

    intens_array = LSQUnivariateSpline(
        isolist.sma, isolist.intens, nodes)(finely_spaced_sma)
    eps_array = LSQUnivariateSpline(
        isolist.sma, isolist.eps, nodes)(finely_spaced_sma)
    pa_array = LSQUnivariateSpline(
        isolist.sma, isolist.pa, nodes)(finely_spaced_sma)
    x0_array = LSQUnivariateSpline(
        isolist.sma, isolist.x0, nodes)(finely_spaced_sma)
    y0_array = LSQUnivariateSpline(
        isolist.sma, isolist.y0, nodes)(finely_spaced_sma)
    grad_array = LSQUnivariateSpline(
        isolist.sma, isolist.grad, nodes)(finely_spaced_sma)
    a3_array = LSQUnivariateSpline(
        isolist.sma, isolist.a3, nodes)(finely_spaced_sma)
    b3_array = LSQUnivariateSpline(
        isolist.sma, isolist.b3, nodes)(finely_spaced_sma)
    a4_array = LSQUnivariateSpline(
        isolist.sma, isolist.a4, nodes)(finely_spaced_sma)
    b4_array = LSQUnivariateSpline(
        isolist.sma, isolist.b4, nodes)(finely_spaced_sma)

    # Return deviations from ellipticity to their original amplitude meaning
    a3_array = -a3_array * grad_array * finely_spaced_sma
    b3_array = -b3_array * grad_array * finely_spaced_sma
    a4_array = -a4_array * grad_array * finely_spaced_sma
    b4_array = -b4_array * grad_array * finely_spaced_sma

    # correct deviations cased by fluctuations in spline solution
    eps_array[np.where(eps_array < 0.)] = 0.
    eps_array[np.where(eps_array < 0.)] = 0.05
    
    # convert everything to C-type array (pointers)
    c_fss_array = ct.c_void_p(finely_spaced_sma.ctypes.data)
    c_intens_array = ct.c_void_p(intens_array.ctypes.data)
    c_eps_array = ct.c_void_p(eps_array.ctypes.data)
    c_pa_array = ct.c_void_p(pa_array.ctypes.data)
    c_x0_array = ct.c_void_p(x0_array.ctypes.data)
    c_y0_array = ct.c_void_p(y0_array.ctypes.data)
    c_a3_array = ct.c_void_p(a3_array.ctypes.data)
    c_b3_array = ct.c_void_p(b3_array.ctypes.data)
    c_a4_array = ct.c_void_p(a4_array.ctypes.data)
    c_b4_array = ct.c_void_p(b4_array.ctypes.data)

    # initialize result and weight arrays, also as 1D ctype array
    result = np.zeros(shape=(shape[1]*shape[0],))
    weight = np.zeros(shape=(shape[1]*shape[0],))
    c_result = ct.c_void_p(result.ctypes.data)
    c_weight = ct.c_void_p(weight.ctypes.data)
    
    # convert high_harmnics bool flag to int,
    # convert all other ints to ctype
    c_high_harm = ct.c_int(int(high_harmonics))
    c_N = ct.c_int(len(finely_spaced_sma))
    c_nrows = ct.c_int(shape[0])
    c_ncols = ct.c_int(shape[1])

    # load into C worker function (worker.so should be in same directory)
    lib = ct.cdll.LoadLibrary(os.path.dirname(os.path.abspath(__file__)) + '/worker.so')
    lib.worker.restype = None
    lib.worker(c_result, c_weight, c_nrows, c_ncols, c_N, c_high_harm,
               c_fss_array, c_intens_array, c_eps_array, c_pa_array,
               c_x0_array, c_y0_array, c_a3_array, c_b3_array,
               c_a4_array, c_b4_array, ct.c_int(nthreads))
               
     # zero weight values must be set to 1.
    weight[np.where(weight <= 0.)] = 1.

    # normalize
    result /= weight

    # fill value
    result[np.where(result == 0.)] = fill

    # reshape
    result = result.reshape(shape[0], shape[1])
    
    return result    
