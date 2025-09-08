# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Define tools for building a model elliptical galaxy image from a list of
isophotes.
"""

import numpy as np
from scipy.interpolate import LSQUnivariateSpline

from .ellipse_model import build_ellipse_model_c

__all__ = ['build_ellipse_model']


def build_ellipse_model(
    shape,
    isolist,
    fill: float = 0.0,
    high_harmonics=False,
    sma_interval: float = 0.1,
):
    """
    Build a model elliptical galaxy image from a list of isophotes.

    For each ellipse in the input isophote list the algorithm fills
    the output image array with the corresponding isophotal intensity.
    Pixels in the output array are in general only partially covered by
    the isophote "pixel". The algorithm takes care of this partial pixel
    coverage by keeping track of how much intensity was added to each
    pixel by storing the partial area information in an auxiliary array.
    The information in this array is then used to normalize the pixel
    intensities.

    Parameters
    ----------
    shape : 2-tuple
        The (ny, nx) shape of the array used to generate the input
        ``isolist``.

    isolist : `~photutils.isophote.IsophoteList` instance
        The isophote list created by the `~photutils.isophote.Ellipse`
        class.

    fill : float, optional
        The constant value to fill empty pixels. If an output pixel has
        no contribution from any isophote, it will be assigned this
        value. The default is 0.

    high_harmonics : bool, optional
        Whether to add the higher-order harmonics (i.e., ``a3``, ``b3``,
        ``a4``, and ``b4``; see `~photutils.isophote.Isophote` for
        details) to the result.

    sma_interval : optional, float
        The interval between node values of the semi-major axis, which is used
        to spline interpolate values of other shape parameters.

    Returns
    -------
    result : 2D `~numpy.ndarray`
        The image with the model galaxy.
    """
    if len(isolist) == 0:
        msg = 'isolist must not be empty'
        raise ValueError(msg)

    # the target grid is spaced in 0.1 pixel intervals so as
    # to ensure no gaps will result on the output array.
    finely_spaced_sma = np.arange(
        isolist[0].sma, isolist[-1].sma, sma_interval,
    )

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
    if high_harmonics:
        a3_array = LSQUnivariateSpline(
            isolist.sma, isolist.a3, nodes)(finely_spaced_sma)
        b3_array = LSQUnivariateSpline(
            isolist.sma, isolist.b3, nodes)(finely_spaced_sma)
        a4_array = LSQUnivariateSpline(
            isolist.sma, isolist.a4, nodes)(finely_spaced_sma)
        b4_array = LSQUnivariateSpline(
            isolist.sma, isolist.b4, nodes)(finely_spaced_sma)

        grad_sma = -grad_array * finely_spaced_sma

        # Return deviations from ellipticity to their original amplitude
        # meaning
        kwargs_harm = {
            'a3_array': a3_array * grad_sma,
            'b3_array': b3_array * grad_sma,
            'a4_array': a4_array * grad_sma,
            'b4_array': b4_array * grad_sma,
        }
    else:
        kwargs_harm = {}

    # correct deviations caused by fluctuations in spline solution
    eps_array[np.where(eps_array < 0.0)] = 0.0

    # for each interpolated isophote, generate intensity values on the
    # output image array
    result, weight = build_ellipse_model_c(
        shape[0],
        shape[1],
        finely_spaced_sma,
        intens_array,
        eps_array,
        pa_array,
        x0_array,
        y0_array,
        **kwargs_harm,
    )

    # zero weight values must be set to 1.0
    weight[np.where(weight <= 0.0)] = 1.0

    # normalize
    result /= weight

    # fill value
    result[np.where(result == 0.0)] = fill

    return result
