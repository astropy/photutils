# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
This module profiles tools for building a model elliptical galaxy image
from a list of isophotes.
"""


import numpy as np

from .geometry import EllipseGeometry

__all__ = ['build_ellipse_model']


def build_ellipse_model(shape, isolist, fill=0., high_harmonics=False):
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

    result = np.zeros(shape=shape)
    weight = np.zeros(shape=shape)

    eps_array[np.where(eps_array < 0.)] = 0.05

    # for each interpolated isophote, generate intensity values on the
    # output image array
    # for index in range(len(finely_spaced_sma)):
    for index in range(1, len(finely_spaced_sma)):
        sma0 = finely_spaced_sma[index]
        eps = eps_array[index]
        pa = pa_array[index]
        x0 = x0_array[index]
        y0 = y0_array[index]
        geometry = EllipseGeometry(x0, y0, sma0, eps, pa)

        intens = intens_array[index]

        # scan angles. Need to go a bit beyond full circle to ensure
        # full coverage.
        r = sma0
        phi = 0.
        while phi <= 2*np.pi + geometry._phi_min:
            # we might want to add the third and fourth harmonics
            # to the basic isophotal intensity.
            harm = 0.
            if high_harmonics:
                harm = (a3_array[index] * np.sin(3.*phi) +
                        b3_array[index] * np.cos(3.*phi) +
                        a4_array[index] * np.sin(4.*phi) +
                        b4_array[index] * np.cos(4.*phi)) / 4.

            # get image coordinates of (r, phi) pixel
            x = r * np.cos(phi + pa) + x0
            y = r * np.sin(phi + pa) + y0
            i = int(x)
            j = int(y)

            if (i > 0 and i < shape[1] - 1 and j > 0 and j < shape[0] - 1):
                # get fractional deviations relative to target array
                fx = x - float(i)
                fy = y - float(j)

                # add up the isophote contribution to the overlapping pixels
                result[j, i] += (intens + harm) * (1. - fy) * (1. - fx)
                result[j, i + 1] += (intens + harm) * (1. - fy) * fx
                result[j + 1, i] += (intens + harm) * fy * (1. - fx)
                result[j + 1, i + 1] += (intens + harm) * fy * fx

                # add up the fractional area contribution to the
                # overlapping pixels
                weight[j, i] += (1. - fy) * (1. - fx)
                weight[j, i + 1] += (1. - fy) * fx
                weight[j + 1, i] += fy * (1. - fx)
                weight[j + 1, i + 1] += fy * fx

                # step towards next pixel on ellipse
                phi = max((phi + 0.75 / r), geometry._phi_min)
                r = max(geometry.radius(phi), 0.5)
            # if outside image boundaries, ignore.
            else:
                break

    # zero weight values must be set to 1.
    weight[np.where(weight <= 0.)] = 1.

    # normalize
    result /= weight

    # fill value
    result[np.where(result == 0.)] = fill

    return result
