# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tools for building a model elliptical galaxy image from a list of
isophotes.
"""

import numpy as np
from scipy.interpolate import LSQUnivariateSpline

from photutils.isophote._ellipse_model import build_ellipse_model_c
from photutils.utils._deprecation import (deprecated,
                                          deprecated_positional_kwargs)

__all__ = ['build_ellipse_model']


def _make_model_image(isolist, shape, *, fill_value, high_harmonics,
                      sma_interval):
    """
    Make a model image from an `~photutils.isophote.IsophoteList`.

    See `~photutils.isophote.IsophoteList.make_model_image` for the
    parameters.
    """
    if len(isolist) == 0:
        msg = 'isolist must not be empty'
        raise ValueError(msg)

    # A nonfinite intensity would make the interpolated intensities
    # nonfinite over a range of semimajor axis lengths.
    finite = np.isfinite(isolist.intensity)
    if not np.all(finite):
        isolist = type(isolist)([isophote for isophote, keep
                                 in zip(isolist, finite, strict=True)
                                 if keep])

    # The spline interpolation below requires at least 4 data points
    # (cubic splines) with strictly increasing sma values.
    if len(isolist) < 4:
        msg = ('isolist must contain at least 4 isophotes with a finite '
               'intensity')
        raise ValueError(msg)
    if np.any(np.diff(isolist.sma) <= 0):
        msg = ('The isophotes in isolist must be sorted by strictly '
               'increasing semimajor axis length')
        raise ValueError(msg)

    # The target grid is spaced in sma_interval intervals so as to
    # ensure no gaps will result on the output array. The last
    # isophote is appended because np.arange excludes the stop value.
    finely_spaced_sma = np.append(
        np.arange(isolist[0].sma, isolist[-1].sma, sma_interval),
        isolist[-1].sma)

    # Interpolate ellipse parameters.

    # End points must be discarded, but how many?
    # This seems to work so far.
    nodes = isolist.sma[2:-2]

    intens_array = LSQUnivariateSpline(
        isolist.sma, isolist.intensity, nodes)(finely_spaced_sma)
    eps_array = LSQUnivariateSpline(
        isolist.sma, isolist.ellipticity, nodes)(finely_spaced_sma)
    # The orientation is defined modulo pi. Fitted values near 0
    # or pi can jump by almost pi between neighboring isophotes, so
    # unwrap them to prevent the spline from sweeping through all the
    # angles in between.
    # The orientations are read in radians without a round trip
    # through degrees.
    pa = isolist._collect_as_array('_orientation_rad')
    pa_unwrapped = np.unwrap(pa, period=np.pi)
    pa_array = LSQUnivariateSpline(
        isolist.sma, pa_unwrapped, nodes)(finely_spaced_sma)
    x0_array = LSQUnivariateSpline(
        isolist.sma, isolist.x0, nodes)(finely_spaced_sma)
    y0_array = LSQUnivariateSpline(
        isolist.sma, isolist.y0, nodes)(finely_spaced_sma)
    if high_harmonics:
        grad_array = LSQUnivariateSpline(
            isolist.sma, isolist.gradient, nodes)(finely_spaced_sma)

        # Shifting the orientation by pi also shifts the angle
        # origin of the harmonics by pi, which flips the sign of the
        # third-order harmonics and leaves the fourth-order harmonics
        # unchanged.
        n_shifts = np.rint((pa_unwrapped - pa) / np.pi)
        sign3 = np.where(n_shifts % 2 == 0, 1.0, -1.0)

        # Harmonics that could not be computed (NaN) are treated as
        # zero, i.e., no deviation from a perfect ellipse.
        a3, b3, a4, b4 = (np.nan_to_num(values, nan=0.0) for values in
                          (isolist.a3, isolist.b3, isolist.a4, isolist.b4))
        a3_array = LSQUnivariateSpline(
            isolist.sma, sign3 * a3, nodes)(finely_spaced_sma)
        b3_array = LSQUnivariateSpline(
            isolist.sma, sign3 * b3, nodes)(finely_spaced_sma)
        a4_array = LSQUnivariateSpline(
            isolist.sma, a4, nodes)(finely_spaced_sma)
        b4_array = LSQUnivariateSpline(
            isolist.sma, b4, nodes)(finely_spaced_sma)

        # Return deviations from ellipticity to their original amplitude
        # meaning. The isophote harmonics are divided by the absolute
        # value of the gradient, which can be positive.
        grad_sma = np.abs(grad_array) * finely_spaced_sma
        kwargs_harm = {
            'a3_array': a3_array * grad_sma,
            'b3_array': b3_array * grad_sma,
            'a4_array': a4_array * grad_sma,
            'b4_array': b4_array * grad_sma,
        }
    else:
        kwargs_harm = {}

    # Correct deviations caused by fluctuations in spline solution.
    eps_array[np.where(eps_array < 0.0)] = 0.0

    # For each interpolated isophote, generate intensity values on the
    # output image array.
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

    # Pixels with no contribution from any isophote have zero weight.
    # Their weight is set to 1.0 for the normalization and their value
    # is set to the fill value.
    empty = weight <= 0.0
    weight[empty] = 1.0

    # Normalize.
    result /= weight

    # Fill value
    result[empty] = fill_value

    return result


@deprecated('3.2', alternative='IsophoteList.make_model_image',
            until='4.0')
@deprecated_positional_kwargs(since='3.0', until='4.0')
def build_ellipse_model(shape, isolist, fill=0.0, high_harmonics=False,
                        sma_interval=0.1):
    """
    Build a model elliptical galaxy image from a list of isophotes.

    Parameters
    ----------
    shape : 2-tuple
        The (ny, nx) shape of the output model image.

    isolist : `~photutils.isophote.IsophoteList` instance
        The isophote list created by the `~photutils.isophote.Ellipse`
        class.

    fill : float, optional
        The constant value to fill empty pixels.

    high_harmonics : bool, optional
        Whether to add the higher-order harmonics to the result.

    sma_interval : float, optional
        The step in semimajor axis length of the interpolation grid.

    Returns
    -------
    result : 2D `~numpy.ndarray`
        The image with the model galaxy.
    """
    return _make_model_image(isolist, shape, fill_value=fill,
                             high_harmonics=high_harmonics,
                             sma_interval=sma_interval)
