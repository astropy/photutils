# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tools for rounding numpy arrays.
"""

import numpy as np


def round_half_away(a):
    """
    Round a float or array of floats to the nearest integer, rounding
    half away from zero.

    Parameters
    ----------
    a : float or array_like
        The input float or array.

    Returns
    -------
    result : int, float, or array_like
        The rounded values. Finite inputs are returned as integers.
        Non-finite inputs (NaN or infinity) are returned as floats,
        preserving the NaN or infinity value.

    Notes
    -----
    NaN and infinity values are preserved in the output. Arrays
    containing any non-finite value are returned as float arrays;
    all-finite arrays are returned as integer arrays.
    """
    data = np.atleast_1d(np.asarray(a, dtype=float))
    rounded = np.where(data >= 0, np.floor(data + 0.5),
                       np.ceil(data - 0.5))

    if np.isscalar(a):
        val = rounded[0]
        if not np.isfinite(a):
            return val
        return int(val)

    if np.isfinite(data).all():
        return rounded.astype(int)

    return rounded
