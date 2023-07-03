# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
This module provides tools to round numpy arrays.
"""

import numpy as np


def py2intround(a):
    """
    Round the input to the nearest integer.

    If two integers are equally close, rounding is done away from 0.

    Parameters
    ----------
    a : float or array-like
        The input float or array.

    Returns
    -------
    result : float or array-like
        The integer-rounded values.
    """
    data = np.atleast_1d(a)
    value = np.where(data >= 0, np.floor(data + 0.5),
                     np.ceil(data - 0.5)).astype(int)

    if np.isscalar(a):
        value = value[0]

    return value
