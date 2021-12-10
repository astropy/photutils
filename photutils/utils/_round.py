# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
This module provides tools to round numpy arrays.
"""

import numpy as np


def _py2intround(a):
    """
    Round the input to the nearest integer.

    If two integers are equally close, rounding is done away from 0.
    """
    data = np.asanyarray(a)
    value = np.where(data >= 0, np.floor(data + 0.5),
                     np.ceil(data - 0.5)).astype(int)

    if not hasattr(a, '__iter__'):
        value = value.item()

    return value
