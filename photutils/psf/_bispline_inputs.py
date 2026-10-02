# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Constant inputs of the compiled spline kernels for a single spline.

A model with one spline calls the kernels with one coefficient row
that has unit weight and no weight derivatives.
"""

import numpy as np

__all__ = ['ONE_PLANE', 'UNIT_WEIGHT', 'ZERO_WEIGHT']

ONE_PLANE = np.zeros(1, dtype=np.intp)
UNIT_WEIGHT = np.ones(1)
ZERO_WEIGHT = np.zeros(1)
# Every model shares these arrays, so they must never be modified
ONE_PLANE.setflags(write=False)
UNIT_WEIGHT.setflags(write=False)
ZERO_WEIGHT.setflags(write=False)
