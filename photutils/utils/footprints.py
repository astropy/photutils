# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
This module provides tools for generating footprints.
"""

import numpy as np

__all__ = ['circular_footprint']


def circular_footprint(radius, dtype=int):
    """
    Create a circular footprint.

    A pixel is considered to be entirely in or out of the footprint
    depending on whether its center is in or out of the footprint.  The
    size of the output array is the minimal bounding box for the
    footprint.

    Parameters
    ----------
    radius : int
        The radius of the circular footprint.

    dtype : data-type, optional
        The data type of the output `~numpy.ndarray`.

    Returns
    -------
    footprint : `~numpy.ndarray`
        A footprint where array elements are 1 within the footprint and
        0 otherwise.

    Examples
    --------
    >>> from photutils.utils import circular_footprint
    >>> circular_footprint(2)
    array([[0, 0, 1, 0, 0],
           [0, 1, 1, 1, 0],
           [1, 1, 1, 1, 1],
           [0, 1, 1, 1, 0],
           [0, 0, 1, 0, 0]])
    """
    if ~np.isfinite(radius) or radius <= 0 or int(radius) != radius:
        raise ValueError('radius must be a positive, finite integer greater '
                         'than 0')

    x = np.arange(-radius, radius + 1)
    xx, yy = np.meshgrid(x, x)
    return np.array((xx**2 + yy**2) <= radius**2, dtype=dtype)
