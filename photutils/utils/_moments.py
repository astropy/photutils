# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
This module provide tools for calculating image moments.
"""

import numpy as np

__all__ = ['_moments_central', '_moments']


def _moments_central(data, center=None, order=1):
    """
    Calculate the central image moments up to the specified order.

    Parameters
    ----------
    data : 2D array_like
        The input 2D array.

    center : tuple of two floats or `None`, optional
        The ``(x, y)`` center position.  If `None` it will calculated as
        the "center of mass" of the input ``data``.

    order : int, optional
        The maximum order of the moments to calculate.

    Returns
    -------
    moments : 2D `~numpy.ndarray`
        The central image moments.
    """
    data = np.asarray(data).astype(float)

    if data.ndim != 2:
        raise ValueError('data must be a 2D array.')

    if center is None:
        from photutils.centroids import centroid_com

        center = centroid_com(data)

    indices = np.ogrid[tuple(slice(0, i) for i in data.shape)]
    ypowers = (indices[0] - center[1]) ** np.arange(order + 1)
    xpowers = np.transpose(indices[1] - center[0]) ** np.arange(order + 1)

    return np.dot(np.dot(np.transpose(ypowers), data), xpowers)


def _moments(data, order=1):
    """
    Calculate the raw image moments up to the specified order.

    Parameters
    ----------
    data : 2D array_like
        The input 2D array.

    order : int, optional
        The maximum order of the moments to calculate.

    Returns
    -------
    moments : 2D `~numpy.ndarray`
        The raw image moments.
    """
    return _moments_central(data, center=(0, 0), order=order)
