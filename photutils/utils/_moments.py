# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tools for calculating image moments.
"""

import numpy as np


def _image_moments(data, center=(0, 0), order=1):
    """
    Calculate the image moments up to the specified order.

    Parameters
    ----------
    data : 2D array_like
        The input 2D array.

    center : tuple of two floats or `None`, optional
        The ``(x, y)`` center position. If `None` it will be calculated
        as the "center of mass" of the input ``data``. The default is
        ``(0, 0)``, which gives the raw image moments.

    order : int, optional
        The maximum order of the moments to calculate.

    Returns
    -------
    moments : 2D `~numpy.ndarray`
        The image moments.
    """
    data = np.asarray(data).astype(float)

    if data.ndim != 2:
        msg = 'data must be a 2D array'
        raise ValueError(msg)

    if order < 0:
        msg = 'order must be non-negative'
        raise ValueError(msg)

    if center is None:
        from photutils.centroids import centroid_com

        center = centroid_com(data)

    indices = np.ogrid[tuple(slice(0, i) for i in data.shape)]
    ypowers = (indices[0] - center[1]) ** np.arange(order + 1)
    xpowers = np.transpose(indices[1] - center[0]) ** np.arange(order + 1)

    return np.dot(np.dot(np.transpose(ypowers), data), xpowers)
