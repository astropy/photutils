# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Helper functions for centroid tests.
"""

import numpy as np
from astropy.modeling.models import Gaussian2D

__all__ = ['make_gaussian_source']


def make_gaussian_source(shape, amplitude, xc, yc, xstd, ystd, theta):
    """
    Make a 2D Gaussian source.

    Parameters
    ----------
    shape : tuple of 2 int
        The (ny, nx) shape of the output image.

    amplitude : float
        The amplitude of the Gaussian.

    xc, yc : float
        The center of the Gaussian.

    xstd, ystd : float
        The standard deviations of the Gaussian.

    theta : float
        The rotation angle of the Gaussian in radians.

    Returns
    -------
    data : 2D `~numpy.ndarray`
        The 2D image of the Gaussian source.
    """
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    model = Gaussian2D(amplitude, xc, yc, xstd, ystd, theta)
    return model(xx, yy)
