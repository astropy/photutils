# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
This module provides functions for measuring non-parametric morphologies
of sources.
"""

import numpy as np

__all__ = ['gini']


def gini(data):
    r"""
    Calculate the `Gini coefficient
    <https://en.wikipedia.org/wiki/Gini_coefficient>`_ of a 2D array.

    The Gini coefficient is calculated using the prescription from `Lotz
    et al. 2004
    <https://ui.adsabs.harvard.edu/abs/2004AJ....128..163L/abstract>`_
    as:

    .. math::
        G = \frac{1}{\left | \bar{x} \right | n (n - 1)}
        \sum^{n}_{i} (2i - n - 1) \left | x_i \right |

    where :math:`\bar{x}` is the mean over all pixel values
    :math:`x_i`.

    The Gini coefficient is a way of measuring the inequality in a given
    set of values. In the context of galaxy morphology, it measures how
    the light of a galaxy image is distributed among its pixels.  A Gini
    coefficient value of 0 corresponds to a galaxy image with the light
    evenly distributed over all pixels while a Gini coefficient value of
    1 represents a galaxy image with all its light concentrated in just
    one pixel.

    Usually Gini's measurement needs some sort of preprocessing for
    defining the galaxy region in the image based on the quality of the
    input data. As there is not a general standard for doing this, this
    is left for the user.

    Parameters
    ----------
    data : array_like
        The 2D data array or object that can be converted to an array.

    Returns
    -------
    gini : `float`
        The Gini coefficient of the input 2D array.
    """
    flattened = np.sort(np.ravel(data))
    npix = np.size(flattened)
    normalization = np.abs(np.mean(flattened)) * npix * (npix - 1)
    kernel = (2.0 * np.arange(1, npix + 1) - npix - 1) * np.abs(flattened)

    return np.sum(kernel) / normalization
