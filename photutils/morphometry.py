# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Functions for measuring non-parametric morphology.
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import numpy as np


def gini(data):
    """Measures the Gini coefficient on a 2-D array based on perscription
    from [lotz2004]_. It is a way of measuring the inaquality in a
    given set of values. In the context of galaxy morphology, it
    measures how the light of a galaxy image is distributed among its
    pixels. A Gini value of 0 corresponds to a galaxy image with the
    light equally distributed over all pixels while a Gini value of 1
    represents a galaxy image with all its light concentrated in
    just one pixel.

    Usually Gini's measurement needs some sort of pre-processing
    based on the quality of the input data. As there is not a general
    standard for doing this, this is left for the user.

    .. [lotz2004] Lotz et al. 2004,
    A new nonparametric approach to galaxy morphological classification,
    http://arxiv.org/abs/astro-ph/0311352

    Parameters
    ----------
    data : array_like
        The 2-d array with the values for measuring the Gini Coefficient.

    Returns
    -------
    gini : `float`
        Gini coefficient value for given 2-D array.
    """
    flattened = np.sort(np.ravel(data))
    N = np.size(flattened)
    normalization = 1/(np.abs(np.mean(flattened)) * N * (N-1))
    kernel = (2*np.arange(1, N+1) - N - 1) * np.abs(flattened)
    G = normalization * np.sum(kernel)
    return G
