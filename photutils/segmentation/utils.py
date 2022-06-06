# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
This module provides tools for detecting sources in an image.
"""

import numpy as np

__all__ = []


def _make_binary_structure(ndim, connectivity):
    """
    Make a binary structure element.

    Parameters
    ----------
    ndim : int
        The number of array dimensions.

    connectivity : {4, 8}
        For the case of ``ndim=2``, the type of pixel connectivity used
        in determining how pixels are grouped into a detected source.
        The options are 4 or 8 (default). 4-connected pixels touch along
        their edges. 8-connected pixels touch along their edges or
        corners. For reference, SourceExtractor uses 8-connected pixels.

    Returns
    -------
    array : `~numpy.ndarray`
        The binary structure element. If ``ndim <= 2`` an array of int
        is returned, otherwise an array of bool is returned.
    """
    if ndim == 1:
        selem = np.array((1, 1, 1))
    elif ndim == 2:
        if connectivity == 4:
            selem = np.array(((0, 1, 0), (1, 1, 1), (0, 1, 0)))
        elif connectivity == 8:
            selem = np.ones((3, 3), dtype=int)
        else:
            raise ValueError(f'Invalid connectivity={connectivity}.  '
                             'Options are 4 or 8.')
    else:
        from scipy.ndimage import generate_binary_structure
        selem = generate_binary_structure(ndim, 1)

    return selem
