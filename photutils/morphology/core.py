# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Functions for measuring morphological properties of objects in an
astronomical image using image moments.
"""

import numpy as np


__all__ = ['data_properties']


def data_properties(data, mask=None, background=None):
    """
    Calculate the morphological properties (and centroid) of a 2D array
    (e.g. an image cutout of an object) using image moments.

    Parameters
    ----------
    data : array_like or `~astropy.units.Quantity`
        The 2D array of the image.

    mask : array_like (bool), optional
        A boolean mask, with the same shape as ``data``, where a `True`
        value indicates the corresponding element of ``data`` is masked.
        Masked data are excluded from all calculations.

    background : float, array_like, or `~astropy.units.Quantity`, optional
        The background level that was previously present in the input
        ``data``.  ``background`` may either be a scalar value or a 2D
        image with the same shape as the input ``data``.  Inputting the
        ``background`` merely allows for its properties to be measured
        within each source segment.  The input ``background`` does *not*
        get subtracted from the input ``data``, which should already be
        background-subtracted.

    Returns
    -------
    result : `~photutils.segmentation.SourceProperties` instance
        A `~photutils.segmentation.SourceProperties` object.
    """

    from ..segmentation import SourceProperties  # prevent circular imports

    segment_image = np.ones(data.shape, dtype=np.int)

    return SourceProperties(data, segment_image, label=1, mask=mask,
                            background=background)
