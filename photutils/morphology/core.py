# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tools for measuring morphological properties of sources.
"""

import numpy as np

__all__ = ['data_properties']


def data_properties(data, mask=None, background=None, wcs=None):
    """
    Calculate the morphological properties (and centroid) of a 2D array
    (e.g., an image cutout of an object) using image moments.

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
        ``data``. ``background`` may either be a scalar value or a 2D
        image with the same shape as the input ``data``. Inputting the
        ``background`` merely allows for its properties to be measured
        within each source segment. The input ``background`` does *not*
        get subtracted from the input ``data``, which should already be
        background-subtracted.

    wcs : WCS object or `None`, optional
        A world coordinate system (WCS) transformation that
        supports the `astropy shared interface for WCS
        <https://docs.astropy.org/en/stable/wcs/wcsapi.html>`_ (e.g.,
        `astropy.wcs.WCS`, `gwcs.wcs.WCS`). If `None`, then all
        sky-based properties will be set to `None`.

    Returns
    -------
    result : `~photutils.segmentation.SourceCatalog` instance
        A scalar `~photutils.segmentation.SourceCatalog` object (single
        source) containing the morphological properties.

    Raises
    ------
    ValueError
        If ``data`` is not a 2D array.

    ValueError
        If ``mask`` is provided and does not have the same shape as
        ``data``.
    """
    # Prevent circular import
    from photutils.segmentation import SegmentationImage, SourceCatalog

    data = np.asanyarray(data)
    if len(data.shape) != 2:
        msg = 'data must be a 2D array'
        raise ValueError(msg)

    seg_arr = np.ones(data.shape, dtype=int)
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != data.shape:
            msg = 'mask must have the same shape as data'
            raise ValueError(msg)
        seg_arr[mask] = 0

    segment_image = SegmentationImage(seg_arr)

    if background is not None:
        background = np.atleast_1d(background)
        if background.shape == (1,):
            background = np.zeros(data.shape) + background

    return SourceCatalog(data, segment_image, mask=None,
                         background=background, wcs=wcs)[0]
