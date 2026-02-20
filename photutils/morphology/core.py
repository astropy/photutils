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

    background : float or array_like, optional
        The background level previously present in the input ``data``.
        ``background`` may be a scalar value or a 2D array with the same
        shape as ``data``. The input ``background`` is not subtracted
        from ``data``, which should already be background-subtracted;
        providing it only enables background-related properties to be
        measured.

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

    ValueError
        If ``mask`` is provided and all pixels are masked.

    ValueError
        If ``background`` is provided and is not a scalar or a 2D array
        with the same shape as ``data``.
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
        if np.all(mask):
            msg = 'All pixels in data are masked'
            raise ValueError(msg)
        seg_arr[mask] = 0

    segment_image = SegmentationImage(seg_arr)

    if background is not None:
        background = np.asarray(background)
        if background.ndim == 0:
            background = np.full(data.shape, float(background))
        elif background.shape != data.shape:
            msg = ('background must be a scalar or a 2D array '
                   'with the same shape as data')
            raise ValueError(msg)

    # mask is encoded in seg_arr (masked pixels set to 0), so
    # mask=None is intentional here
    return SourceCatalog(data, segment_image, mask=None,
                         background=background, wcs=wcs)[0]
