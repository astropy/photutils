# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Shared synthetic scene for the batch-driver comparator tests.
"""

import numpy as np

from photutils.segmentation import SourceCatalog, detect_sources
from photutils.segmentation.utils import _mask_to_mirrored_value

__all__ = ['make_batch_scene', 'make_catalog', 'reference_aperture_data']


def make_batch_scene(*, seed=0):
    """
    Make a deterministic test scene exercising the batch-driver edge
    cases.

    The scene contains isolated sources, close/overlapping pairs
    (neighbor-segment handling), sources touching every image edge (bbox
    clipping), masked pixels inside sources and spanning a close pair
    (mirror-correction rejection), and non-finite data values inside a
    source.

    Parameters
    ----------
    seed : int, optional
        The seed for the random number generator used for the
        background noise.

    Returns
    -------
    result : dict
        Keys ``data``, ``error``, ``mask``, ``segm`` (a
        `~photutils.segmentation.SegmentationImage`).
    """
    rng = np.random.default_rng(seed)
    ny = nx = 151
    yy, xx = np.mgrid[0:ny, 0:nx]
    data = rng.normal(0.0, 0.1, (ny, nx))
    positions = [(20, 20), (24, 26), (75, 75), (75, 81), (140, 20),
                 (5, 100), (100, 4), (147, 147), (50, 120), (120, 50),
                 (100, 100)]
    for i, (xc, yc) in enumerate(positions):
        amp = 5.0 + i
        sig = 1.5 + 0.2 * (i % 4)
        data += amp * np.exp(-((xx - xc) ** 2 + (yy - yc) ** 2)
                             / (2 * sig ** 2))
    error = np.full((ny, nx), 0.1)
    error[::17, ::13] = 0.3
    mask = np.zeros((ny, nx), dtype=bool)
    mask[18:20, 22:24] = True  # inside a source of a close pair
    mask[73:75, 72:84] = True  # spans a close pair
    data[77, 78] = np.nan  # non-finite inside a source
    data[75, 83] = np.inf
    segm = detect_sources(data, 1.0, n_pixels=5)
    return {'data': data, 'error': error, 'mask': mask, 'segm': segm}


def make_catalog(scene, *, aperture_mask_method='correct',
                 with_error=True, with_mask=True):
    """
    Make a `SourceCatalog` from a scene with the given options.

    Parameters
    ----------
    scene : dict
        A scene from `make_batch_scene`.

    aperture_mask_method : {'correct', 'mask', 'none'}, optional
        The segmentation masking method.

    with_error : bool, optional
        Whether to pass the scene error array to the catalog.

    with_mask : bool, optional
        Whether to pass the scene mask array to the catalog.

    Returns
    -------
    result : `~photutils.segmentation.SourceCatalog`
        The source catalog.
    """
    return SourceCatalog(
        scene['data'], scene['segm'],
        error=scene['error'] if with_error else None,
        mask=scene['mask'] if with_mask else None,
        aperture_mask_method=aperture_mask_method)


def reference_aperture_data(cat, label, x_centroid, y_centroid,
                            aperture_bbox, local_background, *,
                            make_error=True):
    """
    Make cutouts of the data, error, and mask arrays for aperture
    photometry.

    This is a verbatim port of the per-source ``_make_aperture_data``
    method that the batch Cython drivers replaced. It is the
    numerical reference for the neighbor handling of the drivers.

    Parameters
    ----------
    cat : `~photutils.segmentation.SourceCatalog`
        The source catalog.

    label : int
        The source label.

    x_centroid, y_centroid : float
        The aperture center.

    aperture_bbox : `~photutils.aperture.BoundingBox`
        The aperture bounding box.

    local_background : float
        The local background to subtract.

    make_error : bool, optional
        Whether to return the error cutout.

    Returns
    -------
    result : tuple
        The ``(data, error, mask, cutout_xycen, slc_sm, flag_masks)``
        values, or ``(None,) * 6`` if the bounding box does not overlap
        the data.
    """
    slc_lg, slc_sm = aperture_bbox.get_overlap_slices(cat._data.shape)
    if slc_lg is None:
        return (None,) * 6

    data = cat._data[slc_lg].astype(float) - local_background

    mask_cutout = None if cat._mask is None else cat._mask[slc_lg]
    data_mask = ~np.isfinite(data)
    if mask_cutout is not None:
        data_mask |= mask_cutout

    if make_error and cat._error is not None:
        error = cat._error[slc_lg]
    else:
        error = None

    cutout_xycen = (x_centroid - max(0, aperture_bbox.ixmin),
                    y_centroid - max(0, aperture_bbox.iymin))

    segm_mask = None
    uncorrected_mask = None
    if cat.aperture_mask_method == 'none':
        mask = data_mask
    else:
        segment_img = cat._segmentation_image.data[slc_lg]
        segm_mask = np.logical_and(segment_img != label,
                                   segment_img != 0)
        if cat.aperture_mask_method == 'mask':
            mask = data_mask | segm_mask
        else:
            mask = data_mask

    if cat.aperture_mask_method == 'correct':
        data, uncorrected_mask = _mask_to_mirrored_value(
            data, segm_mask, cutout_xycen, mask=mask,
            return_uncorrected=True)
        if error is not None:
            error = _mask_to_mirrored_value(error, segm_mask, cutout_xycen,
                                            mask=mask)

    flag_masks = {'data_mask': data_mask,
                  'segm_mask': segm_mask,
                  'uncorrected_mask': uncorrected_mask}

    return data, error, mask, cutout_xycen, slc_sm, flag_masks
