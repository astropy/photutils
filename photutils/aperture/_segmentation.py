# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tools for segmentation-based masking of aperture photometry.

These helpers validate the ``segmentation_image``,
``labels``, and ``mask_method`` keywords shared by
`~photutils.aperture.PixelAperture._photometry`,
`~photutils.aperture.AperturePhotometry`, and
`~photutils.aperture.ApertureStats`. They also apply the local masking
or symmetric-correction behavior to per-aperture cutouts.
"""

import numpy as np

from photutils.utils._round import round_half_away

__all__ = []

# The valid ``mask_method`` values and their integer codes used
# by the batch Cython driver (see ``_batch_photometry.pyx``).
MASK_METHODS = ('none', 'mask', 'source_only', 'background_only',
                'correct')
SEG_METHOD_CODES = {'none': 0, 'mask': 1, 'source_only': 2, 'correct': 3,
                    'background_only': 4}


def process_segmentation_inputs(segmentation_image, labels,
                                mask_method, positions, data_shape):
    """
    Validate the segmentation-masking inputs and resolve the
    per-aperture source labels.

    Parameters
    ----------
    segmentation_image : `~photutils.segmentation.SegmentationImage` \
            or `None`
        The segmentation image, where background pixels are zero and
        sources have positive integer labels. A
        `~photutils.segmentation.SegmentationImage` is required (rather
        than a plain array) because its cached labels make the
        per-call validation of ``labels`` inexpensive.

    labels : int, 1D array_like, or `None`
        The source label(s) associated with the aperture
        ``positions``. ``labels`` is required (must not be `None`)
        when ``segmentation_image`` is input and ``mask_method`` is
        ``'mask'``, ``'source_only'``, or ``'correct'``. Each nonzero
        label must be present in the ``segmentation_image``. A label of
        0 disables the masking for that aperture. ``labels`` is not used
        when ``mask_method`` is ``'background_only'``. In that case the
        labels need not be present in the ``segmentation_image``, but if
        input they must still have one entry per aperture position.

    mask_method : {'none', 'mask', 'source_only', 'background_only', \
            'correct'}
        The segmentation masking method.

    positions : 2D array_like
        The ``(x, y)`` aperture positions with shape ``(n_positions, 2)``.

    data_shape : tuple of int
        The shape of the data array.

    Returns
    -------
    segmentation : 2D `~numpy.ndarray` (intp) or `None`
        The validated segmentation array, or `None` if
        ``mask_method`` is ``'none'``.

    labels : 1D `~numpy.ndarray` (intp) or `None`
        The resolved per-aperture source labels, or `None` if
        ``mask_method`` is ``'none'``. For ``'background_only'``,
        which does not use the labels, an array of zeros.
    """
    if mask_method not in MASK_METHODS:
        msg = f'mask_method must be one of {MASK_METHODS}'
        raise ValueError(msg)

    if mask_method == 'none':
        return None, None

    if segmentation_image is None:
        msg = ('segmentation_image must be input when mask_method '
               "is not 'none'")
        raise ValueError(msg)

    # Local import to avoid a circular import with photutils.segmentation
    from photutils.segmentation import SegmentationImage

    if not isinstance(segmentation_image, SegmentationImage):
        msg = 'segmentation_image must be a SegmentationImage'
        raise TypeError(msg)

    segm = segmentation_image.data
    if segm.shape != tuple(data_shape):
        msg = 'segmentation_image must have the same shape as the data'
        raise ValueError(msg)

    segm = np.ascontiguousarray(segm, dtype=np.intp)

    positions = np.atleast_2d(positions)
    n_positions = positions.shape[0]

    if labels is None:
        if mask_method == 'background_only':
            # The labels are not used, but the batch kernels require a
            # per-source labels array.
            return segm, np.zeros(n_positions, dtype=np.intp)
        msg = ('labels must be input when segmentation_image is input '
               "and mask_method is not 'none'")
        raise ValueError(msg)

    labels = np.atleast_1d(labels)
    if labels.ndim != 1:
        msg = 'labels must be a 1D array'
        raise ValueError(msg)
    if labels.shape[0] != n_positions:
        msg = ('labels must have the same length as the number of '
               'aperture positions')
        raise ValueError(msg)
    labels = np.ascontiguousarray(labels, dtype=np.intp)

    if mask_method == 'background_only':
        # The labels are not used, so they need not be present in the
        # segmentation image. Their shape is still validated above
        # because the input labels are stored and sliced per aperture.
        return segm, np.zeros(n_positions, dtype=np.intp)

    # Each nonzero label must be present in the segmentation image.
    # Otherwise, every labeled pixel would silently be treated as a
    # neighbor of the target source. The SegmentationImage labels are
    # cached, so this check is inexpensive.
    nonzero = labels[labels != 0]
    present = np.isin(nonzero, segmentation_image.labels)
    bad_labels = np.unique(nonzero[~present])
    if bad_labels.size > 0:
        msg = (f'labels {bad_labels.tolist()} are not present in the '
               'segmentation_image')
        raise ValueError(msg)

    return segm, labels


def make_segmentation_exclusion(mask_method, segmentation_cutout,
                                label, *, data=None, error=None,
                                base_mask=None, cutout_xycen=None):
    """
    Build the segmentation-based pixel exclusion mask for a single
    aperture cutout, optionally applying the symmetric ``'correct'``
    replacement to the ``data`` and ``error`` cutouts.

    Parameters
    ----------
    mask_method : {'none', 'mask', 'source_only', 'background_only', \
            'correct'}
        The segmentation masking method.

    segmentation_cutout : 2D `~numpy.ndarray`
        The segmentation cutout for the aperture, aligned with ``data``.

    label : int
        The target source label for this aperture. If ``label`` is 0,
        masking is disabled and no pixels are excluded. The label is
        ignored by the ``'background_only'`` method, which has no
        target source.

    data : 2D `~numpy.ndarray`, optional
        The data cutout, required for the ``'correct'`` method.

    error : 2D `~numpy.ndarray`, optional
        The error cutout, mirror-corrected alongside ``data`` for the
        ``'correct'`` method.

    base_mask : 2D bool `~numpy.ndarray`, optional
        Pixels that are already masked (e.g., the global ``mask`` or
        non-finite values). Used by the ``'correct'`` method to avoid
        mirroring from masked pixels.

    cutout_xycen : tuple of float, optional
        The ``(x, y)`` aperture center relative to the cutout origin,
        required for the ``'correct'`` method.

    Returns
    -------
    data : 2D `~numpy.ndarray`
        The (possibly mirror-corrected) data cutout.

    error : 2D `~numpy.ndarray` or `None`
        The (possibly mirror-corrected) error cutout.

    exclude : 2D bool `~numpy.ndarray`
        A boolean mask where `True` indicates a pixel to exclude from
        the photometry.

    affected : 2D bool `~numpy.ndarray`
        A boolean mask where `True` indicates a pixel excluded,
        replaced, or excluded-instead-of-replaced due to a neighboring
        source (i.e., a labeled pixel that is not the target source).
        Background pixels excluded by the ``'source_only'`` method are
        not marked. For the ``'background_only'`` method, every labeled
        pixel is a neighbor pixel. For the ``'correct'`` method,
        ``exclude`` marks exactly the neighbor pixels that could not be
        corrected. ``exclude`` and ``affected`` may be the same array
        object, so neither must be modified in place.
    """
    exclude = np.zeros(segmentation_cutout.shape, dtype=bool)

    # The 'none' and 'background_only' methods return one array as both
    # the exclude and affected masks. Callers must not modify either
    # mask in place.
    if mask_method == 'none':
        return data, error, exclude, exclude

    if mask_method == 'background_only':
        labeled = segmentation_cutout > 0
        return data, error, labeled, labeled

    if label == 0:
        return data, error, exclude, exclude

    neighbor = ((segmentation_cutout > 0)
                & (segmentation_cutout != label))

    if mask_method == 'mask':
        return data, error, neighbor, neighbor

    if mask_method == 'source_only':
        exclude = segmentation_cutout != label
        return data, error, exclude, neighbor

    # The remaining method is the symmetric 'correct' replacement.
    return _correct_neighbors(segmentation_cutout, neighbor, data,
                              error, base_mask, cutout_xycen)


def _correct_neighbors(segmentation_cutout, neighbor, data, error,
                       base_mask, cutout_xycen):
    """
    Replace neighbor-source pixels with their values mirrored across the
    aperture center.

    A neighbor pixel whose mirror pixel is unavailable (outside the
    cutout, itself a neighbor, or in ``base_mask``) is excluded from the
    photometry instead of being replaced.
    """
    ny, nx = segmentation_cutout.shape
    exclude = np.zeros(segmentation_cutout.shape, dtype=bool)

    data = data.copy()
    if error is not None:
        error = error.copy()

    cx = int(round_half_away(cutout_xycen[0]))
    cy = int(round_half_away(cutout_xycen[1]))

    yidx, xidx = np.nonzero(neighbor)
    xmirror = 2 * cx - xidx
    ymirror = 2 * cy - yidx

    out_of_bounds = ((xmirror < 0) | (ymirror < 0)
                     | (xmirror >= nx) | (ymirror >= ny))
    exclude[yidx[out_of_bounds], xidx[out_of_bounds]] = True

    keep = ~out_of_bounds
    yidx = yidx[keep]
    xidx = xidx[keep]
    xmirror = xmirror[keep]
    ymirror = ymirror[keep]

    bad_mirror = neighbor[ymirror, xmirror]
    if base_mask is not None:
        bad_mirror |= base_mask[ymirror, xmirror]
    exclude[yidx[bad_mirror], xidx[bad_mirror]] = True

    good = ~bad_mirror
    gy = yidx[good]
    gx = xidx[good]
    gmy = ymirror[good]
    gmx = xmirror[good]
    data[gy, gx] = data[gmy, gmx]
    if error is not None:
        error[gy, gx] = error[gmy, gmx]

    return data, error, exclude, neighbor
