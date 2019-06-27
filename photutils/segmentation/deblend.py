# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
This module provides tools for deblending overlapping sources labeled in
a segmentation image.
"""

from copy import deepcopy
import warnings

from astropy.utils.exceptions import AstropyUserWarning
import numpy as np

from .core import SegmentationImage
from .detect import detect_sources
from ..utils.convolution import filter_data
from ..utils.exceptions import NoDetectionsWarning

__all__ = ['deblend_sources']


def deblend_sources(data, segment_img, npixels, filter_kernel=None,
                    labels=None, nlevels=32, contrast=0.001,
                    mode='exponential', connectivity=8, relabel=True):
    """
    Deblend overlapping sources labeled in a segmentation image.

    Sources are deblended using a combination of multi-thresholding and
    `watershed segmentation
    <https://en.wikipedia.org/wiki/Watershed_(image_processing)>`_.  In
    order to deblend sources, there must be a saddle between them.

    Parameters
    ----------
    data : array_like
        The data array.

    segment_img : `~photutils.segmentation.SegmentationImage` or array_like (int)
        A segmentation image, either as a
        `~photutils.segmentation.SegmentationImage` object or an
        `~numpy.ndarray`, with the same shape as ``data`` where sources
        are labeled by different positive integer values.  A value of
        zero is reserved for the background.

    npixels : int
        The number of connected pixels, each greater than ``threshold``,
        that an object must have to be detected.  ``npixels`` must be a
        positive integer.

    filter_kernel : array-like or `~astropy.convolution.Kernel2D`, optional
        The array of the kernel used to filter the image before
        thresholding.  Filtering the image will smooth the noise and
        maximize detectability of objects with a shape similar to the
        kernel.

    labels : int or array-like of int, optional
        The label numbers to deblend.  If `None` (default), then all
        labels in the segmentation image will be deblended.

    nlevels : int, optional
        The number of multi-thresholding levels to use.  Each source
        will be re-thresholded at ``nlevels`` levels spaced
        exponentially or linearly (see the ``mode`` keyword) between its
        minimum and maximum values within the source segment.

    contrast : float, optional
        The fraction of the total (blended) source flux that a local
        peak must have (at any one of the multi-thresholds) to be
        considered as a separate object.  ``contrast`` must be between 0
        and 1, inclusive.  If ``contrast = 0`` then every local peak
        will be made a separate object (maximum deblending).  If
        ``contrast = 1`` then no deblending will occur.  The default is
        0.001, which will deblend sources with a 7.5 magnitude
        difference.

    mode : {'exponential', 'linear'}, optional
        The mode used in defining the spacing between the
        multi-thresholding levels (see the ``nlevels`` keyword).  The
        default is 'exponential'.

    connectivity : {8, 4}, optional
        The type of pixel connectivity used in determining how pixels
        are grouped into a detected source.  The options are 8 (default)
        or 4.  8-connected pixels touch along their edges or corners.
        4-connected pixels touch along their edges.  For reference,
        SExtractor uses 8-connected pixels.

    relabel : bool
        If `True` (default), then the segmentation image will be
        relabeled such that the labels are in consecutive order starting
        from 1.

    Returns
    -------
    segment_image : `~photutils.segmentation.SegmentationImage`
        A segmentation image, with the same shape as ``data``, where
        sources are marked by different positive integer values.  A
        value of zero is reserved for the background.

    See Also
    --------
    :func:`photutils.detect_sources`
    """

    if not isinstance(segment_img, SegmentationImage):
        segment_img = SegmentationImage(segment_img)

    if segment_img.shape != data.shape:
        raise ValueError('The data and segmentation image must have '
                         'the same shape')

    if labels is None:
        labels = segment_img.labels
    labels = np.atleast_1d(labels)
    segment_img.check_labels(labels)

    data = filter_data(data, filter_kernel, mode='constant', fill_value=0.0)

    last_label = segment_img.max_label
    segm_deblended = deepcopy(segment_img)
    for label in labels:
        source_slice = segment_img.slices[segment_img.get_index(label)]
        source_data = data[source_slice]
        source_segm = SegmentationImage(np.copy(
            segment_img.data[source_slice]))
        source_segm.keep_labels(label)  # include only one label
        source_deblended = _deblend_source(
            source_data, source_segm, npixels, nlevels=nlevels,
            contrast=contrast, mode=mode, connectivity=connectivity)

        if not np.array_equal(source_deblended.data.astype(bool),
                              source_segm.data.astype(bool)):
            raise ValueError('Deblending failed for source "{0}".  Please '
                             'ensure you used the same pixel connectivity '
                             'in detect_sources and deblend_sources.  If '
                             'this issue persists, then please inform the '
                             'developers.'.format(label))

        if source_deblended.nlabels > 1:
            # replace the original source with the deblended source
            source_mask = (source_deblended.data > 0)
            segm_tmp = segm_deblended.data
            segm_tmp[source_slice][source_mask] = (
                source_deblended.data[source_mask] + last_label)
            segm_deblended.data = segm_tmp  # needed to call data setter
            last_label += source_deblended.nlabels

    if relabel:
        segm_deblended.relabel_consecutive()

    return segm_deblended


def _deblend_source(data, segment_img, npixels, nlevels=32, contrast=0.001,
                    mode='exponential', connectivity=8):
    """
    Deblend a single labeled source.

    Parameters
    ----------
    data : array_like
        The cutout data array for a single source.  ``data`` should also
        already be smoothed by the same filter used in
        :func:`~photutils.detect_sources`, if applicable.

    segment_img : `~photutils.segmentation.SegmentationImage`
        A cutout `~photutils.segmentation.SegmentationImage` object with
        the same shape as ``data``.  ``segment_img`` should contain only
        *one* source label.

    npixels : int
        The number of connected pixels, each greater than ``threshold``,
        that an object must have to be detected.  ``npixels`` must be a
        positive integer.

    nlevels : int, optional
        The number of multi-thresholding levels to use.  Each source
        will be re-thresholded at ``nlevels`` levels spaced
        exponentially or linearly (see the ``mode`` keyword) between its
        minimum and maximum values within the source segment.

    contrast : float, optional
        The fraction of the total (blended) source flux that a local
        peak must have (at any one of the multi-thresholds) to be
        considered as a separate object.  ``contrast`` must be between 0
        and 1, inclusive.  If ``contrast = 0`` then every local peak
        will be made a separate object (maximum deblending).  If
        ``contrast = 1`` then no deblending will occur.  The default is
        0.001, which will deblend sources with a 7.5 magnitude
        difference.

    mode : {'exponential', 'linear'}, optional
        The mode used in defining the spacing between the
        multi-thresholding levels (see the ``nlevels`` keyword).  The
        default is 'exponential'.

    connectivity : {8, 4}, optional
        The type of pixel connectivity used in determining how pixels
        are grouped into a detected source.  The options are 8 (default)
        or 4.  8-connected pixels touch along their edges or corners.
        4-connected pixels touch along their edges.  For reference,
        SExtractor uses 8-connected pixels.

    Returns
    -------
    segment_image : `~photutils.segmentation.SegmentationImage`
        A segmentation image, with the same shape as ``data``, where
        sources are marked by different positive integer values.  A
        value of zero is reserved for the background.
    """

    from scipy import ndimage
    from skimage.morphology import watershed

    if nlevels < 1:
        raise ValueError('nlevels must be >= 1, got "{0}"'.format(nlevels))
    if contrast < 0 or contrast > 1:
        raise ValueError('contrast must be >= 0 and <= 1, got '
                         '"{0}"'.format(contrast))

    ndim = data.ndim
    if ndim == 1:
        selem = ndimage.generate_binary_structure(ndim, 1)
    else:
        if connectivity == 4:
            selem = ndimage.generate_binary_structure(ndim, 1)
        elif connectivity == 8:
            selem = ndimage.generate_binary_structure(ndim, 2)
        else:
            raise ValueError('Invalid connectivity={0}.  '
                             'Options are 4 or 8'.format(connectivity))

    segm_mask = (segment_img.data > 0)
    source_values = data[segm_mask]
    source_sum = float(np.nansum(source_values))
    source_min = np.nanmin(source_values)
    source_max = np.nanmax(source_values)
    if source_min == source_max:
        return segment_img  # no deblending

    if mode == 'exponential' and source_min < 0:
        warnings.warn('Source "{0}" contains negative values, setting '
                      'deblending mode to "linear"'.format(
                          segment_img.labels[0]), AstropyUserWarning)
        mode = 'linear'

    steps = np.arange(1., nlevels + 1)
    if mode == 'exponential':
        if source_min == 0:
            source_min = source_max * 0.01
        thresholds = source_min * ((source_max / source_min) **
                                   (steps / (nlevels + 1)))
    elif mode == 'linear':
        thresholds = source_min + ((source_max - source_min) /
                                   (nlevels + 1)) * steps
    else:
        raise ValueError('"{0}" is an invalid mode; mode must be '
                         '"exponential" or "linear"'.format(mode))

    # suppress NoDetectionsWarning during deblending
    warnings.filterwarnings('ignore', category=NoDetectionsWarning)

    level_segms = []
    mask = ~segm_mask
    for level in thresholds:
        segm_tmp = detect_sources(data, level, npixels=npixels,
                                  connectivity=connectivity, mask=mask)

        # NOTE: higher threshold levels may not meet 'npixels' criterion
        # resulting in no detections
        if segm_tmp is None or segm_tmp.nlabels == 1:
            continue

        fluxes = np.array([np.nansum(data[segm_tmp == i])
                           for i in segm_tmp.labels])
        idx = np.where((fluxes / source_sum) >= contrast)[0]

        # at least 2 segment meet the contrast requirement
        if idx.size >= 2:
            # keep only the labels that meet the contrast criterion
            segm_tmp.keep_labels(segm_tmp.labels[idx])
            level_segms.append(segm_tmp)

    nlevels = len(level_segms)
    if nlevels == 0:  # no deblending
        return segment_img
    else:
        for i in range(nlevels - 1):
            segm_lower = level_segms[i].data
            segm_upper = level_segms[i + 1].data
            relabel = False
            # if the are more sources at the upper level, then
            # remove the parent source(s) from the lower level,
            # but keep any sources in the lower level that do not have
            # multiple children in the upper level
            for label in level_segms[i].labels:
                mask = (segm_lower == label)
                # checks for 1-to-1 label mapping n -> m (where m >= 0)
                upper_labels = segm_upper[mask]
                upper_labels = np.unique(upper_labels[upper_labels != 0])
                if upper_labels.size >= 2:
                    relabel = True
                    segm_lower[mask] = segm_upper[mask]

            if relabel:
                level_segms[i + 1] = SegmentationImage(
                    ndimage.label(segm_lower, structure=selem)[0])
            else:
                level_segms[i + 1] = level_segms[i]

        return SegmentationImage(watershed(-data, level_segms[-1].data,
                                           mask=segment_img.data.astype(bool),
                                           connectivity=selem))
