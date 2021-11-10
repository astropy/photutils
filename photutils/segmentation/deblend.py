# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
This module provides tools for deblending overlapping sources labeled in
a segmentation image.
"""

import warnings

from astropy.utils.decorators import deprecated_renamed_argument
from astropy.utils.exceptions import AstropyUserWarning
import numpy as np

from .core import SegmentationImage
from .detect import _make_binary_structure, _detect_sources
from ..utils._convolution import _filter_data
from ..utils.exceptions import NoDetectionsWarning

__all__ = ['deblend_sources']


@deprecated_renamed_argument('filter_kernel', 'kernel', '1.2')
def deblend_sources(data, segment_img, npixels, kernel=None, labels=None,
                    nlevels=32, contrast=0.001, mode='exponential',
                    connectivity=8, relabel=True):
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

    kernel : array-like or `~astropy.convolution.Kernel2D`, optional
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
        SourceExtractor uses 8-connected pixels.

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
    :func:`photutils.segmentation.detect_sources`
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

    if kernel is not None:
        data = _filter_data(data, kernel, mode='constant', fill_value=0.0)

    last_label = segment_img.max_label
    segm_deblended = object.__new__(SegmentationImage)
    segm_deblended._data = np.copy(segment_img.data)
    for label in labels:
        source_slice = segment_img.slices[segment_img.get_index(label)]
        source_data = data[source_slice]

        source_segm = object.__new__(SegmentationImage)
        source_segm._data = np.copy(segment_img.data[source_slice])

        source_segm.keep_labels(label)  # include only one label
        source_deblended = _deblend_source(
            source_data, source_segm, npixels, nlevels=nlevels,
            contrast=contrast, mode=mode, connectivity=connectivity)

        if not np.array_equal(source_deblended.data.astype(bool),
                              source_segm.data.astype(bool)):
            raise ValueError(f'Deblending failed for source "{label}".  '
                             'Please ensure you used the same pixel '
                             'connectivity in detect_sources and '
                             'deblend_sources.  If this issue persists, '
                             'then please inform the developers.')

        if source_deblended.nlabels > 1:
            source_deblended.relabel_consecutive(start_label=1)

            # replace the original source with the deblended source
            source_mask = (source_deblended.data > 0)
            segm_tmp = segm_deblended.data
            segm_tmp[source_slice][source_mask] = (
                source_deblended.data[source_mask] + last_label)

            segm_deblended.__dict__ = {}  # reset cached properties
            segm_deblended._data = segm_tmp

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
        :func:`~photutils.segmentation.detect_sources`, if applicable.

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
        SourceExtractor uses 8-connected pixels.

    Returns
    -------
    segment_image : `~photutils.segmentation.SegmentationImage`
        A segmentation image, with the same shape as ``data``, where
        sources are marked by different positive integer values.  A
        value of zero is reserved for the background.  Note that the
        returned `SegmentationImage` may *not* have consecutive labels.
    """
    from scipy.ndimage import label as ndilabel
    from skimage.segmentation import watershed

    if nlevels < 1:
        raise ValueError(f'nlevels must be >= 1, got "{nlevels}"')
    if contrast < 0 or contrast > 1:
        raise ValueError(f'contrast must be >= 0 and <= 1, got "{contrast}"')

    segm_mask = (segment_img.data > 0)
    source_values = data[segm_mask]
    source_sum = float(np.nansum(source_values))
    source_min = np.nanmin(source_values)
    source_max = np.nanmax(source_values)
    if source_min == source_max:
        return segment_img  # no deblending

    if mode == 'exponential' and source_min < 0:
        warnings.warn(f'Source "{segment_img.labels[0]}" contains negative '
                      'values, setting deblending mode to "linear"',
                      AstropyUserWarning)
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
        raise ValueError(f'"{mode}" is an invalid mode; mode must be '
                         '"exponential" or "linear"')

    # suppress NoDetectionsWarning during deblending
    warnings.filterwarnings('ignore', category=NoDetectionsWarning)

    mask = ~segm_mask
    segments = _detect_sources(data, thresholds, npixels=npixels,
                               connectivity=connectivity, mask=mask,
                               deblend_skip=True)

    selem = _make_binary_structure(data.ndim, connectivity)

    # define the sources (markers) for the watershed algorithm
    nsegments = len(segments)
    if nsegments == 0:  # no deblending
        return segment_img
    else:
        for i in range(nsegments - 1):
            segm_lower = segments[i].data
            segm_upper = segments[i + 1].data
            relabel = False
            # if the are more sources at the upper level, then
            # remove the parent source(s) from the lower level,
            # but keep any sources in the lower level that do not have
            # multiple children in the upper level
            for label in segments[i].labels:
                mask = (segm_lower == label)
                # checks for 1-to-1 label mapping n -> m (where m >= 0)
                upper_labels = segm_upper[mask]
                upper_labels = np.unique(upper_labels[upper_labels != 0])
                if upper_labels.size >= 2:
                    relabel = True
                    segm_lower[mask] = segm_upper[mask]

            if relabel:
                segm_new = object.__new__(SegmentationImage)
                segm_new._data = ndilabel(segm_lower, structure=selem)[0]
                segments[i + 1] = segm_new
            else:
                segments[i + 1] = segments[i]

        # Deblend using watershed.  If any sources do not meet the
        # contrast criterion, then remove the faintest such source and
        # repeat until all sources meet the contrast criterion.
        markers = segments[-1].data
        mask = segment_img.data.astype(bool)
        remove_marker = True
        while remove_marker:
            markers = watershed(-data, markers, mask=mask, connectivity=selem)

            labels = np.unique(markers[markers != 0])
            flux_frac = np.array([np.sum(data[markers == label])
                                  for label in labels]) / source_sum
            remove_marker = any(flux_frac < contrast)

            if remove_marker:
                # remove only the faintest source (one at a time)
                # because several faint sources could combine to meet the
                # contrast criterion
                markers[markers == labels[np.argmin(flux_frac)]] = 0.

        segm_new = object.__new__(SegmentationImage)
        segm_new._data = markers
        return segm_new
