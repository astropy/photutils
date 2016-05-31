# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Functions for deblending sources."""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from copy import deepcopy
import warnings
import numpy as np
from astropy.utils.exceptions import AstropyUserWarning
from .core import _convolve_data, detect_sources
from ..segmentation import SegmentationImage


__all__ = ['deblend_sources']


def deblend_sources(data, segment_img, npixels, filter_kernel=None,
                    labels=None, nlevels=32, contrast=0.001,
                    mode='exponential', connectivity=8, relabel=True):
    """
    Deblend overlapping sources labeled in a segmentation image.

    Sources are deblended using a combination of multi-thresholding and
    `watershed segmentation
    <https://en.wikipedia.org/wiki/Watershed_(image_processing)>`_.  In
    order to deblend sources, they must be separated enough such that
    there is a saddle between them.

    .. note::
        This function is experimental.  Please report any issues on the
        `Photutils GitHub issue tracker
        <https://github.com/astropy/photutils/issues>`_

    Parameters
    ----------
    data : array_like
        The 2D array of the image.

    segment_img : `~photutils.segmentation.SegmentationImage` or array_like (int)
        A 2D segmentation image, either as a
        `~photutils.segmentation.SegmentationImage` object or an
        `~numpy.ndarray`, with the same shape as ``data`` where sources
        are labeled by different positive integer values.  A value of
        zero is reserved for the background.

    npixels : int
        The number of connected pixels, each greater than ``threshold``,
        that an object must have to be detected.  ``npixels`` must be a
        positive integer.

    filter_kernel : array-like (2D) or `~astropy.convolution.Kernel2D`, optional
        The 2D array of the kernel used to filter the image before
        thresholding.  Filtering the image will smooth the noise and
        maximize detectability of objects with a shape similar to the
        kernel.

    labels : int or array-like of int, optional
        The label numbers to deblend.  If `None` (default), then all
        labels in the segmentation image will be deblended.

    nlevels : int, optional
        The number of multi-thresholding levels to use.  Each source
        will be re-thresholded at ``nlevels``, spaced exponentially or
        linearly (see the ``mode`` keyword), between its minimum and
        maximum values within the source segment.

    contrast : float, optional
        The fraction of the total (blended) source flux that a local
        peak must have to be considered as a separate object.
        ``contrast`` must be between 0 and 1, inclusive.  If ``contrast
        = 0`` then every local peak will be made a separate object
        (maximum deblending).  If ``contrast = 1`` then no deblending
        will occur.  The default is 0.001, which will deblend sources with
        a magnitude differences of about 7.5.

    mode : {'exponential', 'linear'}, optional
        The mode used in defining the spacing between the
        multi-thresholding levels (see the ``nlevels`` keyword).

    connectivity : {4, 8}, optional
        The type of pixel connectivity used in determining how pixels
        are grouped into a detected source.  The options are 4 or 8
        (default).  4-connected pixels touch along their edges.
        8-connected pixels touch along their edges or corners.  For
        reference, SExtractor uses 8-connected pixels.

    relabel : bool
        If `True` (default), then the segmentation image will be
        relabeled such that the labels are in sequential order starting
        from 1.

    Returns
    -------
    segment_image : `~photutils.segmentation.SegmentationImage`
        A 2D segmentation image, with the same shape as ``data``, where
        sources are marked by different positive integer values.  A
        value of zero is reserved for the background.

    See Also
    --------
    :func:`photutils.detection.detect_sources`
    """

    if not isinstance(segment_img, SegmentationImage):
        segment_img = SegmentationImage(segment_img)

    if segment_img.shape != data.shape:
        raise ValueError('The data and segmentation image must have '
                         'the same shape')

    if labels is None:
        labels = segment_img.labels
    labels = np.atleast_1d(labels)

    data = _convolve_data(data, filter_kernel, mode='constant',
                          fill_value=0.0)

    last_label = segment_img.max
    segm_deblended = deepcopy(segment_img)
    for label in labels:
        segment_img.check_label(label)
        source_slice = segment_img.slices[label - 1]
        source_data = data[source_slice]
        source_segm = SegmentationImage(np.copy(
            segment_img.data[source_slice]))
        source_segm.keep_labels(label)    # include only one label
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
            data = segm_deblended.data
            data[source_slice][source_mask] = (
                source_deblended.data[source_mask] + last_label)
            segm_deblended.data = data    # needed to call data setter
            last_label += source_deblended.nlabels

    if relabel:
        segm_deblended.relabel_sequential()
    return segm_deblended


def _deblend_source(data, segment_img, npixels, nlevels=32, contrast=0.001,
                    mode='exponential', connectivity=8):
    """
    Deblend a single labeled source.

    Parameters
    ----------
    data : array_like
        The 2D array of the image.  The should be a cutout for a single
        source.  ``data`` should already be smoothed by the same filter
        used in :func:`~photutils.detect_sources`, if applicable.

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
        will be re-thresholded at ``nlevels``, spaced exponentially or
        linearly (see the ``mode`` keyword), between its minimum and
        maximum values within the source segment.

    contrast : float, optional
        The fraction of the total (blended) source flux that a local
        peak must have to be considered as a separate object.
        ``contrast`` must be between 0 and 1, inclusive.  If ``contrast
        = 0`` then every local peak will be made a separate object
        (maximum deblending).  If ``contrast = 1`` then no deblending
        will occur.  The default is 0.001, which will deblend sources with
        a magnitude differences of about 7.5.

    mode : {'exponential', 'linear'}, optional
        The mode used in defining the spacing between the
        multi-thresholding levels (see the ``nlevels`` keyword).

    connectivity : {4, 8}, optional
        The type of pixel connectivity used in determining how pixels
        are grouped into a detected source.  The options are 4 or 8
        (default).  4-connected pixels touch along their edges.
        8-connected pixels touch along their edges or corners.  For
        reference, SExtractor uses 8-connected pixels.

    Returns
    -------
    segment_image : `~photutils.segmentation.SegmentationImage`
        A 2D segmentation image, with the same shape as ``data``, where
        sources are marked by different positive integer values.  A
        value of zero is reserved for the background.
    """

    from scipy import ndimage
    from skimage.morphology import watershed

    if nlevels < 1:
        raise ValueError('nlevels must be >= 1, got "{0}"'.format(nlevels))
    if contrast < 0 or contrast > 1:
        raise ValueError('contrast must be >= 0 or <= 1, got '
                         '"{0}"'.format(contrast))

    if connectivity == 4:
        selem = ndimage.generate_binary_structure(2, 1)
    elif connectivity == 8:
        selem = ndimage.generate_binary_structure(2, 2)
    else:
        raise ValueError('Invalid connectivity={0}.  '
                         'Options are 4 or 8'.format(connectivity))

    segm_mask = (segment_img.data > 0)
    source_values = data[segm_mask]
    source_min = np.min(source_values)
    source_max = np.max(source_values)
    if source_min == source_max:
        return segment_img     # no deblending
    if source_min < 0:
        warnings.warn('Source "{0}" contains negative values, setting '
                      'deblending mode to "linear"'.format(
                          segment_img.labels[0]), AstropyUserWarning)
        mode = 'linear'
    source_sum = float(np.sum(source_values))

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
                         '"exponential" or "linear"')

    # create top-down tree of local peaks
    segm_tree = []
    for level in thresholds[::-1]:
        segm_tmp = detect_sources(data, level, npixels=npixels,
                                  connectivity=connectivity)
        if segm_tmp.nlabels >= 2:
            fluxes = []
            for i in segm_tmp.labels:
                fluxes.append(np.sum(data[segm_tmp == i]))
            idx = np.where((np.array(fluxes) / source_sum) >= contrast)[0]
            if len(idx >= 2):
                segm_tree.append(segm_tmp)

    nbranch = len(segm_tree)
    if nbranch == 0:
        return segment_img
    else:
        for j in np.arange(nbranch - 1, 0, -1):
            intersect_mask = (segm_tree[j].data *
                              segm_tree[j - 1].data).astype(bool)
            intersect_labels = np.unique(segm_tree[j].data[intersect_mask])

            if segm_tree[j - 1].nlabels <= len(intersect_labels):
                segm_tree[j - 1] = segm_tree[j]
            else:
                # If a higher tree level has more peaks than in the
                # intersected label(s) with the level below, then remove
                # the intersected label(s) in the lower level, add the
                # higher level, and relabel.
                segm_tree[j].remove_labels(intersect_labels)
                new_segments = segm_tree[j].data + segm_tree[j - 1].data
                new_segm, nsegm = ndimage.label(new_segments)
                segm_tree[j - 1] = SegmentationImage(new_segm)

        return SegmentationImage(watershed(-data, segm_tree[0].data,
                                           mask=segment_img.data,
                                           connectivity=selem))
