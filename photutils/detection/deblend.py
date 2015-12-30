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


def deblend_sources(data, segment_img, npixels, labels=None,
                    filter_kernel=None,
                    nlevels=32, contrast=0.001, mode='exponential',
                    connectivity=8, relabel=True):

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

        if source_deblended.nlabels > 1:
            # replace the original source with the deblended source
            source_mask = (source_deblended.data > 0)
            segm_deblended._data[source_slice][source_mask] = (
                source_deblended.data[source_mask] + last_label)
            last_label += source_deblended.nlabels

    if relabel:
        segm_deblended.relabel_sequential()
    return segm_deblended


def _deblend_source(data, segment_img, npixels, nlevels=32, contrast=0.001,
                    mode='exponential', connectivity=8):
    """
    Deblend a single labeled source.

    data : `~numpy.ndarray`
        data should be smoothed by the same filter used in ``detect_sources``.

    segment_img : `SegmentationImage`
        A cutout from a segmentation image.  Must contain only **one** non-zero label.

    mode : {'exponential', 'linear'}
    """

    from scipy import ndimage
    from skimage.morphology import watershed

    if nlevels < 1:
        raise ValueError('nlevels must be >= 1, got "{0}"'.format(nlevels))
    if contrast < 0 or contrast > 1:
        raise ValueError('contrast must be >= 0 or <= 1, got '
                         '"{0}"'.format(contrast))

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
    all_segm = []     # tmp

    for level in thresholds[::-1]:
        segm_tmp = detect_sources(data, level, npixels=npixels,
                                  connectivity=connectivity)
        all_segm.append(segm_tmp)   # tmp

        if segm_tmp.nlabels >= 2:
            fluxes = []
            for i in segm_tmp.labels:
                fluxes.append(np.sum(data[segm_tmp == i]))
            idx = np.where((np.array(fluxes) / source_sum) >= contrast)[0]
            if len(idx >= 2):
                segm_tree.append(segm_tmp)

    segm_tree_orig = deepcopy(segm_tree)     # tmp
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
                # If higher tree level has more peaks, then remove the
                # intersecting labels in the lower level, add to the higher
                # level, and relabel.
                segm_tree[j].remove_labels(intersect_labels)
                new_segments = segm_tree[j].data + segm_tree[j - 1].data
                new_segm, nsegm = ndimage.label(new_segments)
                segm_tree[j - 1] = SegmentationImage(new_segm)

        return SegmentationImage(watershed(-data, segm_tree[0].data,
                                           mask=segment_img.data))
        #return result, all_segm, segm_tree_orig, segm_tree
