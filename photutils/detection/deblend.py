# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Functions for detecting sources in an astronomical image."""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from copy import deepcopy
import warnings
import numpy as np
from astropy import log
from astropy.utils.exceptions import AstropyUserWarning
from .core import _convolve_data, detect_sources
from ..segmentation import remove_segments


__all__ = ['deblend_sources']


def deblend_sources(data, segment_image, npixels, labels=None,
                    filter_kernel=None,
                    nlevels=32, contrast=0.001, mode='exponential',
                    connectivity=8):

    from scipy import ndimage

    nsources = np.max(segment_image)
    if labels is None:
        labels = np.arange(1, nsources+1)

    label_slices = ndimage.find_objects(segment_image)

    data = _convolve_data(data, filter_kernel, mode='constant',
                          fill_value=0.0)

    segm_deblended = deepcopy(segment_image)
    new_label = nsources
    for label in labels:
        #log.info('Deblending source {0}'.format(label))
        if label > nsources:
            raise ValueError('label "{0}" is not in the input segmentation '
                             'image'.format(label))
        source_slice = label_slices[label - 1]
        if source_slice is None:
            raise ValueError('label "{0}" is not in the input segmentation '
                             'image'.format(label))
        source_data = data[source_slice]
        source_segm = segment_image[source_slice]

        source_segm_deblended, nsources_deblended, a, b, c = _deblend_source(
            source_data, source_segm, npixels, label, nlevels=nlevels,
            contrast=contrast, mode=mode, connectivity=connectivity)

        if nsources_deblended > 1:
            print(label, 'deblended, nsources_deblended={0}'.format(
                nsources_deblended))
            source_mask = (source_segm_deblended > 0)
            segm_deblended[source_slice] = (source_segm_deblended +
                                            new_label) * source_mask
            new_label += nsources_deblended

    return segm_deblended, a, b, c


def _deblend_source(data, segm, npixels, label, nlevels=32, contrast=0.001,
                    mode='exponential', connectivity=8):
    """
    Deblend a single source segment.

    data : `~numpy.ndarray`
        data should be smoothed by the same filter used in ``detect_sources``.

    mode : {'exponential', 'linear'}
    """

    from skimage.morphology import watershed

    mask = (segm > 0)
    source_values = data[mask]
    source_min = np.min(source_values)
    source_max = np.max(source_values)
    if source_min == source_max:
        return segm     # no deblending
    if source_min < 0:
        warnings.warn('Source "{0}" contains negative values, setting '
                      'deblending mode="linear"'.format(label),
                      AstropyUserWarning)
        mode = 'linear'
    source_sum = float(np.sum(source_values))

    steps = np.arange(1., nlevels+1)
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
        all_segm.append(segm_tmp)
        npeaks = np.max(segm_tmp)
        if npeaks >= 2:
            fluxes = []
            for i in np.arange(1, npeaks+1):
                fluxes.append(np.sum(data[segm_tmp == i]))
            idx = np.where((np.array(fluxes) / source_sum) >= contrast)[0]
            if len(idx >= 2):
                segm_tree.append(segm_tmp)

    segm_tree_orig = deepcopy(segm_tree)     # tmp
    nbranch = len(segm_tree)
    if nbranch == 0:
        return segm, 1, None, None, None
    else:
        for j in np.arange(1, nbranch):
            intersection = (segm_tree[j] * segm_tree[j - 1]).astype(bool)
            labels = np.unique(segm_tree[j][intersection])

            if np.max(segm_tree[j-1]) > len(labels):
                # If higher tree level has more peaks, then remove the
                # intersecting labels in the lower level, add to the higher
                # level, and relabel.
                segm_tmp = remove_segments(segm_tree[j], labels,
                                           relabel=False)
                segm_tree[j] = segm_tree[j-1] + segm_tmp
                segm_tree[j] = detect_sources(segm_tree[j], 0, npixels=1,
                                              connectivity=connectivity)

        segm_deblend = watershed(-data, segm_tree[-1], mask=segm)
        return (segm_deblend, np.max(segm_deblend), all_segm,
                segm_tree_orig, segm_tree)
