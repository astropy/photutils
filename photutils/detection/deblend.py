# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Functions for detecting sources in an astronomical image."""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from copy import deepcopy
import numpy as np
from .core import _convolve_data, detect_sources
from ..segmentation import remove_segments


__all__ = ['deblend_source']


def deblend_source(data, segm, label, threshold, filter_kernel=None,
                   nlevels=32, contrast=0.001, mode='exponential',
                   connectivity=8):
    """
    Deblend a single source segment.

    data : `~numpy.ndarray`
        data should be smoothed by the same filter used in ``detect_sources``.

    threshold : float or 2D array

    mode : {'exponential', 'linear'}
    """

    from skimage.morphology import watershed

    mask = (segm == label)
    source = data[mask]
    source_min = np.min(source)
    source_max = np.max(source)
    if source_min == source_max:
        return segm     # no deblending
    if source_min < 0:
        warning.warn('Source "{0}" contains negative values, setting '
                     'deblending mode="linear"'.format(label))
        mode = 'linear'
    source_sum = float(np.sum(source))

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

    data = _convolve_data(data, filter_kernel, mode='constant',
                          fill_value=0.0)

    # create top-down tree of local peaks
    segm_tree = []
    all_segm = []
    for level in thresholds[::-1]:
        segm_tmp = detect_sources(source, level, npixels=1,
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

    segm_tree_orig = deepcopy(segm_tree)
    nbranch = len(segm_tree)
    if nbranch == 0:
        return segm, None, None, None
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
        return segm_deblend, all_segm, segm_tree_orig, segm_tree
