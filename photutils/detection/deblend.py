# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Functions for detecting sources in an astronomical image."""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from copy import deepcopy
import numpy as np
from astropy.table import Column, Table
from astropy.convolution import Kernel2D
from astropy.stats import sigma_clipped_stats
from .core import _convolve_data, detect_sources
from ..morphology import cutout_footprint, fit_2dgaussian
from ..segmentation import remove_segments
from ..utils.wcs_helpers import pixel_to_icrs_coords


__all__ = ['deblend_source']


def deblend_source(data, segm, label, threshold, filter_kernel=None,
                   nlevels=32, contrast=0.001, mode='exponential',
                   connectivity=8):
    """
    Deblend a single source segment.

    data : `~numpy.ndarray`
        data should be smoothed by the same filter used in ``detect_sources``.

    mode : {'exponential', 'linear'}
    """

    from skimage.morphology import watershed

    mask = (segm == label)
    source = data * mask
    source_max = np.max(source)
    source_sum = float(np.sum(source))

    steps = np.arange(1., nlevels+1)
    if mode == 'exponential':
        thresholds = threshold * (source_max / threshold)**(steps / nlevels)
    elif mode == 'linear':
        thresholds = threshold + ((source_max - threshold) / nlevels) * steps
    else:
        raise ValueError('"{0}" is an invalid mode; mode must be '
                         '"exponential" or "linear"')

    data = _convolve_data(data, filter_kernel, mode='constant',
                          fill_value=0.0)

    # create top-down tree of local peaks
    segm_tree = []
    all_segm = []
    for level in thresholds[::-1]:
        segm_tmp = detect_sources(data, level, npixels=1,
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
        return
    elif nbranch == 1:
        return
    else:
        for j in np.arange(nbranch-1):
            intersection = (segm_tree[j] * segm_tree[j + 1]).astype(bool)
            labels = np.unique(segm_tree[j + 1][intersection])
            print(labels, len(labels), np.max(segm_tree[j + 1]))

            if np.max(segm_tree[j]) > len(labels):
                # remove intersecting labels in lower level, if higher level
                # has more peaks.  Note that segm_tree will not be
                # labeled properly until the end.
                segm_tmp = remove_segments(segm_tree[j+1], labels,
                                           relabel=False)
                segm_tree[j] = segm_tree[j] + segm_tmp

                print(j, 'need to remove labeled image in intersections')
            else:
                print(j, 'no change')

        semg_deblend = watershed(-data, segm_tree[0], mask=segm)
        return segm_deblend, all_segm, segm_tree_orig, segm_tree
