# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
This module provides tools for deblending overlapping sources labeled in
a segmentation image.
"""

import warnings
from multiprocessing import cpu_count, get_context

import numpy as np
from astropy.units import Quantity
from astropy.utils.exceptions import AstropyUserWarning

from photutils.segmentation.core import SegmentationImage
from photutils.segmentation.detect import _detect_sources
from photutils.segmentation.utils import _make_binary_structure
from photutils.utils._progress_bars import add_progress_bar

__all__ = ['deblend_sources']


def deblend_sources(data, segment_img, npixels, *, labels=None, nlevels=32,
                    contrast=0.001, mode='exponential', connectivity=8,
                    relabel=True, nproc=1, progress_bar=True):
    """
    Deblend overlapping sources labeled in a segmentation image.

    Sources are deblended using a combination of multi-thresholding and
    `watershed segmentation
    <https://en.wikipedia.org/wiki/Watershed_(image_processing)>`_.  In
    order to deblend sources, there must be a saddle between them.

    Parameters
    ----------
    data : 2D `~numpy.ndarray`
        The 2D array of the image. If filtering is desired, please input
        a convolved image here. This array should be the same array used
        in `~photutils.segmentation.detect_sources`.

    segment_img : `~photutils.segmentation.SegmentationImage`
        The segmentation image to deblend.

    npixels : int
        The minimum number of connected pixels, each greater than
        ``threshold``, that an object must have to be deblended.
        ``npixels`` must be a positive integer.

    labels : int or array_like of int, optional
        The label numbers to deblend.  If `None` (default), then all
        labels in the segmentation image will be deblended.

    nlevels : int, optional
        The number of multi-thresholding levels to use for deblending.
        Each source will be re-thresholded at ``nlevels`` levels spaced
        between its minimum and maximum values (non-inclusive). The
        ``mode`` keyword determines how the levels are spaced.

    contrast : float, optional
        The fraction of the total source flux that a local peak must
        have (at any one of the multi-thresholds) to be deblended
        as a separate object. ``contrast`` must be between 0 and 1,
        inclusive. If ``contrast=0`` then every local peak will be made
        a separate object (maximum deblending). If ``contrast=1`` then
        no deblending will occur. The default is 0.001, which will
        deblend sources with a 7.5 magnitude difference.

    mode : {'exponential', 'linear', 'sinh'}, optional
        The mode used in defining the spacing between the
        multi-thresholding levels (see the ``nlevels`` keyword) during
        deblending. The ``'exponential'`` and ``'sinh'`` modes have
        more threshold levels near the source minimum and less near
        the source maximum. The ``'linear'`` mode evenly spaces the
        threshold levels between the source minimum and maximum.
        The ``'exponential'`` and ``'sinh'`` modes differ in that
        the ``'exponential'`` levels are dependent on the source
        maximum/minimum ratio (smaller ratios are more linear; larger
        ratios are more exponential), while the ``'sinh'`` levels
        are not. Also, the ``'exponential'`` mode will be changed to
        ``'linear'`` for sources with non-positive minimum data values.

    connectivity : {8, 4}, optional
        The type of pixel connectivity used in determining how pixels
        are grouped into a detected source. The options are 8 (default)
        or 4. 8-connected pixels touch along their edges or corners.
        4-connected pixels touch along their edges. The ``connectivity``
        must be the same as that used to create the input segmentation
        image.

    relabel : bool, optional
        If `True` (default), then the segmentation image will be
        relabeled such that the labels are in consecutive order starting
        from 1.

    nproc : int, optional
        The number of processes to use for multiprocessing (if larger
        than 1). If set to 1, then a serial implementation is used
        instead of a parallel one. If `None`, then the number of
        processes will be set to the number of CPUs detected on the
        machine. Please note that due to overheads, multiprocessing may
        be slower than serial processing. This is especially true if one
        only has a small number of sources to deblend. The benefits of
        multiprocessing require ~1000 or more sources to deblend, with
        larger gains as the number of sources increase.

    progress_bar : bool, optional
        Whether to display a progress bar. Note that if multiprocessing
        is used (``nproc > 1``), the estimation times (e.g., time per
        iteration and time remaining, etc) may be unreliable. The
        progress bar requires that the `tqdm <https://tqdm.github.io/>`_
        optional dependency be installed. Note that the progress
        bar does not currently work in the Jupyter console due to
        limitations in ``tqdm``.

    Returns
    -------
    segment_image : `~photutils.segmentation.SegmentationImage`
        A segmentation image, with the same shape as ``data``, where
        sources are marked by different positive integer values.  A
        value of zero is reserved for the background.

    See Also
    --------
    :func:`photutils.segmentation.detect_sources`
    :class:`photutils.segmentation.SourceFinder`
    """
    if isinstance(data, Quantity):
        data = data.value

    if not isinstance(segment_img, SegmentationImage):
        raise ValueError('segment_img must be a SegmentationImage')

    if segment_img.shape != data.shape:
        raise ValueError('The data and segmentation image must have '
                         'the same shape')

    if nlevels < 1:
        raise ValueError('nlevels must be >= 1')
    if contrast < 0 or contrast > 1:
        raise ValueError('contrast must be >= 0 and <= 1')

    if contrast == 1:  # no deblending
        return segment_img.copy()

    if mode not in ('exponential', 'linear', 'sinh'):
        raise ValueError('mode must be "exponential", "linear", or "sinh"')

    if labels is None:
        labels = segment_img.labels
    else:
        labels = np.atleast_1d(labels)
        segment_img.check_labels(labels)

    # include only sources that have at least (2 * npixels);
    # this is required for it to be deblended into multiple sources,
    # each with a minimum of npixels
    mask = (segment_img.areas[segment_img.get_indices(labels)]
            >= (npixels * 2))
    labels = labels[mask]

    footprint = _make_binary_structure(data.ndim, connectivity)

    if nproc is None:
        nproc = cpu_count()  # pragma: no cover

    segm_deblended = object.__new__(SegmentationImage)
    segm_deblended._data = np.copy(segment_img.data)
    last_label = segment_img.max_label
    indices = segment_img.get_indices(labels)

    all_source_data = []
    all_source_segments = []
    all_source_slices = []
    for label, idx in zip(labels, indices):
        source_slice = segment_img.slices[idx]
        source_data = data[source_slice]
        source_segment = object.__new__(SegmentationImage)
        source_segment._data = segment_img.data[source_slice]
        source_segment.keep_labels(label)  # include only one label
        all_source_data.append(source_data)
        all_source_segments.append(source_segment)
        all_source_slices.append(source_slice)

    if nproc == 1:
        if progress_bar:
            desc = 'Deblending'
            all_source_data = add_progress_bar(all_source_data, desc=desc)  # pragma: no cover

        all_source_deblends = []
        for source_data, source_segment in zip(all_source_data,
                                               all_source_segments):
            deblender = _Deblender(source_data, source_segment, npixels,
                                   footprint, nlevels, contrast, mode)
            source_deblended = deblender.deblend_source()
            all_source_deblends.append(source_deblended)

    else:
        nlabels = len(labels)
        args_all = zip(all_source_data, all_source_segments,
                       (npixels,) * nlabels, (footprint,) * nlabels,
                       (nlevels,) * nlabels, (contrast,) * nlabels,
                       (mode,) * nlabels)

        if progress_bar:
            desc = 'Deblending'
            args_all = add_progress_bar(args_all, total=nlabels, desc=desc)  # pragma: no cover

        with get_context('spawn').Pool(processes=nproc) as executor:
            all_source_deblends = executor.starmap(_deblend_source, args_all)

    nonposmin_labels = []
    nmarkers_labels = []
    for (label, source_deblended, source_slice) in zip(
            labels, all_source_deblends, all_source_slices):

        if source_deblended is not None:
            # replace the original source with the deblended source
            segment_mask = (source_deblended.data > 0)
            segm_deblended._data[source_slice][segment_mask] = (
                source_deblended.data[segment_mask] + last_label)
            last_label += source_deblended.nlabels

            if hasattr(source_deblended, 'warnings'):
                if source_deblended.warnings.get('nonposmin',
                                                 None) is not None:
                    nonposmin_labels.append(label)
                if source_deblended.warnings.get('nmarkers',
                                                 None) is not None:
                    nmarkers_labels.append(label)

    if nonposmin_labels or nmarkers_labels:
        segm_deblended.info = {'warnings': {}}
        warnings.warn('The deblending mode of one or more source labels from '
                      'the input segmentation image was changed from '
                      f'"{mode}" to "linear". See the "info" attribute '
                      'for the list of affected input labels.',
                      AstropyUserWarning)

        if nonposmin_labels:
            warn = {'message': f'Deblending mode changed from {mode} to '
                    'linear due to non-positive minimum data values.',
                    'input_labels': np.array(nonposmin_labels)}
            segm_deblended.info['warnings']['nonposmin'] = warn

        if nmarkers_labels:
            warn = {'message': f'Deblending mode changed from {mode} to '
                    'linear due to too many potential deblended sources.',
                    'input_labels': np.array(nmarkers_labels)}
        segm_deblended.info['warnings']['nmarkers'] = warn

    if relabel:
        segm_deblended.relabel_consecutive()

    return segm_deblended


def _deblend_source(source_data, source_segment, npixels, footprint, nlevels,
                    contrast, mode):
    """
    Convenience function to deblend a single labeled source with
    multiprocessing.
    """
    deblender = _Deblender(source_data, source_segment, npixels, footprint,
                           nlevels, contrast, mode)
    return deblender.deblend_source()


class _Deblender:
    """
    Class to deblend a single labeled source.

    Parameters
    ----------
    source_data : 2D `~numpy.ndarray`
        The cutout data array for a single source. ``data`` should
        also already be smoothed by the same filter used in
        :func:`~photutils.segmentation.detect_sources`, if applicable.

    source_segment : `~photutils.segmentation.SegmentationImage`
        A cutout `~photutils.segmentation.SegmentationImage` object with
        the same shape as ``data``. ``segment_img`` should contain only
        *one* source label.

    npixels : int
        The number of connected pixels, each greater than ``threshold``,
        that an object must have to be detected.  ``npixels`` must be a
        positive integer.

    nlevels : int
        The number of multi-thresholding levels to use.  Each source
        will be re-thresholded at ``nlevels`` levels spaced
        exponentially or linearly (see the ``mode`` keyword) between its
        minimum and maximum values within the source segment.

    contrast : float
        The fraction of the total (blended) source flux that a local
        peak must have (at any one of the multi-thresholds) to be
        considered as a separate object.  ``contrast`` must be between 0
        and 1, inclusive.  If ``contrast = 0`` then every local peak
        will be made a separate object (maximum deblending).  If
        ``contrast = 1`` then no deblending will occur.  The default is
        0.001, which will deblend sources with a 7.5 magnitude
        difference.

    mode : {'exponential', 'linear', 'sinh'}
        The mode used in defining the spacing between the
        multi-thresholding levels (see the ``nlevels`` keyword).  The
        default is 'exponential'.

    Returns
    -------
    segment_image : `~photutils.segmentation.SegmentationImage`
        A segmentation image, with the same shape as ``data``, where
        sources are marked by different positive integer values.  A
        value of zero is reserved for the background.  Note that the
        returned `SegmentationImage` will have consecutive labels
        starting with 1.
    """

    def __init__(self, source_data, source_segment, npixels, footprint,
                 nlevels, contrast, mode):

        self.source_data = source_data
        self.source_segment = source_segment
        self.npixels = npixels
        self.footprint = footprint
        self.nlevels = nlevels
        self.contrast = contrast
        self.mode = mode
        self.warnings = {}

        self.segment_mask = source_segment.data.astype(bool)
        self.source_values = source_data[self.segment_mask]
        self.source_min = np.nanmin(self.source_values)
        self.source_max = np.nanmax(self.source_values)
        self.source_sum = np.nansum(self.source_values)
        self.label = source_segment.labels[0]  # should only be 1 label

        # NOTE: this includes the source min/max, but we exclude those
        # later, giving nlevels thresholds between min and max
        # (noninclusive; i.e., nlevels + 1 parts)
        self.linear_thresholds = np.linspace(self.source_min, self.source_max,
                                             self.nlevels + 2)

    def normalized_thresholds(self):
        return ((self.linear_thresholds - self.source_min)
                / (self.source_max - self.source_min))

    def compute_thresholds(self):
        """
        Compute the multi-level detection thresholds for the source.
        """
        if self.mode == 'exponential' and self.source_min <= 0:
            self.warnings['nonposmin'] = 'non-positive minimum'
            self.mode = 'linear'

        if self.mode == 'linear':
            thresholds = self.linear_thresholds
        elif self.mode == 'sinh':
            a = 0.25
            minval = self.source_min
            maxval = self.source_max
            thresholds = self.normalized_thresholds()
            thresholds = np.sinh(thresholds / a) / np.sinh(1.0 / a)
            thresholds *= (maxval - minval)
            thresholds += minval
        elif self.mode == 'exponential':
            minval = self.source_min
            maxval = self.source_max
            thresholds = self.normalized_thresholds()
            thresholds = minval * (maxval / minval) ** thresholds

        return thresholds[1:-1]  # do not include source min and max

    def multithreshold(self):
        """
        Perform multithreshold detection for each source.
        """
        thresholds = self.compute_thresholds()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
            segments = _detect_sources(self.source_data, thresholds,
                                       self.npixels, self.footprint,
                                       self.segment_mask, deblend_mode=True)
        return segments

    def make_markers(self, segments):
        """
        Make markers (possible sources) for the watershed algorithm.

        Parameters
        ----------
        segments : list of `~photutils.segmentation.SegmentationImage`
            A list of segmentation images, one for each threshold.

        Returns
        -------
        markers : list of `~photutils.segmentation.SegmentationImage`
            A list of segmentation images that contain possible sources
            as markers. The last list element contains all of the
            potential source markers.
        """
        from scipy.ndimage import label as ndi_label

        for i in range(len(segments) - 1):
            segm_lower = segments[i].data
            segm_upper = segments[i + 1].data
            markers = segm_lower.astype(bool)
            relabel = False
            # if the are more sources at the upper level, then
            # remove the parent source(s) from the lower level,
            # but keep any sources in the lower level that do not have
            # multiple children in the upper level
            for label in segments[i].labels:
                mask = (segm_lower == label)
                # find label mapping from the lower to upper level
                upper_labels = segm_upper[mask]
                upper_labels = np.unique(upper_labels[upper_labels != 0])
                if upper_labels.size >= 2:
                    relabel = True
                    markers[mask] = segm_upper[mask].astype(bool)

            if relabel:
                segm_data, nlabels = ndi_label(markers,
                                               structure=self.footprint)
                segm_new = object.__new__(SegmentationImage)
                segm_new._data = segm_data
                segm_new.__dict__['labels'] = np.arange(nlabels) + 1
                segments[i + 1] = segm_new
            else:
                segments[i + 1] = segments[i]

        return segments

    def apply_watershed(self, markers):
        """
        Apply the watershed algorithm to the source markers.

        Parameters
        ----------
        markers : list of `~photutils.segmentation.SegmentationImage`
            A list of segmentation images that contain possible sources
            as markers. The last list element contains all of the
            potential source markers.

        Returns
        -------
        segment_data : 2D int `~numpy.ndarray`
            A 2D int array containing the deblended source labels. Note
            that the source labels may not be consecutive.
        """
        from scipy.ndimage import sum_labels
        from skimage.segmentation import watershed

        # all markers are at the top level
        markers = markers[-1].data

        # Deblend using watershed. If any source does not meet the contrast
        # criterion, then remove the faintest such source and repeat until
        # all sources meet the contrast criterion.
        remove_marker = True
        while remove_marker:
            markers = watershed(-self.source_data, markers,
                                mask=self.segment_mask,
                                connectivity=self.footprint)

            labels = np.unique(markers[markers != 0])
            if labels.size == 1:  # only 1 source left
                remove_marker = False
            else:
                flux_frac = sum_labels(self.source_data, markers,
                                       index=labels) / self.source_sum
                remove_marker = any(flux_frac < self.contrast)

                if remove_marker:
                    # remove only the faintest source (one at a time)
                    # because several faint sources could combine to meet
                    # the contrast criterion
                    markers[markers == labels[np.argmin(flux_frac)]] = 0.0

        return markers

    def deblend_source(self):
        """
        Deblend a single labeled source.
        """
        if self.source_min == self.source_max:  # no deblending
            return None

        segments = self.multithreshold()
        if len(segments) == 0:  # no deblending
            return None

        # define the markers (possible sources) for the watershed algorithm
        markers = self.make_markers(segments)

        # If there are too many markers (e.g., due to low threshold
        # and/or small npixels), the watershed step can be very slow
        # (the threshold of 200 is arbitrary, but seems to work well).
        # This mostly affects the "exponential" mode, where there are
        # many levels at low thresholds, so here we try again with
        # "linear" mode.
        if self.mode != 'linear' and markers[-1].nlabels > 200:
            self.warnings['nmarkers'] = 'too many markers'
            self.mode = 'linear'
            segments = self.multithreshold()
            if len(segments) == 0:  # no deblending
                return None
            markers = self.make_markers(segments)

        # deblend using the watershed algorithm using the markers as seeds
        markers = self.apply_watershed(markers)

        if not np.array_equal(self.segment_mask, markers.astype(bool)):
            raise ValueError(f'Deblending failed for source "{self.label}". '
                             'Please ensure you used the same pixel '
                             'connectivity in detect_sources and '
                             'deblend_sources.')

        labels = np.unique(markers[markers != 0])
        if len(labels) == 1:  # no deblending
            return None

        segm_new = object.__new__(SegmentationImage)
        segm_new._data = markers
        segm_new.__dict__['labels'] = labels
        segm_new.relabel_consecutive(start_label=1)

        if self.warnings:
            segm_new.warnings = self.warnings

        return segm_new
