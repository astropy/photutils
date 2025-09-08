# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Define tools for deblending overlapping sources labeled in a
segmentation image.
"""

import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from functools import partial
from multiprocessing import cpu_count, get_context

import numpy as np
from astropy.units import Quantity
from astropy.utils import lazyproperty
from astropy.utils.exceptions import AstropyUserWarning
from scipy.ndimage import label as ndi_label
from scipy.ndimage import sum_labels

from photutils.segmentation.core import SegmentationImage
from photutils.segmentation.detect import _detect_sources
from photutils.segmentation.utils import _make_binary_structure
from photutils.utils._progress_bars import add_progress_bar, tqdm
from photutils.utils._stats import nanmax, nanmin, nansum

__all__ = ['deblend_sources']


@dataclass
class _DeblendParams:
    npixels: int
    footprint: np.ndarray
    nlevels: int
    contrast: float
    mode: str


def deblend_sources(data, segment_img, npixels, *, labels=None, nlevels=32,
                    contrast=0.001, mode='exponential', connectivity=8,
                    relabel=True, nproc=1, progress_bar=True):
    """
    Deblend overlapping sources labeled in a segmentation image.

    Sources are deblended using a combination of multi-thresholding and
    `watershed segmentation
    <https://en.wikipedia.org/wiki/Watershed_(image_processing)>`_. In
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
        The label numbers to deblend. If `None` (default), then all
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
        be slower than serial processing if only a small number of
        sources are to be deblended. The benefits of multiprocessing
        require ~1000 or more sources to deblend, with larger gains as
        the number of sources increase.

    progress_bar : bool, optional
        Whether to display a progress bar. If ``nproc = 1``, then the
        ID shown after the progress bar is the source label being
        deblended. If multiprocessing is used (``nproc > 1``), the ID
        shown is the last source label that was deblended. The progress
        bar requires that the `tqdm <https://tqdm.github.io/>`_ optional
        dependency be installed.

    Returns
    -------
    segment_image : `~photutils.segmentation.SegmentationImage`
        A segmentation image, with the same shape as ``data``, where
        sources are marked by different positive integer values. A value
        of zero is reserved for the background.

    See Also
    --------
    :func:`photutils.segmentation.detect_sources`
    :class:`photutils.segmentation.SourceFinder`
    """
    if isinstance(data, Quantity):
        data = data.value

    if not isinstance(segment_img, SegmentationImage):
        msg = 'segment_img must be a SegmentationImage'
        raise TypeError(msg)

    if segment_img.shape != data.shape:
        msg = 'segment_img must have the same shape as data'
        raise ValueError(msg)

    if nlevels < 1:
        msg = 'nlevels must be >= 1'
        raise ValueError(msg)
    if contrast < 0 or contrast > 1:
        msg = 'contrast must be >= 0 and <= 1'
        raise ValueError(msg)

    if contrast == 1:  # no deblending
        return segment_img.copy()

    if mode not in ('exponential', 'linear', 'sinh'):
        msg = 'mode must be "exponential", "linear", or "sinh"'
        raise ValueError(msg)

    if labels is None:
        labels = segment_img.labels
    else:
        labels = np.atleast_1d(labels)
        segment_img.check_labels(labels)

    # include only sources that have at least (2 * npixels);
    # this is required for a source to be deblended into multiple
    # sources, each with a minimum of npixels
    mask = (segment_img.areas[segment_img.get_indices(labels)]
            >= (npixels * 2))
    labels = labels[mask]

    footprint = _make_binary_structure(data.ndim, connectivity)
    deblend_params = _DeblendParams(npixels, footprint, nlevels, contrast,
                                    mode)

    segm_deblended = segment_img.data.copy()
    label_indices = segment_img.get_indices(labels)

    if nproc is None:
        nproc = cpu_count()  # pragma: no cover

    deblend_label_map = {}
    max_label = segment_img.max_label
    if nproc == 1:
        if progress_bar:  # pragma: no cover
            desc = 'Deblending'
            label_indices = add_progress_bar(label_indices, desc=desc)

        nonposmin_labels = []
        nmarkers_labels = []
        for label, label_idx in zip(labels, label_indices, strict=True):
            if not isinstance(label_indices, np.ndarray):
                label_indices.set_postfix_str(f'ID: {label}')
            source_slice = segment_img.slices[label_idx]
            source_data = data[source_slice]
            source_segment = segment_img.data[source_slice]
            source_deblended, warns = _deblend_source(source_data,
                                                      source_segment,
                                                      label,
                                                      deblend_params)

            if warns:
                if 'nonposmin' in warns:
                    nonposmin_labels.append(label)
                if 'nmarkers' in warns:
                    nmarkers_labels.append(label)

            if source_deblended is not None:
                source_mask = source_deblended > 0
                new_segm = source_deblended[source_mask]  # min label = 1
                segm_deblended[source_slice][source_mask] = (
                    new_segm + max_label)
                new_labels = _get_labels(new_segm) + max_label
                deblend_label_map[label] = new_labels
                max_label += len(new_labels)

    else:
        # Use multiprocessing to deblend sources

        # Prepare the arguments for the worker function
        all_source_data = []
        all_source_segments = []
        all_source_slices = []
        for label_idx in label_indices:
            source_slice = segment_img.slices[label_idx]
            source_data = data[source_slice]
            source_segment = segment_img.data[source_slice]
            all_source_data.append(source_data)
            all_source_segments.append(source_segment)
            all_source_slices.append(source_slice)

        args_all = zip(all_source_data, all_source_segments, labels,
                       strict=True)

        # Create a partial function to pass the deblend_params to the
        # worker function
        worker = partial(_deblend_source, deblend_params=deblend_params)

        # Prepare to store futures and results to preserve the input
        # order of the labels when using as_completed()
        futures_dict = {}
        results = [None] * len(labels)

        disable_pbar = not progress_bar
        mp_context = get_context('spawn')
        with ProcessPoolExecutor(mp_context=mp_context,
                                 max_workers=nproc) as executor:
            # Submit all jobs at once
            for index, args in enumerate(args_all):
                futures_dict[executor.submit(worker, *args)] = index

            with tqdm(total=len(labels), desc='Deblending',
                      disable=disable_pbar) as pbar:
                # Process the results as they are completed
                for future in as_completed(futures_dict):
                    pbar.update(1)
                    idx = futures_dict[future]
                    pbar.set_postfix_str(f'ID: {labels[idx]}')
                    results[idx] = future.result()

        # Process the results
        nonposmin_labels = []
        nmarkers_labels = []
        for label, source_slice, source_deblended in zip(labels,
                                                         all_source_slices,
                                                         results, strict=True):
            source_deblended, warns = source_deblended

            if warns:
                if 'nonposmin' in warns:
                    nonposmin_labels.append(label)
                if 'nmarkers' in warns:
                    nmarkers_labels.append(label)

            if source_deblended is not None:
                source_mask = source_deblended > 0
                new_segm = source_deblended[source_mask]  # min label = 1
                segm_deblended[source_slice][source_mask] = (
                    new_segm + max_label)
                new_labels = _get_labels(new_segm) + max_label
                deblend_label_map[label] = new_labels
                max_label += len(new_labels)

    # process any warnings during deblending
    warning_info = {}
    if nonposmin_labels or nmarkers_labels:
        msg = ('The deblending mode of one or more source labels from the '
               f'input segmentation image was changed from "{mode}" to '
               '"linear". See the "info" attribute for the list of affected '
               'input labels.')
        warnings.warn(msg, AstropyUserWarning)

        if nonposmin_labels:
            nonposmin_labels = np.array(nonposmin_labels)
            msg = (f'Deblending mode changed from {mode} to linear due to '
                   'non-positive minimum data values.')
            warn = {'message': msg, 'input_labels': nonposmin_labels}
            warning_info['nonposmin'] = warn

        if nmarkers_labels:
            nmarkers_labels = np.array(nmarkers_labels)
            msg = (f'Deblending mode changed from {mode} to linear due to '
                   'too many potential deblended sources.')
            warn = {'message': msg, 'input_labels': nmarkers_labels}
            warning_info['nmarkers'] = warn

    if relabel:
        relabel_map = _create_relabel_map(segm_deblended, start_label=1)
        if relabel_map is not None:
            segm_deblended = relabel_map[segm_deblended]
            deblend_label_map = _update_deblend_label_map(deblend_label_map,
                                                          relabel_map)

    segm_img = object.__new__(SegmentationImage)
    segm_img._data = segm_deblended
    segm_img._deblend_label_map = deblend_label_map

    # store the warnings in the output SegmentationImage info attribute
    if warning_info:
        segm_img.info = {'warnings': warning_info}

    return segm_img


def _deblend_source(data, segment_data, label, deblend_params):
    """
    Convenience function to deblend a single labeled source.
    """
    deblender = _SingleSourceDeblender(data, segment_data, label,
                                       deblend_params)
    return deblender.deblend_source(), deblender.warnings


class _SingleSourceDeblender:
    """
    Class to deblend a single labeled source.

    Parameters
    ----------
    data : 2D `~numpy.ndarray`
        The cutout data array for a single source. ``data`` should
        also already be smoothed by the same filter used in
        :func:`~photutils.segmentation.detect_sources`, if applicable.

    segment_data : 2D int `~numpy.ndarray`
        The cutout segmentation image for a single source. Must have the
        same shape as ``data``.

    label : int
        The label of the source to deblend. This is needed because there
        may be more than one source label within the cutout.

    npixels : int
        The number of connected pixels, each greater than ``threshold``,
        that an object must have to be detected. ``npixels`` must be a
        positive integer.

    nlevels : int
        The number of multi-thresholding levels to use. Each source
        will be re-thresholded at ``nlevels`` levels spaced between its
        minimum and maximum values within the source segment. See the
        ``mode`` keyword for how the levels are spaced.

    contrast : float
        The fraction of the total (blended) source flux that a local
        peak must have (at any one of the multi-thresholds) to be
        considered as a separate object. ``contrast`` must be between 0
        and 1, inclusive. If ``contrast = 0`` then every local peak will
        be made a separate object (maximum deblending). If ``contrast =
        1`` then no deblending will occur. The default is 0.001, which
        will deblend sources with a 7.5 magnitude difference.

    mode : {'exponential', 'linear', 'sinh'}
        The mode used in defining the spacing between the
        multi-thresholding levels (see the ``nlevels`` keyword).

    Returns
    -------
    segment_image : `~photutils.segmentation.SegmentationImage`
        A segmentation image, with the same shape as ``data``, where
        sources are marked by different positive integer values. A value
        of zero is reserved for the background. Note that the returned
        `SegmentationImage` will have consecutive labels starting with
        1.
    """

    def __init__(self, data, segment_data, label, deblend_params):
        self.data = data
        self.segment_data = segment_data
        self.label = label
        self.npixels = deblend_params.npixels
        self.footprint = deblend_params.footprint
        self.nlevels = deblend_params.nlevels
        self.contrast = deblend_params.contrast
        self.mode = deblend_params.mode

        self.segment_mask = segment_data == label
        data_values = data[self.segment_mask]
        self.source_min = nanmin(data_values)
        self.source_max = nanmax(data_values)
        self.source_sum = nansum(data_values)
        self.warnings = {}

    @lazyproperty
    def linear_thresholds(self):
        """
        Linearly spaced thresholds between the source minimum and
        maximum (inclusive).

        The source min/max are excluded later, giving nlevels thresholds
        between min and max (noninclusive).
        """
        return np.linspace(self.source_min, self.source_max, self.nlevels + 2)

    @lazyproperty
    def normalized_thresholds(self):
        """
        Normalized thresholds (from 0 to 1) between the source minimum
        and maximum (inclusive).
        """
        return ((self.linear_thresholds - self.source_min)
                / (self.source_max - self.source_min))

    def compute_thresholds(self):
        """
        Compute the multi-level detection thresholds for the source.

        Returns
        -------
        thresholds : 1D `~numpy.ndarray`
            The multi-level detection thresholds for the source.
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
            thresholds = self.normalized_thresholds
            thresholds = np.sinh(thresholds / a) / np.sinh(1.0 / a)
            thresholds *= (maxval - minval)
            thresholds += minval
        elif self.mode == 'exponential':
            minval = self.source_min
            maxval = self.source_max
            thresholds = self.normalized_thresholds
            thresholds = minval * (maxval / minval) ** thresholds

        return thresholds[1:-1]  # do not include source min and max

    def multithreshold(self):
        """
        Perform multithreshold detection for each source.

        This method is useful for debugging and testing.

        Parameters
        ----------
        deblend_mode : bool, optional
            If `True` then only segmentation images with more than one
            label will be returned. If `False` then all segmentation
            images will be returned.

        Returns
        -------
        segments : list of 2D `~numpy.ndarray`
            A list of segmentation images, one for each threshold.
            Only segmentation images with more than one label will be
            returned.
        """
        thresholds = self.compute_thresholds()
        segms = []
        for threshold in thresholds:
            segm = _detect_sources(self.data, threshold, self.npixels,
                                   self.footprint, self.segment_mask,
                                   relabel=False, return_segmimg=False)
            segms.append(segm)
        return segms

    def make_markers(self, return_all=False):
        """
        Make markers (possible sources) for the watershed algorithm.

        Parameters
        ----------
        return_all : bool, optional
            If `False` then return only the final segmentation marker
            image. If `True` then return all segmentation marker images.
            This keyword is useful for debugging and testing.

        Returns
        -------
        markers : 2D `~numpy.ndarray` or list of 2D `~numpy.ndarray`
            A segmentation image that contain markers for possible
            sources. If ``return_all=True`` then a list of all
            segmentation marker images is returned. `None` is returned
            if there is only one source at every threshold.
        """
        thresholds = self.compute_thresholds()
        segm_lower = _detect_sources(self.data, thresholds[0], self.npixels,
                                     self.footprint, self.segment_mask,
                                     relabel=False, return_segmimg=False)

        if return_all:
            all_segms = [segm_lower]

        for threshold in thresholds[1:]:
            segm_upper = _detect_sources(self.data, threshold, self.npixels,
                                         self.footprint, self.segment_mask,
                                         relabel=False, return_segmimg=False)
            if segm_upper is None:  # 0 or 1 labels
                continue

            segm_lower = self.make_marker_segment(segm_lower, segm_upper)

            if return_all:
                all_segms.append(segm_lower)

        if return_all:
            return all_segms

        return segm_lower

    def make_marker_segment(self, segment_lower, segment_upper):
        """
        Make markers (possible sources) for the watershed algorithm.

        Parameters
        ----------
        segment_lower : 2D `~numpy.ndarray`
            The "lower" threshold level segmentation image.

        segment_upper : 2D `~numpy.ndarray`
            The next-highest threshold level segmentation image.

        Returns
        -------
        markers : 2D `~numpy.ndarray`
            A segmentation image that contain markers for possible
            sources.

        Notes
        -----
        For a given label in the lower level, find the labels in the
        upper level (higher threshold value) that are its children
        (i.e., the labels within the same mask as the lower level). If
        there are multiple children, then the lower-level parent label
        is replaced by its children. Parent labels that do not have
        multiple children in the upper level are kept as is (maximizing
        the marker size).
        """
        if segment_lower is None:
            return segment_upper

        labels = _get_labels(segment_lower)
        new_markers = False
        markers = segment_lower.astype(bool)
        for label in labels:
            mask = (segment_lower == label)
            # find label mapping from the lower to upper level
            upper_labels = _get_labels(segment_upper[mask])
            if upper_labels.size >= 2:  # new child markers found
                new_markers = True
                markers[mask] = segment_upper[mask].astype(bool)

        if new_markers:
            # convert bool markers to integer labels
            return ndi_label(markers, structure=self.footprint)[0]

        return segment_lower

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
            that the source labels may not be consecutive if a label was
            removed.
        """
        from skimage.segmentation import watershed

        # Deblend using watershed. If any source does not meet the contrast
        # criterion, then remove the faintest such source and repeat until
        # all sources meet the contrast criterion.
        remove_marker = True
        while remove_marker:
            markers = watershed(-self.data, markers, mask=self.segment_mask,
                                connectivity=self.footprint)

            labels = _get_labels(markers)
            if labels.size == 1:  # only 1 source left
                remove_marker = False
            else:
                flux_frac = (sum_labels(self.data, markers, index=labels)
                             / self.source_sum)
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

        Returns
        -------
        segment_data : 2D int `~numpy.ndarray`
            A 2D int array containing the deblended source labels. The
            source labels are consecutive starting at 1.
        """
        if self.source_min == self.source_max:  # no deblending
            return None

        # define the markers (possible sources) for the watershed algorithm
        markers = self.make_markers()
        if markers is None:
            return None

        # If there are too many markers (e.g., due to low threshold
        # and/or small npixels), the watershed step can be very slow
        # (the threshold of 200 is arbitrary, but seems to work well).
        # This mostly affects the "exponential" mode, where there are
        # many levels at low thresholds, so here we try again with
        # "linear" mode.
        nlabels = len(_get_labels(markers))
        if self.mode != 'linear' and nlabels > 200:
            del markers  # free memory
            self.warnings['nmarkers'] = 'too many markers'
            self.mode = 'linear'
            markers = self.make_markers()
            if markers is None:
                return None

        # deblend using the watershed algorithm using the markers as seeds
        markers = self.apply_watershed(markers)

        if not np.array_equal(self.segment_mask, markers.astype(bool)):
            msg = (f'Deblending failed for source {self.label!r}. '
                   'Please ensure you used the same pixel connectivity '
                   'in detect_sources and deblend_sources.')
            raise ValueError(msg)

        if len(_get_labels(markers)) == 1:  # no deblending
            return None

        # markers may not be consecutive if a label was removed due to
        # the contrast criterion
        relabel_map = _create_relabel_map(markers, start_label=1)
        if relabel_map is not None:
            markers = relabel_map[markers]
        return markers


def _get_labels(array):
    """
    Get the unique labels greater than zero in an array.

    Parameters
    ----------
    array : `~numpy.ndarray`
        The array to get the unique labels from.

    Returns
    -------
    labels : int `~numpy.ndarray`
        The unique labels in the array.
    """
    labels = np.unique(array)
    return labels[labels != 0]


def _create_relabel_map(array, start_label=1):
    """
    Create a mapping of original labels to new labels that are
    consecutive integers.

    By default, the new labels start from 1.

    Parameters
    ----------
    array : 2D `~numpy.ndarray`
        The 2D array to relabel.

    start_label : int, optional
        The starting label number. Must be >= 1. The default is 1.

    Returns
    -------
    relabel_map : 1D `~numpy.ndarray` or None
        The array mapping the original labels to the new labels. If the
        labels are already consecutive starting from ``start_label``,
        then `None` is returned.
    """
    labels = _get_labels(array)

    # check if the labels are already consecutive starting from
    # start_label
    if (labels[0] == start_label
            and (labels[-1] - start_label + 1) == len(labels)):
        return None

    # Create an array to map old labels to new labels
    relabel_map = np.zeros(labels.max() + 1, dtype=array.dtype)
    relabel_map[labels] = np.arange(len(labels)) + start_label

    return relabel_map


def _update_deblend_label_map(deblend_label_map, relabel_map):
    """
    Update the deblend_label_map to reflect the new labels that are
    consecutive integers.

    Parameters
    ----------
    deblend_label_map : dict
        A dictionary mapping the original labels to the new deblended
        labels.

    relabel_map : 1D `~numpy.ndarray`
        The array mapping the original labels to the new labels.

    Returns
    -------
    deblend_label_map : dict
        The updated deblend_label_map.
    """
    for old_label, new_labels in deblend_label_map.items():
        deblend_label_map[old_label] = relabel_map[new_labels]
    return deblend_label_map
