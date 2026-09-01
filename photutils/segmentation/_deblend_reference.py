# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Pure-Python reference implementation of the deblending pipeline.

This module is a pure-Python mirror of the compiled deblending
pipeline used by :func:`~photutils.segmentation.deblend_sources`,
kept for verification and debugging. It is not used in
production. It must track the compiled (Cython) semantics
exactly, which is enforced by the cross-implementation tests in
``photutils/segmentation/tests/test_deblend.py``.
"""

from functools import cached_property

import numpy as np
from scipy.ndimage import label as ndi_label
from scipy.ndimage import sum_labels

from photutils.segmentation._deblend_markers import make_deblend_markers
from photutils.segmentation._deblend_watershed import deblend_watershed
from photutils.segmentation.core import _get_labels
from photutils.segmentation.deblend import _create_relabel_map
from photutils.utils._stats import nanmax, nanmin, nansum


def _detect_sources_deblend(data, threshold, n_pixels, *, footprint,
                            segment_mask):
    """
    Detect sources for a single multithreshold level during deblending.

    This is the deblending analogue of
    `photutils.segmentation.detect._detect_sources`. It differs in
    that the detected segments keep their (possibly non-consecutive)
    label numbers from `~scipy.ndimage.label`, the small segments are
    removed with a bincount-based area filter (the per-label cutout
    loop used by ``_detect_sources`` has a fixed per-label overhead
    that dominates for the small cutouts and the many calls made
    during deblending), and `None` is returned when fewer than two
    segments are found.

    Parameters
    ----------
    data : 2D `~numpy.ndarray`
        The cutout data array for a single source.

    threshold : float
        The data value to be used for the detection threshold.

    n_pixels : int
        The minimum number of connected pixels, each greater than
        ``threshold``, that an object must have to be detected.

    footprint : array_like
        A footprint that defines feature connections.

    segment_mask : 2D bool `~numpy.ndarray`
        A boolean mask of the source segment, with the same shape as
        ``data``. Pixels outside the segment will not be included in
        any source.

    Returns
    -------
    segment_img : 2D int `~numpy.ndarray` or `None`
        A 2D segmentation image, with the same shape as ``data``,
        where sources are marked by different positive integer
        values. A value of zero is reserved for the background. If
        fewer than two sources are found then `None` is returned.
    """
    # NaN values compare as False, so NaN pixels are never included
    # in any source. The comparison is never empty because the
    # deblending thresholds are strictly below the source maximum.
    segment_img = data > threshold
    segment_img &= segment_mask

    segment_img, n_labels = ndi_label(segment_img, structure=footprint)

    # Remove objects with less than n_pixels
    areas = np.bincount(segment_img.ravel())
    keep = areas >= n_pixels
    keep[0] = False
    n_keep = np.count_nonzero(keep)
    if n_keep <= 1:
        return None

    if n_keep < n_labels:
        label_map = np.where(
            keep, np.arange(areas.size, dtype=segment_img.dtype), 0)
        segment_img = label_map[segment_img]

    return segment_img


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

    deblend_params : `~photutils.segmentation.deblend._DeblendParams`
        The parameters for deblending the source.
    """

    def __init__(self, data, segment_data, label, deblend_params):
        self.data = data
        self.segment_data = segment_data
        self.label = label
        self.n_pixels = deblend_params.n_pixels
        self.footprint = deblend_params.footprint
        self.n_levels = deblend_params.n_levels
        self.contrast = deblend_params.contrast
        self.mode = deblend_params.mode

        self.segment_mask = segment_data == label
        data_values = data[self.segment_mask]
        self.source_min = nanmin(data_values)
        self.source_max = nanmax(data_values)
        self.source_sum = nansum(data_values)
        self.warnings = {}

    @cached_property
    def linear_thresholds(self):
        """
        Linearly spaced thresholds between the source minimum and
        maximum (inclusive).

        The source min/max are excluded later, giving n_levels
        thresholds between min and max (noninclusive).
        """
        return np.linspace(self.source_min, self.source_max, self.n_levels + 2)

    @cached_property
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

        Note that this method has side effects. When the mode is
        "exponential" and the source minimum is non-positive, it
        changes ``self.mode`` to "linear" and records the fallback in
        ``self.warnings``. Later calls (e.g., from ``make_markers``)
        therefore use the fallback mode, mirroring the sticky per-source
        mode fallback in the compiled pipeline.

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

        Returns
        -------
        segments : list of 2D `~numpy.ndarray` or `None`
            A list of segmentation images, one for each threshold.
            `None` is returned for thresholds that do not have more than
            one label.
        """
        thresholds = self.compute_thresholds()
        segms = []
        for threshold in thresholds:
            segm = _detect_sources_deblend(self.data, threshold,
                                           self.n_pixels,
                                           footprint=self.footprint,
                                           segment_mask=self.segment_mask)
            segms.append(segm)
        return segms

    def make_markers(self, *, return_all=False):
        """
        Make markers (possible sources) for the watershed algorithm.

        The markers are built from a single component-tree pass over
        the level-quantized cutout (see
        `~photutils.segmentation._deblend_markers.make_deblend_markers`),
        which produces markers identical to the per-level
        multithreshold construction.

        Parameters
        ----------
        return_all : bool, optional
            If `False` then return only the final segmentation marker
            image. If `True` then compute the markers with the
            per-level reference implementation instead and return all
            segmentation marker images. This keyword is useful for
            debugging and testing.

        Returns
        -------
        markers : 2D `~numpy.ndarray` or list of 2D `~numpy.ndarray`
            A segmentation image that contain markers for possible
            sources. If ``return_all=True`` then a list of all
            segmentation marker images is returned. `None` is returned
            if there is only one source at every threshold.
        """
        thresholds = self.compute_thresholds()

        if return_all:
            segm_lower = _detect_sources_deblend(
                self.data, thresholds[0], self.n_pixels,
                footprint=self.footprint, segment_mask=self.segment_mask)
            all_segms = [segm_lower]
            for threshold in thresholds[1:]:
                segm_upper = _detect_sources_deblend(
                    self.data, threshold, self.n_pixels,
                    footprint=self.footprint,
                    segment_mask=self.segment_mask)
                if segm_upper is None:  # 0 or 1 labels
                    continue
                segm_lower = self.make_marker_segment(segm_lower,
                                                      segm_upper)
                all_segms.append(segm_lower)
            return all_segms

        # A pixel is above threshold level i if i < quantized; NaN
        # pixels compare as False against every threshold.
        quantized = np.searchsorted(thresholds, self.data.ravel(),
                                    side='left')
        quantized = quantized.reshape(self.data.shape).astype(np.int32)
        quantized[~self.segment_mask | np.isnan(self.data)] = 0
        connectivity = 8 if self.footprint[0, 0] else 4
        markers, n_markers = make_deblend_markers(quantized,
                                                  self.n_pixels,
                                                  connectivity)
        if n_markers == 0:
            return None
        return markers

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

        # Count the upper-level children of each lower-level label from
        # the unique (lower, upper) label pairs, encoded as a combined
        # integer key.
        both = (segment_lower > 0) & (segment_upper > 0)
        stride = np.int64(np.max(segment_upper)) + 1
        keys = segment_lower[both].astype(np.int64) * stride
        keys += segment_upper[both]
        parents, n_children = np.unique(np.unique(keys) // stride,
                                        return_counts=True)
        multi_parents = parents[n_children >= 2]
        if multi_parents.size == 0:
            return segment_lower

        # Replace each multi-child parent by its children. Pixels of
        # the parent mask that are below the upper threshold are unset.
        # Single-child parents are kept as is (maximizing the marker
        # size).
        replace_lut = np.zeros(np.max(segment_lower) + 1, dtype=bool)
        replace_lut[multi_parents] = True
        replace = replace_lut[segment_lower]
        markers = segment_lower > 0
        markers[replace] = segment_upper[replace] > 0

        # Convert bool markers to integer labels
        return ndi_label(markers, structure=self.footprint)[0]

    def apply_watershed(self, markers):
        """
        Apply the watershed algorithm to the source markers.

        Parameters
        ----------
        markers : list of `~photutils.segmentation.SegmentationImage`
            A list of segmentation images that contain possible sources
            as markers. The last list element contains all the potential
            source markers.

        Returns
        -------
        segment_data : 2D int `~numpy.ndarray`
            A 2D int array containing the deblended source labels. Note
            that the source labels may not be consecutive if a label was
            removed.
        """
        # Deblend using watershed. If any source does not meet the
        # contrast criterion, then remove the faintest such source(s)
        # and repeat until all sources meet the contrast criterion.
        data_neg = np.ascontiguousarray(-self.data, dtype=np.float64)
        connectivity = 8 if self.footprint[0, 0] else 4
        remove_marker = True
        while remove_marker:
            markers = deblend_watershed(data_neg, markers,
                                        self.segment_mask, connectivity)

            labels = _get_labels(markers)
            if labels.size == 1:  # only 1 source left
                remove_marker = False
            else:
                flux_frac = (sum_labels(self.data, markers, index=labels)
                             / self.source_sum)
                remove_marker = any(flux_frac < self.contrast)

                if remove_marker:
                    self._remove_faint_markers(markers, labels, flux_frac)

        return markers

    def _remove_faint_markers(self, markers, labels, flux_frac):
        """
        Remove the faintest below-contrast marker(s) in place.

        The faintest marker is always removed. When the source data
        values are all nonnegative, the largest batch of the faintest
        markers is removed whose total flux fraction is below both
        the contrast and the next-faintest marker flux fraction.
        Removing such a batch in one step is equivalent to removing
        its markers one at a time. Every batch member stays below the
        contrast no matter how the flux of the other removed members is
        redistributed (their total is below the contrast), and markers
        outside the batch can only become brighter, so the faintest
        below-contrast marker always lies inside the batch until the
        batch is exhausted.

        Faint markers that do not fit in such a batch are removed one at
        a time because several faint sources could combine to meet the
        contrast criterion.

        Parameters
        ----------
        markers : 2D int `~numpy.ndarray`
            The watershed-labeled marker image, modified in place.

        labels : 1D `~numpy.ndarray`
            The sorted marker labels.

        flux_frac : 1D `~numpy.ndarray`
            The flux fraction in each marker basin, in the same
            order as ``labels``.
        """
        if self.source_min >= 0 and labels.size > 2:
            order = np.argsort(flux_frac)
            sorted_frac = flux_frac[order]
            csum = np.cumsum(sorted_frac)
            # A batch of the n faintest markers (2 <= n < N) is valid
            # if its total flux fraction is below both the contrast
            # and the next-faintest marker flux fraction.
            batch_ok = ((csum[1:-1] < self.contrast)
                        & (csum[1:-1] < sorted_frac[2:]))
            valid = np.nonzero(batch_ok)[0]
            if valid.size > 0:
                n_remove = int(valid[-1]) + 2
                remove_lut = np.zeros(int(labels[-1]) + 1, dtype=bool)
                remove_lut[labels[order[:n_remove]]] = True
                markers[remove_lut[markers]] = 0
                return

        markers[markers == labels[np.argmin(flux_frac)]] = 0

    def deblend_from_markers(self, markers):
        """
        Deblend the source given its precomputed watershed markers.

        Parameters
        ----------
        markers : 2D int `~numpy.ndarray`
            The marker image for the source.

        Returns
        -------
        segment_data : 2D int `~numpy.ndarray` or `None`
            A 2D int array containing the deblended source labels.
            The source labels are consecutive starting at 1. `None`
            is returned if only one source remains after applying
            the contrast criterion.
        """
        # Deblend using the watershed algorithm using the markers as seeds
        markers = self.apply_watershed(markers)

        if not np.array_equal(self.segment_mask, markers.astype(bool)):
            msg = (f'Deblending failed for source {self.label!r}. '
                   'Please ensure you used the same pixel connectivity '
                   'in detect_sources and deblend_sources.')
            raise ValueError(msg)

        if len(_get_labels(markers)) == 1:  # no deblending
            return None

        # Markers may not be consecutive if a label was removed due to
        # the contrast criterion
        relabel_map = _create_relabel_map(markers, start_label=1)
        if relabel_map is not None:
            markers = relabel_map[markers]
        return markers

    def deblend_source(self):
        """
        Deblend a single labeled source.

        This method computes the markers and the watershed steps
        entirely in Python, mirroring what ``deblend_sources``
        computes through the compiled chunk driver. It is useful for
        debugging and testing.

        Returns
        -------
        segment_data : 2D int `~numpy.ndarray` or `None`
            A 2D int array containing the deblended source labels. The
            source labels are consecutive starting at 1.
        """
        if self.source_min == self.source_max:  # no deblending
            return None

        # Define the markers (possible sources) for the watershed algorithm
        markers = self.make_markers()
        if markers is None:
            return None

        # If there are too many markers (e.g., due to low threshold
        # and/or small n_pixels), the watershed step can be very slow
        # (the threshold of 200 is arbitrary, but seems to work well).
        # This mostly affects the "exponential" mode, where there are
        # many levels at low thresholds, so here we try again with
        # "linear" mode.
        n_labels = len(_get_labels(markers))
        if self.mode != 'linear' and n_labels > 200:
            del markers  # free memory
            self.warnings['n_markers'] = 'too many markers'
            self.mode = 'linear'
            markers = self.make_markers()
            if markers is None:
                return None

        return self.deblend_from_markers(markers)
