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
from .detect import _detect_sources
from .utils import _make_binary_structure
from ..utils._convolution import _filter_data
from ..utils._optional_deps import HAS_TQDM  # noqa

__all__ = ['deblend_sources']


@deprecated_renamed_argument('kernel', None, '1.5', message='"kernel" was '
                             'deprecated in version 1.5 and will be removed '
                             'in a future version. Instead, if filtering is '
                             'desired, please input a convolved image '
                             'directly into the "data" parameter.')
def deblend_sources(data, segment_img, npixels, kernel=None, labels=None,
                    nlevels=32, contrast=0.001, mode='exponential',
                    connectivity=8, relabel=True, progress_bar=True):
    """
    Deblend overlapping sources labeled in a segmentation image.

    Sources are deblended using a combination of multi-thresholding and
    `watershed segmentation
    <https://en.wikipedia.org/wiki/Watershed_(image_processing)>`_.  In
    order to deblend sources, there must be a saddle between them.

    Parameters
    ----------
    data : 2D `~numpy.ndarray`
        The 2D data array. This array should be the same array used in
        `~photutils.segmentation.detect_sources`.

        .. note::
           It is strongly recommended that the user convolve the data
           with ``kernel`` and input the convolved data directly
           into the ``data`` parameter. In this case do not input a
           ``kernel``, otherwise the data will be convolved twice.

    segment_img : `~photutils.segmentation.SegmentationImage`
        The segmentation image to deblend.

    npixels : int
        The number of connected pixels, each greater than ``threshold``,
        that an object must have to be detected.  ``npixels`` must be a
        positive integer.

    kernel : 2D `~numpy.ndarray` or `~astropy.convolution.Kernel2D`, optional
        Deprecated. If filtering is desired, please input a convolved
        image directly into the ``data`` parameter.

        The 2D kernel used to filter the image before thresholding.
        Filtering the image will smooth the noise and maximize
        detectability of objects with a shape similar to the kernel.
        ``kernel`` must be `None` if the input ``data`` are already
        convolved.

    labels : int or array-like of int, optional
        The label numbers to deblend.  If `None` (default), then all
        labels in the segmentation image will be deblended.

    nlevels : int, optional
        The number of multi-thresholding levels to use for deblending.
        Each source will be re-thresholded at ``nlevels`` levels spaced
        exponentially or linearly (see the ``mode`` keyword) between its
        minimum and maximum values.

    contrast : float, optional
        The fraction of the total source flux that a local peak must
        have (at any one of the multi-thresholds) to be deblended
        as a separate object. ``contrast`` must be between 0 and 1,
        inclusive. If ``contrast=0`` then every local peak will be made
        a separate object (maximum deblending). If ``contrast=1`` then
        no deblending will occur. The default is 0.001, which will
        deblend sources with a 7.5 magnitude difference.

    mode : {'exponential', 'linear'}, optional
        The mode used in defining the spacing between the
        multi-thresholding levels (see the ``nlevels`` keyword) during
        deblending.

    connectivity : {8, 4}, optional
        The type of pixel connectivity used in determining how pixels
        are grouped into a detected source. The options are 8 (default)
        or 4. 8-connected pixels touch along their edges or corners.
        4-connected pixels touch along their edges. For reference,
        SourceExtractor uses 8-connected pixels.

    relabel : bool
        If `True` (default), then the segmentation image will be
        relabeled such that the labels are in consecutive order starting
        from 1.

     progress_bar : bool, optional
        Whether to display a progress bar. The progress bar requires
        that the `tqdm <https://tqdm.github.io/>`_ optional dependency
        be installed. Note that the progress bar does not currently work
        in the Jupyter console due to limitations in ``tqdm``.

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
    if not isinstance(segment_img, SegmentationImage):
        raise ValueError('segment_img must be a SegmentationImage')

    if segment_img.shape != data.shape:
        raise ValueError('The data and segmentation image must have '
                         'the same shape')

    if nlevels < 1:
        raise ValueError('nlevels must be >= 1')
    if contrast < 0 or contrast > 1:
        raise ValueError('contrast must be >= 0 and <= 1')

    if mode not in ('exponential', 'linear'):
        raise ValueError('mode must be "exponential" or "linear"')

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

    selem = _make_binary_structure(data.ndim, connectivity)

    if kernel is not None:
        data = _filter_data(data, kernel, mode='constant', fill_value=0.0)

    segm_deblended = object.__new__(SegmentationImage)
    segm_deblended._data = np.copy(segment_img.data)
    last_label = segment_img.max_label

    indices = segment_img.get_indices(labels)
    if progress_bar and HAS_TQDM:
        from tqdm.auto import tqdm
        labels = tqdm(labels)

    warn_negval_labels = []
    for label, idx in zip(labels, indices):
        source_slice = segment_img.slices[idx]
        source_data = data[source_slice]
        source_segment = object.__new__(SegmentationImage)
        source_segment._data = np.copy(segment_img.data[source_slice])
        source_segment.keep_labels(label)  # include only one label

        deblender = _Deblender(source_data, source_segment, npixels, selem,
                               nlevels, contrast, mode)
        source_deblended = deblender.deblend_source()

        if source_deblended is not None:
            # replace the original source with the deblended source
            segment_mask = (source_deblended.data > 0)
            segm_deblended._data[source_slice][segment_mask] = (
                source_deblended.data[segment_mask] + last_label)
            last_label += source_deblended.nlabels

            if hasattr(source_deblended, 'info'):
                if source_deblended.info.get('negval', None) is not None:
                    warn_negval_labels.append(label)

    if warn_negval_labels:
        warnings.warn('The deblending mode of one or more source labels from '
                      'the input segmentation image was changed from '
                      '"exponential" to "linear". See the "info" attribute '
                      'for the list of affected input labels.',
                      AstropyUserWarning)

        segm_deblended.info = {'warnings': {}}
        negval = {'message': 'Deblending mode changed from exponential to '
                  'linear due to negative data values.',
                  'input_labels': np.array(warn_negval_labels)}
        segm_deblended.info['warnings']['negval'] = negval

    if relabel:
        segm_deblended.relabel_consecutive()

    return segm_deblended


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

    mode : {'exponential', 'linear'}
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

    def __init__(self, source_data, source_segment, npixels, selem, nlevels,
                 contrast, mode):

        self.source_data = source_data
        self.source_segment = source_segment
        self.npixels = npixels
        self.selem = selem
        self.nlevels = nlevels
        self.contrast = contrast
        self.mode = mode
        self.info = {}

        self.segment_mask = source_segment.data.astype(bool)
        self.source_values = source_data[self.segment_mask]
        self.source_min = np.nanmin(self.source_values)
        self.source_max = np.nanmax(self.source_values)
        self.source_sum = np.nansum(self.source_values)
        self.label = source_segment.labels[0]  # should only be 1 label

        # NOTE: this includes the source min/max, but we exclude those
        # later, giving nlevels thresholds between min and max
        # (nlevels + 1 parts)
        self.linear_thresholds = np.linspace(self.source_min, self.source_max,
                                             self.nlevels + 2)
        self.thresholds = self.compute_thresholds()

    def normalized_thresholds(self):
        return ((self.linear_thresholds - self.source_min)
                / (self.source_max - self.source_min))

    def compute_thresholds(self):
        """
        Compute the multi-level detection thresholds for the source.
        """
        if self.mode == 'exponential' and self.source_min < 0:
            self.info['negval'] = 'negative data values'
            self.mode = 'linear'

        if self.mode == 'linear':
            thresholds = self.linear_thresholds
        elif self.mode == 'exponential':
            minval = self.source_min
            maxval = self.source_max
            if minval == 0:
                minval = maxval * 0.01
            thresholds = self.normalized_thresholds()
            thresholds = minval * (maxval / minval) ** thresholds

        return thresholds[1:-1]  # do not include source min and max

    def multithreshold(self):
        """
        Perform multithreshold detection for each source.
        """
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
            segments = _detect_sources(self.source_data, self.thresholds,
                                       self.npixels, self.selem,
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
        from scipy.ndimage import label as ndilabel

        for i in range(len(segments) - 1):
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
                segm_new._data = ndilabel(segm_lower, structure=self.selem)[0]
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
                                connectivity=self.selem)

            labels = np.unique(markers[markers != 0])
            flux_frac = (sum_labels(self.source_data, markers, index=labels)
                         / self.source_sum)
            remove_marker = any(flux_frac < self.contrast)

            if remove_marker:
                # remove only the faintest source (one at a time) because
                # several faint sources could combine to meet the contrast
                # criterion
                markers[markers == labels[np.argmin(flux_frac)]] = 0.

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

        if self.info:
            segm_new.info = self.info

        return segm_new
