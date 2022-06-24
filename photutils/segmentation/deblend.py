# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
This module provides tools for deblending overlapping sources labeled in
a segmentation image.
"""

from multiprocessing import cpu_count, Pool, RawArray
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


def _to_shared_array(arr):
    ctype = np.ctypeslib.as_ctypes_type(arr.dtype)
    shared_array = RawArray(ctype, arr.size)
    # temp = np.frombuffer(shared_array, dtype=arr.dtype).reshape(arr.shape)
    temp = np.ctypeslib.as_array(shared_array).reshape(arr.shape)
    np.copyto(temp, arr)
    return shared_array


def _to_numpy_array(shared_array, shape, dtype):
    """
    Create a numpy array backed by a shared memory Array.
    """
    # return np.frombuffer(shared_array, dtype=dtype).reshape(shape)
    return np.ctypeslib.as_array(shared_array).reshape(shape)


def _init_pool(data, segment_data, shape, data_dtype, segm_dtype):
    global shared_data
    global shared_segment_data
    shared_data = _to_numpy_array(data, shape, data_dtype)
    shared_segment_data = _to_numpy_array(segment_data, shape, segm_dtype)


@deprecated_renamed_argument('kernel', None, '1.5', message='"kernel" was '
                             'deprecated in version 1.5 and will be removed '
                             'in a future version. Instead, if filtering is '
                             'desired, please input a convolved image '
                             'directly into the "data" parameter.')
def deblend_sources(data, segment_img, npixels, kernel=None, labels=None,
                    nlevels=32, contrast=0.001, mode='exponential',
                    connectivity=8, relabel=True, nproc=1, progress_bar=True):
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
        4-connected pixels touch along their edges. For reference,
        SourceExtractor uses 8-connected pixels.

    relabel : bool, optional
        If `True` (default), then the segmentation image will be
        relabeled such that the labels are in consecutive order starting
        from 1.

    nproc : int, optional
        The number of processes to use for multiprocessing (if larger
        than 1). If set to 1, then a serial implementation is used
        instead of a parallel one. If `None`, then the number of
        processes will be set to the number of CPUs detected on the
        machine.

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
    if not isinstance(segment_img, SegmentationImage):
        raise ValueError('segment_img must be a SegmentationImage')

    if segment_img.shape != data.shape:
        raise ValueError('The data and segmentation image must have '
                         'the same shape')

    if nlevels < 1:
        raise ValueError('nlevels must be >= 1')
    if contrast < 0 or contrast > 1:
        raise ValueError('contrast must be >= 0 and <= 1')

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

    selem = _make_binary_structure(data.ndim, connectivity)

    if kernel is not None:
        data = _filter_data(data, kernel, mode='constant', fill_value=0.0)

    if nproc is None:
        nproc = cpu_count()

    if progress_bar and HAS_TQDM:
        from tqdm.auto import tqdm

    if nproc > 1:
        # Multiprocssing with shared memory does not work (pickling
        # errors) with arrays that have '>' (big endian) or '<' (little
        # endian) dtypes. Data read from FITS files are always big
        # endian, even if the machine is little endian.
        data = data.astype(data.dtype.name, copy=False)
        segment_img._data = segment_img._data.astype(
            segment_img._data.dtype.name, copy=False)

    segm_deblended = object.__new__(SegmentationImage)
    segm_deblended._data = np.copy(segment_img.data)
    last_label = segment_img.max_label

    if nproc == 1:
        if progress_bar and HAS_TQDM:
            labels = tqdm(labels)

        all_source_deblends = []
        for label in labels:
            deblender = _Deblender(label, npixels, selem, nlevels, contrast,
                                   mode, None, None, data=data,
                                   segment_img=segment_img)
            source_deblended = deblender.deblend_source()
            all_source_deblends.append(source_deblended)

    else:
        args_all = []
        for label in labels:
            args_all.append((label, npixels, selem, nlevels, contrast, mode,
                             segment_img.labels, segment_img.slices))

        shared_data = _to_shared_array(data)
        shared_segment_data = _to_shared_array(segment_img.data)
        shape = data.shape
        initargs = (shared_data, shared_segment_data, shape, data.dtype,
                    segment_img.data.dtype)

        if progress_bar and HAS_TQDM:
            # no progress bar for multiprocessing
            pass

        with Pool(processes=nproc, initializer=_init_pool,
                  initargs=initargs) as pool:
            all_source_deblends = pool.map(_deblend_source, args_all)

    nonposmin_labels = []
    for (label, source_deblended) in zip(labels, all_source_deblends):
        if source_deblended is not None:
            # replace the original source with the deblended source
            source_slice = segment_img.slices[segment_img.get_index(label)]
            segment_mask = (source_deblended.data > 0)
            segm_deblended._data[source_slice][segment_mask] = (
                source_deblended.data[segment_mask] + last_label)
            last_label += source_deblended.nlabels

            if hasattr(source_deblended, 'warnings'):
                if source_deblended.warnings.get('nonposmin',
                                                 None) is not None:
                    nonposmin_labels.append(label)

    if nonposmin_labels:
        warnings.warn('The deblending mode of one or more source labels from '
                      'the input segmentation image was changed from '
                      '"exponential" to "linear". See the "info" attribute '
                      'for the list of affected input labels.',
                      AstropyUserWarning)

        segm_deblended.info = {'warnings': {}}
        nonposmin = {'message': 'Deblending mode changed from exponential to '
                     'linear due to non-positive minimum data values.',
                     'input_labels': np.array(nonposmin_labels)}
        segm_deblended.info['warnings']['nonposmin'] = nonposmin

    if relabel:
        segm_deblended.relabel_consecutive()

    return segm_deblended


def _deblend_source(args):
    """
    Convenience function to deblend a single labeled source.

    "args" needs to be a single argument (which is unpacked here)
    because ProcessPoolExecutor does not have a starmap method.
    """
    (label, npixels, selem, nlevels, contrast, mode, shsegm_labels,
     shsegm_slices) = args

    deblender = _Deblender(label, npixels, selem, nlevels, contrast, mode,
                           shsegm_labels, shsegm_slices)
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

    def __init__(self, label, npixels, selem, nlevels, contrast, mode,
                 shsegm_labels, shsegm_slices, data=None, segment_img=None):

        self.label = label
        self.npixels = npixels
        self.selem = selem
        self.nlevels = nlevels
        self.contrast = contrast
        self.mode = mode
        self.warnings = {}
        self.data = data
        self.segment_img = segment_img

        if data is None:
            # global from _init_workers
            self.data = shared_data
            segment_img = object.__new__(SegmentationImage)
            segment_img._data = shared_segment_data
            segment_img.__dict__['slices'] = shsegm_slices
            segment_img.__dict__['labels'] = shsegm_labels
            self.segment_img = segment_img

        self.source_data, source_segment = self.make_cutouts()

        self.segment_mask = source_segment.data.astype(bool)
        self.source_values = self.source_data[self.segment_mask]
        self.source_min = np.nanmin(self.source_values)
        self.source_max = np.nanmax(self.source_values)
        self.source_sum = np.nansum(self.source_values)

        # NOTE: this includes the source min/max, but we exclude those
        # later, giving nlevels thresholds between min and max
        # (noninclusive; i.e., nlevels + 1 parts)
        self.linear_thresholds = np.linspace(self.source_min, self.source_max,
                                             self.nlevels + 2)

    def make_cutouts(self):
        idx = self.segment_img.get_index(self.label)
        source_slice = self.segment_img.slices[idx]
        source_data = self.data[source_slice]
        source_segment = object.__new__(SegmentationImage)
        source_segment._data = self.segment_img.data[source_slice]
        source_segment.keep_labels(self.label)  # include only one label
        return source_data, source_segment

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
