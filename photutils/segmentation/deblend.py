# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tools for deblending overlapping sources labeled in a segmentation
image.
"""

import warnings
from dataclasses import dataclass

import numpy as np
from astropy.units import Quantity

from photutils.segmentation._deblend_markers import (deblend_markers_chunk,
                                                     deblend_source_extrema)
from photutils.segmentation._deblend_watershed import deblend_source_contrast
from photutils.segmentation.core import (SegmentationImage, _get_labels,
                                         _remap_deblend_label_map)
from photutils.segmentation.flags import SEGMENTATION_FLAGS
from photutils.segmentation.utils import _make_binary_structure
from photutils.utils._deprecation import deprecated_renamed_argument
from photutils.utils._stats import nansum
from photutils.utils.exceptions import DeblendWarning

__all__ = ['deblend_sources']

# The number of markers above which a source is deblended again with
# linearly spaced threshold levels, which have fewer levels at low
# thresholds. The value is arbitrary but works well in practice
_MAX_MARKERS = 200


@dataclass
class _DeblendParams:
    n_pixels: int
    footprint: np.ndarray
    n_levels: int
    contrast: float
    mode: str


@deprecated_renamed_argument('segment_img', 'segmentation_image', '3.0',
                             until='4.0')
@deprecated_renamed_argument('npixels', 'n_pixels', '3.0', until='4.0')
@deprecated_renamed_argument('nlevels', 'n_levels', '3.0', until='4.0')
@deprecated_renamed_argument('nproc', None, '3.0', until='4.0')
@deprecated_renamed_argument('n_processes', None, '3.1', until='4.0')
@deprecated_renamed_argument('progress_bar', None, '3.1', until='4.0')
def deblend_sources(data, segmentation_image, n_pixels, *, labels=None,
                    n_levels=32, contrast=0.001, mode='exponential',
                    connectivity=8, relabel=True,
                    nproc=1,  # noqa: ARG001
                    n_processes=1,  # noqa: ARG001
                    progress_bar=True):  # noqa: ARG001
    """
    Deblend overlapping sources labeled in a segmentation image.

    Sources are deblended using a combination of
    multi-thresholding and `watershed segmentation
    <https://en.wikipedia.org/wiki/Watershed_(image_processing)>`_. In
    order to deblend sources, they must be separated enough that there
    is a saddle point between them.

    Parameters
    ----------
    data : 2D `~numpy.ndarray`
        The 2D array of the image. If filtering is desired, please input
        a convolved image here. This array should be the same array used
        in `~photutils.segmentation.detect_sources`. NaN pixels within
        a source segment are excluded from the multithreshold levels
        and the source flux, and are assigned to a neighboring
        deblended source after all finite pixels have been assigned.

    segmentation_image : `~photutils.segmentation.SegmentationImage`
        The segmentation image to deblend.

    n_pixels : int
        The minimum number of connected pixels, each greater than
        ``threshold``, that an object must have to be deblended.
        ``n_pixels`` must be a positive integer.

    labels : int or array_like of int, optional
        The label numbers to deblend. If `None` (default), then all
        labels in the segmentation image will be deblended.

    n_levels : int, optional
        The number of multi-thresholding levels to use for deblending.
        Each source will be re-thresholded at ``n_levels`` levels spaced
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
        multi-thresholding levels (see the ``n_levels`` keyword)
        during deblending. The ``'exponential'`` and ``'sinh'`` modes
        have more threshold levels near the source minimum and less
        near the source maximum. The ``'linear'`` mode evenly spaces
        the threshold levels between the source minimum and maximum.
        The ``'exponential'`` and ``'sinh'`` modes differ in that
        the ``'exponential'`` levels are dependent on the source
        maximum/minimum ratio (smaller ratios are more linear and
        larger ratios are more exponential), while the ``'sinh'`` levels
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
        This keyword is deprecated and has no effect. It was the name of
        the ``n_processes`` keyword before version 3.0.

        .. deprecated:: 3.0
            The ``nproc`` keyword is deprecated and will be removed in
            version 4.0.

    n_processes : int, optional
        This keyword is deprecated and has no effect. Multiprocessing
        no longer provides any benefit. The deblending computation is
        now dominated by compiled code, and the process startup and
        data-pickling overheads of multiprocessing made it slower than
        the serial implementation.

        .. deprecated:: 3.1
            The ``n_processes`` keyword is deprecated and will be
            removed in version 4.0.

    progress_bar : bool, optional
        This keyword is deprecated and has no effect. Deblending no
        longer displays a progress bar.

        .. deprecated:: 3.1
            The ``progress_bar`` keyword is deprecated and will be
            removed in version 4.0.

    Returns
    -------
    segment_image : `~photutils.segmentation.SegmentationImage`
        A segmentation image, with the same shape as ``data``, where
        sources are marked by different positive integer values. A value
        of zero is reserved for the background. The ``info`` attribute
        of the returned segmentation image is a dictionary that stores
        the input labels for which the deblending mode was changed to
        "linear" as arrays under ``'nonposmin_labels'`` (non-positive
        minimum data values) and ``'n_markers_labels'`` (too many
        potential deblended sources) keys. The dictionary is empty if no
        mode fallbacks occurred. The ``flags`` attribute of the returned
        segmentation image records per-source deblending provenance. See
        `~photutils.segmentation.decode_segmentation_flags`.

    Warns
    -----
    DeblendWarning
        If the deblending mode for one or more sources was changed from
        ``mode`` to "linear" due to non-positive minimum data values or
        too many potential deblended sources.

    See Also
    --------
    :func:`photutils.segmentation.detect_sources`
    :class:`photutils.segmentation.SourceFinder`
    """
    if isinstance(data, Quantity):
        data = data.value

    if not isinstance(segmentation_image, SegmentationImage):
        msg = 'segmentation_image must be a SegmentationImage'
        raise TypeError(msg)

    if segmentation_image.shape != data.shape:
        msg = 'segmentation_image must have the same shape as data'
        raise ValueError(msg)

    if segmentation_image.n_labels == 0:
        msg = 'segmentation_image must have at least one non-zero label'
        raise ValueError(msg)

    if (n_pixels <= 0) or (int(n_pixels) != n_pixels):
        msg = f'n_pixels must be a positive integer, got {n_pixels!r}'
        raise ValueError(msg)

    if (n_levels < 1) or (int(n_levels) != n_levels):
        msg = f'n_levels must be a positive integer, got {n_levels!r}'
        raise ValueError(msg)
    if contrast < 0 or contrast > 1:
        msg = 'contrast must be >= 0 and <= 1'
        raise ValueError(msg)

    if mode not in ('exponential', 'linear', 'sinh'):
        msg = "mode must be 'exponential', 'linear', or 'sinh'"
        raise ValueError(msg)

    if contrast == 1:  # no deblending
        segm_img = segmentation_image.copy()
        if relabel:
            segm_img.relabel_consecutive()
        return segm_img

    if labels is None:
        labels = segmentation_image.labels
    else:
        labels = np.atleast_1d(labels)
        segmentation_image.check_labels(labels)

    # Include only sources that have at least (2 * n_pixels).
    # This is required for a source to be deblended into multiple
    # sources, each with a minimum of n_pixels
    mask = (segmentation_image.areas[
            segmentation_image.get_indices(labels)]
            >= (n_pixels * 2))
    labels = labels[mask]

    footprint = _make_binary_structure(data.ndim, connectivity)
    deblend_params = _DeblendParams(n_pixels, footprint, n_levels, contrast,
                                    mode)

    segm_deblended = segmentation_image.data.copy()
    label_indices = segmentation_image.get_indices(labels)
    all_slices = [segmentation_image.slices[idx] for idx in label_indices]

    # Contiguous, native-byte-order views of the inputs for the
    # compiled kernels. Casting the non-float data dtypes to float64 is
    # exact for the threshold comparisons, matching the NumPy promotion
    # rules
    if data.dtype.type in (np.float32, np.float64):
        driver_dtype = data.dtype.newbyteorder('=')
    else:
        driver_dtype = np.float64
    driver_data = np.ascontiguousarray(data, dtype=driver_dtype)
    segm_data = segmentation_image.data
    if segm_data.dtype.type in (np.int32, np.int64):
        driver_dtype = segm_data.dtype.newbyteorder('=')
    else:
        driver_dtype = np.int64
    driver_segm = np.ascontiguousarray(segm_data, dtype=driver_dtype)

    results = _deblend_sources_chunk(data, segm_data, driver_data,
                                     driver_segm, labels, all_slices,
                                     deblend_params)

    deblend_label_map = {}
    max_label = segmentation_image.max_label
    nonposmin_labels = []
    n_markers_labels = []
    for label, source_slice, (source_deblended, warns) in zip(
            labels, all_slices, results, strict=True):
        if warns:
            if 'nonposmin' in warns:
                nonposmin_labels.append(label)
            if 'n_markers' in warns:
                n_markers_labels.append(label)

        if source_deblended is not None:
            source_mask = source_deblended > 0
            new_segm = source_deblended[source_mask]  # min label = 1
            segm_deblended[source_slice][source_mask] = (
                new_segm + max_label)
            new_labels = _get_labels(new_segm) + max_label
            deblend_label_map[label] = new_labels
            max_label += len(new_labels)

    if nonposmin_labels or n_markers_labels:
        msg = ('The deblending mode of one or more source labels from the '
               f'input segmentation image was changed from "{mode}" to '
               '"linear". See the "info" attribute of the returned '
               'segmentation image for the affected input labels.')
        warnings.warn(msg, DeblendWarning)

    relabel_map = None
    if relabel:
        relabel_map = _create_relabel_map(segm_deblended, start_label=1)
        if relabel_map is not None:
            segm_deblended = relabel_map[segm_deblended]
            deblend_label_map = _remap_deblend_label_map(deblend_label_map,
                                                         relabel_map)

    segm_img = SegmentationImage._from_data(
        segm_deblended, deblend_label_map=deblend_label_map)

    # Store the input labels affected by deblending mode fallbacks in
    # the info attribute
    if nonposmin_labels:
        segm_img.info['nonposmin_labels'] = np.array(nonposmin_labels)
    if n_markers_labels:
        segm_img.info['n_markers_labels'] = np.array(n_markers_labels)

    segm_img._flags_map = _make_flags_map(
        deblend_label_map, nonposmin_labels, n_markers_labels, relabel_map)

    return segm_img


def _linspace_rows(start, stop, num):
    """
    Evaluate ``np.linspace(start[i], stop[i], num)`` for every row.

    This replicates the NumPy implementation operation for operation
    (the evaluation dtype, the step computation, and the zero-step
    special case), so that every row is bitwise identical to the
    scalar ``np.linspace`` call of the pure-Python reference
    implementation.

    Parameters
    ----------
    start, stop : 1D `~numpy.ndarray`
        The start and stop values of each row.

    num : int
        The number of samples per row.

    Returns
    -------
    result : 2D `~numpy.ndarray`
        The samples, with shape ``(len(start), num)``.
    """
    dtype = np.result_type(start, stop)
    if not np.issubdtype(dtype, np.inexact):
        dtype = np.float64
    div = num - 1
    delta = np.subtract(stop, start, dtype=dtype)
    samples = np.arange(0, num, dtype=dtype)
    step = delta / div
    result = samples[None, :] * step[:, None]
    zero_step = step == 0
    if np.any(zero_step):
        # The np.linspace special case for subnormal steps
        result[zero_step] = (samples / div)[None, :] * delta[zero_step, None]
    result += start[:, None]
    result[:, -1] = stop
    return result


def _compute_thresholds(source_min, source_max, n_levels, mode):
    """
    Compute the multithreshold levels of a set of sources.

    This is the vectorized form of the per-source threshold
    computation of the pure-Python reference implementation
    (``_SingleSourceDeblender.compute_thresholds``). It performs the
    same NumPy operations in the same dtypes, so the levels are bitwise
    identical to the reference implementation on every platform.

    Parameters
    ----------
    source_min, source_max : 1D `~numpy.ndarray`
        The minimum and maximum data value of each source segment, in
        the data dtype. Each maximum must be larger than its minimum.

    n_levels : int
        The number of levels per source.

    mode : {'exponential', 'linear', 'sinh'}
        The level spacing. For the ``'exponential'`` mode, the sources
        with a non-positive minimum use the ``'linear'`` spacing.

    Returns
    -------
    thresholds : 2D float64 `~numpy.ndarray`
        The levels of each source, with shape ``(n_sources,
        n_levels)`` and ascending along the second axis. The source
        minimum and maximum are excluded.

    nonposmin : 1D bool `~numpy.ndarray`
        Whether each source fell back to the ``'linear'`` spacing
        because of a non-positive minimum.
    """
    source_min = np.asarray(source_min)
    source_max = np.asarray(source_max)
    nonposmin = np.zeros(source_min.shape, dtype=bool)
    if mode == 'exponential':
        nonposmin = source_min <= 0

    thresholds = _linspace_rows(source_min, source_max, n_levels + 2)
    if mode != 'linear':
        delta = source_max - source_min
        normalized = (thresholds - source_min[:, None]) / delta[:, None]
        if mode == 'sinh':
            a = 0.25
            thresholds = np.sinh(normalized / a) / np.sinh(1.0 / a)
            thresholds *= delta[:, None]
            thresholds += source_min[:, None]
        else:
            keep = ~nonposmin
            ratio = source_max[keep] / source_min[keep]
            thresholds[keep] = (source_min[keep, None]
                                * ratio[:, None] ** normalized[keep])

    # Do not include the source minimum and maximum
    return (np.ascontiguousarray(thresholds[:, 1:-1], dtype=np.float64),
            nonposmin)


def _deblend_sources_chunk(data, segm_data, driver_data, driver_segm,
                           labels, slices, deblend_params):
    """
    Deblend a chunk of labeled sources.

    The data extrema of every source in the chunk are computed by a
    compiled kernel, the multithreshold levels and the mode fallbacks
    are computed for all sources at once in NumPy, the level
    quantization and the marker construction run in the compiled chunk
    driver (which releases the GIL and reuses one workspace across the
    chunk), and the watershed and contrast steps then run for the
    sources that split into two or more markers.

    Parameters
    ----------
    data : 2D `~numpy.ndarray`
        The data array.

    segm_data : 2D int `~numpy.ndarray`
        The segmentation array.

    driver_data, driver_segm : 2D `~numpy.ndarray`
        Contiguous views (or exact casts) of ``data`` and
        ``segm_data`` with dtypes supported by the compiled kernels.

    labels : 1D `~numpy.ndarray`
        The labels of the sources in the chunk.

    slices : list of tuple of slice
        The bounding-box slices of the sources in the chunk.

    deblend_params : `_DeblendParams`
        The parameters for deblending the sources.

    Returns
    -------
    results : list of (2D `~numpy.ndarray` or `None`, dict)
        For each source, the deblended cutout (`None` if the source
        did not deblend) and the mode-fallback warnings dictionary.
    """
    labels = np.asarray(labels, dtype=np.int64)
    y0 = np.array([slc[0].start for slc in slices], dtype=np.int64)
    y1 = np.array([slc[0].stop for slc in slices], dtype=np.int64)
    x0 = np.array([slc[1].start for slc in slices], dtype=np.int64)
    x1 = np.array([slc[1].stop for slc in slices], dtype=np.int64)
    connectivity = 8 if deblend_params.footprint[0, 0] else 4
    n_levels = int(deblend_params.n_levels)
    mode = deblend_params.mode
    chunk_kwargs = {'n_pixels': int(deblend_params.n_pixels),
                    'connectivity': connectivity}

    source_min, source_max = deblend_source_extrema(
        driver_data, driver_segm, labels, y0, y1, x0, x1)

    # Constant (or all-NaN) sources do not deblend. The multithreshold
    # levels of the other sources are computed in the data dtype, as
    # the pure-Python reference implementation does
    active = np.flatnonzero(source_min < source_max)
    markers_list = [None] * len(labels)
    nonposmin = np.zeros(len(labels), dtype=bool)
    n_markers_fallback = np.zeros(len(labels), dtype=bool)
    if active.size > 0:
        values_dtype = data.dtype.newbyteorder('=')
        smin = source_min[active].astype(values_dtype)
        smax = source_max[active].astype(values_dtype)
        thresholds, fallback = _compute_thresholds(smin, smax, n_levels,
                                                   mode)
        nonposmin[active] = fallback
        # Sources with too many markers are only retried with linearly
        # spaced levels (below), so only then may the kernel skip
        # building their markers.
        can_retry = mode != 'linear'
        max_markers = _MAX_MARKERS if can_retry else -1
        markers, n_markers = deblend_markers_chunk(
            driver_data, driver_segm, labels[active], y0[active],
            y1[active], x0[active], x1[active], thresholds,
            max_markers=max_markers, **chunk_kwargs)
        for index, source_markers in zip(active, markers, strict=True):
            markers_list[index] = source_markers

        # Too many markers make the watershed step very slow, so such
        # sources are deblended again with linearly spaced levels. A
        # source that already fell back to the linear spacing keeps its
        # markers
        if can_retry:
            retry = np.flatnonzero((n_markers > _MAX_MARKERS) & ~fallback)
            if retry.size > 0:
                thresholds, _ = _compute_thresholds(
                    smin[retry], smax[retry], n_levels, 'linear')
                retry = active[retry]
                markers, _ = deblend_markers_chunk(
                    driver_data, driver_segm, labels[retry], y0[retry],
                    y1[retry], x0[retry], x1[retry], thresholds,
                    max_markers=-1, **chunk_kwargs)
                for index, source_markers in zip(retry, markers,
                                                 strict=True):
                    markers_list[index] = source_markers
                n_markers_fallback[retry] = True

    results = []
    for index, (label, slc, markers) in enumerate(
            zip(labels, slices, markers_list, strict=True)):
        warns = {}
        if nonposmin[index]:
            warns['nonposmin'] = 'non-positive minimum'
        if n_markers_fallback[index]:
            warns['n_markers'] = 'too many markers'
        if markers is None:
            results.append((None, warns))
            continue

        # The total source flux is computed with the same reduction as
        # the per-source Python path (its rounding depends on the
        # summation order) and the source minimum comes from the
        # compiled extrema. Both are passed to the compiled contrast
        # loop.
        values = data[slc][segm_data[slc] == label]
        source_deblended = deblend_source_contrast(
            driver_data, driver_segm, int(label), int(y0[index]),
            int(y1[index]), int(x0[index]), int(x1[index]), markers,
            connectivity=connectivity,
            contrast=float(deblend_params.contrast),
            source_sum=float(nansum(values)),
            source_min=float(source_min[index]))
        results.append((source_deblended, warns))

    return results


def _make_flags_map(deblend_label_map, nonposmin_labels, n_markers_labels,
                    relabel_map):
    """
    Build the per-label flags mapping for a deblended segmentation
    image.

    Deblended children get the "deblended" flag. Mode-fallback flags
    are set on the children of affected parents, or on the parent's own
    (possibly relabeled) output label if it did not split.

    Parameters
    ----------
    deblend_label_map : dict
        Mapping of input parent labels to arrays of output child labels,
        in the final (post-relabel) label frame.

    nonposmin_labels, n_markers_labels : list of int
        Input parent labels affected by each mode fallback.

    relabel_map : `~numpy.ndarray` or `None`
        The relabeling map applied to the deblended image, or `None` if
        no relabeling was applied.

    Returns
    -------
    flags_map : dict
        Mapping of output labels to bitwise flag values.
    """
    flags_map = {}
    for children in deblend_label_map.values():
        for child in children:
            child = int(child)
            flags_map[child] = (flags_map.get(child, 0)
                                | SEGMENTATION_FLAGS.DEBLENDED)

    fallbacks = [
        (nonposmin_labels, SEGMENTATION_FLAGS.DEBLEND_NONPOSMIN),
        (n_markers_labels, SEGMENTATION_FLAGS.DEBLEND_N_MARKERS),
    ]
    for input_labels, bit in fallbacks:
        for label in input_labels:
            label = int(label)
            if label in deblend_label_map:
                targets = [
                    int(child) for child in deblend_label_map[label]
                ]
            else:
                # The source did not split. Translate its input label to
                # the output label frame.
                if relabel_map is None:
                    out_label = label
                else:
                    out_label = int(relabel_map[label])
                targets = [out_label]
            for target in targets:
                flags_map[target] = flags_map.get(target, 0) | bit
    return flags_map


def _create_relabel_map(array, *, start_label=1):
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

    # Check if the labels are already consecutive starting from
    # start_label
    if (labels[0] == start_label
            and (labels[-1] - start_label + 1) == len(labels)):
        return None

    # Create an array to map old labels to new labels
    relabel_map = np.zeros(labels.max() + 1, dtype=array.dtype)
    relabel_map[labels] = np.arange(len(labels)) + start_label

    return relabel_map
