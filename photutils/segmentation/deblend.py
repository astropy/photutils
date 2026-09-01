# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tools for deblending overlapping sources labeled in a segmentation
image.
"""

import warnings
from dataclasses import dataclass

import numpy as np
from astropy.units import Quantity

from photutils.segmentation._deblend_markers import (DEBLEND_FLAG_NMARKERS,
                                                     DEBLEND_FLAG_NONPOSMIN,
                                                     deblend_markers_chunk)
from photutils.segmentation._deblend_watershed import deblend_source_contrast
from photutils.segmentation.core import (SegmentationImage, _get_labels,
                                         _remap_deblend_label_map)
from photutils.segmentation.flags import SEGMENTATION_FLAGS
from photutils.segmentation.utils import _make_binary_structure
from photutils.utils._deprecation import deprecated_renamed_argument
from photutils.utils._stats import nanmin, nansum
from photutils.utils.exceptions import DeblendWarning

__all__ = ['deblend_sources']

# Mode codes accepted by the compiled chunk driver
_MODE_CODES = {'linear': 0, 'exponential': 1, 'sinh': 2}


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
@deprecated_renamed_argument('nproc', 'n_processes', '3.0', until='4.0')
@deprecated_renamed_argument('n_processes', None, '3.1', until='4.0')
@deprecated_renamed_argument('progress_bar', None, '3.1', until='4.0')
def deblend_sources(data, segmentation_image, n_pixels, *, labels=None,
                    n_levels=32, contrast=0.001, mode='exponential',
                    connectivity=8, relabel=True,
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
        in `~photutils.segmentation.detect_sources`.

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

    n_processes : int, optional
        This keyword is deprecated and has no effect. Multiprocessing
        no longer provides any benefit: the deblending computation
        is now dominated by compiled code and its process startup
        and data-pickling overheads made it slower than the serial
        implementation.

        .. deprecated:: 3.1
            The ``n_processes`` keyword is deprecated and will be
            removed in a future version.

    progress_bar : bool, optional
        This keyword is deprecated and has no effect. Deblending no
        longer displays a progress bar.

        .. deprecated:: 3.1
            The ``progress_bar`` keyword is deprecated and will be
            removed in a future version.

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

    if n_levels < 1:
        msg = 'n_levels must be >= 1'
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

    # Include only sources that have at least (2 * n_pixels);
    # this is required for a source to be deblended into multiple
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

    # Contiguous, driver-compatible views of the inputs; casting the
    # non-float data dtypes to float64 is exact for the threshold
    # comparisons, matching the NumPy promotion rules
    if data.dtype in (np.float32, np.float64):
        driver_data = np.ascontiguousarray(data)
    else:
        driver_data = np.ascontiguousarray(data, dtype=np.float64)
    segm_data = segmentation_image.data
    if segm_data.dtype in (np.int32, np.int64):
        driver_segm = np.ascontiguousarray(segm_data)
    else:
        driver_segm = np.ascontiguousarray(segm_data, dtype=np.int64)

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


def _deblend_sources_chunk(data, segm_data, driver_data, driver_segm,
                           labels, slices, deblend_params):
    """
    Deblend a chunk of labeled sources.

    The multithreshold markers of every source in the chunk are
    built by the compiled chunk driver (which releases the GIL and
    reuses one workspace across the chunk), and the watershed and
    contrast steps then run for the sources that split into two or
    more markers.

    Parameters
    ----------
    data : 2D `~numpy.ndarray`
        The data array.

    segm_data : 2D int `~numpy.ndarray`
        The segmentation array.

    driver_data, driver_segm : 2D `~numpy.ndarray`
        Contiguous views (or exact casts) of ``data`` and
        ``segm_data`` with dtypes supported by the chunk driver.

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
    y0 = np.array([slc[0].start for slc in slices], dtype=np.int64)
    y1 = np.array([slc[0].stop for slc in slices], dtype=np.int64)
    x0 = np.array([slc[1].start for slc in slices], dtype=np.int64)
    x1 = np.array([slc[1].stop for slc in slices], dtype=np.int64)
    connectivity = 8 if deblend_params.footprint[0, 0] else 4
    markers_list, flags = deblend_markers_chunk(
        driver_data, driver_segm, np.asarray(labels, dtype=np.int64),
        y0, y1, x0, x1, n_pixels=int(deblend_params.n_pixels),
        connectivity=connectivity,
        n_levels=int(deblend_params.n_levels),
        mode=_MODE_CODES[deblend_params.mode])

    results = []
    for index, (label, slc, markers, flag) in enumerate(
            zip(labels, slices, markers_list, flags, strict=True)):
        warns = {}
        if flag & DEBLEND_FLAG_NONPOSMIN:
            warns['nonposmin'] = 'non-positive minimum'
        if flag & DEBLEND_FLAG_NMARKERS:
            warns['n_markers'] = 'too many markers'
        if markers is None:
            results.append((None, warns))
            continue

        # The total source flux and minimum are computed here, with the
        # same reductions as the per-source Python path, and passed to
        # the compiled contrast loop.
        values = data[slc][segm_data[slc] == label]
        source_deblended = deblend_source_contrast(
            driver_data, driver_segm, int(label), int(y0[index]),
            int(y1[index]), int(x0[index]), int(x1[index]), markers,
            connectivity=connectivity,
            contrast=float(deblend_params.contrast),
            source_sum=float(nansum(values)),
            source_min=float(nanmin(values)))
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
