# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tools for finding local peaks in an astronomical image.
"""

import inspect
import warnings

import numpy as np
from astropy.table import QTable
from scipy.ndimage import label, maximum_filter
from scipy.spatial import cKDTree

from photutils.utils._deprecation import deprecated_renamed_argument
from photutils.utils._misc import _get_meta
from photutils.utils._parameters import as_pair
from photutils.utils._quantity_helpers import process_quantities
from photutils.utils._stats import nanmin
from photutils.utils.exceptions import NoDetectionsWarning

__all__ = ['find_peaks']

# The radius in pixels below which the circular footprint filter is
# faster than the box filters with circular verification
_MIN_FAST_RADIUS = 4.0

# The mean number of too-close pairs per equal-valued peak above which
# the neighbors of only the kept peaks are found, instead of all the
# pairs at once
_MAX_PAIRS_PER_PEAK = 4

# The number of equal-valued peaks used to estimate the mean number of
# too-close pairs per peak
_N_SAMPLE_PEAKS = 1000


def _collapse_plateaus(peak_mask, min_separation):
    """
    Find the peak pixels, with each plateau reduced to a single pixel.

    A peak pixel is the maximum within the circle of radius
    ``min_separation`` centered on it, so connected peak pixels always
    have equal values (e.g., the pixels of a saturated star). Each
    group of connected peak pixels is replaced by its pixel that is
    nearest to the centroid of the group. For equal distances, the
    first pixel in row-major order is kept.

    Parameters
    ----------
    peak_mask : 2D bool `~numpy.ndarray`
        The boolean mask of peak pixels. It is not modified.

    min_separation : float
        The radius in pixels of the circular region that was used to
        find the peaks.

    Returns
    -------
    y_peaks, x_peaks : 1D `~numpy.ndarray`
        The pixel coordinates of the peaks, in row-major order
        (ascending y, then x).

    n_pixels : 1D `~numpy.ndarray` or `None`
        The number of pixels in the plateau of each peak. `None` is
        returned if there are no plateaus.
    """
    if min_separation < 1:
        # Adjacent pixels are outside of each other's circular region,
        # so they are separate peaks that can have different values
        return *peak_mask.nonzero(), None

    # Diagonal neighbors are within the circular region, and therefore
    # are equal valued, only if the radius is at least sqrt(2) pixels
    use_diagonals = min_separation**2 >= 2

    # Checking for adjacent peak pixels is much faster than labeling
    # the peak pixels, and most images have no plateaus
    has_plateau = (np.any(peak_mask[:, 1:] & peak_mask[:, :-1])
                   or np.any(peak_mask[1:] & peak_mask[:-1]))
    if not has_plateau and use_diagonals:
        has_plateau = (np.any(peak_mask[1:, 1:] & peak_mask[:-1, :-1])
                       or np.any(peak_mask[1:, :-1] & peak_mask[:-1, 1:]))
    if not has_plateau:
        return *peak_mask.nonzero(), None

    structure = np.ones((3, 3)) if use_diagonals else None
    labels, n_labels = label(peak_mask, structure=structure)
    y_peaks, x_peaks = peak_mask.nonzero()
    idx = labels[y_peaks, x_peaks] - 1
    counts = np.bincount(idx, minlength=n_labels)

    # The squared distance of a pixel from the centroid of its label
    # is (y**2 + x**2) - 2 * (y * y_sum + x * x_sum) / count plus a
    # constant. It is multiplied by the count and calculated with
    # integers so that equal distances compare as equal. The sums are
    # integer valued and are exact as floats.
    y_sum = np.bincount(idx, weights=y_peaks, minlength=n_labels)
    x_sum = np.bincount(idx, weights=x_peaks, minlength=n_labels)
    y_sum = y_sum.astype(np.int64)[idx]
    x_sum = x_sum.astype(np.int64)[idx]
    y_pix = y_peaks.astype(np.int64)
    x_pix = x_peaks.astype(np.int64)
    dist = (counts[idx] * (y_pix**2 + x_pix**2)
            - 2 * (y_pix * y_sum + x_pix * x_sum))

    # Sort by label and then by the distance from the label centroid.
    # The first pixel of each label is then the one to keep.
    order = np.lexsort((dist, idx))
    first = np.ones(len(order), dtype=bool)
    first[1:] = idx[order[1:]] != idx[order[:-1]]
    keep = np.sort(order[first])  # row-major order

    return y_peaks[keep], x_peaks[keep], counts[idx[keep]]


def _remove_tied_peaks(x_peaks, y_peaks, peak_values, min_separation,
                       n_pixels=None):
    """
    Remove equal-valued peaks that are too close to each other.

    The input peaks must each be the maximum within the circle of
    radius ``min_separation`` centered on it. Two such peaks can be
    separated by ``min_separation`` or less only if they have equal
    values, so only the peaks with a repeated value are searched. This
    function is therefore not valid for peaks found in any other way.

    Of the equal-valued peaks that are too close to each other, the
    peak of the larger plateau is kept, so that a saturated star is
    preferred over a nearby isolated saturated pixel. For equal-sized
    plateaus, the first peak in row-major order (ascending y, then x)
    is kept. The peaks are selected greedily in that order, so a peak
    is removed only if it is too close to a peak that is kept.

    Parameters
    ----------
    x_peaks : 1D `~numpy.ndarray`
        The x pixel coordinates of the peaks.

    y_peaks : 1D `~numpy.ndarray`
        The y pixel coordinates of the peaks.

    peak_values : 1D `~numpy.ndarray`
        The peak values.

    min_separation : float
        The minimum separation (in pixels) between peaks. Peaks
        separated by exactly this distance are too close, which matches
        the circular region used to find the peaks.

    n_pixels : 1D `~numpy.ndarray` or `None`, optional
        The number of pixels in the plateau of each peak. If `None`,
        every peak is a single pixel.

    Returns
    -------
    x_peaks, y_peaks, peak_values : tuple of `~numpy.ndarray`
        The filtered arrays, in the same row-major order (ascending y,
        then x) as the input arrays.
    """
    # Sorting the values is much faster than finding the index of the
    # unique values, and most images have no repeated peak values
    sorted_values = np.sort(peak_values)
    if not np.any(sorted_values[1:] == sorted_values[:-1]):
        return x_peaks, y_peaks, peak_values

    _, inverse, counts = np.unique(peak_values, return_inverse=True,
                                   return_counts=True)
    repeated = np.nonzero(counts[inverse] > 1)[0]

    # Sort by descending intensity and then by descending plateau size
    # (the stable sort preserves the row-major order for ties). The
    # index of the unique values is used as the intensity rank because
    # unsigned integer values cannot be negated.
    if n_pixels is None:
        n_pixels = np.ones(len(x_peaks), dtype=int)
    order = repeated[np.lexsort((-n_pixels[repeated], -inverse[repeated]))]
    coords = np.column_stack((x_peaks[order], y_peaks[order]))

    # An unbalanced tree is much faster to build, and is as fast to
    # query, for peaks that are spread over the image
    tree = cKDTree(coords, balanced_tree=False, compact_nodes=False)

    # Estimate the mean number of pairs per peak from a sample of the
    # peaks, which is much faster than counting all the pairs. The
    # count for a peak includes the peak itself, and each pair is
    # counted from both of its peaks.
    n_peaks = len(order)
    step = max(1, n_peaks // _N_SAMPLE_PEAKS)
    n_close = tree.query_ball_point(coords[::step], min_separation,
                                    return_length=True)
    pairs_per_peak = (np.mean(n_close) - 1) / 2

    keep = np.ones(n_peaks, dtype=bool)
    if pairs_per_peak <= _MAX_PAIRS_PER_PEAK:
        # Each row is a pair of peaks (i, j) with i < j
        pairs = tree.query_pairs(min_separation, output_type='ndarray')
        if len(pairs) == 0:
            return x_peaks, y_peaks, peak_values

        pairs = pairs[np.argsort(pairs[:, 0], kind='stable')]
        first, starts = np.unique(pairs[:, 0], return_index=True)
        neighbors = np.split(pairs[:, 1], starts[1:])
        for i, later in zip(first, neighbors, strict=True):
            if keep[i]:
                keep[later] = False
    else:
        # With many pairs per peak, the array of all the pairs can need
        # more memory than the image, but few peaks are kept. Finding
        # the neighbors of only the kept peaks is then faster and needs
        # little memory.
        for i in range(n_peaks):
            if keep[i]:
                neighbors = np.array(tree.query_ball_point(coords[i],
                                                           min_separation))
                keep[neighbors[neighbors > i]] = False

    good = np.ones(len(x_peaks), dtype=bool)
    good[order[~keep]] = False

    return x_peaks[good], y_peaks[good], peak_values[good]


def _verify_ring_candidates(data, peak_mask, needs_verify, footprint_bool,
                            half, footprint_size):
    """
    Verify ring candidates against the exact circular footprint.

    Ring candidates are pixels that are the local maximum within the
    inscribed box but not in the circumscribed box. These need per-pixel
    verification against the actual circular footprint.

    Parameters
    ----------
    data : 2D `~numpy.ndarray`
        The 2D image array.

    peak_mask : 2D bool `~numpy.ndarray`
        Boolean mask to update in place. `True` indicates a confirmed
        local maximum.

    needs_verify : 2D bool `~numpy.ndarray`
        Boolean mask of candidate pixels that require verification.

    footprint_bool : 2D bool `~numpy.ndarray`
        The circular footprint boolean mask.

    half : int
        Half the footprint size (``footprint_size // 2``), used to
        center the footprint on each candidate pixel.

    footprint_size : int
        The size of the circular footprint array along each axis.
    """
    y_maybe, x_maybe = needs_verify.nonzero()
    if len(y_maybe) == 0:
        return

    ny, nx = data.shape
    for y, x in zip(y_maybe, x_maybe, strict=True):
        # Map footprint onto data, clipping to image boundaries
        y0 = y - half
        y1 = y0 + footprint_size
        x0 = x - half
        x1 = x0 + footprint_size

        dy0, dy1 = max(0, y0), min(ny, y1)
        dx0, dx1 = max(0, x0), min(nx, x1)

        fy0 = dy0 - y0
        fy1 = footprint_size - (y1 - dy1)
        fx0 = dx0 - x0
        fx1 = footprint_size - (x1 - dx1)

        local = data[dy0:dy1, dx0:dx1]
        fp_local = footprint_bool[fy0:fy1, fx0:fx1]
        local_max = local[fp_local].max()

        # Footprint extends beyond image: include cval=0.0
        if (fy0 > 0 or fy1 < footprint_size or fx0 > 0
                or fx1 < footprint_size):
            local_max = max(local_max, 0.0)

        # peak_mask is updated in place
        if data[y, x] == local_max:
            peak_mask[y, x] = True


def _circular_footprint(radius):
    """
    Make a circular footprint centered on a pixel.

    Parameters
    ----------
    radius : float
        The radius of the circular region in pixels.

    Returns
    -------
    footprint : 2D bool `~numpy.ndarray`
        A square boolean array with an odd size along each axis. `True`
        indicates a pixel whose center is within ``radius`` pixels of
        the central pixel.
    """
    half = int(radius)
    idx = np.arange(-half, half + 1)
    xx, yy = np.meshgrid(idx, idx)
    return (xx ** 2 + yy ** 2) <= radius ** 2


def _fast_circular_peaks(data, radius):
    """
    Find pixels that are local maxima within circular regions.

    This is equivalent to::

        footprint = _circular_footprint(radius)
        data_max = maximum_filter(data, footprint=footprint,
                                  mode='constant', cval=0.0)
        peaks = (data == data_max)

    For a radius of at least ``_MIN_FAST_RADIUS`` pixels, the result
    is computed with fast separable box filters and targeted circular
    verification, which is much faster for large radii. Smaller radii
    use the footprint filter directly because it is faster for them.

    Parameters
    ----------
    data : 2D `~numpy.ndarray`
        The 2D image array. Must be NaN-free because comparisons with
        NaN are always `False`, which would corrupt the local-maximum
        comparisons.

    radius : float
        The radius of the circular region in pixels.

    Returns
    -------
    peak_mask : 2D bool `~numpy.ndarray`
        Boolean mask where `True` indicates a local maximum within the
        circular region.
    """
    footprint_bool = _circular_footprint(radius)

    if radius < _MIN_FAST_RADIUS:
        data_max = maximum_filter(data, footprint=footprint_bool,
                                  mode='constant', cval=0.0)
        return data == data_max

    footprint_size = footprint_bool.shape[0]
    half = footprint_size // 2

    # Circumscribed box (size = footprint_size): contains the footprint.
    # Any pixel that is the max in this box is definitely the max in the
    # circular footprint, since circle <= box.
    data_max_box = maximum_filter(data, size=footprint_size, mode='constant',
                                  cval=0.0)
    definite = (data == data_max_box)

    # Inscribed box: the largest centered box that fits inside the
    # circle
    half_side = int(radius / np.sqrt(2))
    side_insc = 2 * half_side + 1

    data_max_insc = maximum_filter(data, size=side_insc, mode='constant',
                                   cval=0.0)
    # Candidates from inscribed box are a superset of true peaks
    candidates = (data == data_max_insc)

    # Ring candidates: max in inscribed box but not in circumscribed
    # box. These need per-pixel verification against the actual circular
    # footprint.
    needs_verify = candidates & ~definite
    peak_mask = definite.copy()

    # peak_mask is updated in place
    _verify_ring_candidates(data, peak_mask, needs_verify, footprint_bool,
                            half, footprint_size)

    return peak_mask


@deprecated_renamed_argument('npeaks', 'n_peaks', '3.0', until='4.0')
def find_peaks(data, threshold, *, box_size=3, footprint=None, mask=None,
               border_width=None, n_peaks=np.inf, min_separation=None,
               centroid_func=None, error=None, wcs=None):
    """
    Find local peaks in an image that are above a specified threshold
    value.

    Peaks are the maxima above the ``threshold`` within a local
    region. The local regions are defined by either the ``box_size``
    or ``footprint`` parameters. ``box_size`` defines the local region
    around each pixel as a square box. ``footprint`` is a boolean array
    where `True` values specify the region shape.

    If multiple pixels within a local region have identical intensities,
    then the coordinates of all such pixels are returned. Otherwise,
    there will be only one peak pixel per local region. Thus, the
    defined region effectively imposes a minimum separation between
    peaks unless there are identical peaks within the region.

    When ``min_separation`` is set, a fast algorithm is used that
    finds local maxima within a circular region of the given radius.
    Pixels with identical intensities are then reduced so that all
    returned peaks are separated by more than ``min_separation``
    pixels. When set, ``box_size`` and ``footprint`` are not used for
    peak detection.

    If ``centroid_func`` is input, then it will be used to calculate a
    centroid within the defined local region centered on each detected
    peak pixel. In this case, the centroid will also be returned in the
    output table.

    Parameters
    ----------
    data : array_like
        The 2D array of the image.

    threshold : float, scalar `~astropy.units.Quantity` or array_like
        The data value or pixel-wise data values to be used
        for the detection threshold. A peak is detected only
        if it is strictly greater than the ``threshold``. If
        ``data`` is a `~astropy.units.Quantity` array, then
        ``threshold`` must have the same units as ``data``. A 2D
        ``threshold`` must have the same shape as ``data``. See
        `~photutils.segmentation.detect_threshold` for one way to create
        a ``threshold`` image.

    box_size : scalar or tuple, optional
        The size of the local region to search for peaks at every point
        in ``data``. If ``box_size`` is a scalar, then the region
        shape will be ``(box_size, box_size)``. If ``box_size`` has
        two elements, they must be in ``(ny, nx)`` order. ``box_size``
        must have odd values for both axes if ``centroid_func`` is
        input. Either ``box_size`` or ``footprint`` must be defined. If
        they are both defined, then ``footprint`` overrides
        ``box_size``.

    footprint : `~numpy.ndarray` of bools, optional
        A boolean array where `True` values describe the local
        footprint region within which to search for peaks at every
        point in ``data``. ``box_size=(n, m)`` is equivalent to
        ``footprint=np.ones((n, m))``. Either ``box_size`` or
        ``footprint`` must be defined. If they are both defined, then
        ``footprint`` overrides ``box_size``.

    mask : array_like, bool, optional
        A boolean mask with the same shape as ``data``, where a `True`
        value indicates the corresponding element of ``data`` is masked.
        Masked pixels can neither be detected as peaks nor suppress
        nearby peaks within the local region (they are treated the same
        as NaN pixels).

        .. versionchanged:: 3.1
            Masked pixels no longer suppress nearby peaks within the
            local region. Previously, a masked pixel with a value larger
            than a nearby true peak prevented that peak from being
            detected.

    border_width : int, array_like of int, or None, optional
        The width in pixels to exclude around the border of the
        ``data``. If ``border_width`` is a scalar then ``border_width``
        will be applied to all sides. If ``border_width`` has two
        elements, they must be in ``(ny, nx)`` order. If `None`, then no
        border is excluded. The border width values must be nonnegative
        integers.

    n_peaks : int or `numpy.inf`, optional
        The maximum number of peaks to return. It must be a positive
        integer or `numpy.inf` (the default, meaning all peaks
        are returned). When the number of detected peaks exceeds
        ``n_peaks``, the peaks with the highest peak intensities will be
        returned.

    min_separation : float or None, optional
        The minimum allowed separation (in pixels) between detected
        peaks. When set to a positive value, a fast algorithm first
        identifies the local maxima within a circular region of this
        radius centered on each pixel. A group of connected peak
        pixels, which have identical values (e.g., a saturated star),
        is then replaced by its pixel nearest to the centroid of the
        group. Finally, only one of any remaining equal-valued peaks
        that are separated by ``min_separation`` or less is kept (see
        Notes). All returned peaks are therefore separated by more
        than ``min_separation`` pixels. Peaks separated by exactly
        ``min_separation`` are too close. This is much faster for large
        radii than using an explicit circular ``footprint`` of the
        given radius. When set to a positive value, ``box_size`` and
        ``footprint`` are not used for peak detection. If `None`
        (default) or 0, the peak detection uses ``box_size`` or
        ``footprint`` as specified. To return every pixel that is tied
        for the maximum, use a circular ``footprint`` instead.

        .. versionchanged:: 3.1
            Pixels with identical values that are within
            ``min_separation`` of each other now give only one peak.
            Previously, every pixel that was tied for the maximum was
            returned.

    centroid_func : callable, optional
        A callable object (e.g., function or class) that is used to
        calculate the centroid of a 2D array. The ``centroid_func``
        must accept a 2D `~numpy.ndarray`, have a ``mask`` keyword, and
        optionally an ``error`` keyword. The callable object must return
        two scalar values representing the (x, y) centroid.

    error : array_like, optional
        The 2D array of the 1-sigma errors of the input ``data``.
        ``error`` is used only if ``centroid_func`` is input (the
        ``error`` array is passed directly to the ``centroid_func``).
        If the ``centroid_func`` does not accept an ``error``
        keyword, the ``error`` array is ignored. If ``data`` is a
        `~astropy.units.Quantity` array, then ``error`` must have the
        same units as ``data``.

    wcs : `None` or WCS object, optional
        A world coordinate system (WCS) transformation that
        supports the `astropy shared interface for WCS
        <https://docs.astropy.org/en/stable/wcs/wcsapi.html>`_
        (e.g., `astropy.wcs.WCS`, `gwcs.wcs.WCS`). If `None`, then
        the sky coordinates will not be returned in the output
        `~astropy.table.QTable`.

    Returns
    -------
    output : `~astropy.table.QTable` or `None`
        A table containing the x and y pixel location of the peaks
        and their values, in the ``id``, ``x_peak``, ``y_peak``, and
        ``peak_value`` columns. The table also contains a
        ``skycoord_peak`` column if ``wcs`` is input, ``x_centroid``
        and ``y_centroid`` columns if ``centroid_func`` is input, and a
        ``skycoord_centroid`` column if both are input. If no peaks are
        found then `None` is returned.

    Notes
    -----
    By default, the returned pixel coordinates are the integer indices
    of the maximum pixel value within the input ``box_size`` or
    ``footprint`` (i.e., only the peak pixel is identified).

    When ``min_separation`` is given, peaks are detected in three
    steps. First, the local maxima within a circular ``footprint`` of
    the given radius, centered on each pixel, are identified. For a
    radius of at least 4 pixels, the algorithm uses two fast O(N)
    separable box filters (inscribed and circumscribed squares of the
    circle) to classify most candidates, then verifies only the
    remaining few against the exact circular region. Smaller radii use
    the footprint filter directly.

    The local-maximum step returns every pixel that is tied for the
    maximum, so the next two steps handle pixels with identical
    values. Second, each group of connected peak pixels (a plateau,
    e.g., the pixels of a saturated star) is replaced by its pixel
    nearest to the centroid of the group. A plateau therefore gives
    one peak, even if it is larger than ``min_separation``. Adjacent
    peak pixels are connected only if they are within
    ``min_separation`` of each other. Diagonal neighbors are therefore
    connected only for a ``min_separation`` of at least ``sqrt(2)``
    pixels, and this step is skipped for a ``min_separation`` smaller
    than 1 pixel, for which every pixel above the threshold is a peak.

    Third, for any remaining equal-valued peaks that are separated by
    ``min_separation`` or less, only one is kept. This includes the
    peak of a plateau, which can be removed in favor of a nearby
    equal-valued peak. The peak of the plateau with the most pixels is
    kept, so that a saturated star is preferred over a nearby isolated
    saturated pixel. For plateaus of equal size, including single
    pixels, the first peak in row-major order (ascending y, then x) is
    kept.

    A centroiding function can be input via the ``centroid_func``
    keyword to compute centroid coordinates with subpixel precision
    within the input ``box_size`` or ``footprint``. Note that when
    ``min_separation`` is used, the centroid region is determined by
    ``footprint`` if it is input and otherwise by ``box_size`` (default
    3), not by ``min_separation``.

    The peak detection uses ``mode='constant'`` with ``cval=0.0`` for
    `~scipy.ndimage.maximum_filter`, which means pixels outside the
    image boundary are treated as zero. For images with all-negative
    values, this may suppress legitimate peaks near the borders.

    Masked pixels and any NaN values in the input ``data`` are replaced
    with the minimum non-NaN data value before peak detection, so they
    can neither be detected as peaks nor suppress nearby peaks. NaN
    pixels are automatically excluded from the results.

    The output column names (``x_peak``, ``y_peak``, ``peak_value``)
    differ from the star finder classes (e.g.,
    `~photutils.detection.DAOStarFinder`), which use ``x_centroid``,
    ``y_centroid``, and ``flux``.
    """
    arrays, unit = process_quantities((data, threshold, error),
                                      ('data', 'threshold', 'error'))
    data, threshold, error = arrays
    data = np.asanyarray(data)

    if data.size == 0:
        msg = 'data must not be empty'
        raise ValueError(msg)

    if centroid_func is not None and not callable(centroid_func):
        msg = 'centroid_func must be a callable object'
        raise TypeError(msg)

    if min_separation is not None and min_separation < 0:
        msg = 'min_separation must be >= 0'
        raise ValueError(msg)

    if n_peaks != np.inf:
        if isinstance(n_peaks, bool):
            msg = 'n_peaks must be an integer'
            raise TypeError(msg)
        if n_peaks <= 0:
            msg = 'n_peaks must be > 0'
            raise ValueError(msg)
        n_peaks_int = int(n_peaks)
        if n_peaks_int != n_peaks:
            msg = 'n_peaks must be an integer'
            raise ValueError(msg)
        n_peaks = n_peaks_int

    if np.all(data == data.flat[0]):
        msg = 'Input data is constant. No local peaks can be found.'
        warnings.warn(msg, NoDetectionsWarning)
        return None

    if not np.isscalar(threshold):
        threshold = np.asanyarray(threshold)
        if data.shape != threshold.shape:
            msg = ('threshold array must have the same shape as the '
                   'input data')
            raise ValueError(msg)

    if border_width is not None:
        border_width = as_pair('border_width', border_width,
                               lower_bound=(0, 1), upper_bound=data.shape)

    # Combine the input mask with a mask of NaN pixels. Masked and
    # NaN pixels are replaced by the minimum finite data value before
    # peak detection so that they can neither be detected as peaks nor
    # suppress nearby peaks within the local region.
    nan_mask = np.isnan(data)
    if mask is not None:
        mask = np.asanyarray(mask, dtype=bool)
        if data.shape != mask.shape:
            msg = 'data and mask must have the same shape'
            raise ValueError(msg)
        mask = mask | nan_mask
    elif np.any(nan_mask):
        mask = nan_mask

    if mask is not None and np.any(mask):
        data = np.copy(data)  # do not mutate the input data
        data[mask] = nanmin(data)

    # peak_goodmask: good pixels are True
    use_min_separation = min_separation is not None and min_separation > 0
    if use_min_separation:
        peak_goodmask = _fast_circular_peaks(data, min_separation)
    elif footprint is not None:
        data_max = maximum_filter(data, footprint=footprint, mode='constant',
                                  cval=0.0)
        peak_goodmask = (data == data_max)
    else:
        data_max = maximum_filter(data, size=box_size, mode='constant',
                                  cval=0.0)
        peak_goodmask = (data == data_max)

    # Exclude peaks that are masked
    if mask is not None:
        peak_goodmask = np.logical_and(peak_goodmask, ~mask)

    # Exclude peaks that are too close to the border
    if border_width is not None:
        ny, nx = border_width
        if ny > 0:
            peak_goodmask[:ny, :] = False
            peak_goodmask[-ny:, :] = False
        if nx > 0:
            peak_goodmask[:, :nx] = False
            peak_goodmask[:, -nx:] = False

    # Exclude peaks below the threshold
    peak_goodmask = np.logical_and(peak_goodmask, (data > threshold))

    # The circular local maxima include every pixel that is tied for
    # the maximum. Reduce each plateau of connected peak pixels to one
    # pixel, then remove the remaining equal-valued peaks that are
    # still too close to each other.
    if use_min_separation:
        y_peaks, x_peaks, n_pixels = _collapse_plateaus(peak_goodmask,
                                                        min_separation)
        peak_values = data[y_peaks, x_peaks]
        x_peaks, y_peaks, peak_values = _remove_tied_peaks(
            x_peaks, y_peaks, peak_values, min_separation,
            n_pixels=n_pixels)
    else:
        y_peaks, x_peaks = peak_goodmask.nonzero()
        peak_values = data[y_peaks, x_peaks]

    if unit is not None:
        peak_values <<= unit

    n_x_peaks = len(x_peaks)
    if n_x_peaks == 0:
        msg = 'No local peaks were found.'
        warnings.warn(msg, NoDetectionsWarning)
        return None

    if n_x_peaks > n_peaks:
        idx = np.argsort(peak_values)[::-1][:n_peaks]
        x_peaks = x_peaks[idx]
        y_peaks = y_peaks[idx]
        peak_values = peak_values[idx]

    # Construct the output table
    ids = np.arange(len(x_peaks)) + 1
    colnames = ['id', 'x_peak', 'y_peak', 'peak_value']
    coldata = [ids, x_peaks, y_peaks, peak_values]
    table = QTable(coldata, names=colnames)
    table.meta.update(_get_meta())  # keep table.meta type

    if wcs is not None:
        skycoord_peaks = wcs.pixel_to_world(x_peaks, y_peaks)
        idx = table.colnames.index('y_peak') + 1
        table.add_column(skycoord_peaks, name='skycoord_peak', index=idx)

    # Perform centroiding
    if centroid_func is not None:
        # Prevent circular import
        from photutils.centroids import centroid_sources

        # When a footprint is provided, derive the centroid box_size
        # from the footprint shape so they are consistent. Ensure odd
        # dimensions for centroid_sources.
        if footprint is not None:
            centroid_box_size = tuple(
                s if s % 2 else s + 1 for s in footprint.shape)
        else:
            centroid_box_size = box_size
        # Pass the error array only to centroid functions that accept
        # an error keyword. Otherwise, it is ignored.
        centroid_kwargs = {}
        if error is not None:
            spec = inspect.signature(centroid_func)
            if ('error' in spec.parameters
                    or any(param.kind == inspect.Parameter.VAR_KEYWORD
                           for param in spec.parameters.values())):
                centroid_kwargs['error'] = error

        x_centroids, y_centroids = centroid_sources(
            data, x_peaks, y_peaks, box_size=centroid_box_size,
            footprint=footprint, mask=mask,
            centroid_func=centroid_func, **centroid_kwargs)

        table['x_centroid'] = x_centroids
        table['y_centroid'] = y_centroids

        if wcs is not None:
            skycoord_centroids = wcs.pixel_to_world(x_centroids, y_centroids)
            idx = table.colnames.index('y_centroid') + 1
            table.add_column(skycoord_centroids, name='skycoord_centroid',
                             index=idx)

    return table
