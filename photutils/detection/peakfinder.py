# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tools for finding local peaks in an astronomical image.
"""

import warnings

import numpy as np
from astropy.table import QTable
from scipy.ndimage import maximum_filter

from photutils.utils._deprecation import deprecated_renamed_argument
from photutils.utils._misc import _get_meta
from photutils.utils._parameters import as_pair
from photutils.utils._quantity_helpers import process_quantities
from photutils.utils._stats import nanmin
from photutils.utils.exceptions import NoDetectionsWarning

__all__ = ['find_peaks']


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


def _fast_circular_peaks(data, radius):
    """
    Find pixels that are local maxima within circular regions.

    This is equivalent to::

        idx = np.arange(-radius, radius + 1)
        xx, yy = np.meshgrid(idx, idx)
        footprint = np.array((xx**2 + yy**2) <= radius**2, dtype=int)
        data_max = maximum_filter(data, footprint=footprint,
                                  mode='constant', cval=0.0)
        peaks = (data == data_max)

    but uses fast separable box filters with targeted circular
    verification, which is typically ~10-400x faster (depending on the
    radius).

    Parameters
    ----------
    data : 2D `~numpy.ndarray`
        The 2D image array. Must be NaN-free because
        `~scipy.ndimage.maximum_filter` propagates NaNs, which would
        corrupt the local-maximum comparisons.

    radius : float
        The radius of the circular region in pixels.

    Returns
    -------
    peak_mask : 2D bool `~numpy.ndarray`
        Boolean mask where `True` indicates a local maximum within the
        circular region.
    """
    # Build the circular footprint
    idx = np.arange(-radius, radius + 1)
    radius_sq = radius ** 2
    footprint_size = len(idx)

    xx, yy = np.meshgrid(idx, idx)
    footprint_bool = (xx ** 2 + yy ** 2) <= radius_sq

    # For even-sized footprints (non-integer radius), scipy's
    # maximum_filter places the center at index ``footprint_size // 2``
    # (i.e., the origin is biased by +0.5 pixel). The same convention is
    # used here so that the fast path is bit-identical to the reference
    # maximum_filter(footprint=...) result.
    half = footprint_size // 2

    # Circumscribed box (size = footprint_size): contains the footprint.
    # Any pixel that is the max in this box is definitely the max in the
    # circular footprint, since circle <= box.
    data_max_box = maximum_filter(data, size=footprint_size, mode='constant',
                                  cval=0.0)
    definite = (data == data_max_box)

    # Inscribed box: fits inside the circle. For even-sized footprints,
    # the circle center is shifted by 0.5 from the pixel center. We
    # account for this so the inscribed box stays inside the circle.
    if footprint_size % 2 == 0:
        half_side = int(np.floor(radius / np.sqrt(2) - 0.5))
    else:
        half_side = int(np.floor(radius / np.sqrt(2)))
    side_insc = max(2 * half_side + 1, 3)

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
    produces results equivalent to using a circular ``footprint`` of the
    given radius for `~scipy.ndimage.maximum_filter`, but is typically
    ~10-400x faster (depending on the radius). When set, ``box_size``
    and ``footprint`` are not used for peak detection.

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
        shape will be ``(box_size, box_size)``. Either ``box_size`` or
        ``footprint`` must be defined. If they are both defined, then
        ``footprint`` overrides ``box_size``.

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

    border_width : int, array_like of int, or None, optional
        The width in pixels to exclude around the border of the
        ``data``. If ``border_width`` is a scalar then ``border_width``
        will be applied to all sides. If ``border_width`` has two
        elements, they must be in ``(ny, nx)`` order. If `None`, then no
        border is excluded. The border width values must be non-negative
        integers.

    n_peaks : int, optional
        The maximum number of peaks to return. When the number of
        detected peaks exceeds ``n_peaks``, the peaks with the highest
        peak intensities will be returned.

    min_separation : float or None, optional
        The minimum allowed separation (in pixels) between detected
        peaks, enforced using a circular region of this radius. Each
        detected peak must be the maximum value (or tied for the
        maximum) within a circle of this radius. This is equivalent to
        using a circular ``footprint`` of the given radius but uses a
        fast algorithm that is typically ~10-400x faster (depending on
        the radius). When set, ``box_size`` and ``footprint`` are not
        used for peak detection. If `None` (default), the peak detection
        uses ``box_size`` or ``footprint`` as specified.

    centroid_func : callable, optional
        A callable object (e.g., function or class) that is used to
        calculate the centroid of a 2D array. The ``centroid_func``
        must accept a 2D `~numpy.ndarray`, have a ``mask`` keyword, and
        optionally an ``error`` keyword. The callable object must return
        a tuple of two 1D `~numpy.ndarray` objects, representing the x
        and y centroids, respectively.

    error : array_like, optional
        The 2D array of the 1-sigma errors of the input ``data``.
        ``error`` is used only if ``centroid_func`` is input (the
        ``error`` array is passed directly to the ``centroid_func``). If
        ``data`` is a `~astropy.units.Quantity` array, then ``error``
        must have the same units as ``data``.

    wcs : `None` or WCS object, optional
        A world coordinate system (WCS) transformation that
        supports the `astropy shared interface for WCS
        <https://docs.astropy.org/en/stable/wcs/wcsapi.html>`_
        (e.g., `astropy.wcs.WCS`, `gwcs.wcs.WCS`). If `None`, then
        the sky coordinates will not be returned in the output
        `~astropy.table.Table`.

    Returns
    -------
    output : `~astropy.table.QTable` or `None`
        A table containing the x and y pixel location of the peaks and
        their values. If ``centroid_func`` is input, then the table will
        also contain the centroid position. If no peaks are found then
        `None` is returned.

    Notes
    -----
    By default, the returned pixel coordinates are the integer indices
    of the maximum pixel value within the input ``box_size`` or
    ``footprint`` (i.e., only the peak pixel is identified).

    When ``min_separation`` is given, peaks are detected
    using a fast algorithm that is mathematically equivalent
    to a circular ``footprint`` of the given radius for
    `~scipy.ndimage.maximum_filter`. The algorithm uses two fast O(N)
    separable box filters (inscribed and circumscribed squares of
    the circle) to classify most candidates, then verifies only the
    remaining few against the exact circular region.

    A centroiding function can be input via the ``centroid_func``
    keyword to compute centroid coordinates with subpixel precision
    within the input ``box_size`` or ``footprint``. Note that when
    ``min_separation`` is used, the centroid region size is determined
    by ``box_size`` (default 3), not by ``min_separation``.

    The peak detection uses ``mode='constant'`` with ``cval=0.0`` for
    `~scipy.ndimage.maximum_filter`, which means pixels outside the
    image boundary are treated as zero. For images with all-negative
    values, this may suppress legitimate peaks near the borders.

    Any NaN values in the input ``data`` are replaced with the minimum
    finite value before peak detection, and the corresponding pixels are
    automatically excluded from the results.

    The output column names (``x_peak``, ``y_peak``, ``peak_value``)
    differ from the star finder classes (e.g.,
    `~photutils.detection.DAOStarFinder`), which use ``x_centroid``,
    ``y_centroid``, and ``flux``.
    """
    arrays, unit = process_quantities((data, threshold, error),
                                      ('data', 'threshold', 'error'))
    data, threshold, error = arrays
    data = np.asanyarray(data)

    if centroid_func is not None and not callable(centroid_func):
        msg = 'centroid_func must be a callable object'
        raise TypeError(msg)

    if min_separation is not None and min_separation < 0:
        msg = 'min_separation must be >= 0'
        raise ValueError(msg)

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

    # Remove NaN values to avoid runtime warnings and exclude NaN pixels
    # from peak detection
    nan_mask = np.isnan(data)
    if np.any(nan_mask):
        data = np.copy(data)  # ndarray
        data[nan_mask] = nanmin(data)
        mask = (nan_mask if mask is None
                else np.asanyarray(mask) | nan_mask)

    # peak_goodmask: good pixels are True
    if min_separation is not None and min_separation > 0:
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
        mask = np.asanyarray(mask, dtype=bool)
        if data.shape != mask.shape:
            msg = 'data and mask must have the same shape'
            raise ValueError(msg)
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
        x_centroids, y_centroids = centroid_sources(
            data, x_peaks, y_peaks, box_size=centroid_box_size,
            footprint=footprint, error=error, mask=mask,
            centroid_func=centroid_func)

        table['x_centroid'] = x_centroids
        table['y_centroid'] = y_centroids

        if wcs is not None:
            skycoord_centroids = wcs.pixel_to_world(x_centroids, y_centroids)
            idx = table.colnames.index('y_centroid') + 1
            table.add_column(skycoord_centroids, name='skycoord_centroid',
                             index=idx)

    return table
