# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Define tools for finding local peaks in an astronomical image.
"""

import warnings

import numpy as np
from astropy.table import QTable
from scipy.ndimage import maximum_filter

from photutils.utils._misc import _get_meta
from photutils.utils._parameters import as_pair
from photutils.utils._quantity_helpers import process_quantities
from photutils.utils._stats import nanmin
from photutils.utils.exceptions import NoDetectionsWarning

__all__ = ['find_peaks']


def find_peaks(data, threshold, *, box_size=3, footprint=None, mask=None,
               border_width=None, npeaks=np.inf, centroid_func=None,
               error=None, wcs=None):
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

    If ``centroid_func`` is input, then it will be used to calculate a
    centroid within the defined local region centered on each detected
    peak pixel. In this case, the centroid will also be returned in the
    output table.

    Parameters
    ----------
    data : array_like
        The 2D array of the image.

    threshold : float, scalar `~astropy.units.Quantity` or array_like
        The data value or pixel-wise data values to be used for the
        detection threshold. If ``data`` is a `~astropy.units.Quantity`
        array, then ``threshold`` must have the same units as ``data``.
        A 2D ``threshold`` must have the same shape as ``data``. See
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

    npeaks : int, optional
        The maximum number of peaks to return. When the number of
        detected peaks exceeds ``npeaks``, the peaks with the highest
        peak intensities will be returned.

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
    output : `~astropy.table.Table` or `None`
        A table containing the x and y pixel location of the peaks and
        their values. If ``centroid_func`` is input, then the table will
        also contain the centroid position. If no peaks are found then
        `None` is returned.

    Notes
    -----
    By default, the returned pixel coordinates are the integer indices
    of the maximum pixel value within the input ``box_size`` or
    ``footprint`` (i.e., only the peak pixel is identified). However, a
    centroiding function can be input via the ``centroid_func`` keyword
    to compute centroid coordinates with subpixel precision within the
    input ``box_size`` or ``footprint``.
    """
    arrays, unit = process_quantities((data, threshold, error),
                                      ('data', 'threshold', 'error'))
    data, threshold, error = arrays
    data = np.asanyarray(data)

    if np.all(data == data.flat[0]):
        warnings.warn('Input data is constant. No local peaks can be found.',
                      NoDetectionsWarning)
        return None

    if not np.isscalar(threshold):
        threshold = np.asanyarray(threshold)
        if data.shape != threshold.shape:
            msg = ('A threshold array must have the same shape as the '
                   'input data')
            raise ValueError(msg)

    if border_width is not None:
        border_width = as_pair('border_width', border_width,
                               lower_bound=(0, 0), upper_bound=data.shape)

    # remove NaN values to avoid runtime warnings
    nan_mask = np.isnan(data)
    if np.any(nan_mask):
        data = np.copy(data)  # ndarray
        data[nan_mask] = nanmin(data)

    if footprint is not None:
        data_max = maximum_filter(data, footprint=footprint, mode='constant',
                                  cval=0.0)
    else:
        data_max = maximum_filter(data, size=box_size, mode='constant',
                                  cval=0.0)

    peak_goodmask = (data == data_max)  # good pixels are True

    # Exclude peaks that are masked
    if mask is not None:
        mask = np.asanyarray(mask)
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

    nxpeaks = len(x_peaks)
    if nxpeaks > npeaks:
        idx = np.argsort(peak_values)[::-1][:npeaks]
        x_peaks = x_peaks[idx]
        y_peaks = y_peaks[idx]
        peak_values = peak_values[idx]

    if nxpeaks == 0:
        warnings.warn('No local peaks were found.', NoDetectionsWarning)
        return None

    # construct the output table
    ids = np.arange(len(x_peaks)) + 1
    colnames = ['id', 'x_peak', 'y_peak', 'peak_value']
    coldata = [ids, x_peaks, y_peaks, peak_values]
    table = QTable(coldata, names=colnames)
    table.meta.update(_get_meta())  # keep table.meta type

    if wcs is not None:
        skycoord_peaks = wcs.pixel_to_world(x_peaks, y_peaks)
        idx = table.colnames.index('y_peak') + 1
        table.add_column(skycoord_peaks, name='skycoord_peak', index=idx)

    # perform centroiding
    if centroid_func is not None:
        # prevent circular import
        from photutils.centroids import centroid_sources

        if not callable(centroid_func):
            msg = 'centroid_func must be a callable object'
            raise TypeError(msg)

        x_centroids, y_centroids = centroid_sources(
            data, x_peaks, y_peaks, box_size=box_size,
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
