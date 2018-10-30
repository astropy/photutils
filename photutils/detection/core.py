# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Functions for detecting sources in an astronomical image."""

import warnings

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.stats import sigma_clipped_stats
from astropy.table import Table
from astropy.utils.exceptions import (AstropyDeprecationWarning,
                                      AstropyUserWarning)
from astropy.version import version as astropy_version
from astropy.wcs.utils import pixel_to_skycoord

from ..utils.cutouts import cutout_footprint


__all__ = ['detect_threshold', 'find_peaks']


def detect_threshold(data, snr, background=None, error=None, mask=None,
                     mask_value=None, sigclip_sigma=3.0, sigclip_iters=None):
    """
    Calculate a pixel-wise threshold image that can be used to detect
    sources.

    Parameters
    ----------
    data : array_like
        The 2D array of the image.

    snr : float
        The signal-to-noise ratio per pixel above the ``background`` for
        which to consider a pixel as possibly being part of a source.

    background : float or array_like, optional
        The background value(s) of the input ``data``.  ``background``
        may either be a scalar value or a 2D image with the same shape
        as the input ``data``.  If the input ``data`` has been
        background-subtracted, then set ``background`` to ``0.0``.  If
        `None`, then a scalar background value will be estimated using
        sigma-clipped statistics.

    error : float or array_like, optional
        The Gaussian 1-sigma standard deviation of the background noise
        in ``data``.  ``error`` should include all sources of
        "background" error, but *exclude* the Poisson error of the
        sources.  If ``error`` is a 2D image, then it should represent
        the 1-sigma background error in each pixel of ``data``.  If
        `None`, then a scalar background rms value will be estimated
        using sigma-clipped statistics.

    mask : array_like, bool, optional
        A boolean mask with the same shape as ``data``, where a `True`
        value indicates the corresponding element of ``data`` is masked.
        Masked pixels are ignored when computing the image background
        statistics.

    mask_value : float, optional
        An image data value (e.g., ``0.0``) that is ignored when
        computing the image background statistics.  ``mask_value`` will
        be ignored if ``mask`` is input.

    sigclip_sigma : float, optional
        The number of standard deviations to use as the clipping limit
        when calculating the image background statistics.

    sigclip_iters : int, optional
       The number of iterations to perform sigma clipping, or `None` to
       clip until convergence is achieved (i.e., continue until the last
       iteration clips nothing) when calculating the image background
       statistics.

    Returns
    -------
    threshold : 2D `~numpy.ndarray`
        A 2D image with the same shape as ``data`` containing the
        pixel-wise threshold values.

    See Also
    --------
    :func:`photutils.segmentation.detect_sources`

    Notes
    -----
    The ``mask``, ``mask_value``, ``sigclip_sigma``, and
    ``sigclip_iters`` inputs are used only if it is necessary to
    estimate ``background`` or ``error`` using sigma-clipped background
    statistics.  If ``background`` and ``error`` are both input, then
    ``mask``, ``mask_value``, ``sigclip_sigma``, and ``sigclip_iters``
    are ignored.
    """

    if background is None or error is None:
        if astropy_version < '3.1':
            data_mean, data_median, data_std = sigma_clipped_stats(
                data, mask=mask, mask_value=mask_value, sigma=sigclip_sigma,
                iters=sigclip_iters)
        else:
            data_mean, data_median, data_std = sigma_clipped_stats(
                data, mask=mask, mask_value=mask_value, sigma=sigclip_sigma,
                maxiters=sigclip_iters)

        bkgrd_image = np.zeros_like(data) + data_mean
        bkgrdrms_image = np.zeros_like(data) + data_std

    if background is None:
        background = bkgrd_image
    else:
        if np.isscalar(background):
            background = np.zeros_like(data) + background
        else:
            if background.shape != data.shape:
                raise ValueError('If input background is 2D, then it '
                                 'must have the same shape as the input '
                                 'data.')

    if error is None:
        error = bkgrdrms_image
    else:
        if np.isscalar(error):
            error = np.zeros_like(data) + error
        else:
            if error.shape != data.shape:
                raise ValueError('If input error is 2D, then it '
                                 'must have the same shape as the input '
                                 'data.')

    return background + (error * snr)


def find_peaks(data, threshold, box_size=3, footprint=None, mask=None,
               border_width=None, npeaks=np.inf, centroid_func=None,
               subpixel=False, error=None, wcs=None):
    """
    Find local peaks in an image that are above above a specified
    threshold value.

    Peaks are the maxima above the ``threshold`` within a local region.
    The local regions are defined by either the ``box_size`` or
    ``footprint`` parameters.  ``box_size`` defines the local region
    around each pixel as a square box.  ``footprint`` is a boolean array
    where `True` values specify the region shape.

    If multiple pixels within a local region have identical intensities,
    then the coordinates of all such pixels are returned.  Otherwise,
    there will be only one peak pixel per local region.  Thus, the
    defined region effectively imposes a minimum separation between
    peaks unless there are identical peaks within the region.

    If ``centroid_func`` is input, then it will be used to calculate a
    centroid within the defined local region centered on each detected
    peak pixel.  In this case, the centroid will also be returned in the
    output table.

    Parameters
    ----------
    data : array_like
        The 2D array of the image.

    threshold : float or array-like
        The data value or pixel-wise data values to be used for the
        detection threshold.  A 2D ``threshold`` must have the same
        shape as ``data``.  See `detect_threshold` for one way to create
        a ``threshold`` image.

    box_size : scalar or tuple, optional
        The size of the local region to search for peaks at every point
        in ``data``.  If ``box_size`` is a scalar, then the region shape
        will be ``(box_size, box_size)``.  Either ``box_size`` or
        ``footprint`` must be defined.  If they are both defined, then
        ``footprint`` overrides ``box_size``.

    footprint : `~numpy.ndarray` of bools, optional
        A boolean array where `True` values describe the local footprint
        region within which to search for peaks at every point in
        ``data``.  ``box_size=(n, m)`` is equivalent to
        ``footprint=np.ones((n, m))``.  Either ``box_size`` or
        ``footprint`` must be defined.  If they are both defined, then
        ``footprint`` overrides ``box_size``.

    mask : array_like, bool, optional
        A boolean mask with the same shape as ``data``, where a `True`
        value indicates the corresponding element of ``data`` is masked.

    border_width : bool, optional
        The width in pixels to exclude around the border of the
        ``data``.

    npeaks : int, optional
        The maximum number of peaks to return.  When the number of
        detected peaks exceeds ``npeaks``, the peaks with the highest
        peak intensities will be returned.

    centroid_func : callable, optional
        A callable object (e.g. function or class) that is used to
        calculate the centroid of a 2D array.  The ``centroid_func``
        must accept a 2D `~numpy.ndarray`, have a ``mask`` keyword, and
        optionally an ``error`` keyword.  The callable object must
        return a tuple of two 1D `~numpy.ndarray`\\s, representing the x
        and y centroids, respectively.

    subpixel : bool, optional
        .. warning::

            Note the ``subpixel`` keyword is now deprecated (since
            v0.5).  To get the same centroid values, use the
            ``centroid_func`` keyword with the
            `~photutils.centroids.centroid_2dg` function.

            If `True`, then a cutout of the specified ``box_size`` or
            ``footprint`` will be taken centered on each peak and fit
            with a 2D Gaussian (plus a constant).  In this case, the
            fitted local centroid and peak value (the Gaussian amplitude
            plus the background constant) will also be returned in the
            output table.

    error : array_like, optional
        The 2D array of the 1-sigma errors of the input ``data``.
        ``error`` is used only with the ``centroid_func`` keyword or
        when ``subpixel=True`` (deprecated since v0.5).

    wcs : `~astropy.wcs.WCS`
        The WCS transformation to use to convert from pixel to sky
        coordinates.  If `None`, then the sky coordinates will not be
        returned in the output `~astropy.table.Table`.

    Returns
    -------
    output : `~astropy.table.Table`
        A table containing the x and y pixel location of the peaks and
        their values.  If ``centroid_func`` is input, then the table
        will also contain the centroid position.  If ``subpixel=True``
        (deprecated), then the table will also contain the local
        centroid and fitted peak value.  If no peaks are found then an
        empty table is returned.
    """

    if centroid_func is not None and subpixel:
        raise ValueError('centroid_func and subpixel (deprecated) cannot '
                         'both be used.')

    from scipy.ndimage import maximum_filter

    data = np.asanyarray(data)

    # Define an empty table with the expected column names and types.
    # This is done only so that we can return an empty table if no peaks
    # are found.
    columns = ['x_peak', 'y_peak', 'peak_value']
    coltypes = [np.float_, np.float_, np.float_]

    if centroid_func is not None or subpixel:
        columns.extend(['x_centroid', 'y_centroid'])
        coltypes.extend([np.float_, np.float_])

    if subpixel:
        columns.append('fit_peak_value')
        coltypes.append(np.float_)

    empty_table = Table(names=columns, dtype=coltypes)

    # now add SkyCoord mixin columns
    if wcs is not None:
        empty_table.add_column(SkyCoord([], [], unit='deg'),
                               name='skycoord_peak', index=2)

        if (centroid_func is not None or subpixel):
            idx = empty_table.colnames.index('y_centroid')
            empty_table.add_column(SkyCoord([], [], unit='deg'),
                                   name='skycoord_centroid', index=idx+1)

    if np.all(data == data.flat[0]):
        warnings.warn('Input data is constant. No local peaks can be found.',
                      AstropyUserWarning)
        return empty_table

    if not np.isscalar(threshold):
        threshold = np.asanyarray(threshold)

        if data.shape != threshold.shape:
            raise ValueError('A threshold array must have the same shape as '
                             'the input data.')

    # remove NaN values to avoid runtime warnings
    nan_mask = np.isnan(data)
    if np.any(nan_mask):
        data = np.copy(data)  # ndarray
        data[nan_mask] = np.nanmin(data)

    if footprint is not None:
        data_max = maximum_filter(data, footprint=footprint, mode='constant',
                                  cval=0.0)
    else:
        data_max = maximum_filter(data, size=box_size, mode='constant',
                                  cval=0.0)

    peak_goodmask = (data == data_max)    # good pixels are True

    if mask is not None:
        mask = np.asanyarray(mask)
        if data.shape != mask.shape:
            raise ValueError('data and mask must have the same shape')
        peak_goodmask = np.logical_and(peak_goodmask, ~mask)

    if border_width is not None:
        for i in range(peak_goodmask.ndim):
            peak_goodmask = peak_goodmask.swapaxes(0, i)
            peak_goodmask[:border_width] = False
            peak_goodmask[-border_width:] = False
            peak_goodmask = peak_goodmask.swapaxes(0, i)

    peak_goodmask = np.logical_and(peak_goodmask, (data > threshold))
    y_peaks, x_peaks = peak_goodmask.nonzero()
    peak_values = data[y_peaks, x_peaks]

    nxpeaks = len(x_peaks)
    if nxpeaks > npeaks:
        idx = np.argsort(peak_values)[::-1][:npeaks]
        x_peaks = x_peaks[idx]
        y_peaks = y_peaks[idx]
        peak_values = peak_values[idx]

    if nxpeaks == 0:
        warnings.warn('No local peaks were found.', AstropyUserWarning)
        return empty_table

    # construct the output Table
    colnames = ['x_peak', 'y_peak', 'peak_value']
    coldata = [x_peaks, y_peaks, peak_values]
    table = Table(coldata, names=colnames)

    if wcs is not None:
        skycoord_peaks = pixel_to_skycoord(x_peaks, y_peaks, wcs, origin=0)
        table.add_column(skycoord_peaks, name='skycoord_peak', index=2)

    # perform centroiding
    if centroid_func is not None:
        from ..centroids import centroid_sources  # prevents circular import

        if not callable(centroid_func):
            raise ValueError('centroid_func must be a callable object')

        x_centroids, y_centroids = centroid_sources(
            data, x_peaks, y_peaks, box_size=box_size,
            footprint=footprint, error=error, mask=mask,
            centroid_func=centroid_func)

        table['x_centroid'] = x_centroids
        table['y_centroid'] = y_centroids
    elif subpixel:
        warnings.warn('The subpixel keyword is deprecated and will be '
                      'removed in a future version.  The centroid_func '
                      'keyword can be used to calculate centroid positions.',
                      AstropyDeprecationWarning)

        from ..centroids import fit_2dgaussian  # prevents circular import

        x_centroids, y_centroids = [], []
        fit_peak_values = []
        for (y_peak, x_peak) in zip(y_peaks, x_peaks):
            rdata, rmask, rerror, slc = cutout_footprint(
                data, (x_peak, y_peak), box_size=box_size,
                footprint=footprint, mask=mask, error=error)
            gaussian_fit = fit_2dgaussian(rdata, mask=rmask, error=rerror)
            if gaussian_fit is None:
                x_cen, y_cen, fit_peak_value = np.nan, np.nan, np.nan
            else:
                x_cen = slc[1].start + gaussian_fit.x_mean.value
                y_cen = slc[0].start + gaussian_fit.y_mean.value
                fit_peak_value = (gaussian_fit.constant.value +
                                  gaussian_fit.amplitude.value)
            x_centroids.append(x_cen)
            y_centroids.append(y_cen)
            fit_peak_values.append(fit_peak_value)

        table['x_centroid'] = x_centroids
        table['y_centroid'] = y_centroids
        table['fit_peak_value'] = fit_peak_values

    if (centroid_func is not None or subpixel) and wcs is not None:
        skycoord_centroids = pixel_to_skycoord(x_centroids, y_centroids,
                                               wcs, origin=0)
        idx = table.colnames.index('y_centroid')
        table.add_column(skycoord_centroids, name='skycoord_centroid',
                         index=idx+1)

    return table
