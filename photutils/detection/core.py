# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import numpy as np
from ..extern.imageutils import sigmaclip_stats


__all__ = ['detect_threshold', 'detect_sources', 'find_peaks']


def detect_threshold(data, snr, background=None, error=None, mask=None,
                     mask_val=None, sigclip_sigma=3.0, sigclip_iters=None):
    """
    Calculate a pixel-wise threshold image to be used to detect sources.

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

    mask_val : float, optional
        An image data value (e.g., ``0.0``) that is ignored when
        computing the image background statistics.  ``mask_val`` will be
        ignored if ``mask`` is input.

    sigclip_sigma : float, optional
        The number of standard deviations to use as the clipping limit
        when calculating the image background statistics.

    sigclip_iters : float, optional
       The number of iterations to perform sigma clipping, or `None` to
       clip until convergence is achieved (i.e., continue until the last
       iteration clips nothing) when calculating the image background
       statistics.

    Returns
    -------
    threshold : 2D `numpy.ndarray`
        A 2D image with the same shape as ``data`` containing the
        pixel-wise threshold values.

    See Also
    --------
    detect_sources

    Notes
    -----
    The ``mask``, ``mask_val``, ``sigclip_sigma``, and ``sigclip_iters``
    inputs are used only if it is necessary to estimate ``background``
    or ``error`` using sigma-clipped background statistics.  If
    ``background`` and ``error`` are both input, then ``mask``,
    ``mask_val``, ``sigclip_sigma``, and ``sigclip_iters`` are ignored.
    """

    if background is None or error is None:
        data_mean, data_median, data_std = sigmaclip_stats(
            data, image_mask=mask, mask_val=mask_val, sigma=sigclip_sigma,
            iters=sigclip_iters)
        bkgrd_image = np.broadcast_arrays(data_mean, data)[0]
        bkgrdrms_image = np.broadcast_arrays(data_std, data)[0]

    if background is None:
        background = bkgrd_image
    else:
        if np.isscalar(background):
            background = np.broadcast_arrays(background, data)[0]
        else:
            if background.shape != data.shape:
                raise ValueError('If input background is 2D, then it '
                                 'must have the same shape as the input '
                                 'data.')

    if error is None:
        error = bkgrdrms_image
    else:
        if np.isscalar(error):
           error = np.broadcast_arrays(error, data)[0]
        else:
            if error.shape != data.shape:
                raise ValueError('If input error is 2D, then it '
                                 'must have the same shape as the input '
                                 'data.')

    return background + (error * snr)


def detect_sources(data, threshold, npixels, filter_fwhm=None,
                   filter_kernel=None, connectivity=8):
    """
    Detect sources above a specified threshold value in an image and
    return a segmentation image.

    Detected sources must have ``npixels`` connected pixels that are
    each greater than the ``threshold`` value.  If the filtering option
    is used, then the ``threshold`` is applied to the filtered image.

    This function does not deblend overlapping sources.

    Parameters
    ----------
    data : array_like
        The 2D array of the image.

    threshold : float or array-like
        The data value or pixel-wise data values to be used for the
        detection threshold.  A 2D ``threshold`` must have the same
        shape as ``data``.  See `detect_threshold` for one way to create
        a ``threshold`` image.

    npixels : int
        The number of connected pixels, each greater than ``threshold``,
        that an object must have to be detected.  ``npixels`` must be a
        positive integer.

    filter_fwhm : float, optional
        The FWHM of a circular 2D Gaussian filter that is applied to the
        input image before it is thresholded.  Filtering the image will
        maximize detectability of objects with a FWHM similar to
        ``filter_fwhm``.  ``filter_fwhm`` is ignored if
        ``filter_kernel`` is input.

    filter_kernel : array-like (2D), optional
        The 2D array of the kernel used to filter the image before
        thresholding.  ``filter_kernel`` overrides ``filter_fwhm``.

    connectivity : {4, 8}, optional
        The type of pixel connectivity used in determining how pixels
        are grouped into a detected source.  The options are 4 or 8
        (default).  4-connected pixels touch along their edges.
        8-connected pixels touch along their edges or corners.  For
        reference, SExtractor uses 8-connected pixels.

    Returns
    -------
    segment_image :  `numpy.ndarray`
        A 2D segmentation image, with the same shape as ``data``, where
        sources are marked by different positive integer values.  A
        value of zero is reserved for the background.

    See Also
    --------
    detect_threshold, segment_properties

    Examples
    --------

    .. plot::
        :include-source:

        # make a table of Gaussian sources
        from astropy.table import Table
        table = Table()
        table['amplitude'] = [50, 70, 150, 210]
        table['x_mean'] = [160, 25, 150, 90]
        table['y_mean'] = [70, 40, 25, 60]
        table['x_stddev'] = [15.2, 5.1, 3., 8.1]
        table['y_stddev'] = [2.6, 2.5, 3., 4.7]
        table['theta'] = np.array([145., 20., 0., 60.]) * np.pi / 180.

        # make an image of the sources with Gaussian noise
        from photutils.datasets import make_gaussian_sources
        from photutils.datasets import make_noise_image
        shape = (100, 200)
        sources = make_gaussian_sources(shape, table)
        noise = make_noise_image(shape, type='gaussian', mean=0.,
                                 stddev=5., random_state=12345)
        image = sources + noise

        # detect the sources
        from photutils import detect_threshold, detect_sources
        threshold = detect_threshold(image, snr=3)
        segm_image = detect_sources(image, threshold, npixels=5,
                                    filter_fwhm=3.)
        # plot the image and the segmentation image
        import matplotlib.pyplot as plt
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))
        ax1.imshow(image, origin='lower', interpolation='nearest')
        ax2.imshow(segm_image, origin='lower', interpolation='nearest')
    """

    from scipy import ndimage

    if (npixels <= 0) or (int(npixels) != npixels):
        raise ValueError('npixels must be a positive integer, got '
                         '"{0}"'.format(npixels))

    conv_mode = 'constant'    # SExtractor mode
    conv_val = 0.0
    if filter_kernel is not None:
        image = ndimage.convolve(data, filter_kernel, mode=conv_mode,
                                 cval=conv_val)
    else:
        if filter_fwhm is not None:
            image = ndimage.gaussian_filter(data, filter_fwhm, mode=conv_mode,
                                            cval=conv_val)
        else:
            image = data

    image = (image >= threshold)
    if connectivity == 4:
        selem = ndimage.generate_binary_structure(2, 1)
    elif connectivity == 8:    # e.g., SExtractor
        selem = ndimage.generate_binary_structure(2, 2)
    objlabels, nobj = ndimage.label(image, structure=selem)
    objslices = ndimage.find_objects(objlabels)

    # remove objects with less than npixels
    for objslice in objslices:
        objlabel = objlabels[objslice]
        obj_npix = len(np.where(objlabel.ravel() != 0)[0])
        if obj_npix < npixels:
            objlabels[objslice] = 0

    # relabel (labeled indices must be consecutive)
    objlabels, nobj = ndimage.label(objlabels, structure=selem)
    return objlabels


def find_peaks(data, snr_threshold, min_distance=5, exclude_border=True,
               indices=True, num_peaks=np.inf, footprint=None, labels=None,
               mask=None, mask_val=None, sig=3.0, iters=None):
    """
    Find peaks in an image above above a specified signal-to-noise ratio
    threshold and return them as coordinates or a boolean array.

    Peaks are the local maxima in a region of ``2 * min_distance + 1``
    (i.e. peaks are separated by at least ``min_distance``).

    NOTE: If peaks are flat (i.e. multiple adjacent pixels have
    identical intensities), the coordinates of all such pixels are
    returned.

    Parameters
    ----------
    data : array_like
        The 2D array of the image.

    snr_threshold : float
        The signal-to-noise ratio threshold above which to detect
        sources.  The background rms noise level is computed using
        sigma-clipped statistics, which can be controlled via the
        ``sig`` and ``iters`` keywords.

    min_distance : int
        Minimum number of pixels separating peaks in a region of ``2 *
        min_distance + 1`` (i.e. peaks are separated by at least
        ``min_distance``). If ``exclude_border`` is `True`, this value
        also excludes a border ``min_distance`` from the image boundary.
        To find the maximum number of peaks, use ``min_distance=1``.

    exclude_border : bool
        If `True`, ``min_distance`` excludes peaks from the border of
        the image as well as from each other.

    indices : bool
        If `True`, the output will be an array representing peak
        coordinates.  If `False`, the output will be a boolean array
        shaped as ``data.shape`` with peaks present at `True` elements.

    num_peaks : int
        Maximum number of peaks. When the number of peaks exceeds
        ``num_peaks``, return ``num_peaks`` peaks based on highest peak
        intensity.

    footprint : ndarray of bools, optional
        If provided, ``footprint == 1`` represents the local region
        within which to search for peaks at every point in ``data``.
        Overrides ``min_distance``, except for border exclusion if
        ``exclude_border=True``.

    labels : ndarray of ints, optional
        If provided, each unique region ``labels == value`` represents a
        unique region to search for peaks.  Zero is reserved for
        background.

    mask : array_like, bool, optional
        A boolean mask with the same shape as ``data``, where a `True`
        value indicates the corresponding element of ``data`` is
        invalid.  Masked pixels are ignored when computing the image
        background statistics.

    mask_val : float, optional
        An image data value (e.g., ``0.0``) that is ignored when
        computing the image background statistics.  ``mask_val`` will be
        ignored if ``mask`` is input.

    sig : float, optional
        The number of standard deviations to use as the clipping limit
        when calculating the image background statistics.

    iters : float, optional
       The number of iterations to perform clipping, or `None` to clip
       until convergence is achieved (i.e. continue until the last
       iteration clips nothing) when calculating the image background
       statistics.

    Returns
    -------
    output : ndarray or ndarray of bools

        * If ``indices = True`` : (row, column, ...) coordinates of
          peaks.
        * If ``indices = False`` : Boolean array shaped like ``data``,
          with peaks represented by True values.

    Notes
    -----
    The peak local maximum function returns the coordinates of local
    peaks (maxima) in a image. A maximum filter is used for finding
    local maxima.  This operation dilates the original image. After
    comparison between dilated and original image, peak_local_max
    function returns the coordinates of peaks where dilated image =
    original.
    """
    from skimage.feature import peak_local_max

    bkgrd, median, bkgrd_rms = sigmaclip_stats(
        data, image_mask=mask, mask_val=mask_val, sigma=sig, iters=iters)
    level = bkgrd + (bkgrd_rms * snr_threshold)
    return peak_local_max(data, min_distance=min_distance,
                          threshold_abs=level, threshold_rel=0.0,
                          exclude_border=exclude_border, indices=indices,
                          num_peaks=num_peaks, footprint=footprint,
                          labels=labels)
