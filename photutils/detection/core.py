# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Functions for detecting sources in an astronomical image."""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import numpy as np
from astropy.convolution import Kernel2D
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
    threshold : 2D `~numpy.ndarray`
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
            data, mask=mask, mask_val=mask_val, sigma=sigclip_sigma,
            iters=sigclip_iters)
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


def detect_sources(data, threshold, npixels, filter_kernel=None,
                   connectivity=8):
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

    filter_kernel : array-like (2D) or `~astropy.convolution.Kernel2D`, optional
        The 2D array of the kernel used to filter the image before
        thresholding.  Filtering the image will smooth the noise and
        maximize detectability of objects with a shape similar to the
        kernel.

    connectivity : {4, 8}, optional
        The type of pixel connectivity used in determining how pixels
        are grouped into a detected source.  The options are 4 or 8
        (default).  4-connected pixels touch along their edges.
        8-connected pixels touch along their edges or corners.  For
        reference, SExtractor uses 8-connected pixels.

    Returns
    -------
    segment_image : `~numpy.ndarray` (int)
        A 2D segmentation image, with the same shape as ``data``, where
        sources are marked by different positive integer values.  A
        value of zero is reserved for the background.

    See Also
    --------
    detect_threshold, :class:`photutils.segmentation.segment_properties`

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
        from astropy.convolution import Gaussian2DKernel
        sigma = 3.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))   # FWHM = 3
        filter_kernel = Gaussian2DKernel(sigma, x_size=3, y_size=3)
        segm_image = detect_sources(image, threshold, npixels=5,
                                    filter_kernel=filter_kernel)

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
        if isinstance(filter_kernel, Kernel2D):
            image = ndimage.convolve(data, filter_kernel.array, mode=conv_mode,
                                     cval=conv_val)
        else:
            image = ndimage.convolve(data, filter_kernel, mode=conv_mode,
                                     cval=conv_val)
    else:
        image = data

    image = (image > threshold)
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

    # relabel to make sequential label indices
    objlabels, nobj = ndimage.label(objlabels, structure=selem)
    return objlabels


def find_peaks(data, threshold, min_separation=2, exclude_border=True,
               segment_image=None, npeaks=np.inf, footprint=None):
    """
    Find local peaks in an image that are above above a specified
    threshold value.

    Peaks are the local maxima above the ``threshold`` in a region of
    ``(2 * min_separation) + 1`` (i.e., peaks are separated by at least
    ``min_separation`` pixels).

    If peaks are flat (i.e., multiple adjacent pixels have identical
    intensities), then the coordinates of all such pixels are returned,
    even if they are not separated by at least ``min_separation``.

    Parameters
    ----------
    data : array_like
        The 2D array of the image.

    threshold : float
        The data value to be used for the detection threshold.

    min_separation : int, optional
        Minimum number of pixels separating peaks (i.e., peaks are
        separated by at least ``min_separation`` pixels).  To find the
        maximum number of peaks, use ``min_separation=1``.  If
        ``min_separation`` is not an integer, then it will be truncated.

    exclude_border : bool, optional
        If `True`, exclude peaks within ``min_separation`` from the
        border of the image as well as from each other.

    segment_image : `~numpy.ndarray` (int), optional
        If provided, then search for peaks located only within the
        labeled regions of a 2D segmentation image, where sources are
        marked by different positive integer values.  In the
        segmentation image a value of zero is reserved for the
        background.  ``segment_image`` must have the same shape as
        ``data``.

    npeaks : int, optional
        The maximum number of peaks to return.  When the number of
        detected peaks exceeds ``npeaks``, the peaks with the highest
        peak intensities will be returned.

    footprint : `~numpy.ndarray` of bools, optional
        A boolean array where `True` values describe the local footprint
        region within which to search for peaks at every point in
        ``data``.  Overrides ``min_separation``, except for border
        exclusion when ``exclude_border=True``.

    Returns
    -------
    output : `~numpy.ndarray`
        An ``Nx2`` array where the rows contain the ``(y, x)`` pixel
        coordinates of the local peaks.
    """

    if segment_image is not None:
        if segment_image.shape != data.shape:
            raise ValueError('segment_image and data must have the same '
                             'shape')

    from skimage.feature import peak_local_max
    coords = peak_local_max(data, min_distance=int(min_separation),
                            threshold_abs=threshold, threshold_rel=0.0,
                            exclude_border=exclude_border, indices=True,
                            num_peaks=npeaks, footprint=footprint,
                            labels=segment_image)
    if coords.shape[0] <= npeaks:
        return coords
    else:
        # NOTE: num_peaks is ignored by peak_local_max() if labels are input
        peak_values = data[coords[:, 0], coords[:, 1]]
        idx_maxsort = np.argsort(peak_values)[::-1]
        return coords[idx_maxsort][:npeaks]
