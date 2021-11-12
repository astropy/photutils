# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
This module provides tools for detecting sources in an image.
"""

import warnings

from astropy.convolution import Gaussian2DKernel, convolve
from astropy.stats import gaussian_fwhm_to_sigma, sigma_clipped_stats
from astropy.utils.decorators import deprecated_renamed_argument
from astropy.utils.exceptions import AstropyUserWarning
import numpy as np

from .core import SegmentationImage
from ..utils.exceptions import NoDetectionsWarning

__all__ = ['detect_threshold', 'detect_sources', 'make_source_mask']


def detect_threshold(data, nsigma, background=None, error=None, mask=None,
                     mask_value=None, sigclip_sigma=3.0, sigclip_iters=None):
    """
    Calculate a pixel-wise threshold image that can be used to detect
    sources.

    Parameters
    ----------
    data : array_like
        The 2D array of the image.

    nsigma : float
        The number of standard deviations per pixel above the
        ``background`` for which to consider a pixel as possibly being
        part of a source.

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
        The maximum number of iterations to perform sigma clipping, or
        `None` to clip until convergence is achieved (i.e., continue
        until the last iteration clips nothing) when calculating the
        image background statistics.

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
        data_mean, _, data_std = sigma_clipped_stats(
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

    return background + (error * nsigma)


def _make_binary_structure(ndim, connectivity):
    """
    Make a binary structure element.

    Parameters
    ----------
    ndim : int
        The number of array dimensions.

    connectivity : {4, 8}
        For the case of ``ndim=2``, the type of pixel connectivity used
        in determining how pixels are grouped into a detected source.
        The options are 4 or 8 (default). 4-connected pixels touch along
        their edges. 8-connected pixels touch along their edges or
        corners. For reference, SourceExtractor uses 8-connected pixels.

    Returns
    -------
    array : ndarray of int or bool
        The binary structure element.  If ``ndim <= 2`` an array of int
        is returned, otherwise an array of bool is returned.
    """
    from scipy.ndimage import generate_binary_structure

    if ndim == 1:
        selem = np.array((1, 1, 1))
    elif ndim == 2:
        if connectivity == 4:
            selem = np.array(((0, 1, 0), (1, 1, 1), (0, 1, 0)))
        elif connectivity == 8:
            selem = np.ones((3, 3), dtype=int)
        else:
            raise ValueError(f'Invalid connectivity={connectivity}.  '
                             'Options are 4 or 8.')
    else:
        selem = generate_binary_structure(ndim, 1)

    return selem


def _detect_sources(data, thresholds, npixels, kernel=None, connectivity=8,
                    mask=None, deblend_skip=False):
    """
    Detect sources above a specified threshold value in an image and
    return a `~photutils.segmentation.SegmentationImage` object.

    Detected sources must have ``npixels`` connected pixels that are
    each greater than the ``threshold`` value.  If the filtering option
    is used, then the ``threshold`` is applied to the filtered image.
    The input ``mask`` can be used to mask pixels in the input data.
    Masked pixels will not be included in any source.

    This function does not deblend overlapping sources.  First use this
    function to detect sources followed by
    :func:`~photutils.segmentation.deblend_sources` to deblend sources.

    Parameters
    ----------
    data : array_like
        The 2D array of the image.

    thresholds : array-like of floats or arrays
        The data value or pixel-wise data values to
        be used for the detection thresholds. A 2D
        ``threshold`` must have the same shape as ``data``. See
        `~photutils.segmentation.detect_threshold` for one way to create
        a ``threshold`` image.

    npixels : int
        The number of connected pixels, each greater than ``threshold``,
        that an object must have to be detected.  ``npixels`` must be a
        positive integer.

    kernel : array-like (2D) or `~astropy.convolution.Kernel2D`, optional
        The 2D array of the kernel used to filter the image before
        thresholding. Filtering the image will smooth the noise and
        maximize detectability of objects with a shape similar to the
        kernel.

    connectivity : {4, 8}, optional
        The type of pixel connectivity used in determining how pixels
        are grouped into a detected source. The options are 4 or
        8 (default). 4-connected pixels touch along their edges.
        8-connected pixels touch along their edges or corners. For
        reference, SourceExtractor uses 8-connected pixels.

    mask : array_like of bool, optional
        A boolean mask, with the same shape as the input ``data``, where
        `True` values indicate masked pixels.  Masked pixels will not be
        included in any source.

    deblend_skip : bool, optional
        If `True` do not include the segmentation image in the output
        list for any threshold level where the number of detected
        sources is less than 2.  This is useful for source deblending
        and improves its performance.

    Returns
    -------
    segment_image : list of `~photutils.segmentation.SegmentationImage`
        A list of 2D segmentation images, with the same shape as
        ``data``, where sources are marked by different positive integer
        values.  A value of zero is reserved for the background.  If no
        sources are found for a given threshold, then the output list
        will contain `None` for that threshold.  Also see the
        ``deblend_skip`` keyword.
    """
    from scipy import ndimage

    if (npixels <= 0) or (int(npixels) != npixels):
        raise ValueError('npixels must be a positive integer, got '
                         f'"{npixels}"')

    if mask is not None:
        if mask.shape != data.shape:
            raise ValueError('mask must have the same shape as the input '
                             'image.')

    if kernel is not None:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', AstropyUserWarning)
            data = convolve(data, kernel, mask=mask, normalize_kernel=True)

    selem = _make_binary_structure(data.ndim, connectivity)

    segms = []
    for threshold in thresholds:
        # ignore RuntimeWarning caused by > comparison when data contains NaNs
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
            data2 = data > threshold

        if mask is not None:
            data2 &= ~mask

        # return if threshold was too high to detect any sources
        if np.count_nonzero(data2) == 0:
            warnings.warn('No sources were found.', NoDetectionsWarning)
            if not deblend_skip:
                segms.append(None)
            continue

        segm_img, _ = ndimage.label(data2, structure=selem)

        # remove objects with less than npixels
        # NOTE:  for typical data, making the cutout images is ~10x faster
        # than using segm_img directly
        segm_slices = ndimage.find_objects(segm_img)
        for i, slices in enumerate(segm_slices):
            cutout = segm_img[slices]
            segment_mask = (cutout == (i + 1))
            if np.count_nonzero(segment_mask) < npixels:
                cutout[segment_mask] = 0

        if np.count_nonzero(segm_img) == 0:
            warnings.warn('No sources were found.', NoDetectionsWarning)
            if not deblend_skip:
                segms.append(None)
            continue

        segm = object.__new__(SegmentationImage)
        segm._data = segm_img

        if deblend_skip and segm.nlabels == 1:
            continue

        segm.relabel_consecutive()
        segms.append(segm)

    return segms


@deprecated_renamed_argument('filter_kernel', 'kernel', '1.2')
def detect_sources(data, threshold, npixels, kernel=None, connectivity=8,
                   mask=None):
    """
    Detect sources above a specified threshold value in an image and
    return a `~photutils.segmentation.SegmentationImage` object.

    Detected sources must have ``npixels`` connected pixels that are
    each greater than the ``threshold`` value.  If the filtering option
    is used, then the ``threshold`` is applied to the filtered image.
    The input ``mask`` can be used to mask pixels in the input data.
    Masked pixels will not be included in any source.

    This function does not deblend overlapping sources.  First use this
    function to detect sources followed by
    :func:`~photutils.segmentation.deblend_sources` to deblend sources.

    Parameters
    ----------
    data : array_like
        The 2D array of the image.

    threshold : float or array-like
        The data value or pixel-wise data values to be used for the
        detection threshold. A 2D ``threshold`` must have the same shape
        as ``data``. See `~photutils.segmentation.detect_threshold` for
        one way to create a ``threshold`` image.

    npixels : int
        The number of connected pixels, each greater than ``threshold``,
        that an object must have to be detected.  ``npixels`` must be a
        positive integer.

    kernel : array-like (2D) or `~astropy.convolution.Kernel2D`, optional
        The 2D array of the kernel used to filter the image before
        thresholding. Filtering the image will smooth the noise and
        maximize detectability of objects with a shape similar to the
        kernel.

    connectivity : {4, 8}, optional
        The type of pixel connectivity used in determining how pixels
        are grouped into a detected source. The options are 4 or
        8 (default). 4-connected pixels touch along their edges.
        8-connected pixels touch along their edges or corners. For
        reference, SourceExtractor uses 8-connected pixels.

    mask : array_like of bool, optional
        A boolean mask, with the same shape as the input ``data``, where
        `True` values indicate masked pixels.  Masked pixels will not be
        included in any source.

    Returns
    -------
    segment_image : `~photutils.segmentation.SegmentationImage` or `None`
        A 2D segmentation image, with the same shape as ``data``, where
        sources are marked by different positive integer values.  A
        value of zero is reserved for the background.  If no sources
        are found then `None` is returned.

    See Also
    --------
    :func:`photutils.segmentation.detect_threshold`
    :class:`photutils.segmentation.SegmentationImage`
    :func:`photutils.segmentation.source_properties`
    :func:`photutils.segmentation.deblend_sources`

    Examples
    --------
    .. plot::
        :include-source:

        from astropy.convolution import Gaussian2DKernel
        from astropy.stats import gaussian_fwhm_to_sigma
        from astropy.visualization import simple_norm
        import matplotlib.pyplot as plt
        from photutils.datasets import make_100gaussians_image
        from photutils.segmentation import detect_threshold, detect_sources

        # make a simulated image
        data = make_100gaussians_image()

        # detect the sources
        threshold = detect_threshold(data, nsigma=3)
        sigma = 3.0 * gaussian_fwhm_to_sigma   # FWHM = 3.
        kernel = Gaussian2DKernel(sigma, x_size=3, y_size=3)
        kernel.normalize()
        segm = detect_sources(data, threshold, npixels=5, kernel=kernel)

        # plot the image and the segmentation image
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))
        norm = simple_norm(data, 'sqrt', percent=99.)
        ax1.imshow(data, origin='lower', interpolation='nearest',
                   norm=norm)
        ax2.imshow(segm.data, origin='lower', interpolation='nearest',
                   cmap=segm.make_cmap(seed=1234))
        plt.tight_layout()
    """
    return _detect_sources(data, (threshold,), npixels, kernel=kernel,
                           connectivity=connectivity, mask=mask)[0]


@deprecated_renamed_argument('filter_kernel', 'kernel', '1.2')
def make_source_mask(data, nsigma, npixels, mask=None, filter_fwhm=None,
                     filter_size=3, kernel=None, sigclip_sigma=3.0,
                     sigclip_iters=5, dilate_size=11):
    """
    Make a source mask using source segmentation and binary dilation.

    Parameters
    ----------
    data : array_like
        The 2D array of the image.

    nsigma : float
        The number of standard deviations per pixel above the
        ``background`` for which to consider a pixel as possibly being
        part of a source.

    npixels : int
        The number of connected pixels, each greater than ``threshold``,
        that an object must have to be detected.  ``npixels`` must be a
        positive integer.

    mask : array_like, bool, optional
        A boolean mask with the same shape as ``data``, where a `True`
        value indicates the corresponding element of ``data`` is masked.
        Masked pixels are ignored when computing the image background
        statistics.

    filter_fwhm : float, optional
        The full-width at half-maximum (FWHM) of the Gaussian kernel to
        filter the image before thresholding.  ``filter_fwhm`` and
        ``filter_size`` are ignored if ``kernel`` is defined.

    filter_size : float, optional
        The size of the square Gaussian kernel image.  Used only if
        ``filter_fwhm`` is defined.  ``filter_fwhm`` and ``filter_size``
        are ignored if ``kernel`` is defined.

    kernel : array-like (2D) or `~astropy.convolution.Kernel2D`, optional
        The 2D array of the kernel used to filter the image before
        thresholding. Filtering the image will smooth the noise
        and maximize detectability of objects with a shape similar
        to the kernel. ``kernel`` overrides ``filter_fwhm`` and
        ``filter_size``.

    sigclip_sigma : float, optional
        The number of standard deviations to use as the clipping limit
        when calculating the image background statistics.

    sigclip_iters : int, optional
        The maximum number of iterations to perform sigma clipping, or
        `None` to clip until convergence is achieved (i.e., continue
        until the last iteration clips nothing) when calculating the
        image background statistics.

    dilate_size : int, optional
        The size of the square array used to dilate the segmentation
        image.

    Returns
    -------
    mask : 2D bool `~numpy.ndarray`
        A 2D boolean image containing the source mask.
    """
    from scipy import ndimage

    threshold = detect_threshold(data, nsigma, background=None, error=None,
                                 mask=mask, sigclip_sigma=sigclip_sigma,
                                 sigclip_iters=sigclip_iters)

    if kernel is None and filter_fwhm is not None:
        kernel_sigma = filter_fwhm * gaussian_fwhm_to_sigma
        kernel = Gaussian2DKernel(kernel_sigma, x_size=filter_size,
                                  y_size=filter_size)
    if kernel is not None:
        kernel.normalize()

    segm = detect_sources(data, threshold, npixels, kernel=kernel)
    if segm is None:
        return np.zeros(data.shape, dtype=bool)

    selem = np.ones((dilate_size, dilate_size))
    return ndimage.binary_dilation(segm.data.astype(bool), selem)
