# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
from astropy.stats import gaussian_fwhm_to_sigma
from astropy.convolution import Gaussian2DKernel

from .core import SegmentationImage
from ..detection import detect_threshold
from ..utils.convolution import filter_data


__all__ = ['detect_sources', 'make_source_mask']


def detect_sources(data, threshold, npixels, filter_kernel=None,
                   connectivity=8, mask=None):
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
        detection threshold.  A 2D ``threshold`` must have the same
        shape as ``data``.  See `~photutils.detection.detect_threshold`
        for one way to create a ``threshold`` image.

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

    mask : array_like (bool)
        A boolean mask, with the same shape as the input ``data``, where
        `True` values indicate masked pixels.  Masked pixels will not be
        included in any source.

    Returns
    -------
    segment_image : `~photutils.segmentation.SegmentationImage`
        A 2D segmentation image, with the same shape as ``data``, where
        sources are marked by different positive integer values.  A
        value of zero is reserved for the background.

    See Also
    --------
    :func:`photutils.detection.detect_threshold`,
    :class:`photutils.segmentation.SegmentationImage`,
    :func:`photutils.segmentation.source_properties`
    :func:`photutils.segmentation.deblend_sources`

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
        from photutils.datasets import make_gaussian_sources_image
        from photutils.datasets import make_noise_image
        shape = (100, 200)
        sources = make_gaussian_sources_image(shape, table)
        noise = make_noise_image(shape, type='gaussian', mean=0.,
                                 stddev=5., random_state=12345)
        image = sources + noise

        # detect the sources
        from photutils import detect_threshold, detect_sources
        threshold = detect_threshold(image, snr=3)
        from astropy.convolution import Gaussian2DKernel
        sigma = 3.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))   # FWHM = 3
        kernel = Gaussian2DKernel(sigma, x_size=3, y_size=3)
        kernel.normalize()
        segm = detect_sources(image, threshold, npixels=5,
                              filter_kernel=kernel)

        # plot the image and the segmentation image
        import matplotlib.pyplot as plt
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))
        ax1.imshow(image, origin='lower', interpolation='nearest')
        ax2.imshow(segm.data, origin='lower', interpolation='nearest')
    """

    from scipy import ndimage

    if (npixels <= 0) or (int(npixels) != npixels):
        raise ValueError('npixels must be a positive integer, got '
                         '"{0}"'.format(npixels))

    image = (filter_data(data, filter_kernel, mode='constant', fill_value=0.0,
                         check_normalization=True) > threshold)

    if mask is not None:
        if mask.shape != image.shape:
            raise ValueError('mask must have the same shape as the input '
                             'image.')
        image &= ~mask

    if connectivity == 4:
        selem = ndimage.generate_binary_structure(2, 1)
    elif connectivity == 8:
        selem = ndimage.generate_binary_structure(2, 2)
    else:
        raise ValueError('Invalid connectivity={0}.  '
                         'Options are 4 or 8'.format(connectivity))

    segm_img, nobj = ndimage.label(image, structure=selem)

    # remove objects with less than npixels
    # NOTE:  for typical data, making the cutout images is ~10x faster
    # than using segm_img directly
    segm_slices = ndimage.find_objects(segm_img)
    for i, slices in enumerate(segm_slices):
        cutout = segm_img[slices]
        segment_mask = (cutout == (i+1))
        if np.count_nonzero(segment_mask) < npixels:
            cutout[segment_mask] = 0

    # now relabel to make consecutive label indices
    segm_img, nobj = ndimage.label(segm_img, structure=selem)

    return SegmentationImage(segm_img)


def make_source_mask(data, snr, npixels, mask=None, mask_value=None,
                     filter_fwhm=None, filter_size=3, filter_kernel=None,
                     sigclip_sigma=3.0, sigclip_iters=5, dilate_size=11):
    """
    Make a source mask using source segmentation and binary dilation.

    Parameters
    ----------
    data : array_like
        The 2D array of the image.

    snr : float
        The signal-to-noise ratio per pixel above the ``background`` for
        which to consider a pixel as possibly being part of a source.

    npixels : int
        The number of connected pixels, each greater than ``threshold``,
        that an object must have to be detected.  ``npixels`` must be a
        positive integer.

    mask : array_like, bool, optional
        A boolean mask with the same shape as ``data``, where a `True`
        value indicates the corresponding element of ``data`` is masked.
        Masked pixels are ignored when computing the image background
        statistics.

    mask_value : float, optional
        An image data value (e.g., ``0.0``) that is ignored when
        computing the image background statistics.  ``mask_value`` will
        be ignored if ``mask`` is input.

    filter_fwhm : float, optional
        The full-width at half-maximum (FWHM) of the Gaussian kernel to
        filter the image before thresholding.  ``filter_fwhm`` and
        ``filter_size`` are ignored if ``filter_kernel`` is defined.

    filter_size : float, optional
        The size of the square Gaussian kernel image.  Used only if
        ``filter_fwhm`` is defined.  ``filter_fwhm`` and ``filter_size``
        are ignored if ``filter_kernel`` is defined.

    filter_kernel : array-like (2D) or `~astropy.convolution.Kernel2D`, optional
        The 2D array of the kernel used to filter the image before
        thresholding.  Filtering the image will smooth the noise and
        maximize detectability of objects with a shape similar to the
        kernel.  ``filter_kernel`` overrides ``filter_fwhm`` and
        ``filter_size``.

    sigclip_sigma : float, optional
        The number of standard deviations to use as the clipping limit
        when calculating the image background statistics.

    sigclip_iters : int, optional
       The number of iterations to perform sigma clipping, or `None` to
       clip until convergence is achieved (i.e., continue until the last
       iteration clips nothing) when calculating the image background
       statistics.

    dilate_size : int, optional
        The size of the square array used to dilate the segmentation
        image.

    Returns
    -------
    mask : 2D `~numpy.ndarray`, bool
        A 2D boolean image containing the source mask.
    """

    from scipy import ndimage

    threshold = detect_threshold(data, snr, background=None, error=None,
                                 mask=mask, mask_value=None,
                                 sigclip_sigma=sigclip_sigma,
                                 sigclip_iters=sigclip_iters)

    kernel = None
    if filter_kernel is not None:
        kernel = filter_kernel
    if filter_fwhm is not None:
        sigma = filter_fwhm * gaussian_fwhm_to_sigma
        kernel = Gaussian2DKernel(sigma, x_size=filter_size,
                                  y_size=filter_size)
    if kernel is not None:
        kernel.normalize()

    segm = detect_sources(data, threshold, npixels, filter_kernel=kernel)

    selem = np.ones((dilate_size, dilate_size))
    return ndimage.binary_dilation(segm.data.astype(np.bool), selem)
