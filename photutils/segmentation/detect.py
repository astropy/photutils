# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
This module provides tools for detecting sources in an image.
"""

import warnings

import numpy as np
from astropy.stats import SigmaClip

from photutils.segmentation.core import SegmentationImage
from photutils.segmentation.utils import _make_binary_structure
from photutils.utils._quantity_helpers import process_quantities
from photutils.utils._stats import nanmean, nanstd
from photutils.utils.exceptions import NoDetectionsWarning

__all__ = ['detect_threshold', 'detect_sources']


def detect_threshold(data, nsigma, *, background=None, error=None, mask=None,
                     sigma_clip=SigmaClip(sigma=3.0, maxiters=10)):
    """
    Calculate a pixel-wise threshold image that can be used to detect
    sources.

    This is a simple convenience function that uses sigma-clipped
    statistics to compute a scalar background and noise estimate. In
    general, one should perform more sophisticated estimates, e.g.,
    using `~photutils.background.Background2D`.

    Parameters
    ----------
    data : 2D `~numpy.ndarray`
        The 2D array of the image.

    nsigma : float
        The number of standard deviations per pixel above the
        ``background`` for which to consider a pixel as possibly being
        part of a source.

    background : float or 2D `~numpy.ndarray`, optional
        The background value(s) of the input ``data``. ``background``
        may either be a scalar value or a 2D array with the same
        shape as the input ``data``. If the input ``data`` has been
        background-subtracted, then set ``background`` to ``0.0`` (this
        should be typical). If `None`, then a scalar background value
        will be estimated as the sigma-clipped image mean.

    error : float or 2D `~numpy.ndarray`, optional
        The Gaussian 1-sigma standard deviation of the background
        noise in ``data``. ``error`` should include all sources of
        "background" error, but *exclude* the Poisson error of the
        sources. If ``error`` is a 2D image, then it should represent
        the 1-sigma background error in each pixel of ``data``. If
        `None`, then a scalar background rms value will be estimated
        as the sigma-clipped image standard deviation.

    mask : 2D bool `~numpy.ndarray`, optional
        A boolean mask with the same shape as ``data``, where a `True`
        value indicates the corresponding element of ``data`` is masked.
        Masked pixels are ignored when computing the image background
        statistics.

    sigma_clip : `astropy.stats.SigmaClip` instance, optional
        A `~astropy.stats.SigmaClip` object that defines the sigma
        clipping parameters.

    Returns
    -------
    threshold : 2D `~numpy.ndarray`
        A 2D image with the same shape as ``data`` containing the
        pixel-wise threshold values.

    See Also
    --------
    :class:`photutils.background.Background2D`
    :func:`photutils.segmentation.detect_sources`
    :class:`photutils.segmentation.SourceFinder`

    Notes
    -----
    The ``mask`` and ``sigma_clip`` inputs are used only if it
    is necessary to estimate ``background`` or ``error`` using
    sigma-clipped background statistics. If ``background`` and ``error``
    are both input, then ``mask`` and ``sigma_clip`` are ignored.
    """
    arrays, unit = process_quantities((data, background, error),
                                      ('data', 'background', 'error'))
    data, background, error = arrays

    if not isinstance(sigma_clip, SigmaClip):
        raise TypeError('sigma_clip must be a SigmaClip object')

    if background is None or error is None:
        if mask is not None:
            data = np.ma.MaskedArray(data, mask)

        clipped_data = sigma_clip(data, masked=False, return_bounds=False,
                                  copy=True)

    if background is None:
        background = nanmean(clipped_data)

    if not np.isscalar(background) and background.shape != data.shape:
        raise ValueError('If input background is 2D, then it must have the '
                         'same shape as the input data.')

    if error is None:
        error = nanstd(clipped_data)
    if not np.isscalar(error) and error.shape != data.shape:
        raise ValueError('If input error is 2D, then it must have the same '
                         'shape as the input data.')

    threshold = (np.broadcast_to(background, data.shape)
                 + np.broadcast_to(error * nsigma, data.shape))

    if unit:
        threshold <<= unit

    return threshold


def _detect_sources(data, thresholds, npixels, footprint, inverse_mask, *,
                    deblend_mode=False):
    """
    Detect sources above a specified threshold value in an image.

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
    data : 2D `~numpy.ndarray`
        The 2D array of the image. If filtering is desired, please input
        a convolved image here.

    thresholds : list of 2D `~numpy.ndarray` or 1D array of floats
        The data values (as a 1D array of floats) or pixel-wise data
        values (as a sequence of 2D arrays) to be used for the detection
        thresholds. 2D threshold arrays must have the same shape as
        ``data``.

    npixels : int
        The minimum number of connected pixels, each greater than
        ``threshold``, that an object must have to be detected.
        ``npixels`` must be a positive integer.

    footprint : array_like
        A footprint that defines feature connections. As an example,
        for connectivity along pixel edges only, the footprint is
        ``np.array([[0, 1, 0]], [1, 1, 1], [0, 1, 0]])``.

    inverse_mask : 2D bool `~numpy.ndarray`
        A boolean mask, with the same shape as the input ``data``, where
        `False` values indicate masked pixels (the inverse of usual
        pixel masks). Masked pixels will not be included in any source.

    deblend_mode : bool, optional
        If `True` do not include the segmentation image in the output
        list for any threshold level where the number of detected
        sources is less than 2. The deblend mode also does not relabel
        the output segmentation image to have consecutive label. This
        keyword improves performance of source deblending.

    Returns
    -------
    segment_image : list of `~photutils.segmentation.SegmentationImage`
        A list of 2D segmentation images, one for each input threshold,
        with the same shape as ``data``, where sources are marked
        by different positive integer values. A value of zero is
        reserved for the background. If no sources are found for a given
        threshold, then the output list will contain `None` for that
        threshold. Also see the ``deblend_mode`` keyword.
    """
    from scipy.ndimage import find_objects
    from scipy.ndimage import label as ndi_label

    segms = []
    for threshold in thresholds:
        # RuntimeWarning caused by > comparison when data contains NaNs
        # is ignored when calling _detect_sources
        segment_img = data > threshold

        if inverse_mask is not None:
            segment_img &= inverse_mask

        # return if threshold was too high to detect any sources
        if np.count_nonzero(segment_img) == 0:
            if not deblend_mode:
                warnings.warn('No sources were found.', NoDetectionsWarning)
                segms.append(None)
            continue

        # recasting segment_img to int and using output=segment_img
        # gives similar performance
        segment_img, nlabels = ndi_label(segment_img, structure=footprint)
        labels = np.arange(nlabels) + 1

        # remove objects with less than npixels
        # NOTE: making cutout images and setting their pixels to 0 is
        # ~10x faster than using segment_img directly and ~50% faster
        # than using ndimage.sum_labels.
        slices = find_objects(segment_img)
        segm_labels = []
        segm_slices = []
        for label, slc in zip(labels, slices):
            cutout = segment_img[slc]
            segment_mask = (cutout == label)
            if np.count_nonzero(segment_mask) < npixels:
                cutout[segment_mask] = 0
                continue
            segm_labels.append(label)
            segm_slices.append(slc)

        if np.count_nonzero(segment_img) == 0:
            if not deblend_mode:
                warnings.warn('No sources were found.', NoDetectionsWarning)
                segms.append(None)
            continue

        if not deblend_mode:
            # relabel the segmentation image with consecutive numbers
            nlabels = len(segm_labels)
            if len(labels) != nlabels:
                label_map = np.zeros(np.max(labels) + 1, dtype=int)
                labels = np.arange(nlabels) + 1
                label_map[segm_labels] = labels
                segment_img = label_map[segment_img]
        else:
            labels = segm_labels

        segm = object.__new__(SegmentationImage)
        segm._data = segment_img
        segm.__dict__['labels'] = labels
        segm.__dict__['slices'] = segm_slices

        if deblend_mode and segm.nlabels == 1:
            continue

        segms.append(segm)

    return segms


def detect_sources(data, threshold, npixels, *, connectivity=8, mask=None):
    """
    Detect sources above a specified threshold value in an image.

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
    data : 2D `~numpy.ndarray`
        The 2D array of the image. If filtering is desired, please input
        a convolved image here.

    threshold : float or 2D `~numpy.ndarray`
        The data value or pixel-wise data values to be used for the
        detection threshold. A 2D ``threshold`` array must have the same
        shape and units as ``data``.

    npixels : int
        The minimum number of connected pixels, each greater than
        ``threshold``, that an object must have to be detected.
        ``npixels`` must be a positive integer.

    connectivity : {4, 8}, optional
        The type of pixel connectivity used in determining how pixels
        are grouped into a detected source. The options are 4 or
        8 (default). 4-connected pixels touch along their edges.
        8-connected pixels touch along their edges or corners.

    mask : 2D bool `~numpy.ndarray`, optional
        A boolean mask, with the same shape as the input ``data``, where
        `True` values indicate masked pixels. Masked pixels will not be
        included in any source.

    Returns
    -------
    segment_image : `~photutils.segmentation.SegmentationImage` or `None`
        A 2D segmentation image, with the same shape as ``data``, where
        sources are marked by different positive integer values. A value
        of zero is reserved for the background. If no sources are found
        then `None` is returned.

    See Also
    --------
    :func:`photutils.segmentation.deblend_sources`
    :class:`photutils.segmentation.SourceFinder`

    Examples
    --------
    .. plot::
        :include-source:

        import matplotlib.pyplot as plt
        from astropy.convolution import convolve
        from astropy.stats import sigma_clipped_stats
        from astropy.visualization import simple_norm
        from photutils.datasets import make_100gaussians_image
        from photutils.segmentation import (detect_sources,
                                            make_2dgaussian_kernel)

        # make a simulated image
        data = make_100gaussians_image()

        # use sigma-clipped statistics to (roughly) estimate the background
        # background noise levels
        mean, _, std = sigma_clipped_stats(data)

        # subtract the background
        data -= mean

        # detect the sources
        threshold = 3. * std
        kernel = make_2dgaussian_kernel(3.0, size=3)  # FWHM = 3.
        convolved_data = convolve(data, kernel)
        segm = detect_sources(convolved_data, threshold, npixels=5)

        # plot the image and the segmentation image
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))
        norm = simple_norm(data, 'sqrt', percent=99.)
        ax1.imshow(data, origin='lower', interpolation='nearest',
                   norm=norm)
        ax2.imshow(segm.data, origin='lower', interpolation='nearest',
                   cmap=segm.make_cmap(seed=1234))
        plt.tight_layout()
    """
    _ = process_quantities((data, threshold), ('data', 'threshold'))

    if (npixels <= 0) or (int(npixels) != npixels):
        raise ValueError('npixels must be a positive integer, got '
                         f'"{npixels}"')

    if mask is not None:
        if mask.shape != data.shape:
            raise ValueError('mask must have the same shape as the input '
                             'image.')
        if mask.all():
            raise ValueError('mask must not be True for every pixel. There '
                             'are no unmasked pixels in the image to detect '
                             'sources.')
        inverse_mask = np.logical_not(mask)
    else:
        inverse_mask = None

    footprint = _make_binary_structure(data.ndim, connectivity)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)
        return _detect_sources(data, (threshold,), npixels, footprint,
                               inverse_mask, deblend_mode=False)[0]
