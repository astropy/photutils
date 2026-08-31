# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tools for detecting sources in an image.
"""

import warnings

import numpy as np
from astropy.stats import SigmaClip
from scipy.ndimage import find_objects
from scipy.ndimage import label as ndi_label

from photutils.segmentation.core import SegmentationImage
from photutils.segmentation.utils import _make_binary_structure
from photutils.utils._deprecation import deprecated_renamed_argument
from photutils.utils._parameters import (SigmaClipSentinelDefault,
                                         create_default_sigmaclip)
from photutils.utils._quantity_helpers import check_units, process_quantities
from photutils.utils._stats import nanmean, nanstd
from photutils.utils.exceptions import NoDetectionsWarning

__all__ = ['detect_sources', 'detect_threshold']


SIGMA_CLIP = SigmaClipSentinelDefault(sigma=3.0, maxiters=10)


@deprecated_renamed_argument('nsigma', 'n_sigma', '3.0', until='4.0')
def detect_threshold(data, n_sigma, *, background=None, error=None, mask=None,
                     sigma_clip=SIGMA_CLIP):
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

    n_sigma : float
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

    sigma_clip : `astropy.stats.SigmaClip` or `None`, optional
        A `~astropy.stats.SigmaClip` object that defines the sigma
        clipping parameters. If `None` then no sigma clipping will be
        performed.

    Returns
    -------
    threshold : 2D `~numpy.ndarray`
        A 2D image with the same shape (and units) as ``data``
        containing the pixel-wise threshold values.

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
    inputs = (data, background, error)
    names = ('data', 'background', 'error')
    inputs, unit = process_quantities(inputs, names)
    (data, background, error) = inputs

    if sigma_clip is SIGMA_CLIP:
        sigma_clip = create_default_sigmaclip(sigma=SIGMA_CLIP.sigma,
                                              maxiters=SIGMA_CLIP.maxiters)
    if not isinstance(sigma_clip, SigmaClip):
        msg = 'sigma_clip must be a SigmaClip object'
        raise TypeError(msg)

    if background is None or error is None:
        if mask is not None:
            data = np.ma.MaskedArray(data, mask)

        clipped_data = sigma_clip(data, masked=False, return_bounds=False,
                                  copy=True)

    if background is None:
        background = nanmean(clipped_data)

    if not np.isscalar(background) and background.shape != data.shape:
        msg = ('If input background is 2D, then it must have the same '
               'shape as the input data.')
        raise ValueError(msg)

    if error is None:
        error = nanstd(clipped_data)
    if not np.isscalar(error) and error.shape != data.shape:
        msg = ('If input error is 2D, then it must have the same shape '
               'as the input data.')
        raise ValueError(msg)

    threshold = (np.broadcast_to(background, data.shape)
                 + np.broadcast_to(error * n_sigma, data.shape))

    if unit:
        threshold <<= unit

    return threshold


def _detect_sources(data, threshold, n_pixels, footprint, inverse_mask):
    """
    Detect sources above a specified threshold value in an image.

    Detected sources must have ``n_pixels`` connected pixels that are
    each greater than the ``threshold`` value in the input ``data``.

    This function is the core algorithm for detecting sources in
    an image used by `detect_sources`. This function differs from
    `detect_sources` in that it does not perform any boilerplate checks,
    it accepts a ``footprint`` argument instead of a ``connectivity``
    argument, and it accepts an ``inverse_mask`` argument instead of a
    ``mask`` argument.

    Parameters
    ----------
    data : 2D `~numpy.ndarray`
        The 2D array of the image. If filtering is desired, please input
        a convolved image.

    threshold : float or 2D `~numpy.ndarray`
        The data value or pixel-wise data values to be used for the
        detection threshold. If ``data`` is a `~astropy.units.Quantity`
        array, then ``threshold`` must have the same units as ``data``.
        A 2D ``threshold`` array must have the same shape as ``data``.

    n_pixels : int
        The minimum number of connected pixels, each greater than
        ``threshold``, that an object must have to be detected.
        ``n_pixels`` must be a positive integer.

    footprint : array_like
        A footprint that defines feature connections. As an example,
        for connectivity along pixel edges only, the footprint is
        ``np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])``.

    inverse_mask : 2D bool `~numpy.ndarray`
        A boolean mask, with the same shape as the input ``data``, where
        `False` values indicate masked pixels (the inverse of usual
        pixel masks). Masked pixels will not be included in any source.

    Returns
    -------
    segment_image : `~photutils.segmentation.SegmentationImage` or `None`
        A 2D segmentation image, with the same shape as ``data``, where
        sources are marked by different positive integer values. A value
        of zero is reserved for the background. If no sources are found
        then `None` is returned.
    """
    # NaN values compare as False, so NaN pixels are never included
    # in any source
    segment_img = data > threshold

    if inverse_mask is not None:
        segment_img &= inverse_mask

    # Return None if threshold was too high to detect any sources
    if np.count_nonzero(segment_img) == 0:
        return None

    # NOTE: recasting segment_img to int and using output=segment_img
    # gives similar performance
    segment_img, n_labels = ndi_label(segment_img, structure=footprint)
    labels = np.arange(n_labels, dtype=segment_img.dtype) + 1

    # Remove objects with less than n_pixels
    # NOTE: making cutout images and setting their pixels to 0 is
    # ~10x faster than using segment_img directly and ~50% faster
    # than using ndimage.sum_labels.
    slices = find_objects(segment_img)
    segm_labels = []
    segm_slices = []
    segm_areas = []
    for label, slc in zip(labels, slices, strict=True):
        cutout = segment_img[slc]
        segment_mask = (cutout == label)
        # The pixel count is the segment area, so it is kept to seed
        # the SegmentationImage areas instead of being recomputed
        area = np.count_nonzero(segment_mask)
        if area < n_pixels:
            cutout[segment_mask] = 0
            continue
        segm_labels.append(label)
        segm_slices.append(slc)
        segm_areas.append(area)

    if not segm_labels:
        return None

    # Relabel the segmentation image with consecutive numbers;
    # ndimage.label returns segment_img with dtype = np.int32
    # unless the input array has more than 2**31 - 1 pixels
    n_labels = len(segm_labels)
    if len(labels) != n_labels:
        label_map = np.zeros(np.max(labels) + 1,
                             dtype=segment_img.dtype)
        labels = np.arange(n_labels, dtype=segment_img.dtype) + 1
        label_map[segm_labels] = labels
        segment_img = label_map[segment_img]

    return SegmentationImage._from_data(segment_img, labels=labels,
                                        areas=np.array(segm_areas),
                                        slices=segm_slices)


def _detect_sources_deblend(data, threshold, n_pixels, *, footprint,
                            segment_mask):
    """
    Detect sources for a single multithreshold level during deblending.

    This is the deblending analogue of `_detect_sources`. It differs in
    that the detected segments keep their (possibly non-consecutive)
    label numbers from `~scipy.ndimage.label`, the small segments are
    removed with a bincount-based area filter (the per-label cutout
    loop used by `_detect_sources` has a fixed per-label overhead that
    dominates for the small cutouts and the many calls made during
    deblending), and `None` is returned when fewer than two segments are
    found.

    Parameters
    ----------
    data : 2D `~numpy.ndarray`
        The cutout data array for a single source.

    threshold : float
        The data value to be used for the detection threshold.

    n_pixels : int
        The minimum number of connected pixels, each greater than
        ``threshold``, that an object must have to be detected.

    footprint : array_like
        A footprint that defines feature connections.

    segment_mask : 2D bool `~numpy.ndarray`
        A boolean mask of the source segment, with the same shape as
        ``data``. Pixels outside the segment will not be included in
        any source.

    Returns
    -------
    segment_img : 2D int `~numpy.ndarray` or `None`
        A 2D segmentation image, with the same shape as ``data``,
        where sources are marked by different positive integer
        values. A value of zero is reserved for the background. If
        fewer than two sources are found then `None` is returned.
    """
    # NaN values compare as False, so NaN pixels are never included
    # in any source. The comparison is never empty because the
    # deblending thresholds are strictly below the source maximum.
    segment_img = data > threshold
    segment_img &= segment_mask

    segment_img, n_labels = ndi_label(segment_img, structure=footprint)

    # Remove objects with less than n_pixels
    areas = np.bincount(segment_img.ravel())
    keep = areas >= n_pixels
    keep[0] = False
    n_keep = np.count_nonzero(keep)
    if n_keep <= 1:
        return None

    if n_keep < n_labels:
        label_map = np.where(
            keep, np.arange(areas.size, dtype=segment_img.dtype), 0)
        segment_img = label_map[segment_img]

    return segment_img


@deprecated_renamed_argument('npixels', 'n_pixels', '3.0', until='4.0')
def detect_sources(data, threshold, n_pixels, *, connectivity=8, mask=None):
    """
    Detect sources above a specified threshold value in an image.

    Detected sources must have ``n_pixels`` connected pixels that are
    each greater than the ``threshold`` value in the input ``data``. The
    input ``mask`` can be used to mask pixels in the input data. Masked
    pixels will not be included in any source.

    This function does not deblend overlapping sources.
    First use this function to detect sources followed by
    :func:`~photutils.segmentation.deblend_sources` to deblend sources.
    Alternatively, use the :class:`~photutils.segmentation.SourceFinder`
    class to detect and deblend sources in a single step.

    Parameters
    ----------
    data : 2D `~numpy.ndarray`
        The 2D array of the image. If filtering is desired, please input
        a convolved image.

    threshold : float or 2D `~numpy.ndarray`
        The data value or pixel-wise data values to be used for the
        detection threshold. If ``data`` is a `~astropy.units.Quantity`
        array, then ``threshold`` must have the same units as ``data``.
        A 2D ``threshold`` array must have the same shape as ``data``.

    n_pixels : int
        The minimum number of connected pixels, each greater than
        ``threshold``, that an object must have to be detected.
        ``n_pixels`` must be a positive integer.

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

    Raises
    ------
    NoDetectionsWarning
        If no sources are found.

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
        from astropy.visualization import simple_norm
        from photutils.background import Background2D, MedianBackground
        from photutils.datasets import make_100gaussians_image
        from photutils.segmentation import (detect_sources,
                                            make_2dgaussian_kernel)

        # Make a simulated image
        data = make_100gaussians_image()

        # Estimate the background using Background2D and subtract it
        bkg_estimator = MedianBackground()
        bkg = Background2D(data, (50, 50), filter_size=(3, 3),
                           bkg_estimator=bkg_estimator)
        data -= bkg.background  # subtract the background

        # Convolve the data
        kernel = make_2dgaussian_kernel(3.0, size=5)
        convolved_data = convolve(data, kernel)

        # Detect the sources
        threshold = 1.5 * bkg.background_rms  # set the detection threshold
        segment_map = detect_sources(convolved_data, threshold, n_pixels=10)

        # Plot the image and the segmentation image
        fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(8, 10))
        norm = simple_norm(data, 'sqrt', percent=99.5)
        ax1.imshow(data, norm=norm, origin='lower')
        segment_map.imshow(ax=ax2)
        fig.tight_layout()
    """
    check_units((data, threshold), ('data', 'threshold'))

    if (n_pixels <= 0) or (int(n_pixels) != n_pixels):
        msg = f'n_pixels must be a positive integer, got {n_pixels!r}'
        raise ValueError(msg)

    if mask is not None:
        if mask.shape != data.shape:
            msg = 'mask must have the same shape as the input image'
            raise ValueError(msg)
        if mask.all():
            msg = ('mask must not be True for every pixel. There are no '
                   'unmasked pixels in the image to detect sources.')
            raise ValueError(msg)
        inverse_mask = np.logical_not(mask)
    else:
        inverse_mask = None

    footprint = _make_binary_structure(data.ndim, connectivity)

    segm = _detect_sources(data, threshold, n_pixels, footprint,
                           inverse_mask)

    if segm is None:
        msg = ('No sources were found. Try lowering the threshold or '
               'n_pixels parameters.')
        warnings.warn(msg, NoDetectionsWarning)

    return segm
