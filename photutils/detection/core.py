# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import numpy as np
from ..utils import img_stats

__all__ = ['detect_sources', 'find_peaks']


def detect_sources(image, snr_threshold, npixels, filter_fwhm=None,
                   image_mask=None, mask_val=None, sig=3.0, iters=None):
    """
    Detect sources above a specified signal-to-noise ratio
    in a 2D image and return a 2D segmentation image.

    This routine does not deblend overlapping sources.

    Parameters
    ----------
    image : array_like
        The 2D array of the image.

    snr_threshold : float
        The signal-to-noise ratio threshold above which to detect
        sources.  The background rms noise level is computed using
        sigma-clipped statistics, which can be controlled via the
        ``sig`` and ``iters`` keywords.

    npixels : int
        The number of connected pixels an object must have above the
        threshold level to be detected.  Must be a positive integer.

    filter_fwhm : float, optional
        The FWHM of the circular 2D Gaussian filter that is applied to
        the input image before it is thresholded.  Filtering the image
        will maximize detectability of objects with a FWHM similar to
        ``filter_fwhm``.  Set to `None` (the default) to turn off image
        filtering.

    image_mask : array_like, bool, optional
        A boolean mask with the same shape as ``image``, where a `True`
        value indicates the corresponding element of ``image`` is
        invalid.  Masked pixels are ignored when computing the image
        background statistics.

    mask_val : float, optional
        An image data value (e.g., ``0.0``) that is ignored when
        computing the image background statistics.  ``mask_val`` will be
        ignored if ``image_mask`` is input.

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
    segment_image :  array_like
        A 2D segmentation image of integers indicating segment labels.
    """
    from scipy import ndimage

    bkgrd, median, bkgrd_rms = img_stats(image, image_mask=image_mask,
                                         mask_val=mask_val, sig=sig,
                                         iters=iters)
    assert npixels > 0, 'npixels must be a positive integer'
    assert int(npixels) == npixels, 'npixels must be a positive integer'

    if filter_fwhm is not None:
        img_smooth = ndimage.gaussian_filter(image, filter_fwhm)
    else:
        img_smooth = image

    # threshold the smoothed image
    level = bkgrd + (bkgrd_rms * snr_threshold)
    img_thresh = img_smooth >= level

    struct = ndimage.generate_binary_structure(2, 1)
    objlabels, nobj = ndimage.label(img_thresh, structure=struct)
    objslices = ndimage.find_objects(objlabels)

    # remove objects smaller than npixels size
    for objslice in objslices:
        objlabel = objlabels[objslice]
        obj_npix = len(np.where(objlabel.ravel() != 0)[0])
        if obj_npix < npixels:
            objlabels[objslice] = 0

    # relabel (labeled indices must be consecutive)
    objlabels, nobj = ndimage.label(objlabels, structure=struct)
    return objlabels


def find_peaks(image, snr_threshold, min_distance=5, exclude_border=True,
               indices=True, num_peaks=np.inf, footprint=None, labels=None,
               image_mask=None, mask_val=None, sig=3.0, iters=None):
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
    image : array_like
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
        shaped as ``image.shape`` with peaks present at `True` elements.

    num_peaks : int
        Maximum number of peaks. When the number of peaks exceeds
        ``num_peaks``, return ``num_peaks`` peaks based on highest peak
        intensity.

    footprint : ndarray of bools, optional
        If provided, ``footprint == 1`` represents the local region
        within which to search for peaks at every point in ``image``.
        Overrides ``min_distance``, except for border exclusion if
        ``exclude_border=True``.

    labels : ndarray of ints, optional
        If provided, each unique region ``labels == value`` represents a
        unique region to search for peaks.  Zero is reserved for
        background.

    image_mask : array_like, bool, optional
        A boolean mask with the same shape as ``image``, where a `True`
        value indicates the corresponding element of ``image`` is
        invalid.  Masked pixels are ignored when computing the image
        background statistics.

    mask_val : float, optional
        An image data value (e.g., ``0.0``) that is ignored when
        computing the image background statistics.  ``mask_val`` will be
        ignored if ``image_mask`` is input.

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
        * If ``indices = False`` : Boolean array shaped like ``image``,
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

    bkgrd, median, bkgrd_rms = img_stats(image, image_mask=image_mask,
                                         mask_val=mask_val, sig=sig,
                                         iters=iters)
    level = bkgrd + (bkgrd_rms * snr_threshold)
    return peak_local_max(image, min_distance=min_distance,
                          threshold_abs=level, threshold_rel=0.0,
                          exclude_border=exclude_border, indices=indices,
                          num_peaks=num_peaks, footprint=footprint,
                          labels=labels)
