# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import numpy as np
import imageutils
from astropy import log

__all__ = ['lacosmic']


def lacosmic(data, contrast, cr_threshold, neighbor_threshold,
             uncertainty=None, mask=None, background=None, gain=None,
             readnoise=None, maxiter=4):
    """
    Remove cosmic rays from a single astronomical image using the
    `L.A.Cosmic <http://www.astro.yale.edu/dokkum/lacosmic/>`_
    algorithm.  The algorithm is based on Laplacian edge detection and
    is described in `PASP 113, 1420 (2001)`_.

    .. _PASP 113, 1420 (2001):
        http://adsabs.harvard.edu/abs/2001PASP..113.1420V

    Parameters
    ----------
    data : array_like or `~astropy.nddata.NDData`
        The 2D array of the image.

    contrast : float
        Contrast threshold between the Laplacian image and the
        fine-structure image.  If your image is critically sampled, use
        a value around 2.  If your image is undersampled (e.g. HST
        data), a value of 4 or 5 (or more) is more appropriate.  If your
        image is oversampled, use a value between 1 and 2.  For details,
        please see `PASP 113, 1420 (2001)`_, which calls this parameter
        :math:`f_{\mbox{lim}}`.  In particular, figure 4 shows the
        approximate relationship between the ``contrast`` parameter and
        the pixel full-width half-maximum of stars in your image.

    cr_threshold : float
        The Laplacian signal-to-noise ratio threshold for cosmic ray
        detection.

    neighbor_threshold : float
        The Laplacian signal-to-noise ratio threshold for detection of
        cosmic rays in pixels neighboring the initially-identified
        cosmic rays.

    uncertainty : array_like, optional
        The 2D array of the 1-sigma errors of the input ``data``.
        ``uncertainty`` must have the same shape as ``data``.  If
        ``uncertainty`` is not input, then ``gain`` and ``readnoise``
        will be used to construct an approximate model of the
        ``uncertainty``.  If ``uncertainty`` is input, it will override
        the ``gain`` and ``readnoise`` parameters or ``data.uncertainty``
        for `~astropy.nddata.NDData` inputs..

    mask : array_like, bool, optional
        A boolean mask with the same shape as ``data``, where a `True`
        value indicates the corresponding element of ``data`` is ignored
        when identifying cosmic rays.  It is highly recommended that
        saturated stars be included in ``mask``.  If ``mask`` is input
        it will override ``data.mask`` for `~astropy.nddata.NDData`
        inputs.

    background : float or array_like, optional
        The background level previously subtracted from the input
        ``image``.  ``background`` may either be a scalar value or a 2D
        image with the same shape as the input ``image``.  If the input
        ``image`` has not been background-subtracted, then set
        ``background=None`` (default).

    gain : float, optional
        The gain factor that when multiplied by the input ``image``
        results in an image in units of electrons.  For example, if your
        input ``image`` is in units of ADU, then ``gain`` should
        represent electrons/ADU.  If your input ``image`` is in units of
        electrons/s then ``gain`` should be the exposure time.  ``gain``
        and ``readnoise`` must be specified if an ``error_image`` is not
        input.

    readnoise : float, optional
        The read noise (in electrons) in the input ``image``.  ``gain``
        and ``readnoise`` must be specified if an ``error_image`` is not
        input.

    maxiter : float, optional
        The maximum number of iterations.  The default is ``4``.  The
        routine will automatically exit if no additional cosmic rays are
        identified.  If the routine is still identifying cosmic rays
        after ``4`` iterations, then you are likely digging into sources
        (e.g. saturated stars) and/or the noise.  In that case, try
        inputing a ``mask`` or increasing the value of ``cr_threshold``.

    Returns
    -------
    cleaned_image : array_like
        The cosmic-ray cleaned image.

    crmask : array_like, bool
        A mask image of the identified cosmic rays.  Cosmic-ray pixels
        have a value of `True`.
    """

    from scipy import ndimage
    block_size = 2.0
    kernel = np.array([[0.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 0.0]])

    clean_data = data.copy()
    if background is not None:
        clean_data += background
    final_crmask = np.zeros(data.shape, dtype=bool)

    if uncertainty is not None:
        assert data.shape == uncertainty.shape, \
            'uncertainty must have the same shape as data'
    clean_error_image = uncertainty

    ncosmics, ncosmics_tot = 0, 0
    for iteration in range(maxiter):
        sampl_img = imageutils.upsample(clean_data, block_size)
        conv_img = ndimage.convolve(sampl_img, kernel,
                                    mode='mirror').clip(min=0.0)
        laplacian_img = imageutils.downsample(conv_img, block_size)

        if clean_error_image is None:
            if gain is None or readnoise is None:
                raise AssertionError('gain and readnoise must be input if '
                                     'uncertainty is not input')
            med5_img = ndimage.median_filter(clean_data, size=5,
                                             mode='mirror').clip(min=1.e-5)
            error_image = np.sqrt(gain*med5_img + readnoise**2) / gain
        else:
            error_image = clean_error_image

        snr_img = laplacian_img / (block_size * error_image)
        # used to remove extended structures (larger than ~5x5)
        snr_img -= ndimage.median_filter(snr_img, size=5, mode='mirror')

        # used to remove compact bright objects
        med3_img = ndimage.median_filter(clean_data, size=3, mode='mirror')
        med7_img = ndimage.median_filter(med3_img, size=7, mode='mirror')
        finestruct_img = ((med3_img - med7_img) / error_image).clip(min=0.01)

        cr_mask1 = snr_img > cr_threshold
        # NOTE: to follow the paper exactly, this condition should be
        # "> contrast * block_size".  "lacos_im.cl" uses simply "> contrast"
        cr_mask2 = (snr_img / finestruct_img) > contrast
        cr_mask = cr_mask1 * cr_mask2
        if mask is not None:
            cr_mask = np.logical_and(cr_mask, ~mask)

        # grow cosmic rays by one pixel and check in snr_img
        selem = np.ones((3, 3))
        neigh_mask = ndimage.binary_dilation(cr_mask, selem)
        cr_mask = cr_mask1 * neigh_mask
        # now grow one more pixel and lower the detection threshold
        neigh_mask = ndimage.binary_dilation(cr_mask, selem)
        cr_mask = (snr_img > neighbor_threshold) * neigh_mask

        # previously unknown cosmic rays found in this iteration
        crmask_new = np.logical_and(~final_crmask, cr_mask)
        ncosmics = np.count_nonzero(crmask_new)

        final_crmask = np.logical_or(final_crmask, cr_mask)
        ncosmics_tot += ncosmics
        log.info('Iteration {0}: Found {1} cosmic-ray pixels, '
                 'Total: {2}'.format(iteration + 1, ncosmics, ncosmics_tot))
        if ncosmics == 0:
            if background is not None:
                clean_data -= background
            return clean_data, final_crmask
        clean_data = _clean_masked_pixels(clean_data, final_crmask, size=5,
                                           exclude_mask=mask)

    if background is not None:
        clean_data -= background
    return clean_data, final_crmask


def _clean_masked_pixels(data, mask, size=5, exclude_mask=None):
    """
    Clean masked pixels in an image.  Each masked pixel is replaced by
    the median of unmasked pixels in a 2D window of ``size`` centered on
    it.  If all pixels in the window are masked, then the window is
    increased in size until unmasked pixels are found.

    Pixels in ``exclude_mask`` are not cleaned, but they are excluded
    when calculating the local median.
    """

    assert size % 2 == 1, 'size must be an odd integer'
    assert data.shape == mask.shape, \
        'mask must have the same shape as image'
    ny, nx = data.shape
    mask_coords = np.argwhere(mask)
    if exclude_mask is not None:
        assert data.shape == exclude_mask.shape, \
            'exclude_mask must have the same shape as data'
        maskall = np.logical_or(mask, exclude_mask)
    else:
        maskall = mask
    mask_idx = maskall.nonzero()
    data_nanmask = data.copy()
    data_nanmask[mask_idx] = np.nan

    nexpanded = 0
    for coord in mask_coords:
        y, x = coord
        median_val, expanded = _local_median(data_nanmask, x, y, nx, ny,
                                             size=size)
        data[y, x] = median_val
        if expanded:
            nexpanded += 1
    if nexpanded > 0:
        log.info('    Found {0} {1}x{1} masked regions while '
                 'cleaning.'.format(nexpanded, size))
    return data


def _local_median(data_nanmask, x, y, nx, ny, size=5, expanded=False):
    """Compute the local median in a 2D window, excluding NaN."""
    hy, hx = size // 2, size // 2
    x0, x1 = np.array([x - hx, x + hx + 1]).clip(0, nx)
    y0, y1 = np.array([y - hy, y + hy + 1]).clip(0, ny)
    region = data_nanmask[y0:y1, x0:x1].ravel()
    goodpixels = region[np.isfinite(region)]
    if len(goodpixels) > 0:
        median_val = np.median(goodpixels)
    else:
        newsize = size + 2     # keep size odd
        median_val, expanded = _local_median(data_nanmask, x, y, nx, ny,
                                             size=newsize, expanded=True)
    return median_val, expanded
