# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import numpy as np
from photutils import utils
from photutils.utils import _filtering
import bottleneck as bn

__all__ = ['lacosmic']


#@profile
def lacosmic(image, contrast, cr_threshold, neighbor_threshold,
             error_image=None, mask_image=None, background=None,
             gain=None, readnoise=None, maxiters=4):
    """
    Remove cosmic rays from a single astronomical image using the
    `L.A.Cosmic`_ algorithm.  The algorithm is based on Laplacian edge
    detection and is described in `PASP 113, 1420 (2001)`_.

    .. _L.A.Cosmic: http://www.astro.yale.edu/dokkum/lacosmic/
    .. _PASP 113, 1420 (2001): http://adsabs.harvard.edu/abs/2001PASP..113.1420V

    Parameters
    ----------
    image : array_like
        The 2D array of the image.

    contrast : float
        Contrast threshold between the Laplacian image and the
        fine-structure image.

    cr_threshold : float
        The signal-to-noise ratio threshold for cosmic ray detection.

    neighbor_threshold :
        The signal-to-noise ratio threshold for detection of neighboring
        cosmic rays

    error_image : array_like, optional
        The 2D array of the 1-sigma errors of the input ``image``.
        ``error_image`` must have the same shape as ``image``.  If
        ``error_image`` is not input, then ``gain`` and ``readnoise``
        will be used to construct an approximate model of the
        ``error_image``.  If ``error_image`` is input, it will override
        the ``gain`` and ``readnoise`` parameters.

    mask_image : array_like, bool, optional
        A boolean mask with the same shape as ``image``, where a `True`
        value indicates the corresponding element of ``image`` is
        ignored when identifying cosmic rays.  It is highly recommended
        that saturated stars be included in ``mask_image``.

    background : float, optional
        The background level previously subtracted from the input
        ``image``.  If the input ``image`` has not been
        background-subtracted, then set ``background=None`` (default).

    gain : float, optional
        The gain factor that when multiplied by the input ``image``
        results in an image in units of electrons.  For example, if your
        input ``image`` is in units of ADU, then ``gain`` should be
        electrons/ADU.  If your input ``image`` is in units of
        electrons/s then ``gain`` should be the exposure time.  ``gain``
        and ``readnoise`` must be specified if an ``error_image`` is not
        input.

    readnoise : float, optional
        The read noise (electrons) in the input ``image``.  ``gain`` and
        ``readnoise`` must be specified if an ``error_image`` is not
        input.

    maxiters : float, optional
        The maximum number of interations.  The default is ``4``.  The
        routine will automatically exit if no additional cosmic rays are
        identified.  If the routine is still identifying cosmic rays
        after ``4`` iterations, then you are likely digging into sources
        and/or the noise.  In that case, try increasing the value of
        ``cr_threshold``.

    Returns
    -------
    crmask_image :  array_like
        A 2D mask image indicating the location of detected cosmic rays.
    """

    from scipy import ndimage

    block_size = 2.0
    if background is not None:
        image += background

    sampl_img = utils.upsample(image, block_size)
    kernel = np.array([[0.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 0.0]])
    conv_img = ndimage.convolve(sampl_img, kernel,
                                mode='mirror').clip(min=0.0)
    laplacian_img = utils.downsample(conv_img, block_size)

    if error_image is None:
        if gain is None and readnoise is None:
            print('Warning message')
            import sys
            sys.exit()
        med5_img = ndimage.median_filter(image, size=5,
                                         mode='mirror').clip(min=1.e-5)
        error_image = np.sqrt(gain*med5_img + readnoise**2) / gain
    else:
        assert image.shape == error_image.shape, ('error_image must have the '
                                                  'same shape as image')

    snr_img = laplacian_img / (block_size * error_image)

    # remove extended structures (larger than ~5x5)
    snr_img -= ndimage.median_filter(snr_img, size=5, mode='mirror')

    # remove compact bright objects
    med3_img = ndimage.median_filter(image, size=3, mode='mirror')
    med7_img = ndimage.median_filter(med3_img, size=7, mode='mirror')

    finestruct_img = ((med3_img - med7_img) / error_image).clip(min=0.01)
    cr_mask1 = snr_img > cr_threshold
    # NOTE: to follow the paper exactly, this condition should be
    # "> constract * block_size".  "lacos_im.cl" uses simply "> constrast"
    cr_mask2 = (snr_img / finestruct_img) > contrast
    cr_mask = cr_mask1 * cr_mask2
    #cr_mask = np.logical_and(cr_mask1, cr_mask2)

    # check neighbors in snr_img
    struct = np.ones((3, 3))
    neigh_mask = ndimage.binary_dilation(cr_mask, struct)
    cr_mask = cr_mask1 * neigh_mask

    neigh_mask = ndimage.binary_dilation(cr_mask, struct)
    cr_mask = (snr_img > neighbor_threshold) * neigh_mask

    #return cr_mask, cr_mask

    # create cleaned image
    size = 5.0
    #clean_image = clean(image, cr_mask, size)
    clean_image = clean_masked_pixels(image, cr_mask, size)
    outimage = image
    outimage[cr_mask] = clean_image[cr_mask]

    return outimage, cr_mask


def local_median(image_nanmask, x, y, nx, ny, size=5):
    hy, hx = size // 2, size // 2
    x0, x1 = np.array([x - hx, x + hx]).clip(0, nx)
    y0, y1 = np.array([y - hy, y + hy]).clip(0, ny)
    region = image_nanmask[x0:x1, y0:y1].ravel()
    goodpixels = region[np.isfinite(region)]
    if len(goodpixels) > 0:
        median_val = np.median(goodpixels)
    else:
        size += 2     # keep size odd
        print("Found a 5x5 masked region while cleaning, increasing local window size to {0}x{0}".format(size))
        median_val = local_median(image_nanmask, x, y, nx, ny, size=size)
    return median_val


def clean_masked_pixels(image, mask, size):
    """
    Clean masked pixels in an image.  The input masked pixels
    should include the identified cosmic rays and any user-defined
    mask (e.g. saturated stars).  The masked pixels are replaced by
    the median value of unmasked values in a centered 5x5 window.
    If all pixels in the window are masked, then the window is increased
    in size until XX good pixels are found.
    """
    # assert size is odd
    ny, nx = image.shape
    image_nanmask = image.copy()
    mask_idx = mask.nonzero()
    image_nanmask[mask_idx] = np.nan
    mask_coords = np.transpose(mask_idx)
    for coord in mask_coords:
        x, y = coord
        median_val = local_median(image_nanmask, x, y, nx, ny, size=5)
        image[x, y] = median_val
    return image


#@profile
def clean(image, mask, size):
    #return masked_median_filter_slow(image, mask, size)
    return masked_median_filter(image, mask, size)

    #mask = mask.astype(np.int)
    #return _filtering._masked_median_filter(image, mask, size)

#@profile
def masked_median_filter(image, mask, size):
    import bottleneck as bn
    from scipy import ndimage
    idx = np.where(mask == 1)
    img = image.copy()
    img[idx] = np.nan
    z = ndimage.generic_filter(img, bn.nanmedian, 5)
    img[idx] = z[idx]
    return img


#@profile
def masked_median_filter_slow(image, mask, size):
    # this is likely to be slow -> cython
    outimg = np.zeros_like(image)
    ny, nx = image.shape
    #for jj in range(ny):
    #    for ii in range(nx):
    for ii in range(nx):
        for jj in range(ny):
            # NOTE:  this simply clips image at image boundaries
            minx, maxx = max([ii - size/2, 0]), min([ii + size/2 + 1, nx])
            miny, maxy = max([jj - size/2, 0]), min([jj + size/2 + 1, ny])
            image_region = image[miny:maxy, minx:maxx]
            mask_region = mask[miny:maxy, minx:maxx]
            #print(jj, ii)
            #outimg[jj, ii] = np.median(image_region[mask_region == 0])
            zz = image_region[mask_region == 0]
            #outimg[jj, ii] = zz.mean()
            #outimg[jj, ii] = np.median(zz)
            outimg[jj, ii] = bn.median(zz)
    return outimg

