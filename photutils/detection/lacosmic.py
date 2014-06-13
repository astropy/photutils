# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import numpy as np
from photutils import utils

__all__ = ['lacosmic']


def lacosmic(image, contrast, cr_threshold, neighbor_threshold,
             error_image=None, mask_image=None, background=None,
             gain=None, readnoise=None, maxiters=4):
    """Identify cosmic rays in single images."""
    from scipy import ndimage

    block_size = 2.0
    if background is not None:
        image += background

    sampl_img = utils.upsample(image, block_size)
    kernel = np.array([[0.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 0.0]])
    conv_img = ndimage.convolve(sampl_img, kernel,
                                mode='mirror').clip(min=0.0)
    laplacian_img = utils.downsample(conv_img, 2)

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

    # check neighbors in snr_img
    struct = np.ones((3, 3))
    neigh_mask = ndimage.binary_dilation(cr_mask, struct)
    cr_mask = cr_mask1 * neigh_mask

    neigh_mask = ndimage.binary_dilation(cr_mask, struct)
    cr_mask = (snr_img > neighbor_threshold) * neigh_mask
    return cr_mask
