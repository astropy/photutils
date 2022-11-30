# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
This module provides utility functions for image segmentation.
"""

import numpy as np
from astropy.convolution import Gaussian2DKernel
from astropy.stats import gaussian_fwhm_to_sigma

from photutils.utils._parameters import as_pair

__all__ = ['make_2dgaussian_kernel']


def make_2dgaussian_kernel(fwhm, size, mode='oversample', oversampling=10):
    """
    Make a normalized 2D circular Gaussian kernel.

    The kernel must have odd sizes in both X and Y, be centered in the
    central pixel, and normalized to sum to 1.

    Parameters
    ----------
    fwhm : float
        The full-width at half-maximum (FWHM) of the 2D circular
        Gaussian kernel.

    size : int or (2,) int array_like
        The size of the kernel along each axis. If ``size`` is a scalar
        then a square size of ``size`` will be used. If ``size`` has
        two elements, they must be in ``(ny, nx)`` (i.e., array shape)
        order. ``size`` must have odd values for both axes.

    mode : {'oversample', 'center', 'linear_interp', 'integrate'}, optional
        The mode to use for discretizing the 2D Gaussian model:
            * 'oversample' (default):
              Discretize model by taking the average on an oversampled
              grid.
            * 'center':
              Discretize model by taking the value at the center of the
              bin.
            * 'linear_interp':
              Discretize model by performing a bilinear interpolation
              between the values at the corners of the bin.
            * 'integrate':
              Discretize model by integrating the model over the bin.

    oversampling : int, optional
        The oversampling factor used when ``mode='oversample'``.

    Returns
    -------
    kernel : `astropy.convolution.Kernel2D`
        The output smoothing kernel, normalized such that it sums to 1.
    """
    ysize, xsize = as_pair('size', size, lower_bound=(0, 1), check_odd=True)

    kernel = Gaussian2DKernel(fwhm * gaussian_fwhm_to_sigma,
                              x_size=xsize, y_size=ysize, mode=mode,
                              factor=oversampling)
    kernel.normalize(mode='integral')  # ensure kernel sums to 1

    return kernel


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
    array : `~numpy.ndarray`
        The binary structure element. If ``ndim <= 2`` an array of int
        is returned, otherwise an array of bool is returned.
    """
    if ndim == 1:
        footprint = np.array((1, 1, 1))
    elif ndim == 2:
        if connectivity == 4:
            footprint = np.array(((0, 1, 0), (1, 1, 1), (0, 1, 0)))
        elif connectivity == 8:
            footprint = np.ones((3, 3), dtype=int)
        else:
            raise ValueError(f'Invalid connectivity={connectivity}.  '
                             'Options are 4 or 8.')
    else:
        from scipy.ndimage import generate_binary_structure

        footprint = generate_binary_structure(ndim, 1)

    return footprint


def _mask_to_mirrored_value(data, replace_mask, xycenter, mask=None):
    """
    Replace masked pixels with the value of the pixel mirrored across a
    given center position.

    If the mirror pixel is unavailable (i.e., it is outside of the image
    or masked), then the masked pixel value is set to zero.

    Parameters
    ----------
    data : 2D `~numpy.ndarray`
        A 2D array.

    replace_mask : 2D bool `~numpy.ndarray`
        A boolean mask where `True` values indicate the pixels that
        should be replaced, if possible, by mirrored pixel values. It
        must have the same shape as ``data``.

    xycenter : tuple of two int
        The (x, y) center coordinates around which masked pixels will be
        mirrored.

    mask : 2D bool `~numpy.ndarray`
        A boolean mask where `True` values indicate ``replace_mask``
        *mirrored* pixels that should never be used to fix
        ``replace_mask`` pixels. In other words, if a pixel in
        ``replace_mask`` has a mirror pixel in this ``mask``, then the
        mirrored value is set to zero. Using this keyword prevents
        potential spreading of known non-finite or bad pixel values.

    Returns
    -------
    result : 2D `~numpy.ndarray`
        A 2D array with replaced masked pixels.
    """
    outdata = np.copy(data)

    ymasked, xmasked = np.nonzero(replace_mask)
    xmirror = 2 * int(xycenter[0] + 0.5) - xmasked
    ymirror = 2 * int(xycenter[1] + 0.5) - ymasked

    # Find mirrored pixels that are outside of the image
    badmask = ((xmirror < 0) | (ymirror < 0) | (xmirror >= data.shape[1])
               | (ymirror >= data.shape[0]))

    # remove them from the set of replace_mask pixels and set them to
    # zero
    if np.any(badmask):
        outdata[ymasked[badmask], xmasked[badmask]] = 0.0
        # remove the badmask pixels from pixels to be replaced
        goodmask = ~badmask
        ymasked = ymasked[goodmask]
        xmasked = xmasked[goodmask]
        xmirror = xmirror[goodmask]
        ymirror = ymirror[goodmask]

    outdata[ymasked, xmasked] = outdata[ymirror, xmirror]

    # Find mirrored pixels that are masked and replace_mask pixels that are
    # mirrored to other replace_mask pixels. Set them both to zero.
    mirror_mask = replace_mask[ymirror, xmirror]
    if mask is not None:
        mirror_mask |= mask[ymirror, xmirror]
    xbad = xmasked[mirror_mask]
    ybad = ymasked[mirror_mask]
    outdata[ybad, xbad] = 0.0

    return outdata
