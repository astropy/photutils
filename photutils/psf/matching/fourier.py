# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
This module provides tools for matching PSFs using Fourier methods.
"""

import numpy as np
from numpy.fft import fft2, fftshift, ifft2, ifftshift

__all__ = ['resize_psf', 'create_matching_kernel']


def resize_psf(psf, input_pixel_scale, output_pixel_scale, *, order=3):
    """
    Resize a PSF using spline interpolation of the requested order.

    Parameters
    ----------
    psf : 2D `~numpy.ndarray`
        The 2D data array of the PSF.

    input_pixel_scale : float
        The pixel scale of the input ``psf``.  The units must
        match ``output_pixel_scale``.

    output_pixel_scale : float
        The pixel scale of the output ``psf``.  The units must
        match ``input_pixel_scale``.

    order : float, optional
        The order of the spline interpolation (0-5).  The default is 3.

    Returns
    -------
    result : 2D `~numpy.ndarray`
        The resampled/interpolated 2D data array.
    """
    from scipy.ndimage import zoom

    ratio = input_pixel_scale / output_pixel_scale
    return zoom(psf, ratio, order=order) / ratio**2


def create_matching_kernel(source_psf, target_psf, *, window=None):
    """
    Create a kernel to match 2D point spread functions (PSF) using the
    ratio of Fourier transforms.

    Parameters
    ----------
    source_psf : 2D `~numpy.ndarray`
        The source PSF.  The source PSF should have higher resolution
        (i.e., narrower) than the target PSF.  ``source_psf`` and
        ``target_psf`` must have the same shape and pixel scale.

    target_psf : 2D `~numpy.ndarray`
        The target PSF.  The target PSF should have lower resolution
        (i.e., broader) than the source PSF.  ``source_psf`` and
        ``target_psf`` must have the same shape and pixel scale.

    window : callable, optional
        The window (or taper) function or callable class instance used
        to remove high frequency noise from the PSF matching kernel.
        Some examples include:

        * `~photutils.psf.matching.HanningWindow`
        * `~photutils.psf.matching.TukeyWindow`
        * `~photutils.psf.matching.CosineBellWindow`
        * `~photutils.psf.matching.SplitCosineBellWindow`
        * `~photutils.psf.matching.TopHatWindow`

        For more information on window functions and example usage, see
        :ref:`psf_matching`.

    Returns
    -------
    kernel : 2D `~numpy.ndarray`
        The matching kernel to go from ``source_psf`` to ``target_psf``.
        The output matching kernel is normalized such that it sums to 1.
    """
    # inputs are copied so that they are not changed when normalizing
    source_psf = np.copy(np.asanyarray(source_psf))
    target_psf = np.copy(np.asanyarray(target_psf))

    if source_psf.shape != target_psf.shape:
        raise ValueError('source_psf and target_psf must have the same shape '
                         '(i.e., registered with the same pixel scale).')

    # ensure input PSFs are normalized
    source_psf /= source_psf.sum()
    target_psf /= target_psf.sum()

    source_otf = fftshift(fft2(source_psf))
    target_otf = fftshift(fft2(target_psf))
    ratio = target_otf / source_otf

    # apply a window function in frequency space
    if window is not None:
        ratio *= window(target_psf.shape)

    kernel = np.real(fftshift(ifft2(ifftshift(ratio))))
    return kernel / kernel.sum()
