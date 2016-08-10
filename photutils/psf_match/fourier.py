# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tools for matching PSFs using Fourier methods.
"""
from __future__ import division
import numpy as np
from numpy.fft import fft2, ifft2, fftshift, ifftshift


__all__ = ['create_matching_kernel']


def create_matching_kernel(source_psf, target_psf, window=None):
    """
    Create a kernel to match 2D point spread functions (PSF) using the
    ratio of Fourier transforms.

    Parameters
    ----------
    source_psf : 2D `~numpy.ndarray`
        The source PSF.  The source PSF should have higher resolution
        (i.e. narrower) than the target PSF.

    target_psf : 2D `~numpy.ndarray`
        The target PSF.  The target PSF should have lower resolution
        (i.e. broader) than the source PSF.

    window : callable, optional
        The window (or taper) function or callable class instance used
        to remove high frequency noise from the PSF matching kernel.
        Some examples include:

        * `~photutils.HanningWindow`
        * `~photutils.TukeyWindow`
        * `~photutils.CosineBellWindow`
        * `~photutils.SplitCosineBellWindow`

    Returns
    -------
    kernel : 2D `~numpy.ndarray`
        The matching kernel to go from ``source_psf`` to ``target_psf``.
    """

    source_otf = fftshift(fft2(source_psf))
    target_otf = fftshift(fft2(target_psf))
    ratio = target_otf / source_otf

    # apply a window function in frequency space
    if window is not None:
        ratio *= window(target_psf.shape)

    kernel = np.real(fftshift((ifft2(ifftshift(ratio)))))
    return kernel / kernel.sum()
