# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tools for matching PSFs using Fourier methods.
"""

import warnings

import numpy as np
from astropy.utils.exceptions import AstropyUserWarning
from numpy.fft import fft2, fftshift, ifft2, ifftshift
from scipy.ndimage import zoom

__all__ = ['create_matching_kernel', 'resize_psf']


def _validate_psf(psf, name):
    """
    Validate that a PSF is 2D with odd dimensions and centered.

    Parameters
    ----------
    psf : `~numpy.ndarray`
        The PSF array to validate.

    name : str
        The parameter name used in error messages.

    Raises
    ------
    ValueError
        If the PSF is not 2D or has even dimensions.
    """
    if psf.ndim != 2:
        msg = f'{name} must be a 2D array.'
        raise ValueError(msg)

    if psf.shape[0] % 2 == 0 or psf.shape[1] % 2 == 0:
        msg = (f'{name} must have odd dimensions, got '
               f'shape {psf.shape}.')
        raise ValueError(msg)

    center = ((psf.shape[0] - 1) / 2, (psf.shape[1] - 1) / 2)
    peak = np.unravel_index(np.argmax(psf), psf.shape)
    if peak != center:
        msg = (f'The peak of {name} is not centered. Expected peak at '
               f'{center}, but found peak at {peak}.')
        warnings.warn(msg, AstropyUserWarning)


def resize_psf(psf, input_pixel_scale, output_pixel_scale, *, order=3):
    """
    Resize a PSF using spline interpolation of the requested order.

    The total flux of the PSF is conserved during the resizing.

    Parameters
    ----------
    psf : 2D `~numpy.ndarray`
        The 2D data array of the PSF.

    input_pixel_scale : float
        The pixel scale of the input ``psf``. The units must match
        ``output_pixel_scale``.

    output_pixel_scale : float
        The pixel scale of the output ``psf``. The units must match
        ``input_pixel_scale``.

    order : int, optional
        The order of the spline interpolation (0-5). The default is 3.

    Returns
    -------
    result : 2D `~numpy.ndarray`
        The resampled/interpolated 2D data array.

    Raises
    ------
    ValueError
        If ``psf`` is not a 2D array, has even dimensions, is not
        centered, or if the pixel scales are not positive.
    """
    psf = np.asarray(psf, dtype=float)

    if input_pixel_scale <= 0 or output_pixel_scale <= 0:
        msg = ('input_pixel_scale and output_pixel_scale must be '
               'positive.')
        raise ValueError(msg)

    _validate_psf(psf, 'psf')

    ratio = input_pixel_scale / output_pixel_scale

    # Scale by ratio**2 to conserve total flux
    return zoom(psf, ratio, order=order) / ratio**2


def create_matching_kernel(source_psf, target_psf, *, window=None):
    """
    Create a kernel to match 2D point spread functions (PSF) using the
    ratio of Fourier transforms.

    Parameters
    ----------
    source_psf : 2D `~numpy.ndarray`
        The source PSF. The source PSF should have higher resolution
        (i.e., narrower) than the target PSF. ``source_psf`` and
        ``target_psf`` must have the same shape and pixel scale.

    target_psf : 2D `~numpy.ndarray`
        The target PSF. The target PSF should have lower resolution
        (i.e., broader) than the source PSF. ``source_psf`` and
        ``target_psf`` must have the same shape and pixel scale.

    window : callable, optional
        The window (or taper) function or callable class instance used
        to remove high frequency noise from the PSF matching kernel.
        Some examples include:

        * `~photutils.psf_matching.HanningWindow`
        * `~photutils.psf_matching.TukeyWindow`
        * `~photutils.psf_matching.CosineBellWindow`
        * `~photutils.psf_matching.SplitCosineBellWindow`
        * `~photutils.psf_matching.TopHatWindow`

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
        msg = ('source_psf and target_psf must have the same shape '
               '(i.e., registered with the same pixel scale).')
        raise ValueError(msg)

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
