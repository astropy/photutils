# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Utility functions for the psf_matching subpackage.
"""

import warnings

import numpy as np
from astropy.utils.exceptions import AstropyUserWarning
from scipy.ndimage import zoom

__all__ = ['resize_psf']


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


def _validate_window_array(window_array, expected_shape):
    """
    Validate window function output.

    Parameters
    ----------
    window_array : any
        The array returned by the window function.

    expected_shape : tuple
        The expected shape of the window array.

    Raises
    ------
    ValueError
        If the window array is not a 2D array, has the wrong shape,
        or contains values outside the range [0, 1].
    """
    if not isinstance(window_array, np.ndarray) or window_array.ndim != 2:
        msg = ('window function must return a 2D array, got '
               f'{type(window_array).__name__} with '
               f'ndim={getattr(window_array, "ndim", "undefined")}.')
        raise ValueError(msg)

    if window_array.shape != expected_shape:
        msg = (f'window function must return an array with shape '
               f'{expected_shape}, got {window_array.shape}.')
        raise ValueError(msg)

    if np.any(window_array < 0) or np.any(window_array > 1):
        msg = ('window function values must be in the range [0, 1], '
               f'got range [{np.min(window_array)}, '
               f'{np.max(window_array)}].')
        raise ValueError(msg)


def _convert_psf_to_otf(psf, shape):
    """
    Convert a point-spread function to an optical transfer function.

    This computes the FFT of a PSF array after zero-padding to the
    output shape and circularly shifting the PSF so that its center
    is at [0, 0].

    The zero-padding is needed when the input kernel (e.g., a 3x3
    Laplacian) is smaller than the target shape, so that the resulting
    OTF has the correct size for element-wise operations with other
    same-shaped OTFs.

    The circular shift places the kernel center at position [0, 0],
    which is the standard convention for computing OTFs via FFT. This
    ensures correct complex phase in the resulting OTF for general use.
    Note that when only the power spectrum (|OTF|^2) is needed, the
    shift has no effect because it only changes the phase.

    Parameters
    ----------
    psf : 2D `~numpy.ndarray`
        The PSF array.

    shape : tuple of int
        The desired output shape.

    Returns
    -------
    otf : 2D `~numpy.ndarray`
        The optical transfer function (complex array).
    """
    if np.all(psf == 0):
        return np.zeros_like(psf, dtype=complex)

    inshape = psf.shape

    # Zero-pad to the output shape (corner position)
    padded = np.zeros(shape, dtype=psf.dtype)
    padded[:inshape[0], :inshape[1]] = psf

    # Circularly shift so that the center of the PSF is at [0, 0]
    for axis, axis_size in enumerate(inshape):
        padded = np.roll(padded, -int(axis_size / 2), axis=axis)

    return np.fft.fft2(padded)


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
