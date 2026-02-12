# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tools for matching PSFs using Fourier methods.
"""

import warnings

import numpy as np
from astropy.utils.exceptions import AstropyUserWarning
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


def create_matching_kernel(source_psf, target_psf, *, window=None,
                           fourier_cutoff=1e-4):
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
        The window (taper) function or callable class instance used
        to remove high frequency noise from the PSF matching kernel.
        The window function should be a callable that accepts a single
        ``shape`` parameter (a tuple defining the 2D array shape) and
        returns a 2D array of the same shape. The returned window
        values must be in the range [0, 1], where 1.0 indicates full
        preservation of that spatial frequency and 0.0 indicates
        complete suppression. Built-in window classes include:

        * `~photutils.psf_matching.HanningWindow`
        * `~photutils.psf_matching.TukeyWindow`
        * `~photutils.psf_matching.CosineBellWindow`
        * `~photutils.psf_matching.SplitCosineBellWindow`
        * `~photutils.psf_matching.TopHatWindow`

        For more information on window functions, custom windows, and
        example usage, see :ref:`psf_matching`.

    fourier_cutoff : float, optional
        The fractional cutoff threshold for the Fourier transform of the
        source PSF. Frequencies where the source OTF (Optical Transfer
        Function, the Fourier transform of the PSF) amplitude is below
        ``fourier_cutoff`` times the peak amplitude are set to zero to
        avoid division by near-zero values. Must be in the range [0,
        1], where 0 provides minimum filtering (only exact zeros in the
        OTF) and values closer to 1 apply more aggressive filtering. The
        default is 1e-4.

    Returns
    -------
    kernel : 2D `~numpy.ndarray`
        The matching kernel to go from ``source_psf`` to ``target_psf``.
        The output matching kernel is normalized such that it sums to 1.

    Raises
    ------
    ValueError
        If the PSFs are not 2D arrays, have even dimensions, or do not
        have the same shape, if ``fourier_cutoff`` is not in the range
        [0, 1], or if the window function output is invalid (not a 2D
        array, wrong shape, or values outside [0, 1]).

    TypeError
        If the input ``window`` is not callable.
    """
    # copy as float so in-place normalization doesn't modify inputs
    source_psf = np.array(source_psf, dtype=float)
    target_psf = np.array(target_psf, dtype=float)

    _validate_psf(source_psf, 'source_psf')
    _validate_psf(target_psf, 'target_psf')

    if source_psf.shape != target_psf.shape:
        msg = ('source_psf and target_psf must have the same shape '
               '(i.e., registered with the same pixel scale).')
        raise ValueError(msg)

    if window is not None and not callable(window):
        msg = 'window must be a callable.'
        raise TypeError(msg)

    if not 0 <= fourier_cutoff <= 1:
        msg = (f'fourier_cutoff must be in the range [0, 1], '
               f'got {fourier_cutoff}.')
        raise ValueError(msg)

    # ensure input PSFs are normalized
    source_psf /= source_psf.sum()
    target_psf /= target_psf.sum()

    source_otf = np.fft.fftshift(np.fft.fft2(source_psf))
    target_otf = np.fft.fftshift(np.fft.fft2(target_psf))

    # regularized division to avoid dividing by near-zero values
    max_otf = np.max(np.abs(source_otf))
    mask = np.abs(source_otf) > fourier_cutoff * max_otf
    ratio = np.zeros_like(source_otf, dtype=complex)
    ratio[mask] = target_otf[mask] / source_otf[mask]

    # apply a window function in frequency space
    if window is not None:
        window_array = window(target_psf.shape)
        _validate_window_array(window_array, target_psf.shape)
        ratio *= window_array

    kernel = np.real(np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(ratio))))
    return kernel / kernel.sum()
