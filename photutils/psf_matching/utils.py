# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Utility functions for the psf_matching subpackage.
"""


import numpy as np
from scipy.fft import fft2, fftshift, ifftshift
from scipy.ndimage import map_coordinates

__all__ = ['resize_psf']


def _validate_kernel_inputs(source_psf, target_psf, window):
    """
    Validate and prepare common inputs for kernel-making functions.

    Parameters
    ----------
    source_psf : array-like
        The source PSF array.

    target_psf : array-like
        The target PSF array.

    window : callable or None
        The window function or None.

    Returns
    -------
    source_psf : `~numpy.ndarray`
        The validated and normalized source PSF as a float array.

    target_psf : `~numpy.ndarray`
        The validated and normalized target PSF as a float array.

    Raises
    ------
    ValueError
        If the PSFs are not 2D arrays, have even dimensions, do not have
        the same shape, or contain NaN or Inf values.

    TypeError
        If the input ``window`` is not callable.
    """
    # Copy as float so in-place normalization doesn't modify inputs
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

    # Ensure input PSFs are normalized
    source_psf /= source_psf.sum()
    target_psf /= target_psf.sum()

    return source_psf, target_psf


def _validate_psf(psf, name):
    """
    Validate that a PSF is 2D with odd dimensions.

    Parameters
    ----------
    psf : `~numpy.ndarray`
        The PSF array to validate.

    name : str
        The parameter name used in error messages.

    Raises
    ------
    ValueError
        If the PSF is not 2D, has even dimensions, or contains NaN or
        Inf values.
    """
    if psf.ndim != 2:
        msg = f'{name} must be a 2D array.'
        raise ValueError(msg)

    if psf.shape[0] % 2 == 0 or psf.shape[1] % 2 == 0:
        msg = (f'{name} must have odd dimensions, got '
               f'shape {psf.shape}.')
        raise ValueError(msg)

    if not np.all(np.isfinite(psf)):
        msg = f'{name} contains NaN or Inf values.'
        raise ValueError(msg)

    if np.sum(psf) == 0:
        msg = f'{name} must have a non-zero sum. It cannot be normalized.'
        raise ValueError(msg)


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

    if np.any(np.logical_or(window_array < 0, window_array > 1)):
        msg = ('window function values must be in the range [0, 1], '
               f'got range [{np.min(window_array)}, '
               f'{np.max(window_array)}].')
        raise ValueError(msg)


def _normalize_kernel(kernel):
    """
    Normalize a matching kernel so that it sums to 1.

    Parameters
    ----------
    kernel : 2D `~numpy.ndarray`
        The matching kernel.

    Returns
    -------
    kernel : 2D `~numpy.ndarray`
        The normalized matching kernel.

    Raises
    ------
    ValueError
        If the kernel contains non-finite values or its sum is zero or
        nearly zero.
    """
    kernel_sum = np.sum(kernel)

    if not np.isfinite(kernel_sum):
        msg = ('The computed kernel contains non-finite values. This '
               'can occur when the Fourier-space denominator is zero '
               'at frequencies where the numerator is also zero (e.g., '
               'the source OTF and the penalty OTF are both zero at '
               'the same frequency).')
        raise ValueError(msg)

    if np.isclose(kernel_sum, 0.0):
        msg = ('The computed kernel sums to zero, which likely indicates '
               'that the regularization is too aggressive or that the '
               'window function suppressed all frequencies. Try reducing '
               'the regularization parameter or using a different window '
               'function.')
        raise ValueError(msg)

    return kernel / kernel_sum


def _convert_psf_to_otf(psf, shape):
    """
    Convert a point-spread function to an optical transfer function.

    This computes the FFT of a PSF array after centering it in a
    zero-padded array of the output shape and applying `ifftshift` to
    move the PSF center to position [0, 0].

    The PSF is first placed at the center of the zero-padded array,
    ensuring its center aligns with the array's center. The zero-padding
    is needed when the input kernel (e.g., a 3x3 Laplacian) is smaller
    than the target shape, so that the resulting OTF has the correct
    size for element-wise operations with other same-shaped OTFs.

    The `ifftshift` operation then moves the PSF center from the array
    center to position [0, 0], which is the standard convention for
    computing OTFs via FFT. This ensures correct complex phase in
    the resulting OTF for general use. Note that when only the power
    spectrum (|OTF|^2) is needed, the shift has no effect because it
    only changes the phase.

    Parameters
    ----------
    psf : 2D `~numpy.ndarray`
        The PSF array. The PSF must have odd dimensions and be centered
        on the central pixel. The PSF shape must be smaller than or
        equal to the target shape in both dimensions.

    shape : tuple of int
        The desired output shape.

    Returns
    -------
    otf : 2D `~numpy.ndarray`
        The optical transfer function (complex array).
    """
    if psf.ndim != 2:
        msg = 'psf must be a 2D array.'
        raise ValueError(msg)

    if psf.shape[0] % 2 == 0 or psf.shape[1] % 2 == 0:
        msg = f'psf must have odd dimensions, got shape {psf.shape}.'
        raise ValueError(msg)

    if np.all(psf == 0):
        return np.zeros(shape, dtype=complex)

    inshape = psf.shape

    if any(i > s for i, s in zip(inshape, shape, strict=True)):
        msg = (f'The PSF shape {inshape} is larger than the target '
               f'shape {shape} in at least one dimension.')
        raise ValueError(msg)

    # Zero-pad to the output shape with PSF centered in the array
    padded = np.zeros(shape, dtype=psf.dtype)

    # Calculate where to place PSF so its center aligns with padded
    # array center
    center = tuple(s // 2 for s in shape)
    psf_center = tuple(s // 2 for s in inshape)
    start = tuple(c - pc for c, pc in zip(center, psf_center, strict=True))
    padded[start[0]:start[0] + inshape[0],
           start[1]:start[1] + inshape[1]] = psf

    # Shift the centered PSF so its center moves to [0, 0]
    padded = ifftshift(padded)

    return fft2(padded)


def _apply_window_to_fourier(fourier_array, window, shape):
    """
    Apply a centered window function to a Fourier-domain array.

    The window function is assumed to be defined with the DC component
    at the center of the array. Since Fourier arrays use the standard
    FFT layout with the DC component at the corner, this function shifts
    the array to the center, applies the window, and shifts it back.

    Parameters
    ----------
    fourier_array : 2D `~numpy.ndarray`
        A complex Fourier-domain array with the DC component at the
        corner.

    window : callable
        The window function. Must accept a single ``shape`` tuple and
        return a 2D array with values in [0, 1].

    shape : tuple of int
        The shape passed to the window function and the expected shape
        of the window output.

    Returns
    -------
    result : 2D `~numpy.ndarray`
        The windowed Fourier-domain array, still in standard FFT layout
        (DC at the corner).
    """
    window_array = window(shape)
    _validate_window_array(window_array, shape)
    fourier_array = fftshift(fourier_array)
    fourier_array *= window_array
    return ifftshift(fourier_array)


def resize_psf(psf, input_pixel_scale, output_pixel_scale, *, order=3):
    """
    Resize a PSF using spline interpolation of the requested order.

    The PSF is interpolated at the points of a grid with the output
    pixel scale that is centered on the central pixel of the input PSF.
    The total flux of the PSF is conserved during the resizing.

    Parameters
    ----------
    psf : 2D `~numpy.ndarray`
        The 2D data array of the PSF. The PSF must have odd dimensions.
        It is assumed to be centered on the central pixel.

    input_pixel_scale : float or `~astropy.units.Quantity`
        The pixel scale of the input ``psf``. If a float,
        the units must match ``output_pixel_scale``. If a
        `~astropy.units.Quantity`, the units must be convertible to
        those of ``output_pixel_scale``.

    output_pixel_scale : float or `~astropy.units.Quantity`
        The pixel scale of the output ``psf``. If a float,
        the units must match ``input_pixel_scale``. If a
        `~astropy.units.Quantity`, the units must be convertible to
        those of ``input_pixel_scale``.

    order : int, optional
        The order of the spline interpolation (0-5). The default is 3.

    Returns
    -------
    result : 2D `~numpy.ndarray`
        The resampled/interpolated 2D data array, with a pixel scale of
        ``output_pixel_scale``. The output always has odd dimensions,
        which guarantees that it is centered and usable for PSF
        matching. The size along each axis is ``2 * floor((input_size
        - 1) / 2 * (input_pixel_scale / output_pixel_scale)) + 1``.
        This is the largest odd size for which the output grid lies
        within the outermost pixel centers of the input PSF, so no
        output value is extrapolated.

    Raises
    ------
    ValueError
        If ``psf`` is not a 2D array, has even dimensions, contains NaN
        or Inf values, or has a zero sum, if the pixel scales are not
        positive, or if the resized PSF has a zero sum.

    TypeError
        If the pixel scales have units that are not convertible to each
        other.

    Notes
    -----
    This function changes only the pixel scale. The resized PSF
    generally does not have the same shape as the PSF it will be matched
    to, so the two PSFs may still need to be cropped or padded to a
    common shape before computing a matching kernel.

    The PSF is interpolated and is not integrated over the output
    pixels. If the two PSFs come from images with different detector
    pixel sizes, each PSF should already include the pixel integration
    of its own image. An oversampled PSF that is sampled at points, such
    as the output of an optical model, should first be integrated over
    the detector pixels with :func:`~photutils.psf.make_epsf_from_psf`.

    The input PSF should be well sampled at both the input and output
    pixel scales. Resizing a PSF to a larger pixel scale samples it
    without any smoothing. If the PSF is undersampled at the output
    pixel scale, the result is aliased and a matching kernel computed
    from it will be inaccurate. A spline is also a poor interpolant for
    a PSF that is undersampled at the input pixel scale.
    """
    psf = np.asarray(psf, dtype=float)

    if input_pixel_scale <= 0 or output_pixel_scale <= 0:
        msg = 'input_pixel_scale and output_pixel_scale must be positive.'
        raise ValueError(msg)

    _validate_psf(psf, 'psf')

    # The conversion to a float handles pixel scales that are quantities
    # with different units.
    ratio = float(input_pixel_scale / output_pixel_scale)

    # The output grid is centered on the central input pixel and its
    # spacing is exactly the output pixel scale. Its size is the
    # largest odd size that keeps the grid within the outermost input
    # pixel centers, so nothing is extrapolated. The rounding keeps
    # roundoff in a half size that is a whole number from removing a
    # pixel.
    coords = []
    for n_in in psf.shape:
        center = (n_in - 1) / 2
        half_size = int(np.floor(np.round(center * ratio, 9)))
        offsets = np.arange(-half_size, half_size + 1) / ratio
        if order == 0:
            # Nearest-neighbor interpolation rounds a point midway
            # between two input pixels in one direction. Moving the
            # points slightly toward the center keeps the output
            # symmetric.
            offsets *= 1 - 1e-9
        coords.append(center + offsets)

    result = map_coordinates(psf, np.meshgrid(*coords, indexing='ij'),
                             order=order, mode='nearest')

    result_sum = result.sum()
    if result_sum == 0:
        msg = 'The resized PSF has a zero sum and cannot be normalized.'
        raise ValueError(msg)

    # Normalize the PSF to conserve total flux after resizing.
    return result * (psf.sum() / result_sum)
