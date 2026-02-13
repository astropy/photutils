# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tools for matching PSFs using Fourier methods.
"""

import warnings

import numpy as np
from astropy.utils.decorators import deprecated
from astropy.utils.exceptions import AstropyUserWarning
from scipy.ndimage import zoom

__all__ = [
    'create_matching_kernel',
    'make_kernel',
    'make_wiener_kernel',
    'resize_psf',
]


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


def make_kernel(source_psf, target_psf, *, window=None, otf_threshold=1e-4):
    """
    Make a convolution kernel that matches an input PSF to a target PSF
    using the ratio of Fourier transforms.

    This function computes the matching kernel in the Fourier domain by
    dividing the target PSF's Fourier transform by the source PSF's
    Fourier transform. To avoid division by near-zero values, the Fourier
    ratio is set to zero at frequencies where the source OTF (Optical
    Transfer Function) amplitude falls below a threshold.

    The kernel is computed as:

    .. math::

        K = \\mathcal{F}^{-1} \\left[ W \\cdot R \\right]

    where the Fourier-space ratio :math:`R` is defined as:

    .. math::

        R = \\begin{cases}
            \\frac{T}{S} & \\text{if } |S| > \\epsilon \\cdot \\max(|S|) \\\\
            0 & \\text{otherwise}
            \\end{cases}

    Here, :math:`\\mathcal{F}^{-1}` is the inverse Fourier transform,
    :math:`S` and :math:`T` are the Fourier transforms of the source and
    target PSFs (the optical transfer functions, OTFs), :math:`\\epsilon`
    is the ``otf_threshold`` parameter, and :math:`W` is the optional
    ``window`` function (defaulting to 1 if not provided).

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
        example usage, see :ref:`PSF Matching <psf_matching>`.

    otf_threshold : float, optional
        The fractional amplitude threshold for the source OTF
        (Optical Transfer Function, the Fourier transform of the
        PSF). At frequencies where the source OTF amplitude is below
        ``otf_threshold`` times the peak amplitude, the Fourier ratio is
        set to zero to avoid division by near-zero values. Must be in
        the range [0, 1], where 0 provides no thresholding (only exact
        zeros are excluded) and values closer to 1 apply more aggressive
        thresholding.

    Returns
    -------
    kernel : 2D `~numpy.ndarray`
        The matching kernel to go from ``source_psf`` to ``target_psf``.
        The output matching kernel is normalized such that it sums to 1.

    Raises
    ------
    ValueError
        If the PSFs are not 2D arrays, have even dimensions, or do not
        have the same shape, if ``otf_threshold`` is not in the range
        [0, 1], or if the window function output is invalid (not a 2D
        array, wrong shape, or values outside [0, 1]).

    TypeError
        If the input ``window`` is not callable.

    See Also
    --------
    make_wiener_kernel : Make a matching kernel using Wiener
                         (Tikhonov) regularization instead of
                         hard amplitude thresholding.

    Examples
    --------
    Make a matching kernel between two Gaussian PSFs:

    >>> import numpy as np
    >>> from astropy.modeling.models import Gaussian2D
    >>> from photutils.psf_matching import make_kernel
    >>> y, x = np.mgrid[0:51, 0:51]
    >>> psf1 = Gaussian2D(100, 25, 25, 3, 3)(x, y)
    >>> psf2 = Gaussian2D(100, 25, 25, 5, 5)(x, y)
    >>> psf1 /= psf1.sum()
    >>> psf2 /= psf2.sum()
    >>> kernel = make_kernel(psf1, psf2)
    >>> print(f'{kernel.sum():.1f}')
    1.0
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

    if not 0 <= otf_threshold <= 1:
        msg = (f'otf_threshold must be in the range [0, 1], '
               f'got {otf_threshold}.')
        raise ValueError(msg)

    # ensure input PSFs are normalized
    source_psf /= source_psf.sum()
    target_psf /= target_psf.sum()

    source_otf = np.fft.fftshift(np.fft.fft2(source_psf))
    target_otf = np.fft.fftshift(np.fft.fft2(target_psf))

    # regularized division to avoid dividing by near-zero values
    max_otf = np.max(np.abs(source_otf))
    mask = np.abs(source_otf) > otf_threshold * max_otf
    ratio = np.zeros_like(source_otf, dtype=complex)
    ratio[mask] = target_otf[mask] / source_otf[mask]

    # apply a window function in frequency space
    if window is not None:
        window_array = window(target_psf.shape)
        _validate_window_array(window_array, target_psf.shape)
        ratio *= window_array

    kernel = np.real(np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(ratio))))
    return kernel / kernel.sum()


def _psf2otf(psf, shape):
    """
    Convert a point-spread function to an optical transfer function.

    This computes the FFT of a PSF array, handling the centering by
    circularly shifting the PSF so that the center is at [0, 0].
    If needed, the PSF is zero-padded to the output shape.

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


def make_wiener_kernel(source_psf, target_psf, *, regularization=1e-4,
                       penalty=None, window=None):
    """
    Make a convolution kernel that matches an input PSF to a target PSF
    using Wiener regularization in Fourier space.

    This function computes a Wiener-regularized PSF-matching kernel
    in the Fourier domain. The denominator includes a regularization
    term that stabilizes inversion of the source OTF (Optical Transfer
    Function, the Fourier transform of the PSF) by preventing division
    by small values, thereby suppressing noise amplification at spatial
    frequencies where the source response is weak.

    When no ``penalty`` is provided, the regularization is a
    frequency-independent (zero-order scalar) Tikhonov term expressed as
    a fraction of the peak power in the source OTF. In this case, the
    kernel is computed as:

    .. math::

        K = \\mathcal{F}^{-1} \\left[ W \\cdot
            \\frac{T \\cdot S^{*}}
                  {|S|^{2} + \\epsilon \\cdot \\max(|S|^{2})} \\right]

    When a ``penalty`` array is provided (e.g., a Laplacian operator),
    the regularization becomes frequency-dependent:

    .. math::

        K = \\mathcal{F}^{-1} \\left[ W \\cdot
            \\frac{T \\cdot S^{*}}
                  {|S|^{2} + \\epsilon \\cdot |P|^{2}} \\right]

    where :math:`P` is the OTF of the ``penalty`` operator. This
    penalizes high spatial frequencies more heavily, which is
    particularly effective at suppressing noise amplification.

    In both equations, :math:`\\mathcal{F}^{-1}` is the inverse Fourier
    transform, :math:`S` and :math:`T` are the Fourier transforms of
    the source and target PSFs (the OTFs), :math:`S^{*}` is the complex
    conjugate of :math:`S`, :math:`\\epsilon` is the ``regularization``
    parameter, and :math:`W` is the optional ``window`` function
    (defaulting to 1 if not provided). :math:`|S|^{2}` is the power
    spectrum of the source OTF.

    When the ``penalty`` is set to ``'laplacian'``, the regularization
    reproduces the approach used by the ``pypher`` package (`Boucaud
    et al. 2016`_), which applies a discrete Laplacian operator as
    the penalty. This provides stronger suppression of high spatial
    frequencies, which can be beneficial when working with noisy or
    undersampled PSFs.

    Compared to `~photutils.psf_matching.make_kernel`, which uses a hard
    threshold on Fourier amplitude, this approach provides continuous,
    smooth regularization that is better suited for PSFs that have
    near-zero power at high spatial frequencies. The hard-cutoff
    approach zeros out frequencies where the source amplitude is
    below a threshold, which can introduce discontinuities in Fourier
    space. Wiener regularization instead smoothly down-weights those
    frequencies, producing matching kernels with less ringing.

    .. _Boucaud et al. 2016:
        https://ui.adsabs.harvard.edu/abs/2016A%26A...596A..63B/abstract

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

    regularization : float, optional
        The regularization parameter that controls the strength
        of the Wiener (Tikhonov) regularization. When ``penalty``
        is `None`, this is expressed as a fraction of the peak
        power in the source PSF's Fourier transform. When
        ``penalty`` is provided, this scales the penalty operator's
        power spectrum directly. Larger values produce smoother but
        less accurate matching kernels; smaller values preserve more
        detail but may amplify noise.

    penalty : `None`, ``'laplacian'``, or 2D `~numpy.ndarray`, optional
        The regularization penalty operator. This controls the
        structure of the regularization term in the denominator:

        * `None` (default): Scalar Tikhonov regularization. The
          denominator is :math:`|S|^2 + \\epsilon \\cdot
          \\max(|S|^2)`, providing uniform regularization across
          all spatial frequencies.

        * ``'laplacian'``: Uses a discrete Laplacian operator as
          the penalty, producing frequency-dependent regularization
          that penalizes high spatial frequencies more heavily. The
          denominator becomes :math:`|S|^2 + \\epsilon \\cdot
          |L|^2` where :math:`L` is the OTF of the Laplacian
          kernel ``[[0, -1, 0], [-1, 4, -1], [0, -1, 0]]``. This
          reproduces the regularization used by the ``pypher``
          package (`Boucaud et al. 2016`_).

        * 2D `~numpy.ndarray`: A custom penalty operator array.
          Its OTF will be computed and used in the denominator as
          :math:`|S|^2 + \\epsilon \\cdot |P|^2`.

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

        A window function is generally not needed when using Wiener
        regularization because the regularization itself suppresses
        high-frequency noise. However, a window may still be useful
        when working with noisy or undersampled PSFs.

        For more information on window functions, custom windows, and
        example usage, see :ref:`PSF Matching <psf_matching>`.

    Returns
    -------
    kernel : 2D `~numpy.ndarray`
        The matching kernel to go from ``source_psf`` to ``target_psf``.
        The output matching kernel is normalized such that it sums to 1.

    Raises
    ------
    ValueError
        If the PSFs are not 2D arrays, have even dimensions, do not
        have the same shape, if ``regularization`` is not positive,
        or if ``penalty`` is not a valid value.

    TypeError
        If the input ``window`` is not callable.

    See Also
    --------
    make_kernel : Make a matching kernel using a hard frequency cutoff
                  instead of Wiener regularization.

    Examples
    --------
    Make a matching kernel between two Gaussian PSFs:

    >>> import numpy as np
    >>> from astropy.modeling.models import Gaussian2D
    >>> from photutils.psf_matching import make_wiener_kernel
    >>> y, x = np.mgrid[0:51, 0:51]
    >>> psf1 = Gaussian2D(100, 25, 25, 3, 3)(x, y)
    >>> psf2 = Gaussian2D(100, 25, 25, 5, 5)(x, y)
    >>> psf1 /= psf1.sum()
    >>> psf2 /= psf2.sum()
    >>> kernel = make_wiener_kernel(psf1, psf2)
    >>> print(f'{kernel.sum():.1f}')
    1.0

    Use the Laplacian penalty for frequency-dependent regularization:

    >>> kernel = make_wiener_kernel(psf1, psf2, penalty='laplacian')
    >>> print(f'{kernel.sum():.1f}')
    1.0
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

    if regularization <= 0:
        msg = 'regularization must be a positive number.'
        raise ValueError(msg)

    if window is not None and not callable(window):
        msg = 'window must be a callable.'
        raise TypeError(msg)

    # Validate and build the penalty term
    if penalty is None:
        penalty_array = None
    elif isinstance(penalty, str):
        if penalty == 'laplacian':
            penalty_array = np.array([[0, -1, 0],
                                      [-1, 4, -1],
                                      [0, -1, 0]])
        else:
            msg = f'Invalid penalty string {penalty!r}. Must be "laplacian"'
            raise ValueError(msg)
    elif isinstance(penalty, np.ndarray):
        if penalty.ndim != 2:
            msg = 'penalty array must be 2D.'
            raise ValueError(msg)
        penalty_array = penalty
    else:
        msg = 'penalty must be None, "laplacian", or a 2D numpy array'
        raise ValueError(msg)

    # ensure input PSFs are normalized
    source_psf /= source_psf.sum()
    target_psf /= target_psf.sum()

    source_otf = np.fft.fft2(source_psf)
    target_otf = np.fft.fft2(target_psf)

    source_power = np.abs(source_otf) ** 2

    if penalty_array is not None:
        # Frequency-dependent regularization
        penalty_otf = _psf2otf(penalty_array, source_psf.shape)
        reg_term = regularization * np.abs(penalty_otf) ** 2
    else:
        # Wiener (Tikhonov; scalar/zero-order) regularization.
        # This is frequency-independent and expressed as a fraction of
        # the peak power in the source OTF
        reg_term = regularization * np.max(source_power)

    # Compute the Wiener-regularized kernel in Fourier space
    kernel_otf = (target_otf * np.conj(source_otf)
                  / (source_power + reg_term))

    # Apply a window function in frequency space
    if window is not None:
        kernel_otf = np.fft.fftshift(kernel_otf)
        kernel_otf *= window(target_psf.shape)
        kernel_otf = np.fft.ifftshift(kernel_otf)

    kernel = np.real(np.fft.fftshift(np.fft.ifft2(kernel_otf)))
    return kernel / kernel.sum()


@deprecated('3.0', alternative='make_kernel')
def create_matching_kernel(source_psf, target_psf, *, window=None,
                           otf_threshold=1e-4):
    """
    Create a kernel to match 2D point spread functions (PSF).

    .. deprecated:: 3.0
        ``create_matching_kernel`` is deprecated as of Photutils 3.0 and
        will be removed in a future version. Use `make_kernel` instead.
    """
    return make_kernel(source_psf,  # pragma: no cover
                       target_psf,
                       window=window,
                       otf_threshold=otf_threshold)
