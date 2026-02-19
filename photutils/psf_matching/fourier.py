# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tools for matching PSFs using Fourier methods.
"""

import numpy as np
from astropy.utils.decorators import deprecated

from photutils.psf_matching.utils import (_convert_psf_to_otf,
                                          _validate_kernel_inputs,
                                          _validate_window_array, resize_psf)

__all__ = ['create_matching_kernel', 'make_kernel', 'make_wiener_kernel',
           'resize_psf']


def make_kernel(source_psf, target_psf, *, window=None, regularization=1e-4):
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
            \\frac{T}{S} & \\text{if } |S| > \\lambda \\cdot \\max(|S|) \\\\
            0 & \\text{otherwise}
            \\end{cases}

    Here, :math:`\\mathcal{F}^{-1}` is the inverse Fourier transform,
    :math:`S` and :math:`T` are the Fourier transforms of the source and
    target PSFs (the optical transfer functions, OTFs), :math:`\\lambda`
    is the ``regularization`` parameter, and :math:`W` is the optional
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

    regularization : float, optional
        The regularization parameter that controls the OTF amplitude
        threshold for the source OTF (Optical Transfer Function, the
        Fourier transform of the PSF). At frequencies where the source
        OTF amplitude is below ``regularization`` times the peak
        amplitude, the Fourier ratio is set to zero to avoid division by
        near-zero values. Must be in the range [0, 1], where 0 provides
        no thresholding (only exact zeros are excluded) and values
        closer to 1 apply more aggressive thresholding.

    Returns
    -------
    kernel : 2D `~numpy.ndarray`
        The matching kernel to go from ``source_psf`` to ``target_psf``.
        The output matching kernel is normalized such that it sums to 1.

    Raises
    ------
    ValueError
        If the PSFs are not 2D arrays, have even dimensions, or do not
        have the same shape, if ``regularization`` is not in the range
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
    source_psf, target_psf = _validate_kernel_inputs(
        source_psf, target_psf, window)

    if not 0 <= regularization <= 1:
        msg = (f'regularization must be in the range [0, 1], '
               f'got {regularization}.')
        raise ValueError(msg)

    source_otf = np.fft.fft2(source_psf)
    target_otf = np.fft.fft2(target_psf)

    # Note: the following calculations are performed in the Fourier
    # domain with the DC component at the corner of the array (standard
    # FFT layout).

    # Regularized division to avoid dividing by near-zero values
    max_otf = np.max(np.abs(source_otf))
    mask = np.abs(source_otf) > regularization * max_otf
    ratio = np.zeros_like(source_otf, dtype=complex)
    ratio[mask] = target_otf[mask] / source_otf[mask]

    # Apply a window function in frequency space
    if window is not None:
        # The window function is defined in the Fourier domain with the
        # DC component at the center of the array. The ratio array is
        # computed with the DC component at the corner of the array,
        # so we need to shift it to the center to apply the window
        # function.
        window_array = window(target_psf.shape)
        _validate_window_array(window_array, target_psf.shape)
        ratio = np.fft.fftshift(ratio)
        ratio *= window_array
        ratio = np.fft.ifftshift(ratio)

    kernel = np.real(np.fft.fftshift(np.fft.ifft2(ratio)))
    return kernel / kernel.sum()


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
                  {|S|^{2} + \\lambda \\cdot \\max(|S|^{2})} \\right]

    When a ``penalty`` array is provided (e.g., a Laplacian operator),
    the regularization becomes frequency-dependent:

    .. math::

        K = \\mathcal{F}^{-1} \\left[ W \\cdot
            \\frac{T \\cdot S^{*}}
                  {|S|^{2} + \\lambda \\cdot |P|^{2}} \\right]

    where :math:`P` is the OTF of the ``penalty`` operator. This
    penalizes high spatial frequencies more heavily, which is
    particularly effective at suppressing noise amplification.

    In both equations, :math:`\\mathcal{F}^{-1}` is the inverse Fourier
    transform, :math:`S` and :math:`T` are the Fourier transforms of
    the source and target PSFs (the OTFs), :math:`S^{*}` is the complex
    conjugate of :math:`S`, :math:`\\lambda` is the ``regularization``
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

    penalty : `None`, ``'laplacian'``, ``'biharmonic'``, or 2D \
`~numpy.ndarray`, optional
        The regularization penalty operator. This controls the
        structure of the regularization term in the denominator:

        * `None` (default): Scalar Tikhonov regularization. The
          denominator is :math:`|S|^2 + \\lambda \\cdot \\max(|S|^2)`,
          providing uniform regularization across all spatial
          frequencies. Use this for well-behaved PSFs or when you want
          simple, frequency-independent smoothing.

        * ``'laplacian'``: Uses a discrete Laplacian operator (second
          derivative) as the penalty, producing frequency-dependent
          regularization that penalizes high spatial frequencies more
          heavily. The denominator becomes :math:`|S|^2 + \\lambda
          \\cdot |L|^2` where :math:`L` is the OTF of the Laplacian
          kernel ``[[0, -1, 0], [-1, 4, -1], [0, -1, 0]]``. This
          reproduces the regularization used by the ``pypher`` package
          (`Boucaud et al. 2016`_). This is the most commonly used
          penalty for PSF matching and works well for most applications.
          Requires PSFs to be at least 3x3.

        * ``'biharmonic'``: Uses a biharmonic operator (fourth
          derivative, Laplacian of the Laplacian) as the penalty,
          producing very strong suppression of high spatial frequencies.
          Uses the kernel ``[[0, 0, 1, 0, 0], [0, 2, -8, 2, 0], [1,
          -8, 20, -8, 1], [0, 2, -8, 2, 0], [0, 0, 1, 0, 0]]``. This
          produces the smoothest matching kernels and is useful when
          working with very noisy or poorly sampled PSFs, at the cost
          of reduced accuracy in matching. Requires PSFs to be at least
          5x5.

        * 2D `~numpy.ndarray`: A custom penalty operator array.
          Its OTF will be computed and used in the denominator as
          :math:`|S|^2 + \\lambda \\cdot |P|^2`. The PSFs must be at
          least as large as the penalty array in both dimensions.

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
        If the PSFs are not 2D arrays, have even dimensions, do not have
        the same shape, are too small for the specified penalty, if
        ``regularization`` is not positive, or if ``penalty`` is not a
        valid value.

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

    Use the biharmonic penalty for maximum smoothness:

    >>> kernel = make_wiener_kernel(psf1, psf2, penalty='biharmonic')
    >>> print(f'{kernel.sum():.1f}')
    1.0
    """
    source_psf, target_psf = _validate_kernel_inputs(
        source_psf, target_psf, window)

    if regularization <= 0:
        msg = 'regularization must be a positive number.'
        raise ValueError(msg)

    # Validate and build the penalty term
    if penalty is None:
        penalty_array = None
    elif isinstance(penalty, str):
        if penalty == 'laplacian':
            penalty_array = np.array([[+0, -1, +0],
                                      [-1, +4, -1],
                                      [+0, -1, +0]])
        elif penalty == 'biharmonic':
            penalty_array = np.array([[+0, +0, +1, +0, +0],
                                      [+0, +2, -8, +2, +0],
                                      [+1, -8, 20, -8, +1],
                                      [+0, +2, -8, +2, +0],
                                      [+0, +0, +1, +0, +0]])
        else:
            msg = (f'Invalid penalty string {penalty!r}. '
                   'Must be "laplacian" or "biharmonic"')
            raise ValueError(msg)
    elif isinstance(penalty, np.ndarray):
        if penalty.ndim != 2:
            msg = 'penalty array must be 2D.'
            raise ValueError(msg)
        penalty_array = penalty
    else:
        msg = ('penalty must be None, "laplacian", "biharmonic", or a 2D '
               'numpy array')
        raise ValueError(msg)

    # Validate that PSF is large enough for the penalty
    if penalty_array is not None:
        penalty_shape = penalty_array.shape
        psf_shape = source_psf.shape
        if (psf_shape[0] < penalty_shape[0]
                or psf_shape[1] < penalty_shape[1]):
            msg = (f'PSFs must be at least as large as the penalty '
                   f'operator. PSF shape is {psf_shape}, but penalty '
                   f'shape is {penalty_shape}.')
            raise ValueError(msg)

    # Ensure input PSFs are normalized
    source_psf /= source_psf.sum()
    target_psf /= target_psf.sum()

    source_otf = np.fft.fft2(source_psf)
    target_otf = np.fft.fft2(target_psf)

    source_power = np.abs(source_otf) ** 2

    if penalty_array is not None:
        # Frequency-dependent regularization
        penalty_otf = _convert_psf_to_otf(penalty_array, source_psf.shape)
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
        # The window function is defined in the Fourier domain with the
        # DC component at the center of the array. The kernel OTF is
        # computed with the DC component at the corner of the array,
        # so we need to shift it to the center to apply the window
        # function.
        kernel_otf = np.fft.fftshift(kernel_otf)
        kernel_otf *= window(target_psf.shape)
        kernel_otf = np.fft.ifftshift(kernel_otf)

    kernel = np.real(np.fft.fftshift(np.fft.ifft2(kernel_otf)))
    return kernel / kernel.sum()


@deprecated('3.0', alternative='make_kernel')
def create_matching_kernel(source_psf, target_psf, *, window=None,
                           regularization=1e-4):
    """
    Create a kernel to match 2D point spread functions (PSF).

    .. deprecated:: 3.0
        ``create_matching_kernel`` is deprecated as of Photutils 3.0 and
        will be removed in a future version. Use `make_kernel` instead.
    """
    return make_kernel(source_psf,  # pragma: no cover
                       target_psf,
                       window=window,
                       regularization=regularization)
