.. _psf_matching:

PSF Matching (`photutils.psf_matching`)
=======================================

Introduction
------------

The `photutils.psf_matching` subpackage contains tools to generate
kernels for matching point spread functions (PSFs). It provides two
functions for computing PSF-matching kernels in the Fourier domain:

* :func:`~photutils.psf_matching.make_kernel` — Uses the ratio of
  Fourier transforms with a hard amplitude threshold to regularize
  the division (see e.g., `Gordon et al. 2008`_; `Aniano et al.
  2011`_).

* :func:`~photutils.psf_matching.make_wiener_kernel` — Uses Wiener
  regularization, which smoothly suppresses noise amplification at
  spatial frequencies where the source response is weak.

Both functions take a source PSF and a target PSF and return a matching
kernel that, when convolved with the source PSF, produces the target
PSF. They both support an optional ``window`` function to further
suppress high-frequency noise.


How It Works
^^^^^^^^^^^^

The key idea behind both methods is that convolution in the image
domain corresponds to multiplication in the Fourier domain. If the
source PSF has Fourier transform :math:`S` and the target PSF has
Fourier transform :math:`T`, then the matching kernel :math:`K`
satisfies :math:`T = S \cdot K`, so :math:`K = T / S`.

The Fourier transform of a PSF is called the Optical Transfer Function
(OTF). It describes how different spatial frequencies are transmitted
through the optical system. Low frequencies (coarse image features)
are typically strong in the OTF, while high frequencies (fine details)
are weaker. In practice, dividing by near-zero OTF values amplifies
noise. The two functions handle this differently:

`~photutils.psf_matching.make_kernel` sets the Fourier ratio to zero at
frequencies where the source OTF amplitude is below a fraction of the
peak (controlled by the ``regularization`` parameter, default ``1e-4``):

.. math::

    R = \begin{cases}
        T / S & \text{if } |S| > \lambda \cdot \max(|S|) \\
        0     & \text{otherwise}
    \end{cases}

.. math::

    K = \mathcal{F}^{-1}[W \cdot R]

`~photutils.psf_matching.make_wiener_kernel` instead adds a
regularization term to the denominator, providing continuous, smooth
regularization. Wiener regularization smoothly down-weights frequencies
where the source response is weak, rather than zeroing them out with
a hard threshold. This typically produces matching kernels with less
ringing, especially for PSFs that have near-zero power at high spatial
frequencies.

By default, `~photutils.psf_matching.make_wiener_kernel` uses a
frequency-independent scalar Tikhonov regularization term expressed as a
fraction of the peak power in the source OTF:

.. math::

    K = \mathcal{F}^{-1} \left[ W \cdot
        \frac{T \cdot S^{*}}
             {|S|^{2} + \lambda \cdot \max(|S|^{2})} \right]

where :math:`\mathcal{F}^{-1}` is the inverse Fourier transform,
:math:`S^{*}` is the complex conjugate of :math:`S`, :math:`\lambda` is
the ``regularization`` parameter (default ``1e-4``), and :math:`W` is
the optional ``window`` function (defaulting to 1 if not provided).

When a ``penalty`` operator is provided (e.g., ``penalty='laplacian'``),
the regularization becomes frequency-dependent:

.. math::

    K = \mathcal{F}^{-1} \left[ W \cdot
        \frac{T \cdot S^{*}}
              {|S|^{2} + \lambda \cdot |P|^{2}} \right]

where :math:`P` is the OTF of the penalty operator.

A Laplacian penalty operator suppresses high spatial frequencies
more heavily, which is particularly effective at suppressing noise
amplification. Setting ``penalty='laplacian'`` reproduces the
regularization approach used by the ``pypher`` package (`Boucaud et al.
2016`_).

For additional control, an optional ``window`` function can be applied
to both methods to further suppress high-frequency noise in the
Fourier ratios. This is especially useful for real-world PSFs that
may contain noise, diffraction artifacts, or other features that can
amplify through the division. A window is generally less critical for
`~photutils.psf_matching.make_wiener_kernel` because the regularization
itself suppresses high-frequency noise. For more information about
window functions, please see :ref:`psf_matching_window_functions`.


Choosing a Method
^^^^^^^^^^^^^^^^^

Use `~photutils.psf_matching.make_kernel` when:

- The traditional Fourier-ratio approach with a window function is
  preferred (e.g., see `Aniano et al. 2011`_).
- You want fine-grained control over the spatial frequency-space
  filtering via a window function.

Use `~photutils.psf_matching.make_wiener_kernel` when:

- Working with PSFs that have near-zero power at high spatial
  frequencies (e.g., diffraction-limited PSFs).
- You want to avoid ringing artifacts without needing to carefully
  tune a window function.
- A single regularization parameter is preferred over choosing an OTF
  amplitude threshold plus a window function.
- You want frequency-dependent regularization using a penalty operator
  (e.g., ``penalty='laplacian'`` for ``pypher``-style regularization).


PSF Requirements and Preparation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The input source and target PSFs must satisfy these requirements:

* **Same shape and pixel scale**: Both PSFs must be 2D arrays with
  identical shapes and pixel scales. If your PSFs have different shapes
  or pixel scales, use the :func:`~photutils.psf_matching.resize_psf`
  function to resample one PSF to match the other. This function uses
  spline interpolation and preserves the total flux.

* **Odd dimensions**: PSF arrays should have odd dimensions in both axes
  to ensure a well-defined center point.

* **Normalized**: PSF arrays should be normalized so that the sum of all
  pixels equals 1.

* **Centered** (recommended but not required): The peak of the PSF
  should be at the center of the array.


Noiseless Gaussian Example
---------------------------

For this first simple example, let's assume our source and target PSFs
are noiseless 2D Gaussians. The "high-resolution" PSF will be a Gaussian
with :math:`\sigma=3`. The "low-resolution" PSF will be a Gaussian with
:math:`\sigma=5`::

    >>> import numpy as np
    >>> from photutils.psf import CircularGaussianSigmaPRF
    >>> yy, xx = np.mgrid[0:51, 0:51]
    >>> gm1 = CircularGaussianSigmaPRF(flux=1, x_0=25, y_0=25, sigma=3)
    >>> gm2 = CircularGaussianSigmaPRF(flux=1, x_0=25, y_0=25, sigma=5)
    >>> psf1 = gm1(xx, yy)
    >>> psf2 = gm2(xx, yy)

For these 2D Gaussians, the matching kernel should be a 2D Gaussian with
:math:`\sigma=4` (:math:`\sqrt{5^2 - 3^2}`). Let's create the matching
kernel using both methods.

Using ``make_kernel``::

    >>> from photutils.psf_matching import make_kernel
    >>> kernel1 = make_kernel(psf1, psf2)

Using ``make_wiener_kernel``::

    >>> from photutils.psf_matching import make_wiener_kernel
    >>> kernel2 = make_wiener_kernel(psf1, psf2)

Both output kernels are 2D arrays representing the matching kernel that,
when convolved with the source PSF, produces the target PSF. The output
matching kernels are normalized such that they sum to 1::

    >>> print(kernel1.sum())  # doctest: +FLOAT_CMP
    1.0

    >>> print(kernel2.sum())  # doctest: +FLOAT_CMP
    1.0

Let's plot both results side by side:

.. plot::

    import matplotlib.pyplot as plt
    import numpy as np
    from photutils.psf import CircularGaussianSigmaPRF
    from photutils.psf_matching import make_kernel, make_wiener_kernel

    yy, xx = np.mgrid[0:51, 0:51]
    gm1 = CircularGaussianSigmaPRF(flux=1, x_0=25, y_0=25, sigma=3)
    gm2 = CircularGaussianSigmaPRF(flux=1, x_0=25, y_0=25, sigma=5)
    psf1 = gm1(xx, yy)
    psf2 = gm2(xx, yy)

    kernel1 = make_kernel(psf1, psf2)
    kernel2 = make_wiener_kernel(psf1, psf2)

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    axim1 = ax[0].imshow(kernel1, origin='lower')
    plt.colorbar(axim1, ax=ax[0])
    ax[0].set_title('make_kernel')

    axim2 = ax[1].imshow(kernel2, origin='lower')
    plt.colorbar(axim2, ax=ax[1])
    ax[1].set_title('make_wiener_kernel')

    fig.tight_layout()

As expected, both results are 2D Gaussians with :math:`\sigma=4`. Here
we show 1D cuts across the center of the kernel images to confirm:

.. plot::

    import matplotlib.pyplot as plt
    import numpy as np
    from photutils.psf import CircularGaussianSigmaPRF
    from photutils.psf_matching import make_kernel, make_wiener_kernel

    yy, xx = np.mgrid[0:51, 0:51]
    gm1 = CircularGaussianSigmaPRF(flux=1, x_0=25, y_0=25, sigma=3)
    gm2 = CircularGaussianSigmaPRF(flux=1, x_0=25, y_0=25, sigma=5)
    gm3 = CircularGaussianSigmaPRF(flux=1, x_0=25, y_0=25, sigma=4)
    psf1 = gm1(xx, yy)
    psf2 = gm2(xx, yy)
    psf3 = gm3(xx, yy)

    kernel1 = make_kernel(psf1, psf2)
    kernel2 = make_wiener_kernel(psf1, psf2)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(kernel1[25, :], label='make_kernel', lw=4)
    ax.plot(kernel2[25, :], label='make_wiener_kernel', lw=2, ls='--')
    ax.plot(psf3[25, :], label='$\\sigma=4$ Gaussian', ls=':')
    ax.set_xlabel('x')
    ax.set_ylabel('Flux along y=25')
    plt.legend()
    ax.set_ylim((0.0, 0.011))

For these noiseless Gaussians, both methods produce nearly identical
results. The key differences emerge when working with real-world PSFs
that have significant structure in their power spectra.


.. _psf_matching_window_functions:

Window Functions
----------------

When working with real-world PSFs (e.g., from observations or
optical models), the Fourier ratio can still contain residual
high-frequency spatial noise even after regularization. An optional
`window function <https://en.wikipedia.org/wiki/Window_function>`_
(also called a taper function) can be applied to further suppress
these artifacts. Both :func:`~photutils.psf_matching.make_kernel` and
:func:`~photutils.psf_matching.make_wiener_kernel` accept an optional
``window`` parameter.

A window function multiplies the Fourier ratio by a smooth,
radially-symmetric 2D filter that equals 1.0 in the central
low-frequency region and falls to 0.0 at the edges. This filters out
high spatial frequencies where the signal-to-noise ratio is poorest.
The trade-off is that tapering removes some real information along
with the noise, so the choice of window involves balancing artifact
suppression against fidelity. A window is generally less critical for
`~photutils.psf_matching.make_wiener_kernel` because the regularization
itself suppresses high-frequency noise.

``photutils.psf_matching`` provides five built-in window classes. They
are all subclasses of `~photutils.psf_matching.SplitCosineBellWindow`,
which is parameterized by two values:

* ``alpha``: the fraction of the array radius over which the taper
  occurs (the cosine transition region).
* ``beta``: the fraction of the array radius that remains at 1.0
  (the flat inner region).

The different window classes set these parameters in specific ways,
offering different levels of convenience and control.

`~photutils.psf_matching.HanningWindow`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The `Hann window <https://en.wikipedia.org/wiki/Hann_function>`_
(``alpha=1.0``, ``beta=0.0``) is a raised cosine that equals 1.0 only at
the exact center and smoothly tapers to zero at the edges. The entire
array is tapered. This provides the strongest sidelobe suppression in
Fourier space, at the cost of attenuating most of the data. Use this
when edge artifacts and ringing are a primary concern.

`~photutils.psf_matching.TukeyWindow`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The `Tukey window
<https://en.wikipedia.org/wiki/Window_function#Tukey_window>`_ (``beta
= 1 - alpha``) features a flat central plateau at 1.0 surrounded by a
cosine taper. The ``alpha`` parameter controls the fraction of the array
that is tapered: smaller ``alpha`` preserves more data but provides less
artifact suppression, while larger ``alpha`` tapers more aggressively.
When ``alpha=0`` it becomes a `~photutils.psf_matching.TopHatWindow`;
when ``alpha=1`` it becomes a `~photutils.psf_matching.HanningWindow`.
This window provides a good balance and is a solid general-purpose
choice.

`~photutils.psf_matching.CosineBellWindow`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The cosine bell window (``alpha=alpha``, ``beta=0.0``) equals 1.0 at
the center and begins tapering immediately outward using a cosine
function over a fraction ``alpha`` of the array radius. Beyond
the taper region the window is zero. When ``alpha=1``, this is
equivalent to a `~photutils.psf_matching.HanningWindow`. Compared to
a `~photutils.psf_matching.TukeyWindow` with the same ``alpha``, the
cosine bell has no flat plateau, so the taper starts closer to the
center.

`~photutils.psf_matching.SplitCosineBellWindow`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The split cosine bell is the most general window, taking both ``alpha``
and ``beta`` as independent parameters. The window equals 1.0 for
radii less than ``beta`` times the maximum radius, tapers over the
next ``alpha`` fraction, and is zero beyond. Use this when you need
fine-grained control over both the preserved region and the taper width.

`~photutils.psf_matching.TopHatWindow`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The top hat window (``alpha=0.0``, ``beta=beta``) equals 1.0 inside
a circular region and drops sharply to 0.0 outside with no smooth
transition. This preserves all data within the cutoff radius but the
sharp edge creates strong ringing artifacts in Fourier space. For most
PSF matching applications, `~photutils.psf_matching.TukeyWindow` is
generally preferred over this window.

Custom Window Functions
^^^^^^^^^^^^^^^^^^^^^^^

Users may also define their own custom window function and
pass it to :func:`~photutils.psf_matching.make_kernel` or
:func:`~photutils.psf_matching.make_wiener_kernel`. The window function
should be a callable that takes a single ``shape`` argument (a tuple
defining the 2D array shape) and returns a 2D array of the same shape
containing the window values. The window values should range from 0.0
to 1.0, where 1.0 indicates full preservation of that spatial frequency
and 0.0 indicates complete suppression. The window should be radially
symmetric and centered on the array.

Example Window Function Plots
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Here are plots of 1D cuts across the center of each 2D window function
defined above:

.. plot::

    import matplotlib.pyplot as plt
    from photutils.psf_matching import (CosineBellWindow, HanningWindow,
                                        SplitCosineBellWindow, TopHatWindow,
                                        TukeyWindow)

    w1 = HanningWindow()
    w2 = TukeyWindow(alpha=0.5)
    w3 = CosineBellWindow(alpha=0.5)
    w4 = SplitCosineBellWindow(alpha=0.4, beta=0.3)
    w5 = TopHatWindow(beta=0.4)
    shape = (101, 101)
    y0 = (shape[0] - 1) // 2

    # Initialize figure
    fig = plt.figure(figsize=(10, 7))

    # Create a 2-row, 6-column grid
    gs = fig.add_gridspec(2, 6)

    # First row: 3 plots, each spanning 2 columns
    ax1 = fig.add_subplot(gs[0, 0:2])
    ax2 = fig.add_subplot(gs[0, 2:4])
    ax3 = fig.add_subplot(gs[0, 4:6])

    # Second row: 2 plots, centered (occupying columns 1-2 and 3-4)
    ax4 = fig.add_subplot(gs[1, 1:3])
    ax5 = fig.add_subplot(gs[1, 3:5])

    axes = [ax1, ax2, ax3, ax4, ax5]
    windows = [w1, w2, w3, w4, w5]
    titles = ['Hanning',
              'Tukey\n(alpha=0.5)',
              'Cosine Bell\n(alpha=0.5)',
              'Split Cosine Bell\n(alpha=0.4, beta=0.3)',
              'Top Hat\n(beta=0.4)'
              ]

    # Plot using the OO interface
    for ax, window, title in zip(axes, windows, titles):
        ax.plot(window(shape)[y0, :])
        ax.set_title(title)
        ax.set_xlabel('x')
        ax.set_ylim((0, 1.1))

    fig.tight_layout()


Matching Spitzer IRAC PSFs
--------------------------

For this example, let's generate a matching kernel to go
from the Spitzer/IRAC channel 1 (3.6 microns) PSF to the
channel 4 (8.0 microns) PSF. We load the PSFs using the
:func:`~photutils.datasets.load_irac_psf` convenience function::

    >>> from photutils.datasets import load_irac_psf
    >>> ch1_hdu = load_irac_psf(channel=1)  # doctest: +REMOTE_DATA
    >>> ch4_hdu = load_irac_psf(channel=4)  # doctest: +REMOTE_DATA
    >>> ch1_psf = ch1_hdu.data  # doctest: +REMOTE_DATA
    >>> ch4_psf = ch4_hdu.data  # doctest: +REMOTE_DATA

Let's display the images:

.. plot::

    import matplotlib.pyplot as plt
    from astropy.visualization import SimpleNorm
    from photutils.datasets import load_irac_psf

    ch1_hdu = load_irac_psf(channel=1)
    ch4_hdu = load_irac_psf(channel=4)
    ch1_psf = ch1_hdu.data
    ch4_psf = ch4_hdu.data

    fig, ax = plt.subplots(ncols=2, figsize=(9, 4))

    snorm = SimpleNorm('log', log_a=1000)
    snorm.imshow(ch1_psf, ax=ax[0], origin='lower')
    snorm.imshow(ch4_psf, ax=ax[1], origin='lower')
    ax[0].set_title('IRAC channel 1 PSF')
    ax[1].set_title('IRAC channel 4 PSF')

    fig.tight_layout()

Note that these Spitzer/IRAC channel 1 and 4 PSFs have the same
shape and pixel scale. If that is not the case, one can use the
:func:`~photutils.psf_matching.resize_psf` convenience function
to resize a PSF image. Typically, one would interpolate the
lower-resolution PSF to the same size as the higher-resolution PSF.

For real-world PSFs like these, applying a window function is
recommended for :func:`~photutils.psf_matching.make_kernel` to
suppress residual high-frequency artifacts. Here we use the
:class:`~photutils.psf_matching.SplitCosineBellWindow`:

.. doctest-skip::

    >>> from photutils.psf_matching import (SplitCosineBellWindow,
    ...                                     make_kernel)
    >>> window = SplitCosineBellWindow(alpha=0.15, beta=0.3)
    >>> kernel1 = make_kernel(ch1_psf, ch4_psf, window=window,
    ...                       regularization=0.0001)

With :func:`~photutils.psf_matching.make_wiener_kernel`, the Wiener
regularization itself suppresses high-frequency noise, so a window
function is generally not needed:

.. doctest-skip::

    >>> from photutils.psf_matching import make_wiener_kernel
    >>> kernel2 = make_wiener_kernel(ch1_psf, ch4_psf,
    ...                              regularization=0.0001)

For frequency-dependent regularization using a Laplacian or biharmonic
penalty operator:

.. doctest-skip::

    >>> kernel3 = make_wiener_kernel(ch1_psf, ch4_psf,
    ...                              regularization=0.0001,
    ...                              penalty='laplacian')
    >>> kernel4 = make_wiener_kernel(ch1_psf, ch4_psf,
    ...                              regularization=0.0001,
    ...                              penalty='biharmonic')

Let's display the matching kernel results from all methods:

.. plot::

    import matplotlib.pyplot as plt
    from astropy.visualization import SimpleNorm
    from photutils.datasets import load_irac_psf
    from photutils.psf_matching import (SplitCosineBellWindow, make_kernel,
                                        make_wiener_kernel)

    ch1_hdu = load_irac_psf(channel=1)
    ch4_hdu = load_irac_psf(channel=4)
    ch1_psf = ch1_hdu.data
    ch4_psf = ch4_hdu.data

    window = SplitCosineBellWindow(alpha=0.15, beta=0.3)
    regularization = 0.0001
    kernel1 = make_kernel(ch1_psf, ch4_psf, window=window,
                          regularization=regularization)
    kernel2 = make_wiener_kernel(ch1_psf, ch4_psf,
                                 regularization=regularization)
    kernel3 = make_wiener_kernel(ch1_psf, ch4_psf,
                                 regularization=regularization,
                                 penalty='laplacian')
    kernel4 = make_wiener_kernel(ch1_psf, ch4_psf,
                                 regularization=regularization,
                                 penalty='biharmonic')

    snorm = SimpleNorm('log', log_a=10)
    kernels = [kernel1, kernel2, kernel3, kernel4]
    titles = ['make_kernel',
              'make_wiener_kernel',
              'make_wiener_kernel\n(Laplacian penalty)',
              'make_wiener_kernel\n(biharmonic penalty)']

    fig, axes = plt.subplots(2, 2, figsize=(8, 7))
    for ax, kernel, title in zip(axes.ravel(), kernels, titles):
        axim = snorm.imshow(kernel, ax=ax, origin='lower')
        plt.colorbar(axim, ax=ax)
        ax.set_title(title)

    fig.tight_layout()

Let's now convolve the channel 1 PSF with each matching kernel using
`scipy.signal.fftconvolve` and compare the PSF-matched results with the
channel 4 PSF:

.. plot::

    import matplotlib.pyplot as plt
    from astropy.visualization import SimpleNorm
    from scipy.signal import fftconvolve

    from photutils.datasets import load_irac_psf
    from photutils.psf_matching import (SplitCosineBellWindow, make_kernel,
                                        make_wiener_kernel)

    ch1_hdu = load_irac_psf(channel=1)
    ch4_hdu = load_irac_psf(channel=4)
    ch1_psf = ch1_hdu.data
    ch4_psf = ch4_hdu.data

    window = SplitCosineBellWindow(alpha=0.15, beta=0.3)
    regularization = 0.0001
    kernel1 = make_kernel(ch1_psf, ch4_psf, window=window,
                          regularization=regularization)
    kernel2 = make_wiener_kernel(ch1_psf, ch4_psf,
                                 regularization=regularization)
    kernel3 = make_wiener_kernel(ch1_psf, ch4_psf,
                                 regularization=regularization,
                                 penalty='laplacian')
    kernel4 = make_wiener_kernel(ch1_psf, ch4_psf,
                                 regularization=regularization,
                                 penalty='biharmonic')

    matched1 = fftconvolve(ch1_psf, kernel1, mode='same')
    matched2 = fftconvolve(ch1_psf, kernel2, mode='same')
    matched3 = fftconvolve(ch1_psf, kernel3, mode='same')
    matched4 = fftconvolve(ch1_psf, kernel4, mode='same')

    snorm = SimpleNorm('log', log_a=1000)

    titles = ['Channel 4 PSF',
              'make_kernel',
              'make_wiener_kernel',
              'make_wiener_kernel\n(Laplacian penalty)',
              'make_wiener_kernel\n(biharmonic penalty)']

    images = [ch4_psf, matched1, matched2, matched3, matched4]

    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 1])

    # Column 1 spans both rows
    ax_main = fig.add_subplot(gs[:, 0])

    # Columns 2 & 3: 2x2 grid
    ax_top_left = fig.add_subplot(gs[0, 1])
    ax_top_right = fig.add_subplot(gs[0, 2])
    ax_bot_left = fig.add_subplot(gs[1, 1])
    ax_bot_right = fig.add_subplot(gs[1, 2])

    axes = [ax_main, ax_top_left, ax_top_right, ax_bot_left, ax_bot_right]

    for ax, img, title in zip(axes, images, titles):
        axim = snorm.imshow(img, ax=ax, origin='lower')
        plt.colorbar(axim, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(title)

    fig.suptitle('Channel 1 PSF-matched to Channel 4',
                 x=0.666, y=0.98, ha='center', fontsize=16)

    fig.tight_layout(rect=[0, 0, 1, 0.98])

    plt.show()

Now let's examine the residuals between the PSF-matched results and the
channel 4 PSF target:

.. plot::

    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.signal import fftconvolve

    from photutils.datasets import load_irac_psf
    from photutils.psf_matching import (SplitCosineBellWindow, make_kernel,
                                        make_wiener_kernel)

    ch1_hdu = load_irac_psf(channel=1)
    ch4_hdu = load_irac_psf(channel=4)
    ch1_psf = ch1_hdu.data
    ch4_psf = ch4_hdu.data

    window = SplitCosineBellWindow(alpha=0.15, beta=0.3)
    regularization = 0.0001
    kernel1 = make_kernel(ch1_psf, ch4_psf, window=window,
                          regularization=regularization)
    kernel2 = make_wiener_kernel(ch1_psf, ch4_psf,
                                 regularization=regularization)
    kernel3 = make_wiener_kernel(ch1_psf, ch4_psf,
                                 regularization=regularization,
                                 penalty='laplacian')
    kernel4 = make_wiener_kernel(ch1_psf, ch4_psf,
                                 regularization=regularization,
                                 penalty='biharmonic')

    matched1 = fftconvolve(ch1_psf, kernel1, mode='same')
    matched2 = fftconvolve(ch1_psf, kernel2, mode='same')
    matched3 = fftconvolve(ch1_psf, kernel3, mode='same')
    matched4 = fftconvolve(ch1_psf, kernel4, mode='same')

    resid1 = matched1 - ch4_psf
    resid2 = matched2 - ch4_psf
    resid3 = matched3 - ch4_psf
    resid4 = matched4 - ch4_psf
    vmax = np.abs(
        np.array([resid1, resid2, resid3, resid4])).max()

    titles = ['make_kernel',
              'make_wiener_kernel',
              'make_wiener_kernel\n(Laplacian penalty)',
              'make_wiener_kernel\n(biharmonic penalty)']
    residuals = [resid1, resid2, resid3, resid4]

    fig, axes = plt.subplots(2, 2, figsize=(9, 7))
    for ax, resid, title in zip(axes.ravel(), residuals, titles):
        axim = ax.imshow(resid, origin='lower', cmap='RdBu_r',
                         vmin=-vmax, vmax=vmax)
        plt.colorbar(axim, ax=ax)
        ax.set_title(title)

    fig.suptitle('Residuals: PSF-matched minus channel 4 PSF')
    fig.tight_layout()

The residuals are small relative to the peak PSF values, confirming that
all four methods produce good PSF matches.

Finally, let's compare the encircled energies of the PSF-matched
results with the channel 1 and channel 4 PSFs using the
:class:`~photutils.profiles.CurveOfGrowth` class. A residual subpanel
immediately below the main panel shows how well each PSF-matched curve
agrees with the channel 4 target:

.. plot::

    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.signal import fftconvolve

    from photutils.datasets import load_irac_psf
    from photutils.profiles import CurveOfGrowth
    from photutils.psf_matching import (SplitCosineBellWindow, make_kernel,
                                        make_wiener_kernel)

    ch1_hdu = load_irac_psf(channel=1)
    ch4_hdu = load_irac_psf(channel=4)
    ch1_psf = ch1_hdu.data
    ch4_psf = ch4_hdu.data

    window = SplitCosineBellWindow(alpha=0.15, beta=0.3)
    regularization = 0.0001
    kernel1 = make_kernel(ch1_psf, ch4_psf, window=window,
                          regularization=regularization)
    kernel2 = make_wiener_kernel(ch1_psf, ch4_psf,
                                 regularization=regularization)
    kernel3 = make_wiener_kernel(ch1_psf, ch4_psf,
                                 regularization=regularization,
                                 penalty='laplacian')
    kernel4 = make_wiener_kernel(ch1_psf, ch4_psf,
                                 regularization=regularization,
                                 penalty='biharmonic')

    matched1 = fftconvolve(ch1_psf, kernel1, mode='same')
    matched2 = fftconvolve(ch1_psf, kernel2, mode='same')
    matched3 = fftconvolve(ch1_psf, kernel3, mode='same')
    matched4 = fftconvolve(ch1_psf, kernel4, mode='same')

    xycen = (40.0, 40.0)
    radii = np.arange(1, 40)

    cog_ch1 = CurveOfGrowth(ch1_psf, xycen, radii)
    cog_ch4 = CurveOfGrowth(ch4_psf, xycen, radii)
    cog_m1 = CurveOfGrowth(matched1, xycen, radii)
    cog_m2 = CurveOfGrowth(matched2, xycen, radii)
    cog_m3 = CurveOfGrowth(matched3, xycen, radii)
    cog_m4 = CurveOfGrowth(matched4, xycen, radii)

    for cog in [cog_ch1, cog_ch4, cog_m1, cog_m2, cog_m3, cog_m4]:
        cog.normalize()

    labels = [
        'make_kernel',
        'make_wiener_kernel',
        'make_wiener_kernel (Laplacian)',
        'make_wiener_kernel (biharmonic)',
    ]
    cogs_matched = [cog_m1, cog_m2, cog_m3, cog_m4]
    ls_list = ['--', '-.', ':', (0, (3, 1, 1, 1))]

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(8, 8),
        gridspec_kw={'height_ratios': [3, 1]},
        sharex=True,
    )

    # Main panel
    cog_ch1.plot(ax=ax_top, label='Channel 1 PSF', lw=2,
                 color='C0', ls='-')
    cog_ch4.plot(ax=ax_top, label='Channel 4 PSF', lw=3,
                 color='k')
    for cog, label, ls in zip(cogs_matched, labels, ls_list):
        cog.plot(ax=ax_top, label=label, lw=2, ls=ls)
    ax_top.set_ylabel('Normalized Encircled Energy')
    ax_top.set_title(
        'Encircled Energy: Channel 1 & 4 PSFs vs. PSF-matched results')
    ax_top.legend(fontsize=9)
    ax_top.set_xlabel('')

    # Residual subpanel (matched - ch4)
    for cog, label, ls in zip(cogs_matched, labels, ls_list):
        resid = cog.profile - cog_ch4.profile
        ax_bot.plot(cog.radius, resid, lw=2, ls=ls, label=label)
    ax_bot.axhline(0, color='k', lw=1, ls='-')
    ax_bot.set_xlabel('Radius (pixels)')
    ax_bot.set_ylabel('Residual')
    ax_bot.set_title('Matched $-$ Channel 4')

    fig.tight_layout()

The encircled energy curves for the PSF-matched results closely track
the channel 4 PSF, confirming that the PSF matching has been performed
successfully across all four methods. The residual subpanel quantifies
the small remaining differences between each PSF-matched curve and the
channel 4 target.


API Reference
-------------

:doc:`../reference/psf_matching_api`


.. _Gordon et al. 2008:  https://ui.adsabs.harvard.edu/abs/2008ApJ...682..336G/abstract

.. _Aniano et al. 2011:  https://ui.adsabs.harvard.edu/abs/2011PASP..123.1218A/abstract

.. _Boucaud et al. 2016:  https://ui.adsabs.harvard.edu/abs/2016A%26A...596A..63B/abstract
