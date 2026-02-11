.. _psf_matching:

PSF Matching (`photutils.psf_matching`)
=======================================

Introduction
------------

This subpackage contains tools to generate kernels for matching point
spread functions (PSFs).


Matching PSFs
-------------

Photutils provides a function called
:func:`~photutils.psf_matching.create_matching_kernel` that generates a
matching kernel between two PSFs using the ratio of Fourier transforms
(see e.g., `Gordon et al. 2008`_; `Aniano et al. 2011`_).

The key idea behind this method is that convolution in the image domain
corresponds to multiplication in the Fourier domain. If the source PSF
has Fourier transform :math:`S` and the target PSF has Fourier transform
:math:`T`, then the matching kernel :math:`K` satisfies :math:`T = S
\cdot K`, so :math:`K = T / S`.

The Fourier transform of a PSF is called the Optical Transfer Function
(OTF). It describes how different spatial frequencies are transmitted
through the optical system. Low frequencies (coarse image features)
are typically strong in the OTF, while high frequencies (fine details)
are weaker. In practice, dividing by near-zero OTF values amplifies
noise. The ``fourier_cutoff`` parameter (default ``1e-4``) handles this
by zeroing out frequencies where the source OTF amplitude is below
a fraction of the peak, preventing division by near-zero values and
producing a clean result in most cases.

For additional control, an optional ``window`` function can be applied
to further suppress high-frequency noise in the Fourier ratios. This
is especially useful for real-world PSFs that may contain noise,
diffraction artifacts, or other features that can amplify through the
division. For more information, please see `Aniano et al. 2011`_.


PSF Requirements and Preparation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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

* **Centered** (recommended but not required): The peak of the PSF should
  be at the center of the array. A warning will be issued if the peak is
  off-center, but the function will still work.


Noiseless Gaussian Example
^^^^^^^^^^^^^^^^^^^^^^^^^^

For this first simple example, let's assume our source and target PSFs
are noiseless 2D Gaussians. The "high-resolution" PSF will be a Gaussian
with :math:`\sigma=3`. The "low-resolution" PSF will be a Gaussian with
:math:`\sigma=5`::

    >>> import numpy as np
    >>> from astropy.modeling.models import Gaussian2D
    >>> yy, xx = np.mgrid[0:51, 0:51]
    >>> gm1 = Gaussian2D(100, 25, 25, 3, 3)
    >>> gm2 = Gaussian2D(100, 25, 25, 5, 5)
    >>> psf1 = gm1(x, y)
    >>> psf2 = gm2(x, y)
    >>> psf1 /= psf1.sum()  # normalize the PSF
    >>> psf2 /= psf2.sum()

For these 2D Gaussians, the matching kernel should be a 2D Gaussian with
:math:`\sigma=4` (:math:`\sqrt{5^2 - 3^2}`). Let's create the matching
kernel::

    >>> from photutils.psf_matching import create_matching_kernel
    >>> kernel = create_matching_kernel(psf1, psf2, fourier_cutoff=1e-3)

The output ``kernel`` is a 2D array representing the matching kernel
that, when convolved with the source PSF, produces the target PSF. The
``fourier_cutoff`` parameter ensures a clean result by regularizing the
Fourier division.

The output matching kernel is always normalized such that it sums to 1::

    >>> print(kernel.sum())  # doctest: +FLOAT_CMP
    1.0

Let's plot the result:

.. plot::

    import matplotlib.pyplot as plt
    import numpy as np
    from astropy.modeling.models import Gaussian2D
    from photutils.psf_matching import create_matching_kernel

    y, x = np.mgrid[0:51, 0:51]
    gm1 = Gaussian2D(100, 25, 25, 3, 3)
    gm2 = Gaussian2D(100, 25, 25, 5, 5)
    psf1 = gm1(x, y)
    psf2 = gm2(x, y)
    psf1 /= psf1.sum()
    psf2 /= psf2.sum()

    kernel = create_matching_kernel(psf1, psf2, fourier_cutoff=1e-3)
    fig, ax = plt.subplots()
    axim = ax.imshow(kernel, origin='lower')
    plt.colorbar(axim, ax=ax)
    ax.set_title('Matching kernel')

As expected, the result is a 2D Gaussian with :math:`\sigma=4`. Here we
show 1D cuts across the center of the kernel images to confirm:

.. plot::

    import matplotlib.pyplot as plt
    import numpy as np
    from astropy.modeling.models import Gaussian2D
    from photutils.psf_matching import create_matching_kernel

    y, x = np.mgrid[0:51, 0:51]
    gm1 = Gaussian2D(100, 25, 25, 3, 3)
    gm2 = Gaussian2D(100, 25, 25, 5, 5)
    gm3 = Gaussian2D(100, 25, 25, 4, 4)
    psf1 = gm1(x, y)
    psf2 = gm2(x, y)
    psf3 = gm3(x, y)
    psf1 /= psf1.sum()
    psf2 /= psf2.sum()
    psf3 /= psf3.sum()

    kernel = create_matching_kernel(psf1, psf2)
    kernel /= kernel.sum()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(kernel[25, :], label='Matching kernel', lw=4)
    ax.plot(psf3[25, :], label='$\\sigma=4$ Gaussian')
    ax.set_xlabel('x')
    ax.set_ylabel('Flux along y=25')
    plt.legend()
    ax.set_ylim((0.0, 0.011))


.. _psf_matching_window_functions:

Window Functions
----------------

When working with real-world PSFs (e.g., from observations or optical
models), the Fourier ratio can still contain residual high-frequency
noise even after the ``fourier_cutoff`` regularization. An optional
`window function <https://en.wikipedia.org/wiki/Window_function>`_
(also called a taper function) can be applied to further suppress
these artifacts.

A window function multiplies the Fourier ratio by a smooth,
radially-symmetric 2D filter that equals 1.0 in the central
low-frequency region and falls to 0.0 at the edges. This filters out
high spatial frequencies where the signal-to-noise ratio is poorest.
The trade-off is that tapering removes some real information along
with the noise, so the choice of window involves balancing artifact
suppression against fidelity.

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

Users may also define their own custom window function and pass it to
:func:`~photutils.psf_matching.create_matching_kernel`. The window
function should be a callable that takes a single ``shape`` argument
(a tuple defining the 2D array shape) and returns a 2D array of the
same shape containing the window values. The window values should range
from 0.0 to 1.0, where 1.0 indicates full preservation of that spatial
frequency and 0.0 indicates complete suppression.

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

For this example, let's generate a matching kernel to go from the
Spitzer/IRAC channel 1 (3.6 microns) PSF to the channel 4 (8.0
microns) PSF. We load the PSFs using the
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

For real-world PSFs like these, applying a window function
is recommended to suppress residual high-frequency
artifacts in the matching kernel. Here we use the
:class:`~photutils.psf_matching.CosineBellWindow`. Note that
these Spitzer/IRAC channel 1 and 4 PSFs have the same shape
and pixel scale. If that is not the case, one can use the
:func:`~photutils.psf_matching.resize_psf` convenience function
to resize a PSF image. Typically, one would interpolate the
lower-resolution PSF to the same size as the higher-resolution PSF.

.. doctest-skip::

    >>> from photutils.psf_matching import (CosineBellWindow,
    ...                                     create_matching_kernel)
    >>> window = CosineBellWindow(alpha=0.35)
    >>> kernel = create_matching_kernel(ch1_psf, ch4_psf, window=window)

Let's display the matching kernel result:

.. plot::

    import matplotlib.pyplot as plt
    from astropy.visualization import SimpleNorm
    from photutils.datasets import load_irac_psf
    from photutils.psf_matching import CosineBellWindow, create_matching_kernel

    ch1_hdu = load_irac_psf(channel=1)
    ch4_hdu = load_irac_psf(channel=4)
    ch1_psf = ch1_hdu.data
    ch4_psf = ch4_hdu.data

    window = CosineBellWindow(alpha=0.35)
    kernel = create_matching_kernel(ch1_psf, ch4_psf, window=window)

    fig, ax = plt.subplots()

    snorm = SimpleNorm('log', log_a=10)
    axim = snorm.imshow(kernel, ax=ax, origin='lower')
    plt.colorbar(axim, ax=ax)
    ax.set_title('Matching kernel')

The Spitzer/IRAC channel 1 image could then be convolved with this
matching kernel to produce an image with the same resolution as the
channel-4 image.


API Reference
-------------

:doc:`../reference/psf_matching_api`


.. _Gordon et al. 2008:  https://ui.adsabs.harvard.edu/abs/2008ApJ...682..336G/abstract

.. _Aniano et al. 2011:  https://ui.adsabs.harvard.edu/abs/2011PASP..123.1218A/abstract
