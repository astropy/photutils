.. _psf_matching:

PSF Matching (`photutils.psf.matching`)
=======================================

Introduction
------------

This subpackage contains tools to generate kernels for matching point
spread functions (PSFs).


Matching PSFs
-------------

Photutils provides a function called
:func:`~photutils.psf.matching.create_matching_kernel` that generates
a matching kernel between two PSFs using the ratio of Fourier
transforms (see e.g., `Gordon et al. 2008`_; `Aniano et al. 2011`_).

For this first simple example, let's assume our source and target PSFs
are noiseless 2D Gaussians.  The "high-resolution" PSF will be a
Gaussian with :math:`\sigma=3`.  The "low-resolution" PSF will be a
Gaussian with :math:`\sigma=5`:

    >>> import numpy as np
    >>> from astropy.modeling.models import Gaussian2D
    >>> y, x = np.mgrid[0:51, 0:51]
    >>> gm1 = Gaussian2D(100, 25, 25, 3, 3)
    >>> gm2 = Gaussian2D(100, 25, 25, 5, 5)
    >>> g1 = gm1(x, y)
    >>> g2 = gm2(x, y)
    >>> g1 /= g1.sum()
    >>> g2 /= g2.sum()

For these 2D Gaussians, the matching kernel should be a 2D Gaussian
with :math:`\sigma=4` (``sqrt(5**2 - 3**2)``).  Let's create the
matching kernel using a Fourier ratio method.  Note that the input
source and target PSFs must have the same shape and pixel scale:

Let's plot the result:

We quickly observe that the result is not as expected.  This is
because of high-frequency noise in the Fourier transforms (even though
these are noiseless PSFs, there is floating-point noise in the
ratios).  Using the Fourier ratio method, one must filter the
high-frequency noise from the Fourier ratios.  This is performed by
inputing a `window function
<https://en.wikipedia.org/wiki/Window_function>`_, which may be a
function or a callable object.  In general, the user will need to
exercise some care when defining a window function.  For more
information, please see `Aniano et al. 2011`_.

Photutils provides the following window classes:

* `~photutils.psf.matching.HanningWindow`
* `~photutils.psf.matching.TukeyWindow`
* `~photutils.psf.matching.CosineBellWindow`
* `~photutils.psf.matching.SplitCosineBellWindow`
* `~photutils.psf.matching.TopHatWindow`

Here are plots of 1D cuts across the center of the 2D window
functions:

However, the user may input any function or callable object to
generate a custom window function.

In this example, because these are noiseless PSFs, we will use a
`~photutils.psf.matching.TopHatWindow` object as the low-pass filter:

Note that the output matching kernel from
:func:`~photutils.psf.matching.create_matching_kernel` is always
normalized such that the kernel array sums to 1:

Let's display the new matching kernel:

As desired, the result is indeed a 2D Gaussian with a
:math:`\sigma=4`.  Here we will show 1D cuts across the center of the
kernel images:

Matching IRAC PSFs
------------------

For this example, let's generate a matching kernel to go from the
Spitzer/IRAC channel 1 (3.6 microns) PSF to the channel 4 (8.0
microns) PSF.  We load the PSFs using the

Let's display the images:

For this example, we will use the
:class:`~photutils.psf.matching.CosineBellWindow` for the low-pass
window.  Note that these Spitzer/IRAC channel 1 and 4 PSFs have the
same shape and pixel scale.  If that is not the case, one can use the
:func:`~photutils.psf.matching.resize_psf` convenience function to
resize a PSF image.  Typically, one would interpolate the
lower-resolution PSF to the same size as the higher-resolution PSF.

Let's display the matching kernel result:

The Spitzer/IRAC channel 1 image could then be convolved with this
matching kernel to produce an image with the same resolution as the
channel 4 image.


Reference/API
-------------

.. automodapi:: photutils.psf.matching
    :no-heading:


.. _Gordon et al. 2008:  https://ui.adsabs.harvard.edu/abs/2008ApJ...682..336G/abstract

.. _Aniano et al. 2011:  https://ui.adsabs.harvard.edu/abs/2011PASP..123.1218A/abstract
