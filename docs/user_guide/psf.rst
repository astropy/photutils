.. _psf-photometry:

PSF Photometry (`photutils.psf`)
================================

The `photutils.psf` subpackage contains tools for model-fitting
photometry, often called "PSF photometry".


PSF Photometry Overview
-----------------------

Photutils provides a modular set of tools to perform PSF photometry
for different science cases. The tools are implemented as classes that
perform various subtasks of PSF photometry. High-level classes are also
provided to connect these pieces together.

The two main PSF-photometry classes are `~photutils.psf.PSFPhotometry`
and `~photutils.psf.IterativePSFPhotometry`.
`~photutils.psf.PSFPhotometry` provides the framework for a flexible PSF
photometry workflow that can find sources in an image, optionally group
overlapping sources, fit the PSF model to the sources, and subtract the
fit PSF models from the image.

`~photutils.psf.IterativePSFPhotometry` is an iterative version of
`~photutils.psf.PSFPhotometry` where new sources are detected in the
residual image after the fit sources are subtracted. The iterative
process can be useful for crowded fields where sources are blended. A
``mode`` keyword is provided to control the behavior of the iterative
process, where either all sources or only the newly-detected sources are
fit in subsequent iterations. The process repeats until no additional
sources are detected or a maximum number of iterations has been
reached. When used with the `~photutils.detection.DAOStarFinder`,
`~photutils.psf.IterativePSFPhotometry` is essentially an implementation
of the DAOPHOT algorithm described by Stetson in his `seminal paper
<https://ui.adsabs.harvard.edu/abs/1987PASP...99..191S/abstract>`_ for
crowded-field stellar photometry.

The source-finding step is controlled by the ``finder``
keyword, where one inputs a callable function or class
instance. Typically, this would be one of the source-detection
classes implemented in the `photutils.detection`
subpackage, e.g., `~photutils.detection.DAOStarFinder`,
`~photutils.detection.IRAFStarFinder`, or
`~photutils.detection.StarFinder`.

After finding sources, one can optionally apply a clustering algorithm
to group overlapping sources using the ``grouper`` keyword. Usually,
groups are formed by a distance criterion, which is the case of the
grouping algorithm proposed by Stetson. Sources that are grouped are
fit simultaneously. The reason behind the construction of groups and
not fitting all sources simultaneously is illustrated as follows:
imagine that one would like to fit 300 sources and the model for each
source has three parameters to be fitted. If one constructs a single
model to fit the 300 sources simultaneously, then the optimization
algorithm will have to search for the solution in a 900-dimensional
space, which is computationally expensive and error-prone. Having
smaller groups of sources effectively reduces the dimension of the
parameter space, which facilitates the optimization process. For more
details see :ref:`source-grouping`.

The local background around each source can optionally be subtracted
using the ``local_bkg_estimator`` keyword. This keyword accepts a
`~photutils.background.LocalBackground` instance that estimates the
local statistics in a circular annulus aperture centered on each source.
The size of the annulus and the statistic function can be configured in
`~photutils.background.LocalBackground`.

The next step is to fit the sources and/or groups. This
task is performed using an Astropy fitter, for example
`~astropy.modeling.fitting.TRFLSQFitter`, input via the ``fitter``
keyword. All of the PSF models provided by `photutils.psf`, both the
analytic and the image-based models, provide analytic Jacobians that the
fitters use automatically, avoiding finite-difference approximations
of the parameter derivatives and speeding up the fits. The shape of
the region to be fitted can be configured using the ``fit_shape``
parameter. In general, ``fit_shape`` should be set to a small size
(e.g., (5, 5)) that covers the central part of the source with the
highest flux signal-to-noise. The initial positions are derived from
the ``finder`` algorithm. The initial flux values for the fit are
derived from measuring the flux in a circular aperture with radius
``aperture_radius``. Alternatively, the initial positions and fluxes can
be input in a table via the ``init_params`` keyword when calling the
class.

After sources are fitted, a model image of the fit
sources or a residual image can be generated using the
:meth:`~photutils.psf.PSFPhotometry.make_model_image` and
:meth:`~photutils.psf.PSFPhotometry.make_residual_image` methods,
respectively.

For `~photutils.psf.IterativePSFPhotometry`, the above steps can be
repeated until no additional sources are detected (or until a maximum
number of iterations is reached).

The `~photutils.psf.PSFPhotometry` and
`~photutils.psf.IterativePSFPhotometry` classes provide the structure
in which the PSF-fitting steps described above are performed, but
all the stages can be turned on or off or replaced with different
implementations as the user desires. This makes the tools very flexible.
One can also bypass several of the steps by directly inputting to
``init_params`` an Astropy table containing the initial parameters for
the source centers, fluxes, group identifiers, and local backgrounds.
This is also useful if one is interested in fitting only one or a few
sources in an image.


.. _psf-terminology:

Terminology
-----------

PSF photometry measures the flux and position of a star by fitting a
model of how the light of a point source is distributed over the pixels
of an image. Several closely related terms are used for such models,
including "PSF", "PRF", and "ePSF", and different astronomy subfields
use them in slightly different ways. This section defines the terms as
they are used in `photutils.psf`.

Point Spread Function (PSF)
    The PSF is the distribution of light from a point source at the
    detector, after the light has passed through the atmosphere (for
    ground-based data) and the telescope optics. It is a smooth,
    continuous function of position that does not depend on the detector
    pixels. It is sometimes called the instrumental PSF (iPSF) to make
    that explicit.

Pixel response function
    A detector does not record the PSF itself. Each pixel collects the
    light that falls on its area and reports a single value. The pixel
    response function describes how sensitive a pixel is to light at
    each position within it. The simplest and most common assumption is
    that a pixel is equally sensitive over its whole area. Some authors
    abbreviate the pixel response function as "PRF". We always write it
    out in full, because we use "PRF" for the Point Response Function
    defined next.

Effective PSF (ePSF) or Point Response Function (PRF)
    The ePSF is the PSF as it is recorded by the pixels. Formally, it
    is the convolution of the PSF and the pixel response function.
    Its value at a given offset from a source is the fraction of the
    source flux that is recorded by a pixel centered at that offset.
    For pixels with a uniform response, that is the integral of the PSF
    over the area of the pixel. "ePSF" and "PRF" are two names for the
    same function. The name "ePSF" comes from `Anderson & King 2000
    <https://ui.adsabs.harvard.edu/abs/2000PASP..112.1360A/abstract>`_,
    which describes this formalism in detail. The name "PRF" emphasizes
    that the function describes the response of the detector to a point
    source, rather than the PSF of the optics.

The pixel values of an image are samples of the ePSF, not of the PSF.
Like the PSF, the ePSF is a continuous function, because a pixel could
be centered at any offset from a source. The image of a star samples
the ePSF at a spacing of one pixel, at offsets that depend on where the
center of the star falls within a pixel. An oversampled ePSF samples the
same function on a finer grid. Each of its values is still the flux in a
whole detector pixel, not the flux in a subpixel.

A model that is fit to the pixel values of an image should therefore
represent the ePSF. The names of the models in `photutils.psf` show what
each model represents:

- The analytic models whose names end in ``PSF`` (e.g.,
  `~photutils.psf.CircularGaussianPSF`) give the value of the PSF at the
  input positions. They are not integrated over the pixels.

- The analytic models whose names end in ``PRF`` (e.g.,
  `~photutils.psf.CircularGaussianPRF`) are integrated over the pixels,
  assuming that the response is uniform across a pixel.

- The image-based models (`~photutils.psf.ImagePSF` and
  `~photutils.psf.GriddedPSFModel`) interpolate an input image and do
  not integrate it over the pixels. Despite the "PSF" in their names,
  their input images must be ePSFs. An ePSF that is built from observed
  stars (see :ref:`build-epsf`) includes the actual response of the
  pixels.

The difference between the PSF and the ePSF is largest for undersampled
data, where the PSF is narrow compared with a pixel, and for detectors
with significant sensitivity variations within a pixel. For such data,
the PSF is much sharper than the image of a star, and the sum of its
values at the pixel centers changes with the subpixel position of the
star. The difference is smaller for well-sampled data, but it does
not vanish, because the integration over a pixel always broadens the
profile. A PSF model with a known, fixed width is then narrower than
the image of a star, which biases the fitted flux. The distinction
is largely inconsequential when the model width is measured directly
from the pixelated image of a well-sampled star, as the measured width
already incorporates the effects of broadening.

In common usage, all of these functions are often simply called
the "PSF", and "PSF photometry" refers to the general technique of
model-fitting photometry, regardless of exactly which kind of model is
fit. We use both terms in that broad sense throughout `photutils.psf`
and this documentation, and we use the specific terms defined above
where the distinction matters.


.. _psf-models:

PSF Models
----------

As mentioned above, PSF photometry fundamentally involves fitting
models to data. As such, the PSF model is a critical component of PSF
photometry. For accurate results, both for photometry and astrometry,
the PSF model should be a good representation of the actual data. The
PSF model can be a simple analytic function, such as a 2D Gaussian
or Moffat profile, or it can be a more complex model derived from a
2D PSF image, e.g., an effective PSF (ePSF). The PSF model can also
encapsulate changes in the PSF across the detector, e.g., due to optical
aberrations.

For image-based PSF models, the PSF model is typically derived from
observed data or from detailed optical modeling. The PSF model can be
a single PSF model for the entire image or a grid of PSF models at
fiducial detector positions. Image-based PSF models are also often
oversampled with respect to the pixel grid to increase the accuracy of
fitting the PSF model.

The observatory that obtained the data may provide tools for creating
PSF models for their data or an empirical library of PSF models
based on previous observations. For example, the `Hubble Space
Telescope <https://www.stsci.edu/hst>`_ provides libraries of
empirical PSF models for ACS and WFC3 (e.g., `WFC3 PSF Search
<https://www.stsci.edu/hst/instrumentation/wfc3/data-analysis/psf/psf-search>`_).
Similarly, the `James Webb Space Telescope <https://www.stsci.edu/jwst>`_
and the `Nancy Grace Roman Space Telescope <https://www.stsci.edu/roman>`_
provide the `STPSF <https://stpsf.readthedocs.io/>`_ Python software
for creating PSF models. In particular, STPSF outputs gridded PSF
models directly as Photutils `~photutils.psf.GriddedPSFModel` instances.

If you cannot obtain a PSF model from an empirical library or
observatory-provided tool, Photutils provides tools for creating an
empirical PSF model from the data itself, provided you have a large
number of isolated stars. Please see :ref:`build-epsf` for more
information and an example.

The `photutils.psf` subpackage provides several PSF models that
can be used for PSF photometry. The PSF models are based on the
:ref:`Astropy models and fitting <astropy:astropy-modeling>` framework.
The PSF models are used as input (via the ``psf_model`` parameter)
to the PSF photometry classes `~photutils.psf.PSFPhotometry` and
`~photutils.psf.IterativePSFPhotometry`. The PSF models are fitted to
the data using an Astropy fitter class. Typically, the model position
(``x_0`` and ``y_0``) and flux (``flux``) parameters are varied
during the fitting process. The PSF model can also include additional
parameters, such as the full width at half maximum (FWHM) or sigma of
a Gaussian PSF or the alpha and beta parameters of a Moffat PSF. By
default, these additional parameters are "fixed" (i.e., not varied
during the fitting process). The user can choose to also vary these
parameters by setting the ``fixed`` attribute on the model parameter
to `False`. The position and/or flux parameters can also be fixed
during the fitting process if needed, e.g., for forced photometry (see
:ref:`psf-forced-photometry`). Any of the model parameters can also be
bounded during the fitting process (see :ref:`psf-bounded-parameters`).

You can also create your own custom PSF model using the Astropy modeling
framework. The PSF model must be a 2D model that is a subclass of
`~astropy.modeling.Fittable2DModel`. It must have parameters called
``x_0``, ``y_0``, and ``flux``, specifying the central position and
total integrated flux. The value of the model at a position should be
the flux in a detector pixel centered at that position (see
:ref:`psf-terminology`), especially for data that are undersampled.


Analytic PSF Models
^^^^^^^^^^^^^^^^^^^

The `photutils.psf` subpackage provides the following analytic PSF
models:

- `~photutils.psf.GaussianPSF`: a general 2D Gaussian PSF model
  parameterized in terms of the position, total flux, and full width
  at half maximum (FWHM) along the x and y axes. Rotation can also be
  included.

- `~photutils.psf.CircularGaussianPSF`: a circular 2D Gaussian PSF model
  parameterized in terms of the position, total flux, and FWHM.

- `~photutils.psf.GaussianPRF`: a general 2D Gaussian PRF model
  parameterized in terms of the position, total flux, and FWHM
  along the x and y axes. Rotation can also be included.

- `~photutils.psf.CircularGaussianPRF`: a circular 2D Gaussian PRF model
  parameterized in terms of the position, total flux, and FWHM.

- `~photutils.psf.CircularGaussianSigmaPRF`: a circular 2D Gaussian PRF
  model parameterized in terms of the position, total flux, and sigma
  (standard deviation).

- `~photutils.psf.MoffatPSF`: a 2D Moffat PSF model parameterized in
  terms of the position, total flux, :math:`\alpha`, and :math:`\beta`
  parameters.

- `~photutils.psf.MoffatPRF`: a 2D Moffat PRF model with the same
  parameters as `~photutils.psf.MoffatPSF`.

- `~photutils.psf.AiryDiskPSF`: a 2D Airy disk PSF model parameterized
  in terms of the position, total flux, and radius of the first dark
  ring.

- `~photutils.psf.AiryDiskPRF`: a 2D Airy disk PRF model with the same
  parameters as `~photutils.psf.AiryDiskPSF`.

Note there are two types of defined models, PSF and PRF models. The PSF
models are evaluated by sampling the analytic function at the input (x,
y) coordinates. The PRF models are evaluated by integrating the analytic
function over the pixel areas.

The values of a PSF model on a grid of detector pixels are the values of
the PSF at the pixel centers, not the fluxes in the pixels. For a PSF
that is undersampled by the detector pixels (a FWHM of less than about 2
pixels), a PSF model is sharper than the sources in the data and the sum
of its values over the pixels depends on the subpixel position of the
source. Even for a well-sampled PSF, a PSF model whose shape parameters
are fixed gives fitted fluxes that are biased by a few percent for a
FWHM of 2 to 3 pixels, because the integration over a pixel broadens
the image of a source. The PRF models should therefore be used to fit
the pixel values of an image, e.g., for PSF photometry. Every PSF model
above has a PRF counterpart. The Gaussian PRF models are integrated
over the pixels analytically. The Moffat and Airy disk PRF models are
integrated numerically, so they are several times slower to evaluate
than their PSF counterparts.

If one needs a custom PRF model based on an analytical PSF
model that has no PRF counterpart, evaluate the PSF model on
an oversampled grid (a Moffat profile is used below only as an
illustration) and integrate the result over the detector pixels
with `~photutils.psf.make_epsf_from_psf`. The resulting image
can then be used as the input to `~photutils.psf.ImagePSF` (see
:ref:`psf-image-models` below) with the same oversampling factor to
create an image-based PSF model::

    >>> import numpy as np
    >>> from photutils.psf import ImagePSF, MoffatPSF, make_epsf_from_psf
    >>> oversampling = 4
    >>> yy, xx = np.mgrid[-50:51, -50:51] / oversampling
    >>> psf = MoffatPSF(alpha=1.2, beta=2.5)(xx, yy)
    >>> epsf = make_epsf_from_psf(psf, oversampling=oversampling)
    >>> model = ImagePSF(epsf, oversampling=oversampling)

The values of the oversampled PSF model sum to the square of the
oversampling factor, apart from the flux outside of the grid, which is
the normalization that `~photutils.psf.ImagePSF` requires. Discretizing
the model on the detector pixel grid with
:func:`~astropy.convolution.discretize_model` also gives a
pixel-integrated image, but only for one subpixel position of the
source. A model made from that image is not accurate at other
positions unless the PSF is well sampled.

Note that the non-circular Gaussian and Moffat models above have
additional parameters beyond the standard PSF model parameters of
position and flux (``x_0``, ``y_0``, and ``flux``), which are fixed by
default as described above.

Photutils also provides a convenience function called
:func:`~photutils.psf.make_psf_model` that creates a PSF model from an
Astropy fittable 2D model. However, it is recommended that one use the
PSF models provided by `photutils.psf` as they are optimized for PSF
photometry. If a custom PSF model is needed, one can be created using
the Astropy modeling framework that will provide better performance than
using :func:`~photutils.psf.make_psf_model`. A model made with
:func:`~photutils.psf.make_psf_model` is evaluated at the input
positions and is not integrated over the detector pixels.


.. _psf-image-models:

Image-based PSF Models
^^^^^^^^^^^^^^^^^^^^^^

Image-based PSF models are typically derived from observed data or from
detailed optical modeling. The PSF model can be a single PSF model for
the entire image or a grid of PSF models at fiducial detector positions,
which are then interpolated for specific locations.

The model classes below provide the tools needed to perform PSF
photometry within Photutils using the Astropy modeling and fitting
framework. The user must provide the image-based PSF model as an input
to these classes. The input image(s) can be oversampled to increase the
accuracy of the PSF model.

- `~photutils.psf.ImagePSF`: a general class for image-based PSF models
  that allows for intensity scaling and translations.

- `~photutils.psf.GriddedPSFModel`: a PSF model that contains a grid of
  image-based ePSF models at fiducial detector positions.

These models interpolate the input image(s) and do not integrate them
over the detector pixels. The input must therefore be an effective
PSF (ePSF), in which each value is the fraction of the source flux
that falls in a whole detector pixel centered at that position
relative to the source. An oversampled PSF whose values are samples
of the PSF, such as the output of an optical model, is not an ePSF.
A model made from such an image is sharper than the sources in the
data. If the PSF is also undersampled by the detector pixels, the sum
of the model over the detector pixels changes with the subpixel
position of the source. Use `~photutils.psf.make_epsf_from_psf` to
make an ePSF from a sampled PSF::

    >>> from photutils.psf import ImagePSF, make_epsf_from_psf
    >>> epsf = make_epsf_from_psf(psf, oversampling=4)  # doctest: +SKIP
    >>> model = ImagePSF(epsf, oversampling=4)  # doctest: +SKIP

An optical model with an even oversampling factor typically returns
an image with an even number of points along each axis, with the PSF
centered between the four central grid points. The image-based models
accept such an image, and the ePSF made from it, as they are. If an
ePSF with the PSF center on a grid point is needed, for example to
compare it with an ePSF made by `~photutils.psf.EPSFBuilder`, use the
``midpoints=True`` option of `~photutils.psf.make_epsf_from_psf`. It
makes the ePSF at the points midway between the input grid points, so
the output image has one fewer point along each axis.

The gridded ePSF models that `STPSF <https://stpsf.readthedocs.io/>`_
makes with its ``psf_grid`` method (see
`~photutils.psf.webbpsf_reader`) are already integrated over the
detector pixels, so `~photutils.psf.make_epsf_from_psf` must not be
applied to them. STPSF integrates the sampled PSF with a discrete box
kernel, however, which is a low-order approximation of the integral.
In tests with STPSF 2.2.0 and an oversampling factor of 4, the ePSF
peak was too low by 2.5%, 1.1%, and 0.9% for the JWST NIRCam F115W,
F200W, and F444W filters, and the fluxes fitted with these models
to stars made from an accurate ePSF were too high by 1.3%, 0.6%,
and 0.5%. With an oversampling factor of 5 the errors were about a
third as large and had the opposite sign. When that accuracy matters,
make the ePSF from the oversampled PSF that the STPSF ``calc_psf``
method returns, which is sampled at the grid points::

    import stpsf
    from photutils.psf import ImagePSF, make_epsf_from_psf

    nrc = stpsf.NIRCam()
    nrc.filter = 'F115W'
    oversampling = 4
    hdulist = nrc.calc_psf(fov_pixels=101, oversample=oversampling)
    psf = hdulist['OVERDIST'].data * oversampling**2
    epsf = make_epsf_from_psf(psf, oversampling=oversampling)
    model = ImagePSF(epsf, oversampling=oversampling)

The values that ``calc_psf`` returns sum to the fraction of the flux
that is inside the field of view, so multiplying them by the square
of the oversampling factor gives the normalization that
`~photutils.psf.ImagePSF` requires.

An image-based model is zero outside of its input image, so the flux
of the PSF wings beyond the image is not in the model. The sum of
the model over the detector pixels is then smaller than the model
flux, and it changes with the subpixel position of the source because
the number of detector pixels inside the image changes. For a
simulated JWST NIRCam F115W ePSF, the sum changes by up to 2.5% for
an image that covers 6x6 detector pixels and by up to 0.6% for one
that covers 16x16 pixels. The ePSF image should therefore be large
enough that the ePSF is small at its edges. The fluxes and positions
fitted by the PSF photometry classes are not affected if the fitted
region of each source is well inside the image, but the model and
residual images do not include the flux outside of it.


.. _psf-photometry-examples:

PSF Photometry Examples
-----------------------

Let's start with a simple example using simulated stars whose PSF is
assumed to be Gaussian. We'll create a synthetic image using tools
provided by the `photutils.psf` and :ref:`photutils.datasets <datasets>`
modules::

    >>> import numpy as np
    >>> from photutils.datasets import make_noise_image
    >>> from photutils.psf import CircularGaussianPRF, make_psf_model_image
    >>> psf_model = CircularGaussianPRF(flux=1, fwhm=2.7)
    >>> psf_shape = (9, 9)
    >>> n_sources = 10
    >>> shape = (101, 101)
    >>> data, true_params = make_psf_model_image(shape, psf_model, n_sources,
    ...                                          model_shape=psf_shape,
    ...                                          flux=(500, 700),
    ...                                          min_separation=10, seed=0)
    >>> noise = make_noise_image(data.shape, mean=0, stddev=1, seed=0)
    >>> data += noise
    >>> error = np.full(data.shape, 1.0)

Let's plot the image:

.. plot::

    import matplotlib.pyplot as plt
    from photutils.datasets import make_noise_image
    from photutils.psf import CircularGaussianPRF, make_psf_model_image

    psf_model = CircularGaussianPRF(flux=1, fwhm=2.7)
    psf_shape = (9, 9)
    n_sources = 10
    shape = (101, 101)
    data, true_params = make_psf_model_image(shape, psf_model, n_sources,
                                             model_shape=psf_shape,
                                             flux=(500, 700),
                                             min_separation=10, seed=0)
    noise = make_noise_image(data.shape, mean=0, stddev=1, seed=0)
    data += noise

    fig, ax = plt.subplots()
    axim = ax.imshow(data, origin='lower')
    ax.set_title('Simulated Data')
    fig.colorbar(axim)


Fitting multiple sources
^^^^^^^^^^^^^^^^^^^^^^^^

Now let's use `~photutils.psf.PSFPhotometry` to perform PSF photometry
on the sources in this image. Note that the input image must be
background-subtracted prior to using the photometry classes. See
:ref:`background` for tools to subtract a global background from an
image. This step is not needed for our synthetic image because it does
not include background.

We'll use the `~photutils.detection.DAOStarFinder` class for
source detection. We'll estimate the initial fluxes of each
source using a circular aperture with a radius of 4 pixels. The
central 5x5 pixel region of each source will be fit using a
`~photutils.psf.CircularGaussianPRF` PSF model. First, let's create an
instance of the `~photutils.psf.PSFPhotometry` class::

    >>> from photutils.detection import DAOStarFinder
    >>> from photutils.psf import PSFPhotometry
    >>> psf_model = CircularGaussianPRF(flux=1, fwhm=2.7)
    >>> fit_shape = (5, 5)
    >>> finder = DAOStarFinder(6.0, 2.0)
    >>> psfphot = PSFPhotometry(psf_model, fit_shape, finder=finder,
    ...                         aperture_radius=4)

To perform the PSF fitting, we then call the class instance
on the data array, and optionally an error and mask array. A
`~astropy.nddata.NDData` object holding the data, error, and mask arrays
can also be input into the ``data`` parameter. Note that all non-finite
(e.g., NaN or inf) data values are automatically masked. Here we input
the data and error arrays::

    >>> phot = psfphot(data, error=error)

A table of initial PSF model parameter values can also be input when
calling the class instance. An example of that is shown later.

Equivalently, one can input an `~astropy.nddata.NDData` object with any
uncertainty object that can be converted to standard-deviation errors:

.. doctest-skip::

    >>> from astropy.nddata import NDData, StdDevUncertainty
    >>> uncertainty = StdDevUncertainty(error)
    >>> nddata = NDData(data, uncertainty=uncertainty)
    >>> phot2 = psfphot(nddata)

The result is an Astropy `~astropy.table.QTable` with columns for the
source and group identification numbers, the x, y, and flux initial,
fit, and error values, local background, number of unmasked pixels
fit, the group size, quality-of-fit metrics, and flags. See the
`~photutils.psf.PSFPhotometry` documentation for descriptions of the
output columns.

The full table cannot be shown here as it has many columns, but let's
print the source ID along with the fit x, y, and flux values::

    >>> phot['x_fit'].info.format = '.4f'  # optional format
    >>> phot['y_fit'].info.format = '.4f'
    >>> phot['flux_fit'].info.format = '.4f'
    >>> print(phot[('id', 'x_fit', 'y_fit', 'flux_fit')])
     id  x_fit   y_fit  flux_fit
    --- ------- ------- --------
      1 54.5716  7.7458 513.2209
      2 29.0873 25.6223 534.1826
      3 79.6336 28.7482 619.7632
      4 63.2403 48.6157 560.0202
      5 88.8816 54.1347 616.2282
      6 79.8673 61.1302 650.1411
      7 90.9502 72.0862 597.1119
      8  7.8034 78.5703 638.4794
      9  5.5219 89.8622 542.6817
     10 71.8331 90.5734 692.6518

Let's create the residual image::

    >>> resid = psfphot.make_residual_image(data)

and plot it:

.. plot::

    import matplotlib.pyplot as plt
    import numpy as np
    from astropy.visualization import simple_norm
    from photutils.datasets import make_noise_image
    from photutils.detection import DAOStarFinder
    from photutils.psf import (CircularGaussianPRF, PSFPhotometry,
                               make_psf_model_image)

    psf_model = CircularGaussianPRF(flux=1, fwhm=2.7)
    psf_shape = (9, 9)
    n_sources = 10
    shape = (101, 101)

    data, true_params = make_psf_model_image(shape, psf_model, n_sources,
                                             model_shape=psf_shape,
                                             flux=(500, 700),
                                             min_separation=10, seed=0)
    noise = make_noise_image(data.shape, mean=0, stddev=1, seed=0)
    data += noise
    error = np.full(data.shape, 1.0)

    psf_model = CircularGaussianPRF(flux=1, fwhm=2.7)
    fit_shape = (5, 5)
    finder = DAOStarFinder(6.0, 2.0)
    psfphot = PSFPhotometry(psf_model, fit_shape, finder=finder,
                            aperture_radius=4)
    phot = psfphot(data, error=error)

    resid = psfphot.make_residual_image(data)

    fig, ax = plt.subplots(ncols=3, figsize=(15, 5))
    norm = simple_norm(data, 'sqrt', percent=99)
    ax[0].imshow(data, norm=norm, origin='lower')
    ax[1].imshow(data - resid, norm=norm, origin='lower')
    im = ax[2].imshow(resid, norm=norm, origin='lower')
    ax[0].set_title('Data')
    ax[1].set_title('Model')
    ax[2].set_title('Residual Image')
    fig.tight_layout()

The residual image looks like noise, indicating good fits to the
sources.

Further details about the PSF fitting can be obtained from attributes on
the `~photutils.psf.PSFPhotometry` instance. For example, the results
from the ``finder`` instance called during PSF fitting can be accessed
using the ``finder_results`` attribute (the ``finder`` returns an
Astropy table)::

    >>> psfphot.finder_results['x_centroid'].info.format = '.4f'  # optional format
    >>> psfphot.finder_results['y_centroid'].info.format = '.4f'
    >>> psfphot.finder_results['sharpness'].info.format = '.4f'
    >>> psfphot.finder_results['peak'].info.format = '.4f'
    >>> psfphot.finder_results['flux'].info.format = '.4f'
    >>> psfphot.finder_results['mag'].info.format = '.4f'
    >>> psfphot.finder_results['daofind_mag'].info.format = '.4f'
    >>> print(psfphot.finder_results)
     id x_centroid y_centroid sharpness ...   peak    flux     mag   daofind_mag
    --- ---------- ---------- --------- ... ------- -------- ------- -----------
      1    54.5299     7.7460    0.6006 ... 53.5953 476.3221 -6.6948     -2.1093
      2    29.0927    25.5992    0.5955 ... 57.1982 499.4443 -6.7462     -2.1958
      3    79.6185    28.7515    0.5957 ... 65.7175 574.1382 -6.8975     -2.3401
      4    63.2485    48.6134    0.5802 ... 58.3985 521.4656 -6.7931     -2.2209
      5    88.8820    54.1311    0.5948 ... 69.1869 576.2842 -6.9016     -2.4379
      6    79.8727    61.1208    0.6216 ... 74.0935 612.8353 -6.9684     -2.4799
      7    90.9621    72.0803    0.6167 ... 68.4157 561.7163 -6.8738     -2.4035
      8     7.7962    78.5465    0.5979 ... 66.2173 595.6881 -6.9375     -2.3167
      9     5.5858    89.8664    0.5741 ... 54.3786 505.6093 -6.7595     -2.1188
     10    71.8303    90.5624    0.6038 ... 73.5747 639.9299 -7.0153     -2.4516


Fitting a single source
^^^^^^^^^^^^^^^^^^^^^^^

In some cases, one may want to fit only a single source (or a few
sources) in an image. We can do that by defining a table of the sources
that we want to fit. For this example, let's fit the single source at
``(x, y) = (63, 49)``. We first define a table with this position and
then pass that table into the ``init_params`` keyword when calling the
PSF photometry class on the data::

    >>> from astropy.table import QTable
    >>> init_params = QTable()
    >>> init_params['x'] = [63]
    >>> init_params['y'] = [49]
    >>> phot = psfphot(data, error=error, init_params=init_params)

The PSF photometry class allows for flexible input column names
using a heuristic to identify the x, y, and flux columns. See
`~photutils.psf.PSFPhotometry` for more details.

The output table contains only the fit results for the input source::

    >>> phot['x_fit'].info.format = '.4f'  # optional format
    >>> phot['y_fit'].info.format = '.4f'
    >>> phot['flux_fit'].info.format = '.4f'
    >>> print(phot[('id', 'x_fit', 'y_fit', 'flux_fit')])
     id  x_fit   y_fit  flux_fit
    --- ------- ------- --------
      1 63.2403 48.6157 560.0201

Finally, let's show the residual image. The red circular aperture shows
the location of the source that was fit and subtracted.

.. plot::

    import matplotlib.pyplot as plt
    import numpy as np
    from astropy.table import QTable
    from astropy.visualization import simple_norm
    from photutils.aperture import CircularAperture
    from photutils.datasets import make_noise_image
    from photutils.detection import DAOStarFinder
    from photutils.psf import (CircularGaussianPRF, PSFPhotometry,
                               make_psf_model_image)

    psf_model = CircularGaussianPRF(flux=1, fwhm=2.7)
    psf_shape = (9, 9)
    n_sources = 10
    shape = (101, 101)

    data, true_params = make_psf_model_image(shape, psf_model, n_sources,
                                             model_shape=psf_shape,
                                             flux=(500, 700),
                                             min_separation=10, seed=0)
    noise = make_noise_image(data.shape, mean=0, stddev=1, seed=0)
    data += noise
    error = np.full(data.shape, 1.0)

    psf_model = CircularGaussianPRF(flux=1, fwhm=2.7)
    fit_shape = (5, 5)
    finder = DAOStarFinder(6.0, 2.0)
    psfphot = PSFPhotometry(psf_model, fit_shape, finder=finder,
                            aperture_radius=4)

    init_params = QTable()
    init_params['x'] = [63]
    init_params['y'] = [49]
    phot = psfphot(data, error=error, init_params=init_params)

    resid = psfphot.make_residual_image(data)
    xypos = zip(phot['x_fit'], phot['y_fit'], strict=True)
    aper = CircularAperture(xypos, r=4)

    fig, ax = plt.subplots(ncols=3, figsize=(15, 5))
    norm = simple_norm(data, 'sqrt', percent=99)
    ax[0].imshow(data, norm=norm, origin='lower')
    ax[1].imshow(data - resid, norm=norm, origin='lower')
    im = ax[2].imshow(resid, norm=norm, origin='lower')
    ax[0].set_title('Data')
    aper.plot(ax=ax[0], color='red')
    ax[1].set_title('Model')
    aper.plot(ax=ax[1], color='red')
    ax[2].set_title('Residual Image')
    aper.plot(ax=ax[2], color='red')
    fig.tight_layout()


.. _psf-forced-photometry:

Forced Photometry (Fixed Model Parameters)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In general, the three parameters fit for each source are the x and
y positions and the flux. However, the Astropy modeling and fitting
framework allows any of these parameters to be fixed during the fitting.

Let's say you want to fix the (x, y) position for each source. You can
do that by setting the ``fixed`` attribute on the model parameters::

    >>> psf_model2 = CircularGaussianPRF(flux=1, fwhm=2.7)
    >>> psf_model2.x_0.fixed = True
    >>> psf_model2.y_0.fixed = True
    >>> psf_model2.fixed
    {'flux': False, 'x_0': True, 'y_0': True, 'fwhm': True}

Now when the model is fit, the flux will be varied, but the (x, y)
position will be fixed at its initial position for every source. Let's
just fit a single source (defined in ``init_params``)::

    >>> psfphot = PSFPhotometry(psf_model2, fit_shape, finder=finder,
    ...                         aperture_radius=4)
    >>> phot = psfphot(data, error=error, init_params=init_params)

The output table shows that the (x, y) position was unchanged, with the
fit values being identical to the initial values. However, the flux was
fit::

    >>> phot['flux_init'].info.format = '.4f'  # optional format
    >>> phot['flux_fit'].info.format = '.4f'
    >>> print(phot[('id', 'x_init', 'y_init', 'flux_init', 'x_fit',
    ...             'y_fit', 'flux_fit')])
     id x_init y_init flux_init x_fit y_fit flux_fit
    --- ------ ------ --------- ----- ----- --------
      1     63     49  556.5067  63.0  49.0 539.4673


.. _psf-bounded-parameters:

Bounded Model Parameters
^^^^^^^^^^^^^^^^^^^^^^^^

The Astropy modeling and fitting framework also allows for bounding the
parameter values during the fitting process. However, not all Astropy
"Fitter" classes support parameter bounds. Please see `Fitting Models to
Data <https://docs.astropy.org/en/stable/modeling/fitting.html>`_ for
more details.

The model parameter bounds apply to all sources in the image,
thus this mechanism cannot be used to bound the x and y positions
of individual sources. However, the x and y positions can be
bounded for individual sources during the fitting by using the
``xy_bounds`` keyword in `~photutils.psf.PSFPhotometry` and
`~photutils.psf.IterativePSFPhotometry`. This keyword accepts a tuple of
floats representing the maximum distance in pixels that a fitted source
can be from its initial (x, y) position.

For example, you may want to constrain the flux of a source to be
between certain values or ensure that it is a non-negative value. This
can be done by setting the ``bounds`` attribute on the input PSF model
parameters. Here we constrain the flux to be greater than or equal to
0::

    >>> psf_model3 = CircularGaussianPRF(flux=1, fwhm=2.7)
    >>> psf_model3.flux.bounds = (0, None)
    >>> psf_model3.bounds
    {'flux': (0.0, None), 'x_0': (None, None), 'y_0': (None, None), 'fwhm': (1.1754943508222875e-38, None)}

The model parameter ``bounds`` can also be set using the ``min`` and/or
``max`` attributes. Here we set the minimum flux to be 0::

    >>> psf_model3.flux.min = 0
    >>> psf_model3.bounds
    {'flux': (0.0, None), 'x_0': (None, None), 'y_0': (None, None), 'fwhm': (1.1754943508222875e-38, None)}

For this example, let's constrain the flux value to be between
400 and 600::

    >>> psf_model3 = CircularGaussianPRF(flux=1, fwhm=2.7)
    >>> psf_model3.flux.bounds = (400, 600)
    >>> psf_model3.bounds
    {'flux': (400.0, 600.0), 'x_0': (None, None), 'y_0': (None, None), 'fwhm': (1.1754943508222875e-38, None)}


Source Grouping
^^^^^^^^^^^^^^^

Source grouping is an optional feature. To turn it on, create a
`~photutils.psf.SourceGrouper` instance and input it via the ``grouper``
keyword. Here we'll group sources that are within 20 pixels of each
other::

    >>> from photutils.psf import SourceGrouper
    >>> grouper = SourceGrouper(min_separation=20)
    >>> psfphot = PSFPhotometry(psf_model, fit_shape, finder=finder,
    ...                         grouper=grouper, aperture_radius=4)
    >>> phot = psfphot(data, error=error)

The ``group_id`` column shows that seven groups were identified. The
sources in each group were simultaneously fit::

    >>> print(phot[('id', 'group_id', 'group_size')])
     id group_id group_size
    --- -------- ----------
      1        1          1
      2        2          1
      3        3          1
      4        4          1
      5        5          3
      6        5          3
      7        5          3
      8        6          2
      9        6          2
     10        7          1

Care should be taken in defining the source groups. Simultaneously
fitting very large source groups is computationally expensive and
error-prone, because the number of fitted parameters grows with the
group size. A warning will be raised if the number of sources in a group
exceeds a threshold defined by the ``group_warning_threshold`` keyword.


Local Background Subtraction
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To subtract a local background from each source, define a
`~photutils.background.LocalBackground` instance and input it via
the ``local_bkg_estimator`` keyword. Here we'll use an annulus with
an inner and outer radius of 5 and 10 pixels, respectively, with the
`~photutils.background.MMMBackground` statistic (with its default sigma
clipping)::

    >>> from photutils.background import LocalBackground, MMMBackground
    >>> bkgstat = MMMBackground()
    >>> local_bkg_estimator = LocalBackground(5, 10, bkg_estimator=bkgstat)
    >>> finder = DAOStarFinder(10.0, 2.0)
    >>> psfphot = PSFPhotometry(psf_model, fit_shape, finder=finder,
    ...                         grouper=grouper, aperture_radius=4,
    ...                         local_bkg_estimator=local_bkg_estimator)
    >>> phot = psfphot(data, error=error)

The local background values are output in the table::

    >>> phot['local_bkg'].info.format = '.4f'  # optional format
    >>> print(phot[('id', 'local_bkg')])
     id local_bkg
    --- ---------
      1   -0.0839
      2    0.1784
      3    0.2593
      4   -0.0574
      5    0.2492
      6   -0.0818
      7   -0.1130
      8   -0.2166
      9    0.0102
     10    0.3926

The local background values can also be input directly using the
``init_params`` keyword.


Iterative PSF Photometry
^^^^^^^^^^^^^^^^^^^^^^^^

Now let's use the `~photutils.psf.IterativePSFPhotometry` class to
iteratively fit the sources in the image. This class is useful for
crowded fields where faint sources are very close to bright sources. The
faint sources may not be detected until after the bright sources are
subtracted.

For this simple example, let's input a table of three sources for the
first fit iteration. Subsequent iterations will use the ``finder`` to
find additional sources::

    >>> from photutils.background import LocalBackground, MMMBackground
    >>> from photutils.psf import IterativePSFPhotometry
    >>> fit_shape = (5, 5)
    >>> finder = DAOStarFinder(10.0, 2.0)
    >>> bkgstat = MMMBackground()
    >>> local_bkg_estimator = LocalBackground(5, 10, bkg_estimator=bkgstat)
    >>> init_params = QTable()
    >>> init_params['x'] = [54, 29, 80]
    >>> init_params['y'] = [8, 26, 29]
    >>> psfphot2 = IterativePSFPhotometry(psf_model, fit_shape, finder=finder,
    ...                                   local_bkg_estimator=local_bkg_estimator,
    ...                                   aperture_radius=4)
    >>> phot = psfphot2(data, error=error, init_params=init_params)

The table output from `~photutils.psf.IterativePSFPhotometry` contains a
column called ``iter_detected`` that returns the fit iteration in which
the source was detected::

    >>> phot['x_fit'].info.format = '.4f'  # optional format
    >>> phot['y_fit'].info.format = '.4f'
    >>> phot['flux_fit'].info.format = '.4f'
    >>> print(phot[('id', 'iter_detected', 'x_fit', 'y_fit', 'flux_fit')])
     id iter_detected  x_fit   y_fit  flux_fit
    --- ------------- ------- ------- --------
      1             1 54.5694  7.7454 513.4505
      2             1 29.0875 25.6216 531.3096
      3             1 79.6327 28.7475 615.1111
      4             2 63.2401 48.6159 560.9633
      5             2 88.8813 54.1350 612.0937
      6             2 79.8674 61.1301 651.4973
      7             2 90.9502 72.0861 598.9888
      8             2  7.8037 78.5711 642.0335
      9             2  5.5218 89.8621 542.5141
     10             2 71.8326 90.5721 686.2071


Estimating the FWHM of Sources
------------------------------

The `photutils.psf` package also provides a convenience
function called `~photutils.psf.fit_fwhm` to estimate the
full width at half maximum (FWHM) of one or more sources in
an image. This function fits the source(s) with a circular
2D Gaussian PRF model (`~photutils.psf.CircularGaussianPRF`)
using the `~photutils.psf.PSFPhotometry` class. If your sources
are non-circular or non-Gaussian, you can fit them with the
`~photutils.psf.PSFPhotometry` class and a different PSF model. Because
the model is integrated over the pixels, the returned FWHM is the FWHM
of the Gaussian before that integration, which is smaller than the FWHM
of the pixelated image of a source.

For example, let's estimate the FWHM of the sources in our example image
defined above::

    >>> from photutils.psf import fit_fwhm
    >>> finder = DAOStarFinder(6.0, 2.0)
    >>> finder_tbl = finder(data)
    >>> xypos = list(zip(finder_tbl['x_centroid'],
    ...                  finder_tbl['y_centroid'], strict=True))
    >>> fwhm = fit_fwhm(data, xypos=xypos, error=error, fit_shape=(5, 5),
    ...                 fwhm=2)
    >>> print(fwhm)
    [2.70584007 2.71009548 2.67319293 2.6932673  2.6674289  2.69499608
     2.68722503 2.73280482 2.7200538  2.68340968]


Convenience Gaussian Fitting Function
-------------------------------------

The `photutils.psf` package also provides a convenience function called
:func:`~photutils.psf.fit_2dgaussian` for fitting one or more sources
with a 2D Gaussian PRF model (`~photutils.psf.CircularGaussianPRF`)
using the `~photutils.psf.PSFPhotometry` class. See the function
documentation for more details and examples.


API Reference
-------------

:doc:`../reference/psf_api`
