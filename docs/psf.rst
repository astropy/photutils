.. _psf-photometry:

PSF Photometry (`photutils.psf`)
================================

The `photutils.psf` subpackage contains tools for model-fitting
photometry, often called "PSF photometry".

.. _psf-terminology:


Terminology
-----------
Different astronomy subfields use the terms "PSF", "PRF", or related
terms somewhat differently, especially when colloquial usage is taken
into account. This package aims to be at the very least internally
consistent, following the definitions described here.

We take the Point Spread Function (PSF), or instrumental Point
Spread Function (iPSF) to be the infinite-resolution and
infinite-signal-to-noise flux distribution from a point source on
the detector, after passing through optics, dust, atmosphere, etc.
By contrast, the function describing the responsivity variations
across individual *pixels* is the Pixel Response Function (sometimes
called "PRF", but that acronym is not used here for reasons that will
soon be apparent). The convolution of the PSF and pixel response
function, when discretized onto the detector (i.e., a rectilinear
CCD grid), is the effective PSF (ePSF) or Point Response Function
(PRF) (this latter terminology is the definition used by `Spitzer
<https://irsa.ipac.caltech.edu/data/SPITZER/docs/dataanalysistools/tools
/mopex/mopexusersguide/89/>`_). In many cases the PSF/ePSF/PRF
distinction is unimportant, and the ePSF/PRF are simply called
the "PSF", but the distinction can be critical when dealing
carefully with undersampled data or detectors with significant
intra-pixel sensitivity variations. For a more detailed
description of this formalism, see `Anderson & King 2000
<https://ui.adsabs.harvard.edu/abs/2000PASP..112.1360A/abstract>`_.

All this said, in colloquial usage "PSF photometry" sometimes refers
to the more general task of model-fitting photometry (with the effects
of the PSF either implicitly or explicitly included in the models),
regardless of exactly what kind of model is actually being fit. For
brevity (e.g., ``photutils.psf``), we use "PSF photometry" in this way,
as a shorthand for the general approach.


PSF Photometry
--------------

Photutils provides a modular set of tools to perform PSF photometry for
different science cases. These are implemented as separate classes to do
sub-tasks of PSF photometry. It also provides high-level classes that
connect these pieces together.

The two main PSF-photometry classes are `~photutils.psf.PSFPhotometry`
and `~photutils.psf.IterativePSFPhotometry`.
`~photutils.psf.PSFPhotometry` is a flexible PSF photometry algorithm
that can find sources in an image, optionally group overlapping
sources, fit the PSF model to the sources, and subtract the fit PSF
models from the image. `~photutils.psf.IterativePSFPhotometry` is an
iterative version of `~photutils.psf.PSFPhotometry` where after the
fit sources are subtracted, the process repeats until no additional
sources are detected or a maximum number of iterations has been
performed. When used with the `~photutils.detection.DAOStarfinder`,
`~photutils.psf.IterativePSFPhotometry` is essentially an implementation
of the DAOPHOT algorithm (`~photutils.psf.DAOPhotPSFPhotometry`)
described by Stetson in his `seminal paper
<https://ui.adsabs.harvard.edu/abs/1987PASP...99..191S/abstract>`_ for
crowded-field stellar photometry.

The star-finding step is controlled by the ``finder``
keyword, where one inputs a callable function or class
instance. Typically this would be one of the star-detection
classes implemented in the `~photutils.detection`
subpackage, e.g., `~photutils.detection.DAOStarFinder`,
`~photutils.detection.IRAFStarFinder`, or
`~photutils.detection.StarFinder`.

After finding sources, one can optionally apply a clustering algorithm
in order to label the sources according to groups using the ``grouper``
keyword. Usually, those groups are formed by a distance criterion,
which is the case of the grouping algorithm proposed by Stetson. Stars
that grouped are fit simultaneously. The reason behind the construction
of groups and not fitting all stars simultaneously is illustrated as
follows: imagine that one would like to fit 300 stars and the model
for each star has three parameters to be fitted. If one constructs a
single model to fit the 300 stars simultaneously, then the optimization
algorithm will have to search for the solution in a 900-dimensional
space, which is computationally expensive and error-prone. Reducing
the stars in groups effectively reduces the dimension of the parameter
space, which facilitates the optimization process. For more details see
:ref:`psf-grouping`.

The local background around each source can optionally be subtracted
using the ``localbkg_estimator`` keyword. This keyword accepts a
`~photutils.backgrounds.LocalBackground` instance that estimates the
local statistics in a circular annulus aperture centered on each
source. The size of the annulus and the statistic can be configured in
`~photutils.backgrounds.LocalBackground`.

The next step is to fit the sources and/or groups. This
task is performed using an astropy fitter, for example
`~astropy.modeling.fitting.LevMarLSQFitter`, input via the ``fitter``
keyword.

After sources are fitted, a model image of the source or
a residual image can be generated using, for example, the
:meth:`~photutils.psf.PSFPhotometry.make_model_image` and
:meth:`~photutils.psf.PSFPhotometry.make_residual_image` methods.

For `~photutils.psf.IterativePSFPhotometry`, the above steps can be
repeated until no additional sources are detected (or until a maximum
number of iterations).

The `~photutils.psf.PSFPhotometry` and
`~photutils.psf.IterativePSFPhotometry` classes provide the structure
in which the PSF-fitting steps described above are performed, but
all the stages can be turned on or off or replaced with different
implementations as the user desires. This makes the tools very flexible.
One can also bypass several of the steps by directly inputing an astropy
table of the initial parameters for the source centers, fluxes, group
identifiers, and local backgrounds. This is also useful is one is
interested in fitting only one or a few sources in an image.

Basic Usage
^^^^^^^^^^^

Let's start with a simple example with simulated stars whose PSF is
assumed to be Gaussian. We'll create a synthetic image provided by the
:ref:`photutils.datasets <datasets>` module::

    >>> from photutils.datasets import make_psf_test_data
    >>> data, sources = make_psf_test_data()


References
----------

`Spitzer PSF vs. PRF
<https://irsa.ipac.caltech.edu/data/SPITZER/docs/files/spitzer/PRF_vs_PSF.pdf>`_

`The Kepler Pixel Response Function
<https://ui.adsabs.harvard.edu/abs/2010ApJ...713L..97B/abstract>`_

`Stetson, Astronomical Society of the Pacific, Publications, (ISSN
0004-6280), vol. 99, March 1987, p. 191-222.
<https://ui.adsabs.harvard.edu/abs/1987PASP...99..191S/abstract>`_

`Anderson & King, Astronomical Society of the Pacific, Publications,
Volume 112, Issue 776, pp. 1360-1382, Nov 2000
<https://ui.adsabs.harvard.edu/abs/2000PASP..112.1360A/abstract>`_


Reference/API
-------------

.. automodapi:: photutils.psf
    :no-heading:
