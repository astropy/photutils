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

Photutils provides a modular set of tools to perform PSF photometry
for different science cases. The tools are implemented as classes that
perform various subtasks of PSF photometry. High-level classes are also
provided to connect these pieces together.

The two main PSF-photometry classes are `~photutils.psf.PSFPhotometry`
and `~photutils.psf.IterativePSFPhotometry`.
`~photutils.psf.PSFPhotometry` provides the framework for a flexible PSF
photometry workflow that can find sources in an image, optionally group
overlapping sources, fit the PSF model to the sources, and subtract the
fit PSF models from the image. `~photutils.psf.IterativePSFPhotometry`
is an iterative version of `~photutils.psf.PSFPhotometry` where
after the fit sources are subtracted, the process repeats until no
additional sources are detected or a maximum number of iterations has
been reached. When used with the `~photutils.detection.DAOStarFinder`,
`~photutils.psf.IterativePSFPhotometry` is essentially an implementation
of the DAOPHOT algorithm described by Stetson in his `seminal paper
<https://ui.adsabs.harvard.edu/abs/1987PASP...99..191S/abstract>`_ for
crowded-field stellar photometry.

The star-finding step is controlled by the ``finder``
keyword, where one inputs a callable function or class
instance. Typically, this would be one of the star-detection
classes implemented in the `photutils.detection`
subpackage, e.g., `~photutils.detection.DAOStarFinder`,
`~photutils.detection.IRAFStarFinder`, or
`~photutils.detection.StarFinder`.

After finding sources, one can optionally apply a clustering algorithm
to group overlapping sources using the ``grouper`` keyword. Usually,
groups are formed by a distance criterion, which is the case of the
grouping algorithm proposed by Stetson. Stars that grouped are fit
simultaneously. The reason behind the construction of groups and not
fitting all stars simultaneously is illustrated as follows: imagine
that one would like to fit 300 stars and the model for each star has
three parameters to be fitted. If one constructs a single model to fit
the 300 stars simultaneously, then the optimization algorithm will
have to search for the solution in a 900-dimensional space, which
is computationally expensive and error-prone. Having smaller groups
of stars effectively reduces the dimension of the parameter space,
which facilitates the optimization process. For more details see
:ref:`psf-grouping`.

The local background around each source can optionally be subtracted
using the ``localbkg_estimator`` keyword. This keyword accepts a
`~photutils.background.LocalBackground` instance that estimates the
local statistics in a circular annulus aperture centered on each source.
The size of the annulus and the statistic function can be configured in
`~photutils.background.LocalBackground`.

The next step is to fit the sources and/or groups. This
task is performed using an astropy fitter, for example
`~astropy.modeling.fitting.LevMarLSQFitter`, input via the ``fitter``
keyword. The shape of the region to be fitted can be configured using
the ``fit_shape`` parameter. In general, ``fit_shape`` should be set
to a small size (e.g., (5, 5)) that covers the central star region
with the highest flux signal-to-noise. The initial positions are
derived from the ``finder`` algorithm. The initial flux values for the
fit are derived from measuring the flux in a circular aperture with
radius ``aperture_radius``. The initial positions and fluxes can be
alternatively input via the ``init_params`` keyword when calling the
class.

After sources are fitted, a model image of the fit
sources or a residual image can be generated using the
:meth:`~photutils.psf.PSFPhotometry.make_model_image` and
:meth:`~photutils.psf.PSFPhotometry.make_residual_image` methods,
respectively.

For `~photutils.psf.IterativePSFPhotometry`, the above steps can be
repeated until no additional sources are detected (or until a maximum
number of iterations).

The `~photutils.psf.PSFPhotometry` and
`~photutils.psf.IterativePSFPhotometry` classes provide the structure
in which the PSF-fitting steps described above are performed, but
all the stages can be turned on or off or replaced with different
implementations as the user desires. This makes the tools very flexible.
One can also bypass several of the steps by directly inputting to
``init_params`` an astropy table containing the initial parameters for
the source centers, fluxes, group identifiers, and local backgrounds.
This is also useful if one is interested in fitting only one or a few
sources in an image.

Example Usage
-------------

Let's start with a simple example using simulated stars whose PSF is
assumed to be Gaussian. We'll create a synthetic image using tools
provided by the :ref:`photutils.datasets <datasets>` module:

.. doctest-requires:: scipy

    >>> import numpy as np
    >>> from photutils.datasets import make_test_psf_data, make_noise_image
    >>> from photutils.psf import IntegratedGaussianPRF
    >>> psf_model = IntegratedGaussianPRF(flux=1, sigma=2.7 / 2.35)
    >>> psf_shape = (25, 25)
    >>> nsources = 10
    >>> shape = (101, 101)
    >>> data, true_params = make_test_psf_data(shape, psf_model, psf_shape,
    ...                                        nsources, flux_range=(500, 700),
    ...                                        min_separation=10, seed=0)
    >>> noise = make_noise_image(data.shape, mean=0, stddev=1, seed=0)
    >>> data += noise
    >>> error = np.abs(noise)

Let's plot the image:

.. plot::

    import matplotlib.pyplot as plt
    from photutils.datasets import make_noise_image, make_test_psf_data
    from photutils.psf import IntegratedGaussianPRF

    psf_model = IntegratedGaussianPRF(flux=1, sigma=2.7 / 2.35)
    psf_shape = (25, 25)
    nsources = 10
    shape = (101, 101)
    data, true_params = make_test_psf_data(shape, psf_model, psf_shape,
                                           nsources, flux_range=(500, 700),
                                           min_separation=10, seed=0)
    noise = make_noise_image(data.shape, mean=0, stddev=1, seed=0)
    data += noise
    plt.imshow(data, origin='lower')
    plt.title('Simulated Data')
    plt.colorbar()


Fitting multiple stars
^^^^^^^^^^^^^^^^^^^^^^

Now let's use `~photutils.psf.PSFPhotometry` to perform PSF photometry
on the stars in this image. Note that the input image must be
background-subtracted prior to using the photometry classes. See
:ref:`background` for tools to subtract a global background from an
image. This is not needed for our synthetic image because it does not
include background.

We'll use the `~photutils.detection.DAOStarFinder` class for
source detection. We'll estimate the initial fluxes of each
source using a circular aperture with a radius 4 pixels. The
central 5x5 pixel region of each star will be fit using an
`~photutils.psf.IntegratedGaussianPRF` PSF model. First, let's create an
instance of the `~photutils.psf.PSFPhotometry` class:

.. doctest-requires:: scipy

    >>> from photutils.detection import DAOStarFinder
    >>> from photutils.psf import PSFPhotometry
    >>> psf_model = IntegratedGaussianPRF(flux=1, sigma=2.7 / 2.35)
    >>> fit_shape = (5, 5)
    >>> finder = DAOStarFinder(6.0, 2.0)
    >>> psfphot = PSFPhotometry(psf_model, fit_shape, finder=finder,
    ...                         aperture_radius=4)

To perform the PSF fitting, we then call the class instance
on the data array, and optionally an error and mask array. A
`~astropy.nddata.NDData` object holding the data, error, and mask arrays
can also be input into the ``data`` parameter. Note that all non-finite
(e.g., NaN or inf) data values are automatically masked. Here we input
the data and error arrays:

.. doctest-requires:: scipy

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

The result is an astropy `~astropy.table.Table` with columns for the
source and group identification numbers, the x, y, and flux initial,
fit, and error values, local background, number of unmasked pixels
fit, the group size, quality-of-fit metrics, and flags. See the
`~photutils.psf.PSFPhotometry` documentation for descriptions of the
output columns.

The full table cannot be shown here as it has many columns, but let's
print the source ID along with the fit x, y, and flux values:

.. doctest-requires:: scipy

    >>> phot['x_fit'].info.format = '.4f'  # optional format
    >>> phot['y_fit'].info.format = '.4f'
    >>> phot['flux_fit'].info.format = '.4f'
    >>> print(phot[('id', 'x_fit', 'y_fit', 'flux_fit')])  # doctest: +FLOAT_CMP
     id  x_fit   y_fit  flux_fit
    --- ------- ------- --------
      1 32.7715 12.2210 627.4274
      2 13.2700 14.5841 507.5778
      3 63.6483 22.3907 640.9277
      4 82.2847 25.5223 662.0007
      5 41.5416 35.8893 687.8236
      6 21.5721 41.9480 620.8562
      7 14.1823 65.0090 681.7447
      8 61.8363 67.5560 608.2459
      9 74.6206 68.1855 502.8906
     10 15.1685 78.0373 558.0204

Let's create the residual image:

.. doctest-requires:: scipy

    >>> resid = psfphot.make_residual_image(data, (25, 25))

and plot it:

.. plot::

    import matplotlib.pyplot as plt
    import numpy as np
    from photutils.datasets import make_noise_image, make_test_psf_data
    from photutils.detection import DAOStarFinder
    from photutils.psf import IntegratedGaussianPRF, PSFPhotometry

    psf_model = IntegratedGaussianPRF(flux=1, sigma=2.7 / 2.35)
    psf_shape = (25, 25)
    nsources = 10
    shape = (101, 101)
    data, true_params = make_test_psf_data(shape, psf_model, psf_shape,
                                           nsources, flux_range=(500, 700),
                                           min_separation=10, seed=0)
    noise = make_noise_image(data.shape, mean=0, stddev=1, seed=0)
    data += noise
    error = np.abs(noise)

    psf_model = IntegratedGaussianPRF(flux=1, sigma=2.7 / 2.35)
    fit_shape = (5, 5)
    finder = DAOStarFinder(6.0, 2.0)
    psfphot = PSFPhotometry(psf_model, fit_shape, finder=finder,
                            aperture_radius=4)
    phot = psfphot(data, error=error)

    resid = psfphot.make_residual_image(data, (25, 25))
    plt.imshow(resid, origin='lower')
    plt.title('Residual Image')
    plt.colorbar()

The residual image looks like noise, indicating good fits to the
sources.

Further details about the PSF fitting can be obtained from attributes on
the `~photutils.psf.PSFPhotometry` instance. For example, the results
from the ``finder`` instance called during PSF fitting can be accessed
using the ``finder_results`` attribute (the ``finder`` returns an
astropy table):

.. doctest-requires:: scipy

    >>> psfphot.finder_results[0]['xcentroid'].info.format = '.4f'  # optional format
    >>> psfphot.finder_results[0]['ycentroid'].info.format = '.4f'  # optional format
    >>> psfphot.finder_results[0]['sharpness'].info.format = '.4f'  # optional format
    >>> psfphot.finder_results[0]['peak'].info.format = '.4f'
    >>> psfphot.finder_results[0]['flux'].info.format = '.4f'
    >>> psfphot.finder_results[0]['mag'].info.format = '.4f'
    >>> print(psfphot.finder_results[0])  # doctest: +FLOAT_CMP
     id xcentroid ycentroid sharpness ... sky   peak    flux    mag
    --- --------- --------- --------- ... --- ------- ------- -------
      1   32.7662   12.2009    0.6116 ... 0.0 68.3302  8.7599 -2.3562
      2   13.2605   14.5831    0.5761 ... 0.0 52.5762  6.9429 -2.1039
      3   63.6581   22.4396    0.5810 ... 0.0 63.7761  8.1125 -2.2729
      4   82.2983   25.4851    0.5846 ... 0.0 66.2365  8.5570 -2.3308
      5   41.5040   35.8841    0.5925 ... 0.0 71.3016  9.1493 -2.4035
      6   21.5235   41.9433    0.6014 ... 0.0 64.8142  8.4183 -2.3131
      7   14.1791   64.9962    0.6275 ... 0.0 78.2702 10.3522 -2.5376
      8   61.8323   67.5194    0.5755 ... 0.0 62.9875  8.2963 -2.2972
      9   74.6218   68.1660    0.6043 ... 0.0 53.9404  7.1150 -2.1304
     10   15.1527   78.0361    0.6136 ... 0.0 62.7977  8.2524 -2.2915

The ``fit_results`` attribute contains a dictionary with a wealth of
detailed information, including the fit models and any information
returned from the ``fitter`` for each source:

.. doctest-requires:: scipy

    >>> psfphot.fit_results.keys()
    dict_keys(['local_bkg', 'fit_models', 'fit_infos', 'fit_param_errs', 'fit_error_indices', 'npixfit', 'nmodels', 'cen_res_idx', 'fit_residuals'])

As an example, let's print the covariance matrix of the fit parameters
for the first source (note that not all astropy fitters will return a
covariance matrix):

.. doctest-skip::

    >>> psfphot.fit_results['fit_infos'][0]['param_cov']  # doctest: +FLOAT_CMP
    array([[ 7.27034774e-01,  8.86845334e-04,  3.98593038e-03],
           [ 8.86845334e-04,  2.92871525e-06, -6.36805464e-07],
           [ 3.98593038e-03, -6.36805464e-07,  4.29520185e-05]])


Fitting a single source
^^^^^^^^^^^^^^^^^^^^^^^

In some cases, one may want to fit only a single source (or few sources)
in an image. We can do that by defining a table of the sources that
we want to fit. For this example, let's fit the single star at ``(x,
y) = (42, 36)``. We first define a table with this position and then
pass that table into the ``init_params`` keyword when calling the PSF
photometry class on the data:

.. doctest-requires:: scipy

    >>> from astropy.table import QTable
    >>> init_params = QTable()
    >>> init_params['x'] = [42]
    >>> init_params['y'] = [36]
    >>> phot = psfphot(data, error=error, init_params=init_params)

The PSF photometry class allows for flexible input column names
using a heuristic to identify the x, y, and flux columns. See
`~photutils.psf.PSFPhotometry` for more details.

The output table contains only the fit results for the input source:

.. doctest-requires:: scipy

    >>> phot['x_fit'].info.format = '.4f'  # optional format
    >>> phot['y_fit'].info.format = '.4f'
    >>> phot['flux_fit'].info.format = '.4f'
    >>> print(phot[('id', 'x_fit', 'y_fit', 'flux_fit')])  # doctest: +FLOAT_CMP
     id  x_fit   y_fit  flux_fit
    --- ------- ------- --------
      1 41.5416 35.8893 687.8236

Finally, let's show the residual image. The red circular aperture shows
the location of the star that was fit and subtracted.

.. plot::

    import matplotlib.pyplot as plt
    import numpy as np
    from astropy.table import QTable
    from photutils.aperture import CircularAperture
    from photutils.datasets import make_noise_image, make_test_psf_data
    from photutils.detection import DAOStarFinder
    from photutils.psf import IntegratedGaussianPRF, PSFPhotometry

    psf_model = IntegratedGaussianPRF(flux=1, sigma=2.7 / 2.35)
    psf_shape = (25, 25)
    nsources = 10
    shape = (101, 101)
    data, true_params = make_test_psf_data(shape, psf_model, psf_shape,
                                           nsources, flux_range=(500, 700),
                                           min_separation=10, seed=0)
    noise = make_noise_image(data.shape, mean=0, stddev=1, seed=0)
    data += noise
    error = np.abs(noise)

    psf_model = IntegratedGaussianPRF(flux=1, sigma=2.7 / 2.35)
    fit_shape = (5, 5)
    finder = DAOStarFinder(6.0, 2.0)
    psfphot = PSFPhotometry(psf_model, fit_shape, finder=finder,
                            aperture_radius=4)

    init_params = QTable()
    init_params['x'] = [42]
    init_params['y'] = [36]
    phot = psfphot(data, error=error, init_params=init_params)

    resid = psfphot.make_residual_image(data, (25, 25))
    plt.imshow(resid, origin='lower')

    resid = psfphot.make_residual_image(data, (25, 25))
    aper = CircularAperture(zip(init_params['x'], init_params['y']), r=4)
    plt.imshow(resid, origin='lower')
    aper.plot(color='red')

    plt.title('Residual Image')
    plt.colorbar()


Forced Photometry
^^^^^^^^^^^^^^^^^

In general, the three parameters fit for each source are the x and
y positions and the flux. However, the astropy modeling and fitting
framework allows any of these parameters to be fixed during the fitting.

Let's say you want to fix the (x, y) position for each source. You can
do that by setting the ``fixed`` attribute on the model parameters:

.. doctest-requires:: scipy

    >>> psf_model2 = IntegratedGaussianPRF(flux=1, sigma=2.7 / 2.35)
    >>> psf_model2.x_0.fixed = True
    >>> psf_model2.y_0.fixed = True
    >>> psf_model2.fixed
    {'flux': False, 'x_0': True, 'y_0': True, 'sigma': True}

Now when the model is fit, the flux will be varied but, the (x, y)
position will be fixed at its initial position for every source. Let's
just fit a single source (defined in ``init_params``):

.. doctest-requires:: scipy

    >>> psfphot = PSFPhotometry(psf_model2, fit_shape, finder=finder,
    ...                         aperture_radius=4)
    >>> phot = psfphot(data, error=error, init_params=init_params)

The output table shows that the (x, y) position was unchanged, with the
fit values being identical to the initial values. However, the flux was
fit:

.. doctest-requires:: scipy

    >>> phot['flux_init'].info.format = '.4f'  # optional format
    >>> phot['flux_fit'].info.format = '.4f'
    >>> print(phot[('id', 'x_init', 'y_init', 'flux_init', 'x_fit',
    ...             'y_fit', 'flux_fit')])  # doctest: +FLOAT_CMP
     id x_init y_init flux_init x_fit y_fit flux_fit
    --- ------ ------ --------- ----- ----- --------
      1     42     36  701.6391  42.0  36.0 921.2168


Source Grouping
^^^^^^^^^^^^^^^

Source grouping is an optional feature. To turn it on, create a
`~photutils.psf.SourceGrouper` instance and input it via the ``grouper``
keyword. Here we'll group stars that are within 20 pixels of each
other:

.. doctest-requires:: scipy, sklearn

    >>> from photutils.psf import SourceGrouper
    >>> grouper = SourceGrouper(min_separation=20)
    >>> psfphot = PSFPhotometry(psf_model, fit_shape, finder=finder,
    ...                         grouper=grouper, aperture_radius=4)
    >>> phot = psfphot(data, error=error)

The ``group_id`` column shows that six groups were identified (each with
two stars). The stars in each group were simultaneously fit.

.. doctest-requires:: scipy, sklearn

    >>> print(phot[('id', 'group_id', 'group_size')])
     id group_id group_size
    --- -------- ----------
      1        1          2
      2        1          2
      3        2          2
      4        2          2
      5        3          1
      6        4          1
      7        5          2
      8        6          2
      9        6          2
     10        5          2

Care should be taken in defining the star groups. As noted above,
simultaneously fitting very large star groups is computationally
expensive and error-prone. A warning will be raised if the number of
sources in a group exceeds 25.


Local Background Subtraction
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To subtract a local background from each source, define a
`~photutils.background.LocalBackground` instance and input it via
the ``localbkg_estimator`` keyword. Here we'll use an annulus with
an inner and outer radius of 5 and 10 pixels, respectively, with the
`~photutils.background.MMMBackground` statistic (with its default sigma
clipping):

.. doctest-requires:: scipy, sklearn

    >>> from photutils.background import LocalBackground, MMMBackground
    >>> bkgstat = MMMBackground()
    >>> localbkg_estimator = LocalBackground(5, 10, bkgstat)
    >>> finder = DAOStarFinder(10.0, 2.0)
    >>> psfphot = PSFPhotometry(psf_model, fit_shape, finder=finder,
    ...                         grouper=grouper, aperture_radius=4,
    ...                         localbkg_estimator=localbkg_estimator)
    >>> phot = psfphot(data, error=error)

The local background values are output in the table:

.. doctest-requires:: scipy, sklearn

    >>> phot['local_bkg'].info.format = '.4f'  # optional format
    >>> print(phot[('id', 'local_bkg')])  # doctest: +FLOAT_CMP
     id local_bkg
    --- ---------
      1    0.2869
      2    0.0207
      3   -0.1331
      4    0.2441
      5   -0.0371
      6   -0.2863
      7   -0.1639
      8    0.0544
      9   -0.1431
     10   -0.0414

The local background values can also be input directly using the
``init_params`` keyword.


Iterative PSF Photometry
^^^^^^^^^^^^^^^^^^^^^^^^

Now let's use the `~photutils.psf.IterativePSFPhotometry` class to
iteratively fit the stars in the image. This class is useful for crowded
fields where faint stars are very close to bright stars. The faint stars
may not be detected until after the bright stars are subtracted.

For this simple example, let's input a table of three stars for the
first fit iteration. Subsequent iterations will use the ``finder`` to
find additional stars:

.. doctest-requires:: scipy

    >>> from photutils.background import LocalBackground, MMMBackground
    >>> from photutils.psf import IterativePSFPhotometry
    >>> fit_shape = (5, 5)
    >>> finder = DAOStarFinder(10.0, 2.0)
    >>> bkgstat = MMMBackground()
    >>> localbkg_estimator = LocalBackground(5, 10, bkgstat)
    >>> init_params = QTable()
    >>> init_params['x'] = [33, 13, 64]
    >>> init_params['y'] = [12, 15, 22]
    >>> psfphot2 = IterativePSFPhotometry(psf_model, fit_shape, finder=finder,
    ...                                   localbkg_estimator=localbkg_estimator,
    ...                                   aperture_radius=4)
    >>> phot = psfphot2(data, error=error, init_params=init_params)

The table output from `~photutils.psf.IterativePSFPhotometry` contains a
column called ``iter_detected`` that returns the fit iteration in which
the source was detected:

.. doctest-requires:: scipy

    >>> phot['x_fit'].info.format = '.4f'  # optional format
    >>> phot['y_fit'].info.format = '.4f'
    >>> phot['flux_fit'].info.format = '.4f'
    >>> print(phot[('id', 'iter_detected', 'x_fit', 'y_fit', 'flux_fit')])  # doctest: +FLOAT_CMP
     id iter_detected  x_fit   y_fit  flux_fit
    --- ------------- ------- ------- --------
      1             1 32.7697 12.2179 623.1128
      2             1 13.2674 14.5843 505.6723
      3             1 63.6500 22.3870 644.4719
      4             2 82.2881 25.5224 658.0372
      5             2 41.5420 35.8889 688.7885
      6             2 21.5767 41.9471 625.1697
      7             2 14.1820 65.0087 683.7103
      8             2 61.8352 67.5545 607.3983
      9             2 74.6200 68.1865 506.1033
     10             2 15.1675 78.0376 558.5847


References
----------

`Spitzer PSF vs. PRF
<https://irsa.ipac.caltech.edu/data/SPITZER/docs/files/spitzer/PRF_vs_PSF.pdf>`_

`The Kepler Pixel Response Function
<https://ui.adsabs.harvard.edu/abs/2010ApJ...713L..97B/abstract>`_

`Stetson (1987 PASP 99, 191)
<https://ui.adsabs.harvard.edu/abs/1987PASP...99..191S/abstract>`_

`Anderson and King (2000 PASP 112, 1360)
<https://ui.adsabs.harvard.edu/abs/2000PASP..112.1360A/abstract>`_


Reference/API
-------------

.. automodapi:: photutils.psf
    :no-heading:
