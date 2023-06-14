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
performed. When used with the `~photutils.detection.DAOStarFinder`,
`~photutils.psf.IterativePSFPhotometry` is essentially an implementation
of the DAOPHOT algorithm described by Stetson in his `seminal paper
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
`~photutils.background.LocalBackground` instance that estimates the
local statistics in a circular annulus aperture centered on each
source. The size of the annulus and the statistic can be configured in
`~photutils.background.LocalBackground`.

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

Example Usage
-------------

Let's start with a simple example with simulated stars whose PSF is
assumed to be Gaussian. We'll create a synthetic image provided by the
:ref:`photutils.datasets <datasets>` module::

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
    import numpy as np
    from photutils.datasets import make_test_psf_data, make_noise_image
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
    plt.imshow(data)
    plt.title('Simulated Data')
    plt.colorbar()


Fitting multiple stars
^^^^^^^^^^^^^^^^^^^^^^

Now let's use `~photutils.psf.PSFPhotometry` to perform PSF photometry
on this image. Note that the input image must be background-subtracted
prior to using the photometry classes. See :ref:`background` for tools
to subtract a global background from an image.

We'll use the `~photutils.detection.DAOStarFinder` for source
detection. We'll fit the central 5x5 pixel region of each star using a
`~photutils.psf.IntegratedGaussianPRF` PSF model. We first create
an instance of the `~photutils.psf.PSFPhotometry` class::

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
can also be input into the ``data`` parameter. A table of initial
PSF model parameter values can also be input when calling the class
instance. Here we use the data and error arrays::

    >>> phot = psfphot(data, error=error)

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

The full table cannot be shown here as it has a large number of columns,
but let's print print the source id along with the fit x, y, and flux
values::

    >>> print(phot[('id', 'x_fit', 'y_fit', 'flux_fit')])  # doctest: +FLOAT_CMP
     id       x_fit              y_fit             flux_fit
    --- ------------------ ------------------ ------------------
      1  32.77148399484856  12.22102577825017  627.4273999433956
      2  13.27001582702945  14.58405114790792  507.5777675679552
      3  63.64829088944866 22.390725227343395  640.9277337102149
      4  82.28474959784151 25.522276342326016  662.0007199929463
      5  41.54155347688922  35.88933616834469  687.8236197218502
      6 21.572091809145466  41.94801131801161  620.8561882378236
      7 14.182317247338144  65.00902999557461  681.7447257950179
      8  61.83627242626185   67.5559557331335  608.2459417927776
      9   74.6206253109306  68.18554867674209 502.89058093032907
     10 15.168451808033558  78.03734517330257  558.0203651230795

Let's create the residual image::

    >>> resid = psfphot.make_residual_image(data, (25, 25))

and plot it:

.. plot::

    import matplotlib.pyplot as plt
    import numpy as np
    from photutils.datasets import make_test_psf_data, make_noise_image
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
    error = np.abs(noise)

    from photutils.detection import DAOStarFinder
    from photutils.psf import PSFPhotometry

    psf_model = IntegratedGaussianPRF(flux=1, sigma=2.7 / 2.35)
    fit_shape = (5, 5)
    finder = DAOStarFinder(6.0, 2.0)
    psfphot = PSFPhotometry(psf_model, fit_shape, finder=finder,
                            aperture_radius=4)
    phot = psfphot(data, error=error)

    resid = psfphot.make_residual_image(data, (25, 25))
    plt.imshow(resid)
    plt.title('Residual Image')
    plt.colorbar()

The residual image looks like noise, indicating good fits to the
sources.

Further details about the PSF fitting can be obtained from attributes on
the `~photutils.psf.PSFPhotometry` instance. For example, the results
from the ``finder`` instance (another astropy table) can be obtained::

    >>> print(psfphot.finder_results[0])  # doctest: +FLOAT_CMP
     id     xcentroid      ...        flux                mag
    --- ------------------ ... ------------------ -------------------
      1  32.76616312538923 ...  8.759914568968053 -2.3562496768338215
      2 13.260487336490504 ...  6.942889580709919 -2.1038506457966806
      3  63.65811505587449 ...  8.112512027320118 -2.2728883841959138
      4  82.29828239306062 ...  8.557042352587091  -2.330809203775193
      5 41.503985357241085 ...  9.149300616176507  -2.403469743323483
      6   21.5234588686359 ...   8.41831136093057 -2.3130624614752806
      7  14.17914895242459 ... 10.352214238287955 -2.5375831277724186
      8 61.832258738894474 ...  8.296287915436698 -2.2972095386343816
      9  74.62182677494232 ...  7.114958201196044  -2.130430882609707
     10 15.152687611700179 ...  8.252429063923989 -2.2914544997949413

The ``fit_results`` attribute contains a dictionary with a wealth
of further detailed information, including the fit models and any
information returned from the ``fitter`` for each source:

    >>> psfphot.fit_results.keys()
    dict_keys(['local_bkg', 'fit_models', 'fit_infos', 'fit_param_errs', 'npixfit', 'nmodels', 'cen_res_idx', 'fit_residuals'])

As an example, let's print the fit parameter covariance matrix for the
first source (note that not all astropy fitters will return a covariance
matrix)::

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
photometry class on the data::

    >>> from astropy.table import QTable
    >>> init_params = QTable()
    >>> init_params['x'] = [42]
    >>> init_params['y'] = [36]
    >>> phot = psfphot(data, error=error, init_params=init_params)

The output table contains only the fit results for the input source.
Let's show the residual image:

.. plot::

    import matplotlib.pyplot as plt
    import numpy as np
    from photutils.datasets import make_test_psf_data, make_noise_image
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
    error = np.abs(noise)

    from photutils.detection import DAOStarFinder
    from photutils.psf import PSFPhotometry

    psf_model = IntegratedGaussianPRF(flux=1, sigma=2.7 / 2.35)
    fit_shape = (5, 5)
    finder = DAOStarFinder(6.0, 2.0)
    psfphot = PSFPhotometry(psf_model, fit_shape, finder=finder,
                            aperture_radius=4)

    from astropy.table import QTable
    init_params = QTable()
    init_params['x'] = [42]
    init_params['y'] = [36]
    phot = psfphot(data, error=error, init_params=init_params)

    resid = psfphot.make_residual_image(data, (25, 25))
    plt.imshow(resid)

    from photutils.aperture import CircularAperture

    resid = psfphot.make_residual_image(data, (25, 25))
    aper = CircularAperture(zip(init_params['x'], init_params['y']), r=4)
    plt.imshow(resid)
    aper.plot(color='red')

    plt.title('Residual Image')
    plt.colorbar()

The red circular aperture shows the location of the star that was
subtracted.


Forced Photometry
^^^^^^^^^^^^^^^^^

In general, the three parameters fit for each source are the x and
y positions and the flux. However, the astropy modeling and fitting
framework allows any of these parameters to be fixed during the fitting.

Let's say you want to fix the (x, y) position for each source. You can
do that by setting the ``fixed`` attribute on the model parameter::

    >>> psf_model2 = IntegratedGaussianPRF(flux=1, sigma=2.7 / 2.35)
    >>> psf_model2.x_0.fixed = True
    >>> psf_model2.y_0.fixed = True
    >>> psf_model2.fixed
    {'flux': False, 'x_0': True, 'y_0': True, 'sigma': True}

Now when the model is fit the (x, y) position will be fixed at its
initial position for every source. Let's just fit a single source
(defined in ``init_params``)::

    >>> psfphot = PSFPhotometry(psf_model2, fit_shape, finder=finder,
    ...                         aperture_radius=4)
    >>> phot = psfphot(data, error=error, init_params=init_params)

The output table shows that the (x, y) position was unchanged, with the
fit values being identical to the initial values. However, the flux was
fit::

    >>> print(phot[('id', 'x_init', 'y_init', 'flux_init', 'x_fit',
    ...             'y_fit', 'flux_fit')])  # doctest: +FLOAT_CMP
     id x_init y_init     flux_init     x_fit y_fit      flux_fit
    --- ------ ------ ----------------- ----- ----- -----------------
      1     42     36 701.6390630906382  42.0  36.0 921.2167861548569


Source Grouping
^^^^^^^^^^^^^^^

Source grouping is an optional feature. To turn it on, create a
`~photutils.psf.SourceGrouper` instance and input it via the ``grouper``
keyword. Here we'll group stars that are within 20 pixels of each
other::

    >>> from photutils.psf import SourceGrouper
    >>> grouper = SourceGrouper(min_separation=20)
    >>> psfphot = PSFPhotometry(psf_model, fit_shape, finder=finder,
    ...                         grouper=grouper, aperture_radius=4)
    >>> phot = psfphot(data, error=error)

The ``group_id`` column shows that six groups were identified (each with
two stars). The stars in each group were simultaneously fit.

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


Local Background Subtraction
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To subtract a local background from each source, define a
`~photutils.background.LocalBackground` instance an input it via the
``localbkg_estimator`` keyword. Here we'll use an annulus with an
inner and outer radius of 5 and 10 pixels, respectively, with the
`~photutils.background.MMMBackground` statistic (with its default sigma
clipping)::

    >>> from photutils.background import LocalBackground, MMMBackground
    >>> bkgstat = MMMBackground()
    >>> localbkg_estimator = LocalBackground(5, 10, bkgstat)
    >>> finder = DAOStarFinder(10.0, 2.0)

    >>> psfphot = PSFPhotometry(psf_model, fit_shape, finder=finder,
    ...                         grouper=grouper, aperture_radius=4,
    ...                         localbkg_estimator=localbkg_estimator)
    >>> phot = psfphot(data, error=error)

The local background values are output in the table::

    >>> print(phot[('id', 'local_bkg')])  # doctest: +FLOAT_CMP
     id      local_bkg
    --- --------------------
      1   0.2868767131808984
      2 0.020718326602389925
      3  -0.1331493984320581
      4   0.2440799441238853
      5 -0.03708541210598071
      6  -0.2863497468267567
      7 -0.16388547407177662
      8  0.05435649834620228
      9 -0.14307720372014684
     10 -0.04142177531136823


Iterative PSF Photometry
^^^^^^^^^^^^^^^^^^^^^^^^

Now let's use the `~photutils.psf.IterativePSFPhotometry` class to
iteratively fit the stars in the image. This class is useful for crowded
fields where faint stars are very close to bright stars. The faint stars
may not be detected until after the bright stars are subtracted.

For this simple example, let's input a table of three stars for the
first fit iteration. Subsequent iterations will use the ``finder`` to
find additional stars::

    >>> from photutils.psf import IterativePSFPhotometry
    >>> init_params = QTable()
    >>> init_params['x'] = [33, 13, 64]
    >>> init_params['y'] = [12, 15, 22]
    >>> psfphot2 = IterativePSFPhotometry(psf_model, fit_shape, finder=finder,
    ...                                   localbkg_estimator=localbkg_estimator,
    ...                                   aperture_radius=4)
    >>> phot = psfphot2(data, error=error, init_params=init_params)

The table output from `~photutils.psf.IterativePSFPhotometry` contains a
column called ``iter_detected`` which returns the fit iteration in which
the source was detected::

    >>> print(phot[('id', 'iter_detected', 'x_fit', 'y_fit', 'flux_fit')])  # doctest: +FLOAT_CMP
     id iter_detected       x_fit              y_fit             flux_fit
    --- ------------- ------------------ ------------------ -----------------
      1             1 32.769748521507736 12.217877551785042  623.112845666705
      2             1 13.267368530744182 14.584313177544152 505.6723003886122
      3             1 63.649999091018834 22.386979414024587 644.4718993131787
      4             2  82.28810152282769  25.52243260298307 658.0372017828031
      5             2  41.54196044408549  35.88887512943361 688.7884814263355
      6             2 21.576693854399878  41.94709948442762 625.1697408936903
      7             2 14.181996733367436  65.00871043519591 683.7102916315391
      8             2  61.83517624202557  67.55451682658261 607.3982833181155
      9             2   74.6199614642493  68.18647252595234 506.1033464597233
     10             2 15.167462695939475  78.03763869218587 558.5846710637369


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
