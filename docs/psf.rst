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
    >>> noise = make_noise_image(data.shape, mean=0, stddev=1)
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
    noise = make_noise_image(data.shape, mean=0, stddev=1)
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
`~photutils.psf.IntegratedGaussianPRF` PSF model::

    >>> from photutils.detection import DAOStarFinder
    >>> from photutils.psf import PSFPhotometry

    >>> psf_model = IntegratedGaussianPRF(flux=1, sigma=2.7 / 2.35)
    >>> fit_shape = (5, 5)
    >>> finder = DAOStarFinder(6.0, 2.0)
    >>> psfphot = PSFPhotometry(psf_model, fit_shape, finder=finder,
    >>>                         aperture_radius=4)
    >>> phot = psfphot(data, error=error)

The result is an astropy `~astropy.table.Table` with columns for the
source and group identification numbers, the x, y, and flux initial,
fit, and error values, local background, number of unmasked pixels
fit, the group size, quality-of-fit metrics, and flags. See the
`~photutils.psf.PSFPhotometry` documentation for descriptions of the
output columns.

The full table cannot be shown here as it has a large number of columns,
but let's print print the source id along with the fit x, y, and flux
values::

    >>> print(phot[('id', 'x_fit', 'y_fit', 'flux_fit')]
    id       x_fit              y_fit             flux_fit
    --- ------------------ ------------------ ------------------
      1  32.77404564797249  12.21241587326953  627.4295952266112
      2 13.280576995465728  14.59025837871636   509.587414373434
      3 63.636601583810105  22.40323207475757  647.7941676784337
      4  82.28254515932343 25.529559437743846  663.6261884967792
      5 41.545338645209505  35.89394538939527   687.803302465493
      6 21.570470010710824 41.949846900895494  622.2950919502064
      7 14.179729867606312  65.01034469898062  682.1063539928764
      8 61.832928315721446  67.55176672592506  608.6862651104628
      9  74.62575019365177  68.18507475252673 502.22374019838963
     10 15.152402590000605  78.01918334187057   553.394644171982

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
    noise = make_noise_image(data.shape, mean=0, stddev=1)
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

    >>> print(psfphot.finder_results[0])
     id     xcentroid      ...        flux               mag
    --- ------------------ ... ----------------- -------------------
      1 32.798546744210974 ... 8.961786512586334  -2.380986484874818
      2 13.277691484600824 ... 6.614429496837522  -2.051230979184781
      3 63.631531678983926 ... 8.524071587500329 -2.3266177209553094
      4  82.28629177090498 ... 8.455331590418556 -2.3178266096881077
      5  41.48375684304935 ... 9.508906296394022  -2.445326419379224
      6 21.506684549579933 ... 8.600694332329011  -2.336333782913264
      7  14.18281096109073 ... 9.993015501623889  -2.499241402772193
      8  61.86210314839908 ...  8.40967305502954  -2.311947779876285
      9   74.6331728593996 ... 6.961807305185392  -2.106804995857878
     10 15.152159604120909 ... 8.476286915780012  -2.320514122052738

The ``fit_results`` attribute contains a dictionary with a wealth
of further detailed information, including the fit models and any
information returned from the ``fitter`` for each source:

    >>> psfphot.fit_results.keys()
    dict_keys(['local_bkg', 'fit_models', 'fit_infos', 'fit_param_errs', 'npixfit', 'nmodels', 'cen_res_idx', 'fit_residuals'])

As an example, let's print the fit parameter covariance matrix for the
first source (note that not all astropy fitters will return a covariance
matrix)::

    >>> psfphot.fit_results['fit_infos'][0]['param_cov']
    array([[ 9.55761538e-01,  3.66212195e-04, -1.06887650e-03],
           [ 3.66212195e-04,  1.24115431e-05, -1.75532271e-06],
           [-1.06887650e-03, -1.75532271e-06,  2.97592212e-06]])


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
    noise = make_noise_image(data.shape, mean=0, stddev=1)
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
    ...             'y_fit', 'flux_fit')])
     id x_init y_init     flux_init     x_fit y_fit      flux_fit
    --- ------ ------ ----------------- ----- ----- ------------------
      1     42     36 676.9365873273836  42.0  36.0 496.70185010898047


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

The ``group_id`` column shows that 6 groups were identified (each with
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
    ...                         grouper=grouper, aperture_radius=4
    ...                         localbkg_estimator=localbkg_estimator)
    >>> phot = psfphot(data, error=error)

The local background values are output in the table::

    >>> print(phot[('id', 'local_bkg')])
     id      local_bkg
    --- --------------------
      1  0.05683413339116036
      2   0.1465178770856933
      3  0.05156304353688759
      4  0.06533797895988643
      5 0.015539380434351796
      6   0.2946447148528156
      7 -0.24848306906560644
      8  0.06902761284931344
      9 0.037708674392344696
     10  0.07697120639964465


Iterative PSF Photometry
^^^^^^^^^^^^^^^^^^^^^^^^

Now let's use the `~photutils.psf.IterativePSFPhotometry` class to
iteratively fit the stars in the image. This class is useful for crowded
fields where faint stars are very close to bright stars. The faint stars
may not be detected until after the bright stars are subtracted.

For this simple example, let's input a table of three stars for the
first fit iteration. Subsequent iterations will use the ``finder`` to
find additional stars::

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

    >>> print(phot2[('id', 'iter_detected', 'x_fit', 'y_fit', 'flux_fit')])
      id iter_detected       x_fit              y_fit             flux_fit
    --- ------------- ------------------ ------------------ ------------------
      1             1  32.78080238212221 12.215569632832379  631.3728970528724
      2             1 13.270436821198123  14.58460590684079 508.67012005993433
      3             1  63.63290114928681  22.40686369968043  639.6021481152056
      4             2  82.27790147215153 25.534763865978288  655.2570434776327
      5             2  41.54497971052003 35.889780164594136  687.3208299191051
      6             2   21.5687339044637  41.95064339406346  622.0413056357854
      7             2 14.190712563534408  65.00609713727931  682.9852091170068
      8             2 61.841700204614824  67.55075533597399  617.0243577148906
      9             2   74.6233906521105   68.1802921736959 507.47873637759386
     10             2 15.158481627411087  78.01467993772174  556.6266865096007


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
