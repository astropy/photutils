PSF Photometry
==============

The `photutils.psf` module contains tools for model-fitting photometry, often
called "PSF photometry".

.. warning::
    The PSF photometry API is currently considered *experimental* and may change
    in the future.  We will aim to keep compatibility where practical, but will
    not finalize the API until sufficient user feedback has been accumulated.

.. _psf-terminology:

Terminology
-----------
Different astronomy sub-fields use the terms Point Spread Function (PSF) and
Point Response Function (PRF) somewhat differently, especially when colloquial
usage is taken into account.  For this module we assume that the PRF is an image
of a point source *after discretization* e.g., onto a rectilinear CCD grid. This
is the definition used by `Spitzer
<http://irsa.ipac.caltech.edu/data/SPITZER/docs/dataanalysistools/tools/mopex/mopexusersguide/89/>`_.
Where relevant, we use this terminology for this sort of model, and consider
"PSF" to refer to the underlying model. In many cases this distinction is
unimportant, but can be critical when dealing with undersampled data.

Despite this, in colloquial usage "PSF photometry" often means the same sort of
model-fitting analysis, regardless to exactly what kind of model is actually
being fit.  We take this road, using "PSF photometry" as shorthand for the
general approach.

PSF Photometry in Crowded Fields
--------------------------------

Photutils provides a modular set of tools to perform PSF photometry for
different science cases. These are implemented as separate classes to do
sub-tasks of PSF photometry. It also provides high-level classes that connect
these pieces together. In particular, it contains an implementation of the
DAOPHOT algorithm (`~photutils.psf.DAOPhotPSFPhotometry`) proposed by
`Stetson in his seminal paper <http://adsabs.harvard.edu/abs/1987PASP...99..191S>`_
for crowded-field stellar photometry.

The DAOPHOT algorithm consists in applying the loop FIND, GROUP, NSTAR,
SUBTRACT, FIND until no more stars are detected or a given number of
iterations is reached. Basically, `~photutils.psf.DAOPhotPSFPhotometry` works
as follows. The first step is to estimate the sky background. For this task,
photutils provides several classes to compute scalar and 2D backgrounds, see
`~photutils.background` for details. The next step is to find an initial
estimate of the positions of potential sources.
This can be accomplished by using source detection algorithms,
which are implemented in `~photutils.detection`.

After finding sources one would apply a clustering algorithm in order to label
the sources according to groups. Usually, those groups are formed by a
distance criterion, which is the case of the grouping algorithm proposed
by Stetson. In `~photutils.psf.DAOGroup`, we provide an implementation of
that algorithm. In addition, `~photutils.psf.DBSCANGroup` can also be used to
group sources with more complex distance criteria. The reason behind the
construction of groups is illustrated as follows: imagine that one would like
to fit 300 stars and the model for each star has three parameters to be
fitted. If one constructs a single model to fit the 300 stars simultaneously,
then the optimization algorithm will to search for the solution in a 900
dimensional space, which is very likely to give a wrong solution. Reducing the
stars in groups effectively reduces the dimension of the parameter space,
which facilitates the optimization process.

Provided that the groups are available, the next step is to fit the sources
simultaneously for each group. This task can be done using an astropy
fitter, for instance, `~astropy.modeling.fitting.LevMarLSQFitter`.

After sources are fitted, they are subtracted from the given image
and, after fitting all sources, the residual image is analyzed by the finding
routine again in order to check if there exist any source which has not been
detected previously. This process goes on until no more sources are identified
by the finding routine.

.. note::
    It is important to note the conventions on the column names of the
    input/output astropy Tables which are passed along the source detection
    objects and the photometry objetcs. For instance, all source detection
    objects should output a table with columns named as ``xcentroid`` and
    ``ycentroid`` (check `~photutils.detection`). On the other hand,
    `~photutils.psf.DAOGroup` expects columns named as ``x_0`` and ``y_0``,
    which represents the initial guesses on the sources' centroids.
    Finally, the output of the fitting process shows columns named as
    ``x_fit``, ``y_fit``, ``flux_fit`` for the optimum values and
    ``x_0``, ``y_0``, ``flux_0`` for the initial guesses.
    Although this convention implies that the columns have to be renamed
    along the process, it has the advantage of clarity so that one can
    keep track and easily differentiate from where input/outputs came from.


Basic Usage
^^^^^^^^^^^

The `~photutils.psf.DAOPhotPSFPhotometry` is the core class that implements the
DAOPHOT algorithm for performing PSF photometry in crowded fields.
It basically encapsulates the loop "FIND, GROUP, NSTAR, SUBTRACT, FIND..." in
one place so that one can easily perform PSF photometry just by setting up a
`~photutils.psf.DAOPhotPSFPhotometry` object.

This class and all of the classes it *uses* for the steps in the process are
implemented in such a way that they can be used callable functions. The actual
implementation of the  ``__call__`` method for
`~photutils.psf.DAOPhotPSFPhotometry` is identical to the ``do_photometry``
method (which is why the documentation for ``__call__`` is in
``do_photometry``). This allows subclasses of
`~photutils.psf.DAOPhotPSFPhotometry` to override ``do_photometry`` if they want
to change some behavior, making such code more maintainable.

The basic usage of `~photutils.psf.DAOPhotPSFPhotometry` is as follows:

.. doctest-skip::

    >>> # create a DAOPhotPSFPhotometry object
    >>> from photutils.psf import DAOPhotPSFPhotometry
    >>> my_photometry = DAOPhotPSFPhotometry(finder=my_finder,
    ...                                      group_maker=my_group_maker,
    ...                                      bkg_estimator=my_bkg_estimator,
    ...                                      psf_model=my_psf_model,
    ...                                      fitter=my_fitter, niters=1,
    ...                                      fitshape=(7,7))
    >>> # get photometry results
    >>> photometry_results, residual_image = my_photometry(image=my_image)

Where ``my_finder``, ``my_group_maker``, and ``my_bkg_estimator`` may be any
suitable class or callable function. This approach allows one to customize every
part of the photometry process provided that their input/output are compatible
with the input/ouput expected by `~photutils.psf.DAOPhotPSFPhotometry`.
`photutils.psf` provides all the necessary classes to reproduce the DAOPHOT
algorithm, but any individual part of that algorithm can be swapped for a
user-defined function.  See the API documentation for precise details on what
these classes or functions should look like.

Performing PSF Photometry
^^^^^^^^^^^^^^^^^^^^^^^^^

Let's take a look at a simple example with simulated stars whose PSF is
assumed to be Gaussian.

First let's create an image with four overlapping stars::

    >>> import numpy as np
    >>> from astropy.table import Table
    >>> from photutils.datasets import make_random_gaussians, make_noise_image
    >>> from photutils.datasets import make_gaussian_sources
    >>> sigma_psf = 2.0
    >>> sources = Table()
    >>> sources['flux'] = [700, 800, 700, 800]
    >>> sources['x_mean'] = [12, 17, 12, 17]
    >>> sources['y_mean'] = [15, 15, 20, 20]
    >>> sources['x_stddev'] = sigma_psf*np.ones(4)
    >>> sources['y_stddev'] = sources['x_stddev']
    >>> sources['theta'] = [0, 0, 0, 0]
    >>> sources['id'] = [1, 2, 3, 4]
    >>> tshape = (32, 32)
    >>> image = (make_gaussian_sources(tshape, sources) +
    ...          make_noise_image(tshape, type='poisson', mean=6.,
    ...                           random_state=1) +
    ...          make_noise_image(tshape, type='gaussian', mean=0.,
    ...                           stddev=2., random_state=1))

.. doctest-requires:: matplotlib

    >>> from matplotlib import rcParams
    >>> rcParams['font.size'] = 13
    >>> import matplotlib.pyplot as plt
    >>> plt.imshow(image, cmap='viridis', aspect=1, interpolation='nearest',
    ...            origin='lower') # doctest: +SKIP
    >>> plt.title('Simulated data') # doctest: +SKIP
    >>> plt.colorbar(orientation='horizontal', fraction=0.046, pad=0.04) # doctest: +SKIP

.. plot::

    import numpy as np
    from astropy.table import Table
    from photutils.datasets import make_random_gaussians, make_noise_image, make_gaussian_sources

    sigma_psf = 2.0
    sources = Table()
    sources['flux'] = [700, 800, 700, 800]
    sources['x_mean'] = [12, 17, 12, 17]
    sources['y_mean'] = [15, 15, 20, 20]
    sources['x_stddev'] = sigma_psf*np.ones(4)
    sources['y_stddev'] = sources['x_stddev']
    sources['theta'] = [0, 0, 0, 0]
    sources['id'] = [1, 2, 3, 4]
    tshape = (32, 32)
    image = (make_gaussian_sources(tshape, sources) +
             make_noise_image(tshape, type='poisson', mean=6.,
                              random_state=1) +
             make_noise_image(tshape, type='gaussian', mean=0.,
                              stddev=2., random_state=1))

    from matplotlib import rcParams
    rcParams['font.size'] = 13
    import matplotlib.pyplot as plt

    plt.imshow(image, cmap='viridis', aspect=1, interpolation='nearest',
               origin='lower')
    plt.title('Simulated data')
    plt.colorbar(orientation='horizontal', fraction=0.046, pad=0.04)

Then let's import the required classes to set up a `~photutils.psf.DAOPhotPSFPhotometry` object::

    >>> from photutils.detection import IRAFStarFinder
    >>> from photutils.psf import IntegratedGaussianPRF, DAOGroup
    >>> from photutils.background import MMMBackground, MADStdBackgroundRMS
    >>> from astropy.modeling.fitting import LevMarLSQFitter
    >>> from astropy.stats import gaussian_sigma_to_fwhm

Let's then instantiate and use the objects:

.. doctest-requires:: scipy

    >>> bkgrms = MADStdBackgroundRMS()
    >>> std = bkgrms(image)
    >>> iraffind = IRAFStarFinder(threshold=3.5*std,
    ...                           fwhm=sigma_psf*gaussian_sigma_to_fwhm,
    ...                           minsep_fwhm=0.01, roundhi=5.0, roundlo=-5.0,
    ...                           sharplo=0.0, sharphi=2.0)
    >>> daogroup = DAOGroup(2.0*sigma_psf*gaussian_sigma_to_fwhm)
    >>> mmm_bkg = MMMBackground()
    >>> fitter = LevMarLSQFitter()
    >>> psf_model = IntegratedGaussianPRF(sigma=sigma_psf)
    >>> from photutils.psf import DAOPhotPSFPhotometry
    >>> daophot_photometry = DAOPhotPSFPhotometry(finder=iraffind,
    ...                                           group_maker=daogroup,
    ...                                           bkg_estimator=mmm_bkg,
    ...                                           psf_model=psf_model,
    ...                                           fitter=LevMarLSQFitter(),
    ...                                           niters=1, fitshape=(11,11))
    >>> result_tab, residual_image = daophot_photometry(image=image)

Note that the parameters values for the finder class, i.e.,
`~photutils.detection.IRAFStarFinder`, are completly chosen in an arbitrary
manner and optimum values do vary according to the data.

As mentioned before, the way to actually do the photometry is by using
``daophot_photometry`` as a function-like call.

It's worth noting that ``image`` does not need to be background subtracted.
The subtraction is done during the photometry process with the attribute
``bkg`` that was used to set up ``daophot_photometry``.

Now, let's compare the simulated and the residual images:

.. doctest-skip::

    >>> plt.subplot(1, 2, 1)
    >>> plt.imshow(image, cmap='viridis', aspect=1, interpolation='nearest',
                   origin='lower')
    >>> plt.title('Simulated data')
    >>> plt.colorbar(orientation='horizontal', fraction=0.046, pad=0.04)
    >>> plt.subplot(1 ,2, 2)
    >>> plt.imshow(residual_image, cmap='viridis', aspect=1,
    ...            interpolation='nearest', origin='lower')
    >>> plt.title('Residual Image')
    >>> plt.colorbar(orientation='horizontal', fraction=0.046, pad=0.04)
    >>> plt.show()

.. plot::

    import numpy as np
    from photutils.datasets import make_random_gaussians, make_noise_image, make_gaussian_sources
    from astropy.table import Table

    sigma_psf = 2.0
    sources = Table()
    sources['flux'] = [700, 800, 700, 800]
    sources['x_mean'] = [12, 17, 12, 17]
    sources['y_mean'] = [15, 15, 20, 20]
    sources['x_stddev'] = sigma_psf*np.ones(4)
    sources['y_stddev'] = sources['x_stddev']
    sources['theta'] = [0, 0, 0, 0]
    sources['id'] = [1, 2, 3, 4]
    tshape = (32, 32)
    image = (make_gaussian_sources(tshape, sources) +
             make_noise_image(tshape, type='poisson', mean=6.,
                              random_state=1) +
             make_noise_image(tshape, type='gaussian', mean=0.,
                              stddev=2., random_state=1))

    from photutils.detection import IRAFStarFinder
    from photutils.psf import IntegratedGaussianPRF, DAOGroup
    from photutils.background import MMMBackground, MADStdBackgroundRMS
    from astropy.modeling.fitting import LevMarLSQFitter
    from astropy.stats import gaussian_sigma_to_fwhm

    bkgrms = MADStdBackgroundRMS()
    std = bkgrms(image)
    iraffind = IRAFStarFinder(threshold=3.5*std,
                              fwhm=sigma_psf*gaussian_sigma_to_fwhm,
                              minsep_fwhm=0.01, roundhi=5.0, roundlo=-5.0,
                              sharplo=0.0, sharphi=2.0)
    daogroup = DAOGroup(2.0*sigma_psf*gaussian_sigma_to_fwhm)
    mmm_bkg = MMMBackground()
    psf_model = IntegratedGaussianPRF(sigma=sigma_psf)
    fitter = LevMarLSQFitter()

    from photutils.psf import DAOPhotPSFPhotometry

    daophot_photometry = DAOPhotPSFPhotometry(finder=iraffind,
                                              group_maker=daogroup,
                                              bkg_estimator=mmm_bkg,
                                              psf_model=psf_model,
                                              fitter=LevMarLSQFitter(),
                                              niters=1, fitshape=(11,11))
    result_tab, residual_image = daophot_photometry(image=image)

    from matplotlib import rcParams
    rcParams['font.size'] = 13
    import matplotlib.pyplot as plt

    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap='viridis', aspect=1, interpolation='nearest',
               origin='lower')
    plt.title('Simulated data')
    plt.colorbar(orientation='horizontal', fraction=0.046, pad=0.04)
    plt.subplot(1 ,2, 2)
    plt.imshow(residual_image, cmap='viridis', aspect=1,
               interpolation='nearest', origin='lower')
    plt.title('Residual Image')
    plt.colorbar(orientation='horizontal', fraction=0.046, pad=0.04)
    plt.show()

Performing PSF Photometry with Fixed Centroids
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In case that the centroids positions of the stars are known a priori, then
they can be held fixed during the fitting process and the optimizer will
only consider flux as a variable. To do that, one has to set the ``fixed``
attribute for the centroid parameters in ``psf`` as ``True``.

Consider the previous example after the line
``psf_model = IntegratedGaussianPRF(sigma=sigma_psf)``:

.. doctest-skip::

    >>> psf_model.x_0.fixed = True
    >>> psf_model.y_0.fixed = True
    >>> pos = Table(names=['x_0', 'y_0'], data=[sources['x_mean'],
    ...                                         sources['y_mean']])

Note that we do not need to set the ``finder`` and ``niters`` attributes in
`~photutils.psf.DAOPhotPSFPhotometry` and the positions are passed using the
keyword ``positions``:

.. doctest-skip::

    >>> daophot_photometry = DAOPhotPSFPhotometry(group_maker=daogroup,
    ...                                           bkg_estimator=mmm_bkg,
    ...                                           psf_model=psf_model,
    ...                                           fitter=LevMarLSQFitter(),
    ...                                           fitshape=(11,11))
    >>> result_tab, residual_image = daophot_photometry(image=image,
    ...                                                 positions=pos)

.. doctest-skip::

    >>> plt.subplot(1, 2, 1)
    >>> plt.imshow(image, cmap='viridis', aspect=1,
    ...            interpolation='nearest', origin='lower')
    >>> plt.title('Simulated data')
    >>> plt.colorbar(orientation='horizontal', fraction=0.046, pad=0.04)
    >>> plt.subplot(1 ,2, 2)
    >>> plt.imshow(residual_image, cmap='viridis', aspect=1,
    ...            interpolation='nearest', origin='lower')
    >>> plt.title('Residual Image')
    >>> plt.colorbar(orientation='horizontal', fraction=0.046, pad=0.04)

.. plot::

    import numpy as np
    from photutils.datasets import make_random_gaussians, make_noise_image, make_gaussian_sources
    from astropy.table import Table

    sigma_psf = 2.0
    sources = Table()
    sources['flux'] = [700, 800, 700, 800]
    sources['x_mean'] = [12, 17, 12, 17]
    sources['y_mean'] = [15, 15, 20, 20]
    sources['x_stddev'] = sigma_psf*np.ones(4)
    sources['y_stddev'] = sources['x_stddev']
    sources['theta'] = [0, 0, 0, 0]
    sources['id'] = [1, 2, 3, 4]
    tshape = (32, 32)
    image = (make_gaussian_sources(tshape, sources) +
             make_noise_image(tshape, type='poisson', mean=6.,
                              random_state=1) +
             make_noise_image(tshape, type='gaussian', mean=0.,
                              stddev=2., random_state=1))

    from photutils.detection import IRAFStarFinder
    from photutils.psf import IntegratedGaussianPRF, DAOGroup
    from photutils.background import MMMBackground, MADStdBackgroundRMS
    from astropy.modeling.fitting import LevMarLSQFitter
    from astropy.stats import gaussian_sigma_to_fwhm

    bkgrms = MADStdBackgroundRMS()
    std = bkgrms(image)
    iraffind = IRAFStarFinder(threshold=3.5*std,
                              fwhm=sigma_psf*gaussian_sigma_to_fwhm,
                              minsep_fwhm=0.01, roundhi=5.0, roundlo=-5.0,
                              sharplo=0.0, sharphi=2.0)
    daogroup = DAOGroup(2.0*sigma_psf*gaussian_sigma_to_fwhm)
    mmm_bkg = MMMBackground()
    psf_model = IntegratedGaussianPRF(sigma=sigma_psf)

    psf_model.x_0.fixed = True
    psf_model.y_0.fixed = True

    pos = Table(names=['x_0', 'y_0'], data=[sources['x_mean'],
                                            sources['y_mean']])

    fitter = LevMarLSQFitter()

    from photutils.psf import DAOPhotPSFPhotometry

    daophot_photometry = DAOPhotPSFPhotometry(group_maker=daogroup,
                                              bkg_estimator=mmm_bkg,
                                              psf_model=psf_model,
                                              fitter=LevMarLSQFitter(),
                                              fitshape=(11,11))

    result_tab, residual_image = daophot_photometry(image=image,
                                                    positions=pos)

    from matplotlib import rcParams
    import matplotlib.pyplot as plt
    rcParams['font.size'] = 13

    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap='viridis', aspect=1, interpolation='nearest',
               origin='lower')
    plt.title('Simulated data')
    plt.colorbar(orientation='horizontal', fraction=0.046, pad=0.04)
    plt.subplot(1 ,2, 2)
    plt.imshow(residual_image, cmap='viridis', aspect=1,
               interpolation='nearest', origin='lower')
    plt.title('Residual Image')
    plt.colorbar(orientation='horizontal', fraction=0.046, pad=0.04)

For more examples, also check the online notebook in the next section.

Example Notebooks (online)
--------------------------

* `PSF photometry on artificial Gaussian stars in crowded fields <https://github.com/astropy/photutils-datasets/blob/master/notebooks/ArtificialCrowdedFieldPSFPhotometry.ipynb>`_
* `PSF photometry on artificial Gaussian stars <https://github.com/astropy/photutils-datasets/blob/master/notebooks/GaussianPSFPhot.ipynb>`_
* `PSF/PRF Photometry on Spitzer Data <https://github.com/astropy/photutils-datasets/blob/master/notebooks/PSFPhotometrySpitzer.ipynb>`_


References
----------

`Spitzer PSF vs. PRF
<http://irsa.ipac.caltech.edu/data/SPITZER/docs/files/spitzer/PRF_vs_PSF.pdf>`_

`Kepler PSF calibration
<http://keplerscience.arc.nasa.gov/CalibrationPSF.shtml>`_

`The Kepler Pixel Response Function
<http://adsabs.harvard.edu/abs/2010ApJ...713L..97B>`_

`Stetson, Astronomical Society of the Pacific, Publications, (ISSN 0004-6280),
vol. 99, March 1987, p. 191-222. <http://adsabs.harvard.edu/abs/1987PASP...99..191S>`_

Reference/API
-------------

.. automodapi:: photutils.psf
    :no-heading:


.. automodapi:: photutils.psf.sandbox
