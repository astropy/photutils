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

Photutils provides an implementation of the DAOPHOT algorithm
(`~photutils.psf.DAOPhotPSFPhotometry`) proposed by Stetson in his
seminal paper for crowded-field stellar photometry.
The DAOPHOT algorithm consists in applying the loop FIND, GROUP, NSTAR,
SUBTRACT, FIND until no more stars are detected or a given number of
iterations is reached.

Basically, DAOPHOT works as follows. The first step is to estimate the sky
background. For this task, photutils provides several classes to compute
scalar and 2D backgrounds, see `~photutils.background` for details. The next
step is to find an initial estimate of the positions of potential sources.
This can be accomplished by using source detection algorithms,
which are implemented in `~photutils.detection`.

After finding sources one would apply a clustering algorithm in order to label
the sources according to groups. Usually, those groups are formed by a
distance criterion, which is the case of the grouping algorithm proposed
by Stetson. In `~photutils.psf.DAOGroup`, we provide an implementation of
that algorithm.

Provided that the groups are available, the next step is to fit the sources
simultaneously for each group. This task can be done using an astropy
fitter, for instance, `~astropy.modeling.fitting.LevMarLSQFitter`.

After sources are fitted, they are subtracted from the given image
and, after fitting all sources, the residual image is analyzed by the finding
routine again in order to check if there exist any source which has not been
detected previously. This process goes on until no more sources are identified 
by the finding routine.

Basic Usage
^^^^^^^^^^^

The DAOPhotPSFPhotometry is the core class to implement the DAOPHOT algorithm
to perform PSF photometry in crowded fields.

It basically encapsulates the loop "FIND, GROUP, NSTAR, SUBTRACT" in one place
so that one can easily perform PSF photometry just by setting up a
DAOPhotPSFPhotometry object.

This class was implemented in such a way that it can be used as a callable
function. The basic idea is illustrated as follows::

    >>> # create a DAOPhotPSFPhotometry object
    >>> from photutils.psf import DAOPhotPSFPhotometry 
    >>> my_photometry = DAOPhotPSFPhotometry(find=my_finder, group=my_group,
                        bkg=my_bkg, psf=my_psf_model, fitter=my_fitter,
                        niters=5, fitshape=(7,7)) #doctest: +SKIP
    >>> # get photometry results
    >>> photometry_results, residual_image = my_photometry(image=my_image) # doctest: +SKIP

Where ``my_finder``, ``my_group``, and ``my_bkg`` may be any suitable callable
function. This approach allows one to customize every part of the photometry
process provided that their input/output are compatible with the input/ouput
expected by DAOPhotPSFPhotometry. See the API documentation for details.

Performing PSF Photometry
^^^^^^^^^^^^^^^^^^^^^^^^^

Let's take a look in a simple example with simulated stars whose PSF is
assumed to be Gaussian::
    
    >>> from photutils.datasets import make_random_gaussians
    >>> from photutils.datasets import make_noise_image
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
                 make_noise_image(tshape, type='poisson', mean=6.,
                                  random_state=1) +
                 make_noise_image(tshape, type='gaussian', mean=0.,
                                  stddev=2., random_state=1))

    >>> from photutils.detection import IRAFStarFinder
    >>> from photutils.psf import IntegratedGaussianPRF, DAOGroup
    >>> from photutils.background import MMMBackground
    >>> from photutils.background import MADStdBackgroundRMS
    >>> from astropy.modeling.fitting import LevMarLSQFitter
    >>> from astropy.stats import gaussian_sigma_to_fwhm

    >>> bkgrms = MADStdBackgroundRMS()
    >>> std = bkgrms(image)
    >>> iraffind = IRAFStarFinder(threshold=3.5*std,
                                  fwhm=sigma_psf*gaussian_sigma_to_fwhm,
                                  minsep_fwhm=0.01, roundhi=5.0, roundlo=-5.0,
                                  sharplo=0.0, sharphi=2.0)
    >>> daogroup = DAOGroup(2.0*sigma_psf*gaussian_sigma_to_fwhm)
    >>> mmm_bkg = MMMBackground()
    >>> psf_model = IntegratedGaussianPRF(sigma=sigma_psf)
    >>> fitter = LevMarLSQFitter()
    
    >>> from photutils.psf import DAOPhotPSFPhotometry

    >>> daophot_photometry = DAOPhotPSFPhotometry(find=iraffind, group=daogroup,
                                                  bkg=mmm_bkg, psf=psf_model,
                                                  fitter=LevMarLSQFitter(),
                                                  niters=2, fitshape=(11,11))
    >>> result_tab, residual_image = daophot_photometry(image=image)
    
    >>> from matplotlib import rcParams
    >>> import matplotlib.pyplot as plt
    >>> rcParams['image.cmap'] = 'viridis'
    >>> rcParams['image.aspect'] = 1  # to get images with square pixels
    >>> rcParams['figure.figsize'] = (20,10)
    >>> rcParams['image.interpolation'] = 'nearest'
    >>> rcParams['image.origin'] = 'lower'
    >>> rcParams['font.size'] = 14
    >>> plt.subplot(1, 2, 1)
    >>> plt.imshow(image)
    >>> plt.title('Simulated data')
    >>> plt.colorbar(orientation='horizontal', fraction=0.046, pad=0.04)
    >>> plt.subplot(1 ,2, 2)
    >>> plt.imshow(residual_image)
    >>> plt.title('Residual Image')
    >>> plt.colorbar(orientation='horizontal', fraction=0.046, pad=0.04)    
    >>> plt.show()

.. plot::
    
    from photutils.datasets import make_random_gaussians
    from photutils.datasets import make_noise_image
    from photutils.datasets import make_gaussian_sources

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
    from photutils.background import MMMBackground
    from photutils.background import MADStdBackgroundRMS
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

    daophot_photometry = DAOPhotPSFPhotometry(find=iraffind, group=daogroup,
                                          bkg=mmm_bkg, psf=psf_model,
                                          fitter=LevMarLSQFitter(),
                                          niters=2, fitshape=(11,11))
    result_tab, residual_image = daophot_photometry(image=image)
    
    from matplotlib import rcParams
    import matplotlib.pyplot as plt
    rcParams['image.cmap'] = 'viridis'
    rcParams['image.aspect'] = 1  # to get images with square pixels
    rcParams['figure.figsize'] = (20,10)
    rcParams['image.interpolation'] = 'nearest'
    rcParams['image.origin'] = 'lower'
    rcParams['font.size'] = 14

    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title('Simulated data')
    plt.colorbar(orientation='horizontal', fraction=0.046, pad=0.04)
    plt.subplot(1 ,2, 2)
    plt.imshow(residual_image)
    plt.title('Residual Image')
    plt.colorbar(orientation='horizontal', fraction=0.046, pad=0.04)    


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
