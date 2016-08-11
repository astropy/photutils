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

Basically, DAOPHOT works as follows. The first step is to estimate the sky
background. For this task, photutils provides several classes to compute
scalar and 2D backgrounds, see `~photutils.background`. The next step is
to find an initial estimate of the positions of potential sources. This can
be accomplished by using source detection algorithms, which are implemented
in `~photutils.detection`.

After finding sources one would apply a clustering algorithm in order to label
the sources according to groups. Usually, those groups are formed by a
distance criterion, which is the case of the grouping algorithm proposed
by Stetson. In `~photutils.psf.DAOGroup`, we provide an implementation of
this algorithm.

Provided that the groups are available, the next step is to fit the sources
simultaneously within each group. This task can be done using an astropy
fitter, for instance, `~astropy.modeling.fitting.LevMarLSQFitter`.

After sources are fitted, they are subtracted from the given image
and, after fitting all sources, the residual image is analyzed by the finding
routine again in order to check if there exist any source which has not been
fitted. This process goes on until no more sources are identified by the
finding routine.

Example Notebooks (online)
--------------------------

* `PSF photometry on artificial Gaussian stars in crowded fields <https://github.com/astropy/photutils-datasets/blob/master/notebooks/GaussianPSFPhot.ipynb>>`
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
