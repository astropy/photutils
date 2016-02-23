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


References
----------
`Spitzer PSF vs. PRF
<http://irsa.ipac.caltech.edu/data/SPITZER/docs/files/spitzer/PRF_vs_PSF.pdf>`_

`Kepler PSF calibration
<http://keplerscience.arc.nasa.gov/CalibrationPSF.shtml>`_

`The Kepler Pixel Response Function
<http://adsabs.harvard.edu/abs/2010ApJ...713L..97B>`_


Example Notebooks (online)
--------------------------

* `PSF photometry on artificial Gaussian stars <https://github.com/astropy/photutils-datasets/blob/master/notebooks/GaussianPSFPhot.ipynb>`_
* `PSF/PRF Photometry on Spitzer Data <https://github.com/astropy/photutils-datasets/blob/master/notebooks/PSFPhotometrySpitzer.ipynb>`_

Reference/API
-------------

.. automodapi:: photutils.psf
    :no-heading:
