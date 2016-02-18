PSF Photometry
==============

The `photutils.psf` module contains tools for PSF photometry.

.. warning::
    The PSF photometry API is currently *experimental* and may change
    in the future.  For example, the functions currently accept
    `~numpy.ndarray` objects for the ``data`` parameters, but they may
    be changed to accept `astropy.nddata` objects.


Examples
--------

* :doc:`../notebooks/gaussian_psfs`

Reference/API
-------------

.. automodapi:: photutils.psf
    :no-heading:
