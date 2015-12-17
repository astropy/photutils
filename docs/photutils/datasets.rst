.. _datasets:

Datasets (`photutils.datasets`)
===============================

.. currentmodule:: photutils.datasets

Introduction
------------

`photutils.datasets` gives easy access to a few example datasets
(mostly images, but also e.g. source catalogs or PSF models).

This is useful for the Photutils documentation, tests, and benchmarks,
but also for users that would like to try out Photutils functions or
implement new methods for Photutils or their own scripts.

Functions that start with ``load_*`` load data files from disk.  Very
small data files are bundled in the Photutils code repository and are
guaranteed to be available.  Mid-sized data files are currently
available from a separate `photutils-datasets`_ repository and loaded
into the Astropy cache on the user's machine on first load.

Functions that start with ``make_*`` generate simple simulated data
(e.g. Gaussian sources on flat background with Poisson or Gaussian
noise).  Note that there are other tools like `skymaker`_ that can
simulate much more realistic astronomical images.


Getting Started
---------------

To load an example image with `~photutils.datasets.load_star_image`::

    >>> from photutils import datasets
    >>> hdu = datasets.load_star_image()  # doctest: +REMOTE_DATA
    >>> print(hdu.data.shape)  # doctest: +REMOTE_DATA
    (1059, 1059)

``hdu`` is an `astropy.io.fits.ImageHDU` object and ``hdu.data`` is a
`numpy.array` object that you can analyse with Photutils.

Let's plot the image:

.. plot::
    :include-source:

     from photutils import datasets
     hdu = datasets.load_star_image()
     plt.imshow(hdu.data, origin='lower', cmap='gray')
     plt.tight_layout()
     plt.show()


Reference/API
-------------

.. automodapi:: photutils.datasets
    :no-heading:

.. _photutils-datasets: https://github.com/astropy/photutils-datasets/
.. _skymaker: http://www.astromatic.net/software/skymaker
