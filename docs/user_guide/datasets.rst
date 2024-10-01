.. _datasets:

Datasets (`photutils.datasets`)
===============================

Introduction
------------

`photutils.datasets` gives easy access to load or make a few example
datasets.  The datasets are mostly images, but they also include PSF
models and a source catalog.

These datasets are useful for the Photutils documentation, tests, and
benchmarks, but also for users that would like to try out or implement
new methods for Photutils.

Functions that start with ``load_*`` load data files from disk.  Very
small data files are bundled in the Photutils code repository and are
guaranteed to be available.  Mid-sized data files are currently
available from the `astropy-data`_ repository and loaded into the
Astropy cache on the user's machine on first load.

Functions that start with ``make_*`` generate simple simulated data
(e.g., Gaussian sources on a flat background with Poisson or Gaussian
noise).  Note that there are other tools like `skymaker`_ that can
simulate much more realistic astronomical images.


Getting Started
---------------

Let's load an example image of M67 with
:func:`~photutils.datasets.load_star_image`::

    >>> from photutils.datasets import load_star_image
    >>> hdu = load_star_image()  # doctest: +REMOTE_DATA
    >>> print(hdu.data.shape)  # doctest: +REMOTE_DATA
    (1059, 1059)

``hdu`` is a FITS `~astropy.io.fits.ImageHDU` object and ``hdu.data``
is a `~numpy.ndarray` object.

Let's plot the image:

.. plot::
    :include-source:

    import matplotlib.pyplot as plt
    from photutils.datasets import load_star_image

    hdu = load_star_image()
    plt.imshow(hdu.data, origin='lower', interpolation='nearest')
    plt.tight_layout()
    plt.show()


Reference/API
-------------

.. automodapi:: photutils.datasets
    :no-heading:

.. _astropy-data: https://github.com/astropy/astropy-data/
.. _skymaker: https://github.com/astromatic/skymaker
