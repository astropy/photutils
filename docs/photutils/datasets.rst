.. _datasets:

Datasets (`photutils.datasets`)
===============================

.. currentmodule:: photutils.datasets

Introduction
============

`photutils.datasets` gives easy access to a few example datasets
(mostly images, but also e.g. source catalogs or PSF models).

This is useful for the `photutils` documentation, tests and benchmarks,
but also for users that would like to try out `photutils` functions
or implement new methods for `photutils` or their own scripts.

Functions that start with ``load_*`` load data files from disk.
Very small data files are bundled in the `photutils`_ code repository
and are guaranteed to be available.
Mid-sized data files are currently available from a separate `photutils-datasets`_ repository
and loaded into the Astropy cache on the user's machine on first load.

Functions that start with ``make_*`` generate simple simulated data
(e.g. Gaussian sources on flat background with Poisson or Gaussian noise).
Note that there are other tools like `skymaker`_
that can simulate much more realistic astronomical images.


Getting Started
===============

TODO: write me

Reference/API
=============

.. automodapi:: photutils.datasets
    :no-heading:

.. _photutils: https://github.com/astropy/photutils/
.. _photutils-datasets: https://github.com/astropy/photutils-datasets/
.. _skymaker: http://www.astromatic.net/software/skymaker
