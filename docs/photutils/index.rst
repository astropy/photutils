Source detection and photometry (`photutils`)
=============================================

Introduction
------------

The `photutils` package is destined to implement functions for

* detecting sources on astronomical images
* estimating morphological parameters of those sources
  (e.g., centroid and shape parameters)
* performing photometry (both aperture and PSF)

.. note::

    It is possible that `photutils` will eventually be merged into ``astropy`` as
    ``astropy.photometry``.

.. note::

    `photutils` requires `numpy <http://www.numpy.org/>`__ and
    `astropy <http://www.astropy.org/>`__ to be installed.
    Some functionality is only available if `scipy <http://www.scipy.org/>`__ or
    `scikit-image <http://scikit-image.org/>`__ are installed, users are
    encouraged to install those optional dependencies.

Getting Started
---------------

.. note::
   Eventually this will contain an example showing object detection
   and photometry used in series. For now it just shows an aperture
   photometry function.

Given a list of source locations, sum flux in identical circular apertures:

  >>> import numpy as np
  >>> from photutils import CircularAperture, aperture_photometry
  >>> data = np.ones((100, 100))
  >>> xc = [10., 20., 30., 40.]
  >>> yc = [10., 20., 30., 40.]
  >>> positions = zip(xc, yc)
  >>> apertures = CircularAperture(positions, 3.)
  >>> flux = aperture_photometry(data, apertures)
  >>> flux
  array([ 28.27433388,  28.27433388,  28.27433388,  28.27433388])


Using `photutils`
-----------------

.. toctree::
    :maxdepth: 2

    aperture.rst
    psf.rst
    datasets.rst
    detection.rst
    morphology.rst
    utils.rst

.. toctree::
  :maxdepth: 1

  high-level_API.rst


.. note::
   We also have a series of IPython notebooks that demonstrate how to use photutils.
   You can view them online `here <http://nbviewer.ipython.org/github/astropy/photutils-datasets/tree/master/notebooks/>`__
   or download them `here <https://github.com/astropy/photutils-datasets>`__ if you'd like to execute them on your machine.
   Contributions welcome!


.. _coordinate-conventions:

Coordinate Conventions in `photutils`
-------------------------------------

In this module the coordinates are zero-indexed, meaning that ``(x, y)
= (0., 0.)`` corresponds to the center of the lower-left array
element.  For example, the value of ``data[0, 0]`` is taken as the
value over the range ``-0.5 < x <= 0.5``, ``-0.5 < y <= 0.5``. Note
that this differs from the SourceExtractor_ convention, in which the
center of the lower-left array element is ``(1, 1)``.

The ``x`` coordinate corresponds to the second (fast) array index and
the ``y`` coordinate corresponds to the first (slow) index. So
``data[y, x]`` gives the value at coordinates (x, y). Along with the
zero-indexing, this means that the array is defined over the
coordinate range ``-0.5 < x <= data.shape[1] - 0.5``,
``-0.5 < y <= data.shape[0] - 0.5``.


.. _SourceExtractor: http://www.astromatic.net/software/sextractor
