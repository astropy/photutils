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

    It is possible that `photutils` will eventually be merged into
    ``astropy`` as ``astropy.photometry``.

.. note::

    `photutils` requires `numpy <http://www.numpy.org/>`__ and
    `astropy <http://www.astropy.org/>`__ to be installed.
    Some functionality is only available if `scipy <http://www.scipy.org/>`__ or
    `scikit-image <http://scikit-image.org/>`__ are installed, users are
    encouraged to install those optional dependencies.

Getting Started
---------------

Given a data array, the following example uses `photutils` to find sources
and perform aperture photometry on them.

The dataset in this example is `~photutils.datasets.load_star_image`:

  >>> import numpy as np
  >>> from photutils import datasets
  >>> hdu = datasets.load_star_image()   # doctest: +REMOTE_DATA
  >>> image = hdu.data[500:700, 500:700]   # doctest: +REMOTE_DATA
  >>> image -= np.median(image)   # doctest: +REMOTE_DATA

In this example we assume that the data is background-subtracted.
`photutils` supports different source detection algorithms, this example
uses `~photutils.daofind`. The parameters of the detected sources are returned
in a `~astropy.table.Table`:

  >>> from astropy.stats import median_absolute_deviation as mad
  >>> from photutils import daofind
  >>> bkg_sigma = 1.48 * mad(image)   # doctest: +REMOTE_DATA
  >>> sources = daofind(image, fwhm=4.0, threshold=3*bkg_sigma)   # doctest: +REMOTE_DATA

Given the list of source location, summing the pixel values in identical circular
apertures. The result is returned in a `~astropy.table.Table`, with two
columns named ``'flux'`` and ``'fluxerr'``:

  >>> from photutils import CircularAperture, CircularAnnulus, aperture_photometry
  >>> positions = zip(sources['xcen'], sources['ycen'])   # doctest: +REMOTE_DATA
  >>> apertures = CircularAperture(positions, 4.)   # doctest: +REMOTE_DATA
  >>> fluxtable = aperture_photometry(image, apertures)   # doctest: +REMOTE_DATA

And now check which one is the fainest and brightest source in this dataset:

  >>> faintest = (apertures.positions[fluxtable['flux'].argmin()],
  ...             fluxtable['flux'].min())   # doctest: +REMOTE_DATA
  >>> print(faintest)   # doctest: +REMOTE_DATA
  (array([ 118.71993103,   66.80723769]), -342.91175178365006)
  >>> brightest = (apertures.positions[fluxtable['flux'].argmax()],
  ...             fluxtable['flux'].max())   # doctest: +REMOTE_DATA
  >>> print(brightest)   # doctest: +REMOTE_DATA
  (array([ 57.85429092,  99.22152913]), 387408.0358707984)


Let's plot the image and the apertures. The apertures of all the
sources found in this image are marked with gray circles. The brightest source is
marked with red while the faintest is with blue:

.. doctest-skip::

  >>> import matplotlib.patches as patches
  >>> import matplotlib.pylab as plt
  >>> plt.imshow(image, cmap='gray_r', origin='lower')
  >>> apertures.plot(color='gray', lw=1.5)
  >>> plt.gca().add_patch(patches.Circle(faintest[0], apertures.r, color='blue',
  ...                                    fill=False, lw=1.5))
  >>> plt.gca().add_patch(patches.Circle(brightest[0], apertures.r, color='red',
  ...                                    fill=False, lw=1.5))


.. plot::

  import numpy as np
  import matplotlib.pylab as plt
  import matplotlib.patches as patches
  from astropy.stats import median_absolute_deviation as mad
  from photutils import datasets, daofind, CircularAperture, aperture_photometry
  hdu = datasets.load_star_image()
  image = hdu.data[500:700, 500:700]
  image -= np.median(image)
  bkg_sigma = 1.48 * mad(image)
  sources = daofind(image, fwhm=4.0, threshold=3*bkg_sigma)
  positions = zip(sources['xcen'], sources['ycen'])
  apertures = CircularAperture(positions, 4.)
  fluxtable = aperture_photometry(image, apertures)
  faintest = (apertures.positions[fluxtable['flux'].argmin()], fluxtable['flux'].min())
  brightest = (apertures.positions[fluxtable['flux'].argmax()], fluxtable['flux'].max())
  plt.imshow(image, cmap='gray_r', origin='lower')
  apertures.plot(color='gray', lw=1.5)
  plt.gca().add_patch(patches.Circle(faintest[0], apertures.r, color='blue',
                                     fill=False, lw=1.5))
  plt.gca().add_patch(patches.Circle(brightest[0], apertures.r, color='red',
                                     fill=False, lw=1.5))


Using `photutils`
-----------------

.. toctree::
    :maxdepth: 2

    aperture.rst
    psf.rst
    datasets.rst
    detection.rst
    morphology.rst
    geometry.rst
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
