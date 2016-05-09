Overview
========

Introduction
------------

Photutils contains functions for:

* estimating the background and background rms in astronomical images
* detecting sources in astronomical images
* estimating morphological parameters of those sources (e.g., centroid
  and shape parameters)
* performing aperture and PSF photometry

The code and the documentation are available at the following links:

* Code: https://github.com/astropy/photutils
* Issue Tracker: https://github.com/astropy/photutils/issues
* Documentation: https://photutils.readthedocs.io/


.. _coordinate-conventions:

Coordinate Conventions
----------------------

In Photutils, pixel coordinates are zero-indexed, meaning that ``(x,
y) = (0, 0)`` corresponds to the center of the lowest, leftmost array
element.  This means that the value of ``data[0, 0]`` is taken as the
value over the range ``-0.5 < x <= 0.5``, ``-0.5 < y <= 0.5``.  Note
that this differs from the SourceExtractor_, IRAF_, FITS, and ds9_
conventions, in which the center of the lowest, leftmost array element
is ``(1, 1)``.

The ``x`` (column) coordinate corresponds to the second (fast) array
index and the ``y`` (row) coordinate corresponds to the first (slow)
index.  ``data[y, x]`` gives the value at coordinates (x, y).  Along
with zero-indexing, this means that an array is defined over the
coordinate range ``-0.5 < x <= data.shape[1] - 0.5``, ``-0.5 < y <=
data.shape[0] - 0.5``.

.. _SourceExtractor: http://www.astromatic.net/software/sextractor
.. _IRAF: http://iraf.noao.edu/
.. _ds9: http://ds9.si.edu/


Bundled Datasets
----------------

In this documentation, we use example `datasets <datasets.html>`_
provided by calling functions such as
:func:`~photutils.datasets.load_star_image`.  This function returns an
Astropy :class:`~astropy.io.fits.ImageHDU` object, and is equivalent
to doing:

.. doctest-skip::

    >>> from astropy.io import fits
    >>> hdu = fits.open('dataset.fits')[0]

where the ``[0]`` accesses the first HDU in the FITS file.


Contributors
------------

For the complete list of contributors please see the `Photutils
contributors page on Github
<https://github.com/astropy/photutils/graphs/contributors>`_.
