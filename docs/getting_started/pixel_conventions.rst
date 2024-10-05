Pixel Coordinate Conventions
============================

In Photutils, integer pixel coordinates are located at the center of
pixels, and they are 0-indexed, matching the Python 0-based indexing.
That means the first pixel is considered pixel ``0``, but pixel
coordinate ``0`` is the *center* of that pixel. Hence, the first pixel
spans pixel values ``-0.5`` to ``0.5``.

For a 2-dimensional array, ``(x, y) = (0, 0)`` corresponds to
the *center* of the bottom, leftmost array element. That means
the first pixel spans the ``x`` and ``y`` pixel values from
``-0.5`` to ``0.5``. Note that this differs from the IRAF, `FITS
WCS <https://fits.gsfc.nasa.gov/fits_wcs.html>`_, `ds9`_, and
`SourceExtractor`_ conventions, in which the center of the bottom,
leftmost array element is ``(x, y) = (1, 1)``.

Following Python indexing, two-dimensional arrays are indexed as
``image[yi, xi]``, with 0 being the first index. The ``xi`` (column)
index corresponds to the second (fast) array index and the ``yi`` (row)
index corresponds to the first (slow) index.

.. _SourceExtractor: https://sextractor.readthedocs.io/en/latest/
.. _ds9: http://ds9.si.edu/
