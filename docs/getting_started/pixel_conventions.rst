Pixel Coordinate Conventions
============================

Photutils uses 0-indexed pixel coordinates, consistent with standard
Python and NumPy indexing. In this convention, integer pixel coordinates
represent the *centers* of the pixels. For example, the center of the
first pixel is at coordinate ``0``, meaning that pixel spans the range
from ``-0.5`` to ``0.5``.

For a two-dimensional image, the pixel center at ``(x, y) = (0, 0)``
corresponds to the bottom-left array element. Consequently, this first
pixel spans from ``-0.5`` to ``0.5`` in both the ``x`` and ``y``
directions.

Because image data are standard NumPy arrays, they are accessed as
``image[yi, xi]``, where ``yi`` is the row index (the first, or slow,
array axis) and ``xi`` is the column index (the second, or fast, array
axis). It is important to note that this array indexing order (``[y,
x]``) is the reverse of the spatial coordinate order (``(x, y)``).

This 0-indexed convention differs from the `FITS WCS`_ standard, which
uses 1-based pixel coordinates. In the FITS convention, the center of
the bottom-left pixel is ``(x, y) = (1, 1)``. Software such as `ds9`_,
`SourceExtractor`_, and IRAF follow the FITS convention. Therefore, to
match coordinates from these tools with Photutils, you must subtract 1
from their ``x`` and ``y`` coordinates.

.. _FITS WCS: https://fits.gsfc.nasa.gov/fits_wcs.html
.. _ds9: http://ds9.si.edu/
.. _SourceExtractor: https://sextractor.readthedocs.io/en/latest/
