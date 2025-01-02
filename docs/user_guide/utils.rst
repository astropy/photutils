Utility Functions (`photutils.utils`)
=====================================

Introduction
------------

The `photutils.utils` package contains general-purpose utility functions
that do not fit into any of the other subpackages.

Some functions and classes of note include:

* :class:`~photutils.utils.ImageDepth`: Class to calculate the limiting
  flux and magnitude of an image by placing random circular apertures on
  blank regions.

* :class:`~photutils.utils.ShepardIDWInterpolator`: Class to perform
  inverse distance weighted (IDW) interpolation.

* :func:`~photutils.utils.calc_total_error`: Function to calculate the
  total error in an image by combining a background-only error array with
  the source Poisson error.

* :func:`~photutils.utils.make_random_cmap`: Function to create a
  colormap consisting of random muted colors. This type of colormap is
  useful for plotting segmentation images.


API Reference
-------------

:doc:`../reference/utils_api`
