Source detection and photometry (`photutils`)
=============================================

Introduction
------------

The `photutils` package is destined to implement functions for

* detecting sources on astronomical images
* estimating morphological parameters of those sources
  (e.g., centroid and shape parameters)
* performing photometry (both aperture and PSF)

Currently, only aperture photometry functions have been implemented.

Getting Started
---------------

.. note::
   Eventually this will contain an example showing object detection
   and photometry used in series. For now it just shows an aperture
   photometry function.

Given a list of source locations, sum flux in identical circular apertures:

  >>> import numpy as np
  >>> import photutils
  >>> data = np.ones((100, 100))
  >>> xc = [10., 20., 30., 40.]
  >>> yc = [10., 20., 30., 40.]
  >>> flux = photutils.aperture_circular(data, xc, yc, 3.)
  >>> flux
  array([ 28.04,  28.04,  28.04,  28.04])

Using `photutils`
-----------------

.. toctree::

    aperture.rst

Coordinate Convention
---------------------

In this module the coordinates are zero-indexed, meaning that `(x, y)
= (0., 0.)` corresponds to the center of the lower-left array element.
For example, the value of `data[0, 0]` is taken as the value over the
range `-0.5 < x <= 0.5`, `-0.5 < y <= 0.5`. The array is thus defined
over the range `-0.5 < x <= data.shape[1] - 0.5`, `-0.5 < y <=
data.shape[0] - 0.5`. (Note that this differs from the
SourceExtractor_ convention, in which the center of the lower-left
array element is `(1, 1)`.)


See Also
--------

TODO: references? 

.. automodapi:: photutils

.. _SourceExtractor: http://www.astromatic.net/software/sextractor
