Source Morphological Properties
===============================

.. warning::
    `scikit-image`_ is required for some functionality.

.. _scikit-image: http://scikit-image.org/


`photutils.morphology` provides functions to estimate morphological
properties and the centroid of a source.  The
:func:`~photutils.morphology.data_properties` function calculates the
properties of a single source from a cutout image.  Please see
`~photutils.segmentation.SegmentProperties` for the list of properties
that are calculated.

If working with segmentation images, the
:func:`~photutils.segmentation.segment_properties` function calculates
the properties for all (or a specified subset) of segmented sources.
Please see `photutils.segmentation` for more details.


Centroiding a Source
--------------------

`photutils.morphology` also provides separate functions to calculate
the centroid of a source using three different methods:

    * :func:`~photutils.morphology.centroid_com`: Object center of
      mass determined from 2D image moments.

    * :func:`~photutils.morphology.centroid_1dg`: Fitting 1D Gaussians
      to the marginal x and y distributions of the data.

    * :func:`~photutils.morphology.centroid_2dg`: Fitting a 2D
      Gaussian to the 2D distribution of the data.


Getting Started
---------------


Reference/API
-------------

.. automodapi:: photutils.morphology
    :no-heading:
