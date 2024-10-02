.. doctest-skip-all

.. _api_reference:

*************
API Reference
*************


Importing from Photutils
========================

Photutils is a Python library that provides commonly-used tools and key
functionality for detecting and performing photometry of astronomical
sources. Photutils is organized into subpackages covering different
topics.

Importing the entire ``photutils`` package will not import the tools
in the subpackages. There are no tools available in the top-level
``photutils`` namespace. For example, the following will not work::

    >>> import photutils
    >>> aper = photutils.CircularAperture((10, 20), r=4)

The tools in each subpackage must be imported separately. For example,
to import the aperture photometry tools, use::

    >>> from photutils import aperture
    >>> aper = aperture.CircularAperture((10, 20), r=4)

or::

    >>> from photutils.aperture import CircularAperture
    >>> aper = CircularAperture((10, 20), r=4)


API Documentation
=================

The following is a list of all subpackages and their corresponding modules:

* `photutils.aperture`

* `photutils.background`

* `photutils.centroids`

* `photutils.datasets`

* `photutils.detection`

* `photutils.geometry`

* `photutils.isophote`

* `photutils.morphology`

* `photutils.profiles`

* `photutils.psf`

  - `photutils.psf.matching`

* `photutils.segmentation`

* `photutils.utils`

.. toctree::
    :maxdepth: 1
    :hidden:

    photutils.aperture <aperture_api>
    photutils.background <background_api>
    photutils.centroids <centroids_api>
    photutils.datasets <datasets_api>
    photutils.detection <detection_api>
    photutils.geometry <geometry_api>
    photutils.isophote <isophote_api>
    photutils.morphology <morphology_api>
    photutils.profiles <profiles_api>
    photutils.psf <psf_api>
    photutils.segmentation <segmentation_api>
    photutils.utils <utils_api>
