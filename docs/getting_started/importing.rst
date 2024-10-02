.. doctest-skip-all

.. _importing:

Importing from Photutils
========================

Photutils is organized into subpackages covering different topics.
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
