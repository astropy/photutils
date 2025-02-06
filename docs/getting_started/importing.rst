.. doctest-skip-all

.. _importing:

Importing from Photutils
========================

Photutils is organized into subpackages covering different topics.
Importing only ``photutils`` will not import the tools in the
subpackages. There are no tools available in the top-level ``photutils``
namespace. For example, the following will not work::

    >>> import photutils
    >>> aper = photutils.CircularAperture((10, 20), r=4)

The tools in each subpackage must be imported separately. For example,
to import the aperture photometry tools, use::

    >>> from photutils import aperture
    >>> aper = aperture.CircularAperture((10, 20), r=4)

or::

    >>> from photutils.aperture import CircularAperture
    >>> aper = CircularAperture((10, 20), r=4)


.. warning::

    *Do not import from specific modules of packages.* This is
    unnecessary and the internal organization of the package may change
    without notice. All public tools are available in the package
    top-level namespace. For example, do not import from the ``circle``
    module within the ``aperture`` package::

        >>> from photutils.aperture.circle import CircularAperture
        >>> aper = CircularAperture((10, 20), r=4)


.. warning::

    Modules, functions, classes, methods, and attributes whose names
    begin with a leading underscore are considered private objects and
    should not be imported or accessed. If a module name in a package
    begins with a leading underscore, then none of its members are
    public, regardless of whether they begin with a leading underscore.

    Private objects are not intended for public use and may change
    without notice.
