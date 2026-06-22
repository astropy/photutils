.. doctest-skip-all

.. _importing:

Importing from Photutils
========================

**Photutils** functionality is organized into subpackages that must be
imported explicitly. Importing only the top-level ``photutils`` package
will not expose its tools. For example, the following code will fail::

    >>> import photutils
    >>> aper = photutils.CircularAperture((10, 20), r=4)
    AttributeError: module 'photutils' has no attribute 'CircularAperture'

Instead, you must import tools from their specific subpackage. For
example, to use the aperture photometry tools, you can import a class
directly::

    >>> from photutils.aperture import CircularAperture
    >>> aper = CircularAperture((10, 20), r=4)

Alternatively, you can import the subpackage itself::

    >>> from photutils import aperture
    >>> aper = aperture.CircularAperture((10, 20), r=4)


.. warning::

    **Do not import from a subpackage's internal modules.** This is
    unnecessary and the internal organization of subpackages may change
    without notice. All public tools are available directly at the
    **subpackage level**. For example, do **not** import from the
    internal ``circle`` module within the ``aperture`` subpackage::

        >>> # Do not import internal modules
        >>> from photutils.aperture.circle import CircularAperture
        >>> aper = CircularAperture((10, 20), r=4)


.. warning::

    Modules, functions, classes, methods, and attributes whose names
    begin with a leading underscore are considered private objects and
    should not be imported or accessed. If a module or subpackage name
    begins with a leading underscore, then none of its members are
    public, regardless of whether they begin with a leading underscore.

    **Private objects are not intended for public use and may change
    without notice.**
