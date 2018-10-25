ObjectFinder
============

Existing code documented `here 
<https://photutils.readthedocs.io/en/stable/api/photutils.detection.StarFinderBase.html>`_
-- see the ``find_stars`` function for the basic API. ``Finder`` is a relatively
independent routine in the PSF fitting process, and as such it is documented within its
own documentation blocks, based on ``StarFinderBase``; it is documented here for
completeness.

The object which defines the detection of objects in an image. Subclass finders
may require additional parameters -- see below for the example of ``DAOStarFinder``. The
class is defined entirely by its ``find_stars`` function, with some initialization,
depending on the individual subclass.

``find_stars`` accepts ``data``, the two-dimensional image in which sources should be
found, and returns ``table``, an `~astropy.table.Table` list of sources which passed
the given detection criteria. For example, in `~photutils.detection.DAOStarFinder`
a source must have given ``sharpness`` and ``roundness`` within specified ranges for
acceptance as a detected source.

Parameters
----------

data : array_like
    The 2D image array in which the finder should detection sources.


Returns
-------

table : `~astropy.table.Table`
    The table of detected sources.


Example Usage
-------------

StarFinder currently implements two methods: DAOFind and IRAFFind. For example, daofind
can be run to find objects with FWHM of approximately 3 pixels with a peak 5-sigma above
the background::

    from photutils import DAOStarFinder
    daofind = DAOStarFinder(fwhm=3.0, threshold=5.*std)
    sources = daofind(data)
    