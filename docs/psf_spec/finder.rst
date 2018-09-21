ObjectFinder
============

Existing code documented at
https://photutils.readthedocs.io/en/stable/api/photutils.detection.StarFinderBase.html
- see the ``find_stars`` function for the basic API.

The object which defines the detection of objects in an image. Subclass finders
may require additional parameters -- see below for the example of ``DAOStarFinder``.

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
the background:
::
    from photutils import DAOStarFinder
    daofind = DAOStarFinder(fwhm=3.0, threshold=5.*std)
    sources = daofind(data)
    