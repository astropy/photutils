Source Detection and Extraction
===============================


Introduction
------------

The `photutils` package provides methods to detect sources in an
astronomical image above a specified threshold similar to that used by
SExtractor.  It also provides two functions to detect stars in
astronomical images:

* :func:`~photutils.daofind`, based on the `DAOFIND <http://iraf.net/irafhelp.php?val=daofind&help=Help+Page>`_ algorithm

* :func:`~photutils.irafstarfind`, based on IRAF's `starfind <http://iraf.net/irafhelp.php?val=starfind&help=Help+Page>`_ algorithm


Source Extraction
-----------------
Source extraction refers to the detection of sources in an
astronomical images.  It also known as image segmentation or image
labeling in other python packages.


Getting Started
---------------

Create an image with a single 2D circular Gaussian source to represent
a star and find it in the image using :func:`~photutils.daofind`:

.. doctest-requires:: scipy, skimage

    >>> import numpy as np
    >>> import photutils
    >>> y, x = np.mgrid[-50:51, -50:51]
    >>> img = 100.0 * np.exp(-(x**2/5.0 + y**2/5.0))
    >>> tbl = photutils.daofind(img, threshold=1.0, fwhm=3.0)
    >>> tbl.pprint(max_width=-1)
     id xcentroid ycentroid   sharpness    roundness1 roundness2 npix sky  peak      flux          mag
     --- --------- --------- -------------- ---------- ---------- ---- --- ----- ------------- --------------
       1      50.0      50.0 0.440818817057        0.0        0.0 25.0 0.0 100.0 62.4702758896 -4.48918355985


Finding Stars in an Image
-------------------------

.. toctree:: findstars.rst


Reference/API
-------------

.. automodapi:: photutils.detection.core
    :no-heading:
.. automodapi:: photutils.detection.findstars
    :no-heading:
