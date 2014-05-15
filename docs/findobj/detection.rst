
Source Detection and Segmentation
=================================

Introduction
------------

The ``findobj`` package provides methods to detect sources in an
astronomical image above a specified signal-to-noise threshold similar
to that used by SExtractor.  It also provides methods to detect stars
in astronomical images using two methods:

* `DAOFIND <http://iraf.net/irafhelp.php?val=daofind&help=Help+Page>`_ algorithm
* IRAF's `starfind <http://iraf.net/irafhelp.php?val=starfind&help=Help+Page>`_ algorithm


Getting Started
---------------

Create an image with a single 2D circular Gaussian source to represent
a star and find it in the image using ``daofind``:

  >>> import numpy as np
  >>> import findobj
  >>> y, x = np.mgrid[-50:51, -50:51]
  >>> img = 100.0 * np.exp(-(x**2/5.0 + y**2/5.0))
  >>> tbl = findobj.daofind(img, 3.0, 1.0)
  >>> tbl.pprint(max_width=-1)
    id xcen ycen     sharp      round1 round2 npix sky  peak      flux          mag
   --- ---- ---- -------------- ------ ------ ---- --- ----- ------------- --------------
     1 50.0 50.0 0.440818817057    0.0    0.0 25.0 0.0 100.0 62.4702758896 -4.48918355985


Finding Stars in an Image
-------------------------

.. toctree:: findstars.rst


Reference/API
=============

.. automodapi:: findobj.detection
.. automodapi:: findobj.findstars

