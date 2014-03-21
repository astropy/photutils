Source identification and segmentation
======================================

Introduction
------------

The ``findobj`` package allows one to detect stars in astronomical images
using three methods:

    * `DAOFIND <http://iraf.net/irafhelp.php?val=daofind&help=Help+Page>`_ algorithm

    * IRAF's `starfind <http://iraf.net/irafhelp.php?val=starfind&help=Help+Page>`_ algorithm

    * fit empirical PSFs (work in progress)

It is likely that ``findobj`` will eventually be merged into
`photutils <http://photutils.readthedocs.org/en/latest/>`_ or
`astropy`_.

.. warning::
    ``findobj`` requires `astropy`_ version 0.3.0 (or newer).

.. warning::
    ``findobj`` is currently a work-in-progress, and thus it is
    possible there will be significant API changes in later versions.


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


Using findobj
-------------

.. toctree:: findstars.rst


Centroiding an Object
=====================

The ``findobj`` package allows one to centroid objects using three
different methods:

    * Object center of mass determined from 2D image moments.

    * Fitting 1D Gaussians to the marginal x and y distributions of the data.

    * Fitting a 2D Gaussian to the 2D distribution of the data.


Object Shape Parameters
=======================

The ``findobj`` package allows one to calculate morphological shape
parameters of an object.  These include:

    * centroid
    * eccentricity
    * linear eccentricity
    * major axis length
    * minor axis length
    * position angle
    * covariance matrix


Reference/API
=============

.. automodapi:: findobj
