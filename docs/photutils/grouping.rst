Grouping Algorithms
===================

Introduction
------------

In Point Spread Function (PSF) photometry, a grouping algorithm is used
to decide whether two or more stars belong to the same group, i. e., whether
there are any pixels whose counts are due to the linear combination of counts
from two or more sources.

DAOPHOT GROUP
-------------

In his seminal paper, Stetson provided a grouping algorithm which is able to
decide whether or not a given star is influencing the brightness of any other
star. This goal is achieved by means of a variable called
"critical separation", which Stetson defined as being the distance such that
any two stars separated by less than it would be interfering in each other
counts.

Stetson also gives intutive reasoning to suggest that the critical separation
may be defined as the product of fwhm with some positive integer.

Grouping Sources
^^^^^^^^^^^^^^^^

Photutils provides an implementation of DAOPHOT GROUP in the
:class:`~photutils.DAOGroup` class. Let's take a look at a simple example::

    >>> from photutils.psf import DAOGroup
    >>> fwhm = 2.0
    >>> image = ...
    >>> daogroup = DAOGroup(crit_separation=1.5*fwhm)
    >>> apert = CircularAperture((daogroup['x_0'], daogroup['y_0']), r=fwhm)
    >>> apert.plot(lw=1.5, alpha=0.5)
    >>> plt.imshow(image, origin='lower', interpolation='None')
