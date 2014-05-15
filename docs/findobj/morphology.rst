
Source Morphological Shape Parameters
=====================================

The ``findobj`` package allows one to calculate morphological shape
parameters of an object.  These include:

* centroid
* eccentricity
* linear eccentricity
* major axis length
* minor axis length
* position angle
* covariance matrix


Centroiding an Object
=====================

The ``findobj`` package allows one to centroid objects using three
different methods:

* Object center of mass determined from 2D image moments.
* Fitting 1D Gaussians to the marginal x and y distributions of the data.
* Fitting a 2D Gaussian to the 2D distribution of the data.


Getting Started
===============


Reference/API
=============

.. automodapi:: findobj.morphology

