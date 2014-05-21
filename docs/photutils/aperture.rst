Aperture photometry
===================

.. warning:: The aperture photometry API is currently *experimental*
   and may change in the future. For example, the functions currently
   accept `~numpy.ndarray` objects for the parameters ``data``,
   ``error`` and ``gain``. They may be changed to accept objects that
   encompass all these parameters for a single image.

The following functions are provided for different types of apertures:

.. currentmodule:: photutils


A Simple Example
----------------

Suppose there are 4 sources located at (10, 10), (20, 20), and (30,
30), (40, 40), in pixel coordinates. To sum the flux inside a circular
aperture of radius 3 pixels centered on each object,:

    >>> import numpy as np
    >>> import photutils
    >>> data = np.ones((100, 100))
    >>> xc = [10., 20., 30., 40.]
    >>> yc = [10., 20., 30., 40.]
    >>> apertures = [photutils.CircularAperture(3.)] * len(xc)
    >>> photutils.aperture_photometry(data, xc, yc, apertures)
    array([ 28.27433388,  28.27433388,  28.27433388,  28.27433388])

Since all the data values are 1, we expect the answer to equal the area of
a circle with the same radius, and it does:

    >>> print np.pi * 3. ** 2
    28.2743338823

Precision
---------

There are different ways to sum the pixels. By default, the method
used is ``'exact'``, wherein the exact intersection of the aperture with
each pixel is calculated. There are other options that are faster but
at the expense of less precise answers. For example,:

    >>> photutils.aperture_photometry(data, xc, yc, apertures,
    ...                               method='subpixel', subpixels=5)
    array([ 27.96,  27.96,  27.96,  27.96])

The result differs from the true value because this method subsamples
each pixel according to the keyword ``subpixels`` and either includes
or excludes each subpixel. The default value is ``subpixels=5``,
meaning that each pixel is divided into 25 subpixels. (This is the
method and subsampling used in SourceExtractor_.) The precision can be
increased by increasing ``subpixels`` but note that computation time
will be increased.

Aperture Photometry for Multiple Apertures
------------------------------------------

In the above example, suppose we wished to use a different radius
aperture for each object. Instead of setting ``r = 3.``, we could have
used a list specifying an aperture for each ``xc`` and ``yc`` (the length
of the aperture list must match that of ``xc`` and ``yc`` in this case):

  >>> r = [1., 2., 3., 4.]
  >>> apertures = []
  >>> for index in range(len(r)):
          apertures.append(photutils.CircularAperture(r[index]))
  >>> flux = photutils.aperture_photometry(data, xc, yc, apertures)
  >>> flux
  array([  3.14159265,  12.56637061,  28.27433388,  50.26548246])

Suppose instead that we wish to use 3 apertures of radius 3, 4, and 5
pixels on each source (each source gets the same 3 apertures):

  >>> r = [3., 4., 5.]
  >>> apertures = []
  >>> for index in range(len(r)):
          apertures.append([photutils.CircularAperture(r[index])] * len(xc))
  >>> flux = photutils.aperture_photometry(data, xc, yc, apertures)
  >>> flux
  array([[ 28.27433388,  28.27433388,  28.27433388,  28.27433388],
         [ 50.26548246,  50.26548246,  50.26548246,  50.26548246],
         [ 78.53981634,  78.53981634,  78.53981634,  78.53981634]])


Finally, suppose we wish to use a different set of 3 apertures for each source:

  >>> r = np.asarray([[3., 4., 5., 6.], [4., 5., 6., 7.], [5., 6., 7., 8.]])
  >>> apertures = np.empty(r.shape, dtype=object)
  >>> for index in np.ndindex(r.shape):
          apertures[index] = photutils.CircularAperture(r[index])
  >>> flux = photutils.aperture_photometry(data, xc, yc, apertures)
  >>> flux
  array([[  28.27433388,   50.26548246,   78.53981634,  113.09733553],
         [  50.26548246,   78.53981634,  113.09733553,  153.93804003],
         [  78.53981634,  113.09733553,  153.93804003,  201.06192983]])

These examples illustrate that the ``apertures`` parameter can be an array
of up to two dimensions where the "fast" (or trailing) dimension corresponds
to different objects, and the "slow" (or "leading") dimension corresponds to
multiple apertures per object. The ``apertures`` parameter obeys
broadcasting rules in that the trailing dimension can either be equal to the
number of objects (``len(xc)``) or 1. If 1, the array is effectively
broadcast so that the trailing dimension matches the number of objects.

Other aperture photometry functions have multiple parameters
specifying the apertures. For example, for elliptical apertures, one
must specify ``a``, ``b``, and ``theta``:

  >>> a = 5.
  >>> b = 3.
  >>> theta = np.pi / 4.
  >>> apertures = [photutils.EllipticalAperture(a, b, theta)] * len(xc)
  >>> flux = photutils.aperture_photometry(data, xc, yc, apertures)
  >>> flux
  array([ 47.1238898,  47.1238898,  47.1238898,  47.1238898])

One may prefer to have multiple apertures. For example, to use 4 ellipses of
different sizes but with the same position angle, we could do:

 >>> a = [5., 6., 7., 8.]
 >>> b = [3., 4., 5., 6.]
 >>> theta = np.pi / 4.
 >>> apertures = []
 >>> for index in range(len(a)):
         apertures.append(photutils.EllipticalAperture(a[index], b[index], theta))
 >>> flux = photutils.aperture_photometry(data, xc, yc, apertures)
 >>> flux
 array([  47.1238898 ,   75.39822369,  109.95574288,  150.79644737])


Background Subtraction
----------------------

So that the functions are as flexible as possible, background
subtraction is left up to the user or calling function.

* *Global background subtraction*

  If ``bkg`` is an array representing the background level of the data
  (determined in an external function), simply do

    >>> flux = photutils.aperture_photometry(data - bkg, xc, yc, apertures)

* *Local background subtraction*

  Suppose we want to estimate the local background level around each pixel
  with a circular annulus of inner radius 6 pixels and outer radius 8 pixels:

    >>> apertures = [photutils.CircularAperture(3.)] * len(xc)
    >>> rawflux = photutils.aperture_photometry(data, xc, yc, apertures)
    >>> annulus_apertures = [photutils.CircularAnnulus(6., 8.)] * len(xc)
    >>> bkgflux = photutils.aperture_photometry(data, xc, yc, annulus_apertures)
    >>> aperture_area = np.pi * 3 ** 2
    >>> annulus_area = np.pi * (8 ** 2 - 6 ** 2)
    >>> flux = rawflux - bkgflux * aperture_area / annulus_area
    >>> flux
    array([ -1.77635684e-14,  -1.77635684e-14,  -1.77635684e-14,
            -1.77635684e-14])


  (The result differs from 0 due to inclusion or exclusion of
  subpixels in the apertures.)

Error Estimation
----------------

If, and only if, the ``error`` keyword is specified, the return value
will be ``(flux, fluxerr)`` rather than just ``flux``. ``fluxerr`` is an
array of the same shape as ``flux``, specifying the uncertainty in each
corresponding flux value.

For example, suppose we have previously calculated the error on each
pixel's value and saved it in the array ``data_error``:

  >>> data_error = 0.1 * data  # (100 x 100 array)
  >>> flux, fluxerr = photutils.aperture_photometry(data, xc, yc, apertures,
  >>>                                               error=data_error)
  >>> fluxerr
  array([ 0.53173616,  0.53173616,  0.53173616,  0.53173616])

``fluxerr`` is given by

.. math:: \Delta F = \sqrt{ \sum_i \sigma_i^2}

where :math:`\sigma` is the given error array and the sum is over all
pixels in the aperture.

In the cases above, it is assumed that the ``error`` parameter specifies
the *full* error (either it includes Poisson noise due to individual
sources or such noise is irrelevant). However, it is often the case
that one has previously calculated a smooth "background error" array
which by design doesn't include increased noise on bright pixels. In
such a case, we wish to explicitly include Poisson noise from the
source itself. Specifying the ``gain`` parameter does this. For example,
suppose we have a function ``background()`` that calculates the
position-dependent background level and variance of our data:

  >>> myimagegain = 1.5
  >>> sky_level, sky_sigma = background(data)  # function returns two arrays
  >>> flux, fluxerr = photutils.aperture_photometry(data - sky_level, xc,
  >>>                                               yc, apertures,
  >>>                                               error=sky_sigma,
  >>>                                               gain=myimagegain)

In this case, and indeed whenever ``gain`` is not `None`, then ``fluxerr``
is given by

  .. math:: \Delta F = \sqrt{\sum_i (\sigma_i^2 + f_i / g_i)}

where :math:`f_i` is the value of the data (``data - sky_level`` in this
case) at each pixel and :math:`g_i` is the value of the gain at each
pixel.

.. note::

   In cases where the error and gain arrays are slowly varying across
   the image, it is not necessary to sum the error from every pixel in
   the aperture individually. Instead, we can approximate the error as
   being roughly constant across the aperture and simply take the
   value of :math:`\sigma` at the center of the aperture. This can be
   done by setting the keyword ``pixelwise_errors=False``. This saves
   some computation time. In this case the flux error is

   .. math:: \Delta F = \sqrt{A \sigma^2 + F / g}

   where :math:`\sigma` and :math:`g` are the error and gain at the
   center of the aperture, and :math:`A` is the area of the
   aperture. :math:`F` is the *total* flux in the aperture.


Pixel Masking
-------------

If the ``mask`` keyword is specified, masked pixels are treated in the
following way:

* Find the pixel the same distance from the object center,
  but 180 degrees away ("reflected" through the center).
* If this pixel is unmasked, set the masked pixel to its value.
* If this pixel is also masked, set the masked pixel to 0.


Extension to arbitrary apertures using `~photutils.Aperture` objects
--------------------------------------------------------------------

The photometry function, `~photutils.aperture_photometry`, performs
aperture photometry in arbitrary apertures. This function accepts
`Aperture`-derived objects, such as
`~photutils.CircularAperture`. (The wrappers handle creation of the
`~photutils.Aperture` objects or arrays thereof.) This makes it simple
to extend functionality: a new type of aperture photometry simply
requires the definition of a new `~photutils.Aperture` subclass.

All `~photutils.Aperture` subclasses must implement only two methods,
``encloses(xx, yy)`` and ``extent()``. They can optionally implement a
third method, ``area()``.

* ``encloses(xx, yy)``: Takes two 2-d arrays of x and y positions
  *relative to the object center* and returns a bool array indicating
  whether each position is in the aperture.
* ``extent()``: Returns the maximum extent of the aperture, (x_min,
  x_max, y_min, y_max) *relative to the object center*. This is used
  to determine the portion of the data array to subsample (if
  necessary).
* ``area()``: If convenient to calculate, this returns the area of the
  aperture.  This speeds computation in certain situations (such as a
  scalar error). If not provided, ``~photutils.aperture_photometry`` will
  estimate the area using the result of ``encloses(xx, yy)``.

Note that all x and y coordinates here refer to the fast and slow
(second and first) axis of the data array respectively. See
:ref:`coordinate-conventions`.

See Also
--------

1. `IRAF's APPHOT specification [PDF]`_ (Sec. 3.3.5.8 - 3.3.5.9)

2. `SourceExtractor Manual [PDF]`_ (Sec. 9.4 p. 36)

.. _SourceExtractor: http://www.astromatic.net/software/sextractor
.. _SourceExtractor Manual [PDF]: https://www.astromatic.net/pubsvn/software/sextractor/trunk/doc/sextractor.pdf
.. _IRAF's APPHOT specification [PDF]: http://iraf.net/irafdocs/apspec.pdf
