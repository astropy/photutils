Aperture photometry
===================

.. warning:: The aperture photometry API is currently *experimental*
   and will change in the near future.

The following functions are provided for different types of apertures:

.. currentmodule:: photutils


A Simple Example
----------------

Suppose there are 4 sources located at (10, 10), (20, 20), and (30, 30),
(40, 40), in pixel coordinates. To sum the pixel values (flux) inside a
circular aperture of radius 3 pixels centered on each object:

    >>> import numpy as np
    >>> from photutils import CircularAperture, aperture_photometry
    >>> data = np.ones((100, 100))
    >>> xc = [10., 20., 30., 40.]
    >>> yc = [10., 20., 30., 40.]
    >>> positions = zip(xc, yc)
    >>> apertures = CircularAperture(positions, 3.)
    >>> aperture_photometry(data, apertures)
    <Table rows=4 names=('aperture_sum') units=('')>
    array([(28.274333882308134,), (28.274333882308134,), (28.274333882308134,),
           (28.274333882308134,)],
          dtype=[('aperture_sum', '<f8')])

The results are returned in a `~astropy.table.Table` with a column, named
``'aperture_sum'``.
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

    >>> aperture_photometry(data, apertures,
    ...                     method='subpixel', subpixels=5)
    <Table rows=4 names=('aperture_sum') units=('')>
    array([(27.959999999999997,), (27.959999999999997,), (27.959999999999997,),
           (27.959999999999997,)],
          dtype=[('aperture_sum', '<f8')])

The result differs from the true value because this method subsamples
each pixel according to the keyword ``subpixels`` and either includes
or excludes each subpixel. The default value is ``subpixels=5``,
meaning that each pixel is divided into 25 subpixels. (This is the
method and subsampling used in SourceExtractor_.) The precision can be
increased by increasing ``subpixels`` but note that computation time
will be increased.

Aperture Photometry for Multiple Apertures
------------------------------------------

Currently the `~photutils.Aperture` objects only support single radius
apertures. As a workaround one may loop over different apertures.

Suppose that we wish to use 3 apertures of radius 3, 4, and 5
pixels on each source (each source gets the same 3 apertures):

  >>> r = [3., 4., 5.]
  >>> flux = []
  >>> for radius in r:
  ...     flux.append(aperture_photometry(data, CircularAperture(positions, radius)))

Now we have 3 separate tables containing the photometry results, one for
each aperture. One may use `~astropy.table.hstack` to stack them into one
`~astropy.table.Table`:

  >>> from astropy.table import hstack
  >>> fluxtable = hstack(flux)
  >>> fluxtable    # doctest: +FLOAT_CMP
  <Table rows=4 names=('aperture_sum_1','aperture_sum_2','aperture_sum_3') units=('','','')>
  array([(28.274333882308134, 50.26548245743669, 78.53981633974482),
         (28.274333882308134, 50.26548245743669, 78.53981633974482),
         (28.274333882308134, 50.26548245743669, 78.53981633974482),
         (28.274333882308134, 50.26548245743669, 78.53981633974482)],
        dtype=[('aperture_sum_1', '<f8'), ('aperture_sum_2', '<f8'), ('aperture_sum_3', '<f8')])


Other aperture photometry functions have multiple parameters
specifying the apertures. For example, for elliptical apertures, one
must specify ``a``, ``b``, and ``theta``:

  >>> from photutils import EllipticalAperture
  >>> a = 5.
  >>> b = 3.
  >>> theta = np.pi / 4.
  >>> apertures = EllipticalAperture(positions, a, b, theta)
  >>> fluxtable = aperture_photometry(data, apertures)
  >>> fluxtable   # doctest: +FLOAT_CMP
  <Table rows=4 names=('aperture_sum') units=('')>
  array([(47.1238898038469,), (47.1238898038469,), (47.1238898038469,),
         (47.1238898038469,)],
        dtype=[('aperture_sum', '<f8')])


Again, for multiple apertures one should loop over them.

 >>> a = [5., 6., 7., 8.]
 >>> b = [3., 4., 5., 6.]
 >>> theta = np.pi / 4.
 >>> flux = []
 >>> for index in range(len(a)):
 ...     flux.append(aperture_photometry(data, EllipticalAperture(positions, a[index], b[index], theta)))
 >>> fluxtable = hstack(flux)
 >>> fluxtable   # doctest: +FLOAT_CMP
 <Table rows=4 names=('aperture_sum_1','aperture_sum_2','aperture_sum_3','aperture_sum_4') units=('','','','')>
 array([ (47.1238898038469, 75.39822368615505, 109.9557428756428, 150.7964473723101),
         (47.1238898038469, 75.39822368615505, 109.9557428756428, 150.7964473723101),
         (47.1238898038469, 75.39822368615505, 109.9557428756428, 150.7964473723101),
         (47.1238898038469, 75.39822368615505, 109.9557428756428, 150.7964473723101)],
          dtype=[('aperture_sum_1', '<f8'), ('aperture_sum_2', '<f8'), ('aperture_sum_3', '<f8'), ('aperture_sum_4', '<f8')])


Background Subtraction
----------------------

So that the functions are as flexible as possible, background
subtraction is left up to the user or calling function.

* *Global background subtraction*

  If ``bkg`` is an array representing the background level of the data
  (determined in an external function), simply do

    >>> fluxtable = aperture_photometry(data - bkg, apertures)  # doctest: +SKIP

* *Local background subtraction*

  Suppose we want to estimate the local background level around each pixel
  with a circular annulus of inner radius 6 pixels and outer radius 8 pixels:

    >>> from photutils import CircularAnnulus
    >>> apertures = CircularAperture(positions, 3.)
    >>> rawflux_table = aperture_photometry(data, apertures)
    >>> annulus_apertures = CircularAnnulus(positions, 6., 8.)
    >>> bkgflux_table = aperture_photometry(data, annulus_apertures)
    >>> aperture_area = np.pi * 3 ** 2
    >>> annulus_area = np.pi * (8 ** 2 - 6 ** 2)
    >>> fluxtable = hstack([rawflux_table, bkgflux_table], table_names=['raw', 'bkg'])
    >>> fluxtable['result_aperture_sum'] = fluxtable['aperture_sum_raw'] - fluxtable['aperture_sum_bkg'] * aperture_area / annulus_area
    >>> fluxtable['result_aperture_sum']   # doctest: +FLOAT_CMP
    <Column name='result_aperture_sum' unit=u'' format=None description=None>
    array([ -1.77635684e-14,  -1.77635684e-14,  -1.77635684e-14,
            -1.77635684e-14])

  (The result differs from 0 due to inclusion or exclusion of
  subpixels in the apertures.)

Error Estimation
----------------

If, and only if, the ``error`` keyword is specified, the return table will
have ``'aperture_sum'`` and ``'aperture_sum_err'`` columns rather than just
``'aperture_sum'``. ``'aperture_sum_err'`` has same lenght as
``'aperture_sum'``, specifying the uncertainty in each corresponding flux
value.

For example, suppose we have previously calculated the error on each
pixel's value and saved it in the array ``data_error``:

  >>> data_error = 0.1 * data  # (100 x 100 array)
  >>> fluxtable = aperture_photometry(data, apertures, error=data_error)
  >>> fluxtable   # doctest: +FLOAT_CMP
  <Table rows=4 names=('aperture_sum','aperture_sum_err') units=('','')>
  array([(28.274333882308134, 0.531736155271655),
         (28.274333882308134, 0.531736155271655),
         (28.274333882308134, 0.531736155271655),
         (28.274333882308134, 0.531736155271655)],
        dtype=[('aperture_sum', '<f8'), ('aperture_sum_err', '<f8')])

``'aperture_sum_err'`` values are given by

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
  >>> sky_level, sky_sigma = background(data)  # function returns two arrays   # doctest: +SKIP
  >>> fluxtable = aperture_photometry(data - sky_level, apertures,
  ...                                 error=sky_sigma, gain=myimagegain)   # doctest: +SKIP

In this case, and indeed whenever ``gain`` is not `None`, then ``'aperture_sum_err'``
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
``encloses(extent, nx, ny)`` and ``extent()``. They can optionally implement
a third method, ``area()``.

* ``encloses(extent, nx, ny)``: Takes an extent and two dimensions and
  returns an array of shape (ny, nx) giving the fraction of each pixel
  covered by the aperture.
* ``extent()``: Returns the maximum extent of the aperture, (x_min, x_max,
  y_min, y_max). This may be out of the data area, and thus may be
  different, than the ``extent`` parameter of the ``encloses()`` method. It
  is used to determine the portion of the data array to subsample (if
  necessary).
* ``area()``: If convenient to calculate, this returns the area of the
  aperture.  This speeds computation in certain situations (such as a
  scalar error). If not provided, ``~photutils.aperture_photometry`` will
  estimate the area using the result of ``encloses(extent, nx, ny)``.

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
