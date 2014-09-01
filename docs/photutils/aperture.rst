Aperture photometry
===================

.. currentmodule:: photutils


Introduction
------------

In photutils, the :func:`~photutils.aperture_photometry` function is the main
tool to carry out aperture photometry in an astronomical image for a given
set of apertures. Currently five aperture shapes are supported:

* Circle
* Circular annulus
* Ellipse
* Elliptical annulus
* Rectangle

The positions can be provided either as pixel coordinates, or coordinates on
the sky (provided that the data is specified with a WCS transformation). In
addition, users can create their own aperture classes for use with
``photutils`` (see `Defining your own aperture objects`_).

Creating the aperture objects
-----------------------------

The first step when carrying out photometry is to create an aperture object.
We start off with an example of creating an aperture in pixel coordinates
using the :class:`~photutils.CircularAperture` class:

  >>> from photutils import CircularAperture
  >>> positions = [(30., 30.), (40., 40.)]
  >>> apertures = CircularAperture(positions, r=3.)

The positions should be either a single tuple of ``(x, y)``, a list of ``(x,
y)`` tuples, or an array with shape ``Nx2``, where ``N`` is the number of
positions. In the above example, there are 2 sources located at ``(30, 30)``
and ``(40, 40)``, in pixel coordinates, and the apertures are 3 pixels in
radius.

Creating an aperture object in celestial coordinates is similar, and makes
use of the :class:`~photutils.SkyCircularAperture` class, and the Astropy
:class:`~astropy.coordinates.SkyCoord` class to define celestial coordinates:

  >>> from astropy import units as u
  >>> from astropy.coordinates import SkyCoord
  >>> from photutils import SkyCircularAperture
  >>> positions = SkyCoord(l=[1.2, 2.3] * u.deg, b=[0.1, 0.2] * u.deg,
  ...                      frame='galactic')
  >>> apertures = SkyCircularAperture(positions, r=4. * u.arcsec)

.. note:: At this time, the apertures are not strictly defined completely in
          celestial coordinates in the sense that they simply use celestial
          coordinates to define the central position, and the remaining
          parameters are converted to pixels using the pixel scale of the
          image (so projection distortions are not taken into account). If
          the apertures were truly completely defined in celestial
          coordinates, the shapes would not be preserved when converting to
          pixel coordinates.

Carrying out the photometry
---------------------------

Once the aperture object is created, we can carry out the photometry using
the :func:`~photutils.aperture_photometry` function. We start off by defining
the apertures as described above:

    >>> positions = [(30., 30.), (40., 40.)]
    >>> apertures = CircularAperture(positions, r=3.)

and we then call the :func:`~photutils.aperture_photometry` function with the
data and the apertures:

    >>> import numpy as np
    >>> from photutils import aperture_photometry
    >>> data = np.ones((100, 100))
    >>> phot_table = aperture_photometry(data, apertures)
    >>> print phot_table
     aperture_sum pixel_center [2] input_center [2]
                      pix              pix
    ------------- ---------------- ----------------
    28.2743338823     30.0 .. 30.0     30.0 .. 30.0
    28.2743338823     40.0 .. 40.0     40.0 .. 40.0

This function returns the results of the photometry in an Astropy
`~astropy.table.Table`, In this example, the table has 3 columns, named
``'aperture_sum'``, ``'pixel_center'``, ``'input_center'``.

Since all the data values are 1, the sum in the apertures is equal to the
area of a circle with the same radius:

    >>> print np.pi * 3. ** 2
    28.2743338823

Precision
---------

There are different ways to sum the pixels. By default, the method
used is ``'exact'``, wherein the exact intersection of the aperture with
each pixel is calculated. There are other options that are faster but
at the expense of less precise answers. For example,:

    >>> phot_table = aperture_photometry(data, apertures,
    ...                                  method='subpixel', subpixels=5)
    >>> print phot_table['aperture_sum']
    aperture_sum
    <BLANKLINE>
    ------------
           27.96
           27.96

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

  >>> radii = [3., 4., 5.]
  >>> flux = []
  >>> for radius in radii:
  ...     flux.append(aperture_photometry(data, CircularAperture(positions, radius)))

Now we have 3 separate tables containing the photometry results, one for
each aperture. One may use `~astropy.table.hstack` to stack them into one
`~astropy.table.Table`:

  >>> from astropy.table import hstack
  >>> phot_table = hstack(flux)
  >>> print phot_table['aperture_sum_1', 'aperture_sum_2', 'aperture_sum_3']    # doctest: +FLOAT_CMP
  aperture_sum_1 aperture_sum_2 aperture_sum_3
  <BLANKLINE>
  -------------- -------------- --------------
   28.2743338823  50.2654824574  78.5398163397
   28.2743338823  50.2654824574  78.5398163397


Other aperture photometry functions have multiple parameters
specifying the apertures. For example, for elliptical apertures, one
must specify ``a``, ``b``, and ``theta``:

  >>> from photutils import EllipticalAperture
  >>> a = 5.
  >>> b = 3.
  >>> theta = np.pi / 4.
  >>> apertures = EllipticalAperture(positions, a, b, theta)
  >>> phot_table = aperture_photometry(data, apertures)
  >>> print phot_table['aperture_sum']   # doctest: +FLOAT_CMP
  aperture_sum
  <BLANKLINE>
  -------------
  47.1238898038
  47.1238898038


Again, for multiple apertures one should loop over them.

 >>> a = [5., 6., 7., 8.]
 >>> b = [3., 4., 5., 6.]
 >>> theta = np.pi / 4.
 >>> flux = []
 >>> for index in range(len(a)):
 ...     flux.append(aperture_photometry(data, EllipticalAperture(positions, a[index], b[index], theta)))
 >>> phot_table = hstack(flux)
 >>> print phot_table['aperture_sum_1', 'aperture_sum_2',
 ...                  'aperture_sum_3', 'aperture_sum_4']   # doctest: +FLOAT_CMP
 aperture_sum_1 aperture_sum_2 aperture_sum_3 aperture_sum_4
 <BLANKLINE>
 -------------- -------------- -------------- --------------
  47.1238898038  75.3982236862  109.955742876  150.796447372
  47.1238898038  75.3982236862  109.955742876  150.796447372


Background Subtraction
----------------------

So that the functions are as flexible as possible, background
subtraction is left up to the user or calling function.

* *Global background subtraction*

  If ``bkg`` is an array representing the background level of the data
  (determined in an external function), simply do

    >>> phot_table = aperture_photometry(data - bkg, apertures)  # doctest: +SKIP

* *Local background subtraction*

  Suppose we want to estimate the local background level around each pixel
  with a circular annulus of inner radius 6 pixels and outer radius 8 pixels.
  We can start off by defining the apertures:

    >>> from photutils import CircularAnnulus
    >>> apertures = CircularAperture(positions, r=3)
    >>> annulus_apertures = CircularAnnulus(positions, r_in=6., r_out=8.)

  then we compute the sum of the pixels inside each aperture, and combine
  the tables:

    >>> rawflux_table = aperture_photometry(data, apertures)
    >>> bkgflux_table = aperture_photometry(data, annulus_apertures)
    >>> phot_table = hstack([rawflux_table, bkgflux_table], table_names=['raw', 'bkg'])

  We can then scale the background value to the circular aperture area and
  subtract the background:

    >>> aperture_area = np.pi * 3 ** 2
    >>> annulus_area = np.pi * (8 ** 2 - 6 ** 2)
    >>> final_sum = (phot_table['aperture_sum_raw'] -
    ...              phot_table['aperture_sum_bkg'] * aperture_area / annulus_area)
    >>> phot_table['residual_aperture_sum'] = final_sum
    >>> print phot_table['residual_aperture_sum']   # doctest: +FLOAT_CMP
    residual_aperture_sum
    ---------------------
        2.48689957516e-14
        2.48689957516e-14

  (In this case, the result differs from 0 due to inclusion or exclusion of
  subpixels in the apertures.)

Error Estimation
----------------

If, and only if, the ``error`` keyword is specified, the return table will
have ``'aperture_sum'`` and ``'aperture_sum_err'`` columns rather than just
``'aperture_sum'``. ``'aperture_sum_err'`` has same length as
``'aperture_sum'``, specifying the uncertainty in each corresponding flux
value.

For example, suppose we have previously calculated the error on each
pixel's value and saved it in the array ``data_error``:

  >>> data_error = 0.1 * data  # (100 x 100 array)
  >>> phot_table = aperture_photometry(data, apertures, error=data_error)
  >>> print phot_table   # doctest: +FLOAT_CMP
   aperture_sum aperture_sum_err pixel_center [2] input_center [2]
                                      pix              pix
  ------------- ---------------- ---------------- ----------------
  28.2743338823   0.531736155272     30.0 .. 30.0     30.0 .. 30.0
  28.2743338823   0.531736155272     40.0 .. 40.0     40.0 .. 40.0


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
  >>> phot_table = aperture_photometry(data - sky_level, positions, apertures,
  ...                                  error=sky_sigma, gain=myimagegain)   # doctest: +SKIP

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

If an image mask is specified via the ``mask`` keyword, masked pixels
can either be ignored (``mask_method='skip'``, which is the default)
or interpolated (``mask_method='interpolation'``).  Interpolated
pixels are replaced by the mean value of the neighboring non-masked
pixels.  Some examples are below.

Without a mask image::

  >>> data = np.ones((5, 5))
  >>> aperture = CircularAperture((2, 2), 2.)
  >>> mask = np.zeros_like(data, dtype=bool)
  >>> data[2, 2] = 100.
  >>> mask[2, 2] = True
  >>> t1 = aperture_photometry(data, aperture)
  >>> print t1['aperture_sum']
   aperture_sum
  -------------
  111.566370614

With the mask image and the default ``mask_method``
(``mask_method='skip'``)::

  >>> t2 = aperture_photometry(data, aperture, mask=mask)
  >>> print t2['aperture_sum']
   aperture_sum
  -------------
  11.5663706144

With the mask image and ``mask_method='interpolation'``::

  >>> t3 = aperture_photometry(data, aperture, mask=mask,
  ...                          mask_method='interpolation')
  >>> print t3['aperture_sum']
   aperture_sum
  -------------
  12.5663706144


Photometry using sky coordinates
--------------------------------

As mentioned in `Creating the aperture objects`_, doing photometry using
apertures defined in celestial coordinates simply requires defining a 'sky'
aperture using a :class:`~astropy.coordinates.SkyCoord` object. We show here
an example of photometry on real data in celestial coordinates.

We start off by loading Spitzer 4.5 micron observations of a region of the
Galactic plane:

>>> from photutils import datasets
>>> hdu = datasets.load_spitzer_image()   # doctest: +REMOTE_DATA
>>> catalog = datasets.load_spitzer_catalog()   # doctest: +REMOTE_DATA

The catalog contains (among other things) the galactic coordinates of the
sources in the image as well as the (PSF-fitted) fluxes from the official
Spitzer data reduction. We can then define the apertures based on the
existing catalog positions:

>>> positions = SkyCoord(catalog['l'], catalog['b'], frame='galactic')   # doctest: +REMOTE_DATA
>>> apertures = SkyCircularAperture(positions, r=4.8 * u.arcsec)   # doctest: +REMOTE_DATA
>>> phot_table = aperture_photometry(hdu, apertures)    # doctest: +REMOTE_DATA

The ``hdu`` object is a FITS HDU that contains, in addition to the data, a
header describing the WCS of the image (including the coordinate frame of the
image and the projection from celestial to pixel coordinates). The
`~photutils.aperture_photometry` function uses this information to automatically convert
the apertures defined in celestial coordinates into pixel coordinates.

The Spitzer catalog also contains the official fluxes for the sources that we
can compare to our fluxes. The Spitzer catalog units are mJy while the data is in
MJy/sr, so we have to do the conversion before comparing the results. (The
image data has a pixel scale of 1.2 arcsec / pixel)

>>> import astropy.units as u
>>> factor = (1.2 * u.arcsec) ** 2 / u.pixel
>>> fluxes_catalog = catalog['f4_5']   # doctest: +REMOTE_DATA
>>> converted_aperture_sum = (phot_table['aperture_sum'] * factor).to(u.mJy / u.pixel)   # doctest: +REMOTE_DATA

Finally, we can plot the comparison:

.. doctest-skip::

  >>> import matplotlib.pylab as plt
  >>> plt.scatter(fluxes_catalog, converted_aperture_sum.value)
  >>> plt.xlabel('Spitzer catalog fluxes ')
  >>> plt.ylabel('Aperture photometry fluxes')

.. plot::

  from astropy import units as u
  from astropy.coordinates import SkyCoord
  from photutils import aperture_photometry, SkyCircularAperture

  # Load dataset
  from photutils import datasets
  hdu = datasets.load_spitzer_image()
  catalog = datasets.load_spitzer_catalog()

  # Set up apertures
  positions = SkyCoord(catalog['l'], catalog['b'], frame='galactic')
  apertures = SkyCircularAperture(positions, r=4.8 * u.arcsec)
  phot_table = aperture_photometry(hdu, apertures)

  # Convert to correct units
  factor = (1.2 * u.arcsec) ** 2 / u.pixel
  fluxes_catalog = catalog['f4_5']
  converted_aperture_sum = (phot_table['aperture_sum'] * factor).to(u.mJy / u.pixel)

  # Plot
  import matplotlib.pylab as plt
  plt.scatter(fluxes_catalog, converted_aperture_sum.value)
  plt.xlabel('Spitzer catalog fluxes ')
  plt.ylabel('Aperture photometry fluxes')
  plt.plot([40, 100, 450],[40, 100, 450], color='black', lw=2)

The two catalogs are in good agreement. The Spitzer fluxes were computed
using PSF photometry, and therefore differences are expected between the two.

Defining your own aperture objects
----------------------------------

The photometry function, `~photutils.aperture_photometry`, performs
aperture photometry in arbitrary apertures. This function accepts
`Aperture`-derived objects, such as
`~photutils.CircularAperture`. (The wrappers handle creation of the
`~photutils.Aperture` objects or arrays thereof.) This makes it simple
to extend functionality: a new type of aperture photometry simply
requires the definition of a new `~photutils.Aperture` subclass.

All `~photutils.Aperture` subclasses must implement only two methods,
``do_photometry(data)`` and ``extent()``. They can optionally implement
a third method, ``area()``.

* ``extent()``: Returns the maximum extent of the aperture, (x_min, x_max,
  y_min, y_max). Note that this may be out of the data area. It is used to
  determine the portion of the data array to subsample (if necessary).
* ``area()``: If convenient to calculate, this returns the area of the
  aperture.  This speeds computation in certain situations (such as a
  scalar error).

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
