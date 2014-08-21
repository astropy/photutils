Aperture photometry
===================

.. warning:: The aperture photometry API is currently *experimental*
   and will change in the near future.

.. currentmodule:: photutils


Introduction
------------

In ``photutils`` `aperture_photometry` is the main tool to carry out
aperture photometry in an astronomical image for a given list of
sources. Currently 3 types of apertures are supported: circular, elliptical
and rectangular as well as two types of annulus apertures: circular and
elliptical.  The objects can be identified either by providing a list of
pixel positions, or a list of sky positions along with a wcs transformation.


A Simple Example
----------------

Suppose there are 2 sources located at (30, 30) and (40, 40), in pixel
coordinates. To sum the pixel values (flux) inside a circular aperture of
radius 3 pixels centered on each object:

    >>> import numpy as np
    >>> from photutils import aperture_photometry
    >>> data = np.ones((100, 100))
    >>> xc = [30., 40.]
    >>> yc = [30., 40.]
    >>> positions = zip(xc, yc)
    >>> radius = 3.
    >>> apertures = ('circular', radius)
    >>> phot_table, aux_dict = aperture_photometry(data, positions, apertures)
    >>> print phot_table
     aperture_sum pixel_center [2] input_center [2]
                      pix              pix
    ------------- ---------------- ----------------
    28.2743338823     30.0 .. 30.0     30.0 .. 30.0
    28.2743338823     40.0 .. 40.0     40.0 .. 40.0

    >>> type(aux_dict['apertures'])
    <class 'photutils.aperture_core.CircularAperture'>

`aperture_photometry` returns with a 2-tuple. The first element contains the
result of the photometry in a `~astropy.table.Table`. In this example case
it has 3 columns, named ``'aperture_sum'``, ``'pixel_center'``,
``'input_center'``.  The second element is an auxiliary information
dictionary. The apertures, used during the photometry, are returned as the
``'apertures'`` element of this dictionary.

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

    >>> phot_table = aperture_photometry(data, positions, apertures,
    ...                                  method='subpixel', subpixels=5)[0]
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

  >>> r = [3., 4., 5.]
  >>> flux = []
  >>> for radius in r:
  ...     flux.append(aperture_photometry(data, positions, ('circular', radius))[0])

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
  >>> apertures = ('elliptical', a, b, theta)
  >>> phot_table = aperture_photometry(data, positions, apertures)[0]
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
 ...     flux.append(aperture_photometry(data, positions, ('elliptical', a[index], b[index], theta))[0])
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

    >>> phot_table = aperture_photometry(data - bkg, positions, apertures)[0]  # doctest: +SKIP

* *Local background subtraction*

  Suppose we want to estimate the local background level around each pixel
  with a circular annulus of inner radius 6 pixels and outer radius 8 pixels:

    >>> radius = 3.
    >>> apertures = ('circular', radius)
    >>> rawflux_table = aperture_photometry(data, positions, apertures)[0]
    >>> annulus_apertures = ('circular_annulus', 6., 8.)
    >>> bkgflux_table = aperture_photometry(data, positions, annulus_apertures)[0]
    >>> aperture_area = np.pi * 3 ** 2
    >>> annulus_area = np.pi * (8 ** 2 - 6 ** 2)
    >>> phot_table = hstack([rawflux_table, bkgflux_table], table_names=['raw', 'bkg'])
    >>> phot_table['residual_aperture_sum'] = phot_table['aperture_sum_raw'] - phot_table['aperture_sum_bkg'] * aperture_area / annulus_area
    >>> print phot_table['residual_aperture_sum']   # doctest: +FLOAT_CMP
    residual_aperture_sum
    ---------------------
        2.48689957516e-14
        2.48689957516e-14

  (The result differs from 0 due to inclusion or exclusion of subpixels in
  the apertures.)

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
  >>> phot_table = aperture_photometry(data, positions, apertures, error=data_error)[0]
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
  ...                                 error=sky_sigma, gain=myimagegain)[0]   # doctest: +SKIP

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
  >>> mask = np.zeros_like(data, dtype=bool)
  >>> data[2, 2] = 100.
  >>> mask[2, 2] = True
  >>> t1, d1 = aperture_photometry(data, (2, 2), ('circular', 2))
  >>> print t1['aperture_sum']
   aperture_sum
  -------------
  111.566370614

With the mask image and the default ``mask_method``
(``mask_method='skip'``)::

  >>> t2, d2 = aperture_photometry(data, (2, 2), ('circular', 2), mask=mask)
  >>> print t2['aperture_sum']
   aperture_sum
  -------------
  11.5663706144

With the mask image and ``mask_method='interpolation'``::

  >>> t3, d3 = aperture_photometry(data, (2, 2), ('circular', 2), mask=mask, mask_method='interpolation')
  >>> print t3['aperture_sum']
   aperture_sum
  -------------
  12.5663706144


Photometry using sky coordinates
------------------------------------------

Although internally all the photometry functions use pixel coordinates,
there is a possibility to provide sky coordinates as input positions to
`aperture_photometry`.  In this example we use a Spitzer image stored in
``photutils-dataset``. There is also a catalog provided with the example
fits file, containing, among others, the galactic coordinates of the sources
in the image. Note: if the coordinates are provided as a list of sky
coordinates, but not as a `~astropy.coordinates.SkyCoord` object, they are
assumed to be in the same celestial frame as the wcs transformation.

>>> from astropy.io import fits
>>> from astropy.table import Table
>>> from photutils.datasets import get_path
>>> pathcat = get_path('spitzer_example_catalog.xml', location='remote')   # doctest: +REMOTE_DATA
>>> pathhdu = get_path('spitzer_example_image.fits', location='remote')   # doctest: +REMOTE_DATA
>>> hdu = fits.open(pathhdu)   # doctest: +REMOTE_DATA
>>> catalog = Table.read(pathcat)   # doctest: +REMOTE_DATA
>>> pos_gal = zip(catalog['l'], catalog['b'])   # doctest: +REMOTE_DATA
>>> photometry_pos_gal = aperture_photometry(hdu, pos_gal, ('circular', 4),
...                                          pixelcoord=False)[0]   # doctest: +REMOTE_DATA


The same can be achieved with providing the positions as a
`~astropy.coordinates.SkyCoord` object:

>>> from astropy.coordinates import SkyCoord
>>> pos_skycoord = SkyCoord(catalog['l'], catalog['b'], frame='galactic')   # doctest: +REMOTE_DATA
>>> photometry_skycoord = aperture_photometry(hdu, pos_skycoord,
...                                           ('circular', 4))[0]   # doctest: +REMOTE_DATA

>>> np.all(photometry_skycoord['aperture_sum'] == photometry_pos_gal['aperture_sum'])   # doctest: +REMOTE_DATA
    True

The coordinate catalog also contains the fluxes for the sources. The catalog
units are mJy while the data is in MJy/sr, so we have to do the conversion
before comparing the results. (The image data has the pixel scale of
1.2 arcsec / pixel)

>>> import astropy.units as u
>>> factor = (1.2 * u.arcsec) ** 2 / u.pixel
>>> fluxes_catalog = catalog['f4_5']   # doctest: +REMOTE_DATA
>>> apeture_sum = photometry_skycoord['aperture_sum'] * u.MJy / u.sr   # doctest: +REMOTE_DATA
>>> converted_aperture_sum = (aperture_sum * factor).to(u.mJy / u.pixel)   # doctest: +REMOTE_DATA


.. doctest-skip::

  >>> import matplotlib.pylab as plt
  >>> plt.scatter(fluxes_catalog, converted_aperture_sum.value)

.. plot::

  import matplotlib.pylab as plt
  import astropy.units as u
  from astropy.io import fits
  from astropy.table import Table
  from astropy.coordinates import SkyCoord
  from photutils.datasets import get_path
  from photutils import aperture_photometry
  pathcat = get_path('spitzer_example_catalog.xml', location='remote')
  pathhdu = get_path('spitzer_example_image.fits', location='remote')
  hdu = fits.open(pathhdu)
  catalog = Table.read(pathcat)
  pos_skycoord = SkyCoord(catalog['l'], catalog['b'], frame='galactic')
  photometry_skycoord = aperture_photometry(hdu, pos_skycoord, ('circular', 4))[0]
  factor = (1.2 * u.arcsec) ** 2 / u.pixel
  fluxes_catalog = catalog['f4_5']
  aperture_sum = photometry_skycoord['aperture_sum'] * u.MJy / u.sr
  converted_aperture_sum = (aperture_sum * factor).to(u.mJy / u.pixel)
  plt.scatter(fluxes_catalog, converted_aperture_sum.value)
  plt.xlabel('Fluxes catalog')
  plt.ylabel('Fluxes aperture photometry')
  plt.plot([40,100,450],[40,100,450], color='black', lw=2)


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
``do_photometry(data)`` and ``extent()``. They can optionally implement
a third method, ``area()``.

* ``extent()``: Returns the maximum extent of the aperture, (x_min, x_max,
  y_min, y_max). This may be out of the data area, and thus may be
  different, than the ``extent`` parameter of the ``encloses()`` method. It
  is used to determine the portion of the data array to subsample (if
  necessary).
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
