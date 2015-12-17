Aperture Photometry
===================

.. currentmodule:: photutils

Introduction
------------

In Photutils, the :func:`~photutils.aperture_photometry` function is
the main tool to perform aperture photometry on an astronomical image
for a given set of apertures.  The aperture shapes that are currently
provided are:

* Circle
* Circular annulus
* Ellipse
* Elliptical annulus
* Rectangle
* Rectangular annulus

The positions can be input either as pixel coordinates or sky
coordinates, provided that the data is specified with a WCS
transformation.

Users can also create their own custom apertures (see
:ref:`custom-apertures`).

.. _creating-aperture-objects:

Creating Aperture Objects
-------------------------

The first step in performing aperture photometry is to create an
aperture object.  An aperture object is defined by a position (or a
list of positions) and parameters that define its size and possibly,
orientation (e.g., an elliptical aperture).

We start with an example of creating a circular aperture in pixel
coordinates using the :class:`~photutils.CircularAperture` class::

    >>> from photutils import CircularAperture
    >>> positions = [(30., 30.), (40., 40.)]
    >>> apertures = CircularAperture(positions, r=3.)

The positions should be either a single tuple of ``(x, y)``, a list of
``(x, y)`` tuples, or an array with shape ``Nx2``, where ``N`` is the
number of positions.  In the above example, there are two sources
located at pixel coordinates ``(30, 30)`` and ``(40, 40)``, and the
apertures have a radius of 3 pixels.

Creating an aperture object in celestial coordinates is similar.  One
first uses the :class:`~astropy.coordinates.SkyCoord` class to define
celestial coordinates and then the
:class:`~photutils.SkyCircularAperture` class to define the aperture
object::

    >>> from astropy import units as u
    >>> from astropy.coordinates import SkyCoord
    >>> from photutils import SkyCircularAperture
    >>> positions = SkyCoord(l=[1.2, 2.3] * u.deg, b=[0.1, 0.2] * u.deg,
    ...                      frame='galactic')
    >>> apertures = SkyCircularAperture(positions, r=4. * u.arcsec)

.. note::
    At this time, apertures are not defined completely in celestial
    coordinates.  They simply use celestial coordinates to define the
    central position, and the remaining parameters are converted to pixels
    using the pixel scale of the image at the central position.  Projection
    distortions are not taken into account.  If the apertures were defined
    completely in celestial coordinates, their shapes would not be preserved
    when converting to pixel coordinates.

Performing Aperture Photometry
------------------------------

After the aperture object is created, we can then carry out the
photometry using the :func:`~photutils.aperture_photometry` function.
We start by defining the apertures as described above:

    >>> positions = [(30., 30.), (40., 40.)]
    >>> apertures = CircularAperture(positions, r=3.)

and then we call the :func:`~photutils.aperture_photometry` function
with the data and the apertures::

    >>> import numpy as np
    >>> from photutils import aperture_photometry
    >>> data = np.ones((100, 100))
    >>> phot_table = aperture_photometry(data, apertures)
    >>> print(phot_table)
     aperture_sum xcenter ycenter
                    pix     pix
    ------------- ------- -------
    28.2743338823    30.0    30.0
    28.2743338823    40.0    40.0

This function returns the results of the photometry in an Astropy
`~astropy.table.Table`.  In this example, the table has three columns,
named ``'aperture_sum'``, ``'xcenter'``, and ``'ycenter'``.

Since all the data values are 1.0, the aperture sums are equal to the
area of a circle with a radius of 3::

    >>> print(np.pi * 3. ** 2)    # doctest: +FLOAT_CMP
    28.2743338823

Aperture and Pixel Overlap
--------------------------

The overlap of the apertures with the data pixels can be handled in
different ways.  For the default method (``method='exact'``), the
exact intersection of the aperture with each pixel is calculated.  The
other options, ``'center'`` and ``'subpixel'``, are faster, but with
the expense of less precision.  For ``'center'``, a pixel is
considered to be entirely in or out of the aperture depending on
whether its center is in or out of the aperture.  For ``'subpixel'``,
pixels are divided into a number of subpixels, which are in or out of
the aperture based on their centers.

This example uses the ``'subpixel'`` method where pixels are resampled
by a factor of 5 in each dimension::

    >>> phot_table = aperture_photometry(data, apertures,
    ...                                  method='subpixel', subpixels=5)
    >>> print(phot_table['aperture_sum'])
    aperture_sum
    <BLANKLINE>
    ------------
           27.96
           27.96

Note that the results differ from the true value of 28.274333 (see
above).

For the ``'subpixel'`` method, the default value is ``subpixels=5``,
meaning that each pixel is equally divided into 25 smaller pixels
(this is the method and subsampling factor used in SourceExtractor_.).
The precision can be increased by increasing ``subpixels``, but note
that computation time will be increased.

Multiple Apertures at Each Position
-----------------------------------

While the `~photutils.Aperture` objects support multiple positions,
they currently must have a fixed size and orientation (e.g., defined
by radius for a circular aperture, or axes lengths and orientation for
an elliptical aperture).  To perform photometry in multiple apertures
at each position, one may loop over different aperture size and
orientation parameters.

Suppose that we wish to use three circular apertures, with radii of 3,
4, and 5 pixels, on each source::

    >>> radii = [3., 4., 5.]
    >>> flux = []
    >>> for radius in radii:
    ...     flux.append(aperture_photometry(data, CircularAperture(positions, radius)))

We now have three separate tables containing the photometry results,
one for each aperture.  One may use `~astropy.table.hstack` to stack
them into one `~astropy.table.Table`::

    >>> from astropy.table import hstack
    >>> phot_table = hstack(flux)
    >>> print(phot_table['aperture_sum_1', 'aperture_sum_2', 'aperture_sum_3'])    # doctest: +FLOAT_CMP
    aperture_sum_1 aperture_sum_2 aperture_sum_3
    <BLANKLINE>
    -------------- -------------- --------------
     28.2743338823  50.2654824574  78.5398163397
     28.2743338823  50.2654824574  78.5398163397

Other apertures have multiple parameters specifying the aperture size
and orientation.  For example, for elliptical apertures, one must
specify ``a``, ``b``, and ``theta``::

    >>> from photutils import EllipticalAperture
    >>> a = 5.
    >>> b = 3.
    >>> theta = np.pi / 4.
    >>> apertures = EllipticalAperture(positions, a, b, theta)
    >>> phot_table = aperture_photometry(data, apertures)
    >>> print(phot_table['aperture_sum'])    # doctest: +FLOAT_CMP
    aperture_sum
    <BLANKLINE>
    -------------
    47.1238898038
    47.1238898038

Again, for multiple apertures one should loop over them::

    >>> a = [5., 6., 7., 8.]
    >>> b = [3., 4., 5., 6.]
    >>> theta = np.pi / 4.
    >>> flux = []
    >>> for index in range(len(a)):
    ...     flux.append(aperture_photometry(
    ...         data, EllipticalAperture(positions, a[index], b[index], theta)))
    >>> phot_table = hstack(flux)
    >>> print(phot_table['aperture_sum_1', 'aperture_sum_2',
    ...                  'aperture_sum_3', 'aperture_sum_4'])    # doctest: +FLOAT_CMP
    aperture_sum_1 aperture_sum_2 aperture_sum_3 aperture_sum_4
    <BLANKLINE>
    -------------- -------------- -------------- --------------
     47.1238898038  75.3982236862  109.955742876  150.796447372
     47.1238898038  75.3982236862  109.955742876  150.796447372

Background Subtraction
----------------------

:func:`aperture_photometry` assumes that the data have been
background-subtracted.

Global Background Subtraction
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If ``bkg`` is an array representing the background of the data
(determined by `~photutils.background.Background` or an external
function), simply do::

    >>> phot_table = aperture_photometry(data - bkg, apertures)  # doctest: +SKIP

Local Background Subtraction
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Suppose we want to perform the photometry in a circular aperture with
a radius of 3 pixels and estimate the local background level around
each source with a circular annulus of inner radius 6 pixels and outer
radius 8 pixels.  We start by defining the apertures::

    >>> from photutils import CircularAnnulus
    >>> apertures = CircularAperture(positions, r=3)
    >>> annulus_apertures = CircularAnnulus(positions, r_in=6., r_out=8.)

We then compute the aperture sum in both apertures and combine the two
tables::

    >>> rawflux_table = aperture_photometry(data, apertures)
    >>> bkgflux_table = aperture_photometry(data, annulus_apertures)
    >>> phot_table = hstack([rawflux_table, bkgflux_table], table_names=['raw', 'bkg'])

To calculate the mean local background within the circular annulus
aperture, we need to divide its sum by its area, which can be
calculated using the :meth:`~photutils.CircularAnnulus.area` method::

    >>> bkg_mean = phot_table['aperture_sum_bkg'] / annulus_apertures.area()

The background sum within the circular aperture is then the mean local
background times the circular aperture area::

    >>> bkg_sum = bkg_mean * apertures.area()
    >>> final_sum = phot_table['aperture_sum_raw'] - bkg_sum
    >>> phot_table['residual_aperture_sum'] = final_sum
    >>> print(phot_table['residual_aperture_sum'])    # doctest: +FLOAT_CMP
    residual_aperture_sum
    ---------------------
        -3.5527136788e-15
        -3.5527136788e-15

The result here should be zero because all of the data values are 1.0
(the small difference from 0.0 is due to numerical precision).


.. _error_estimation:

Error Estimation
----------------

If and only if the ``error`` keyword is input to
:func:`aperture_photometry`, the returned table will include a
``'aperture_sum_err'`` column in addition to ``'aperture_sum'``.
``'aperture_sum_err'`` provides the propagated uncertainty associated
with ``'aperture_sum'``.

For example, suppose we have previously calculated the error on each
pixel's value and saved it in the array ``data_error``::

    >>> data_error = 0.1 * data  # (100 x 100 array)
    >>> phot_table = aperture_photometry(data, apertures, error=data_error)
    >>> print(phot_table)    # doctest: +FLOAT_CMP
       aperture_sum aperture_sum_err xcenter ycenter
                                       pix     pix
      ------------- ---------------- ------- -------
      28.2743338823   0.531736155272    30.0    30.0
      28.2743338823   0.531736155272    40.0    40.0

``'aperture_sum_err'`` values are given by

.. math:: \Delta F = \sqrt{ \sum_{i \in A} \sigma_i^2}

where :math:`\sigma` is the given error array and the sum is over
pixels in the aperture :math:`A`.

In the example above, it is assumed that the ``error`` keyword
specifies the *full* error (either it includes Poisson noise due to
individual sources or such noise is irrelevant).  However, it is often
the case that one has previously calculated a smooth "background
error" array which by design doesn't include increased noise on bright
pixels.  In such a case, we wish to explicitly include Poisson noise
from the sources.  Specifying the ``effective_gain`` keyword does
this.  For example, suppose we have a function ``background()`` that
calculates the position-dependent background level and variance of our
data::

    >>> effective_gain = 1.5
    >>> sky_level, sky_sigma = background(data)  # function returns two arrays   # doctest: +SKIP
    >>> phot_table = aperture_photometry(data - sky_level, apertures,
    ...                                  error=sky_sigma,
    ...                                  effective_gain=effective_gain)   # doctest: +SKIP

In this case, and indeed whenever ``effective_gain`` is not `None`, then
``'aperture_sum_err'`` is given by

.. math:: \Delta F = \sqrt{\sum_{i \in A} (\sigma_i^2 + f_i / g_i)}

where :math:`f_i` is the value of the data (``data - sky_level`` in
this case) at each pixel and :math:`g_i` is the value of the
``effective_gain`` at each pixel.

.. note::

    In cases where the ``error`` and ``effective_gain`` arrays are
    slowly varying across the image, it is not necessary to sum the
    error from every pixel in the aperture individually.  Instead, we
    can approximate the error as being roughly constant across the
    aperture and simply take the value of :math:`\sigma` at the center
    of the aperture.  This can be done by setting the keyword
    ``pixelwise_errors=False``.  This saves some computation time.  In
    this case the flux error is

    .. math:: \Delta F = \sqrt{A \sigma^2 + F / g}

    where :math:`\sigma` and :math:`g` are the ``error`` and
    ``effective_gain`` at the center of the aperture, :math:`A` is the
    area of the aperture, and :math:`F` is the *total* flux in the
    aperture.

Pixel Masking
-------------

Pixels can be ignored/excluded (e.g., bad pixels) from the aperture
photometry by providing an image mask via the ``mask`` keyword::

    >>> data = np.ones((5, 5))
    >>> aperture = CircularAperture((2, 2), 2.)
    >>> mask = np.zeros_like(data, dtype=bool)
    >>> data[2, 2] = 100.   # bad pixel
    >>> mask[2, 2] = True
    >>> t1 = aperture_photometry(data, aperture, mask=mask)
    >>> print(t1['aperture_sum'])
     aperture_sum
    -------------
    11.5663706144

The result is very different if a ``mask`` image is not provided::

    >>> t2 = aperture_photometry(data, aperture)
    >>> print(t2['aperture_sum'])
     aperture_sum
    -------------
    111.566370614

Aperture Photometry Using Sky Coordinates
-----------------------------------------

As mentioned in :ref:`creating-aperture-objects`, performing
photometry using apertures defined in celestial coordinates simply
requires defining a 'sky' aperture using a
:class:`~astropy.coordinates.SkyCoord` object.  We show here an
example of photometry on real data in celestial coordinates.

We start by loading a Spitzer 4.5 micron image of a region of the
Galactic plane::

    >>> from photutils import datasets
    >>> hdu = datasets.load_spitzer_image()   # doctest: +REMOTE_DATA
    >>> catalog = datasets.load_spitzer_catalog()   # doctest: +REMOTE_DATA

The catalog contains (among other things) the Galactic coordinates of
the sources in the image as well as the PSF-fitted fluxes from the
official Spitzer data reduction.  We define the apertures positions
based on the existing catalog positions::

    >>> positions = SkyCoord(catalog['l'], catalog['b'], frame='galactic')   # doctest: +REMOTE_DATA
    >>> apertures = SkyCircularAperture(positions, r=4.8 * u.arcsec)   # doctest: +REMOTE_DATA
    >>> phot_table = aperture_photometry(hdu, apertures)    # doctest: +REMOTE_DATA

The ``hdu`` object is a FITS HDU that contains, in addition to the
data, a header describing the WCS of the image (including the
coordinate frame of the image and the projection from celestial to
pixel coordinates).  The `~photutils.aperture_photometry` function
uses this information to automatically convert the apertures defined
in celestial coordinates into pixel coordinates.

The Spitzer catalog also contains the official fluxes for the sources
that we can compare to our fluxes.  The Spitzer catalog units are mJy
while the data are in units of MJy/sr, so we have to do the conversion
before comparing the results.  The image data has a pixel scale of 1.2
arcsec / pixel.

    >>> import astropy.units as u
    >>> factor = (1.2 * u.arcsec) ** 2 / u.pixel
    >>> fluxes_catalog = catalog['f4_5']   # doctest: +REMOTE_DATA
    >>> converted_aperture_sum = (phot_table['aperture_sum'] *
    ...                           factor).to(u.mJy / u.pixel)   # doctest: +REMOTE_DATA

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

Despite using different methods, the two catalogs are in good
agreement.  The aperture photometry fluxes are based on a circular
aperture with a radius of 4.8 arcsec.  The Spitzer catalog fluxes were
computed using PSF photometry.  Therefore, differences are expected
between the two measurements.

.. _custom-apertures:

Defining Your Own Custom Apertures
----------------------------------

The photometry function :func:`~photutils.aperture_photometry` can
perform aperture photometry in arbitrary apertures.  This function
accepts `Aperture`-derived objects, such as
`~photutils.CircularAperture`.  This makes it simple to extend
functionality: a new type of aperture photometry simply requires the
definition of a new `~photutils.Aperture` subclass.

All `~photutils.PixelAperture` subclasses must implement three
methods, ``do_photometry(data)``, ``area()``, and ``plot()``.
`~photutils.SkyAperture` subclasses must implement only
``do_photometry(data)`` and ``plot()``.

* ``do_photometry(data)``: A method to sum the pixel values within the
  defined aperture.
* ``area()``: A method to return the area (pixels**2) of the aperture.
* ``plot()``: A method to plot the aperture on a `matplotlib`_ Axes
  instance.

Note that all x and y coordinates refer to the fast and slow (second
and first) axis of the data array respectively.  See
:ref:`coordinate-conventions`.

.. _matplotlib: http://matplotlib.org

See Also
--------

1. `IRAF's APPHOT specification [PDF]`_ (Sec. 3.3.5.8 - 3.3.5.9)

2. `SourceExtractor Manual [PDF]`_ (Sec. 9.4 p. 36)

.. _SourceExtractor: http://www.astromatic.net/software/sextractor
.. _SourceExtractor Manual [PDF]: https://www.astromatic.net/pubsvn/software/sextractor/trunk/doc/sextractor.pdf
.. _IRAF's APPHOT specification [PDF]: http://iraf.net/irafdocs/apspec.pdf


Reference/API
-------------

.. automodapi:: photutils.aperture_core
    :no-heading:
