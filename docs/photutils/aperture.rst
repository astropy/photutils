Aperture Photometry (`photutils.aperture`)
==========================================

Introduction
------------

In Photutils, the :func:`~photutils.aperture_photometry` function is
the main tool to perform aperture photometry on an astronomical image
for a given set of apertures.

Photutils provides several apertures defined in pixel or sky
coordinates.  The aperture classes that are defined in pixel
coordinates are:

    * `~photutils.CircularAperture`
    * `~photutils.CircularAnnulus`
    * `~photutils.EllipticalAperture`
    * `~photutils.EllipticalAnnulus`
    * `~photutils.RectangularAperture`
    * `~photutils.RectangularAnnulus`

Each of these classes has a corresponding variant defined in celestial
coordinates:

    * `~photutils.SkyCircularAperture`
    * `~photutils.SkyCircularAnnulus`
    * `~photutils.SkyEllipticalAperture`
    * `~photutils.SkyEllipticalAnnulus`
    * `~photutils.SkyRectangularAperture`
    * `~photutils.SkyRectangularAnnulus`

To perform aperture photometry with sky-based apertures, one will need
to specify a WCS transformation.

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
number of positions.  The above example defines two circular apertures
located at pixel coordinates ``(30, 30)`` and ``(40, 40)`` with a
radius of 3 pixels.

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
    Sky apertures are not defined completely in celestial coordinates.
    They simply use celestial coordinates to define the central
    position, and the remaining parameters are converted to pixels
    using the pixel scale of the image at the central position.
    Projection distortions are not taken into account.  If the
    apertures were defined completely in celestial coordinates, their
    shapes would not be preserved when converting to pixel
    coordinates.


Performing Aperture Photometry
------------------------------

After the aperture object is created, we can then perform the
photometry using the :func:`~photutils.aperture_photometry` function.
We start by defining the apertures as described above::

    >>> positions = [(30., 30.), (40., 40.)]
    >>> apertures = CircularAperture(positions, r=3.)

and then we call the :func:`~photutils.aperture_photometry` function
with the data and the apertures::

    >>> import numpy as np
    >>> from photutils import aperture_photometry
    >>> data = np.ones((100, 100))
    >>> phot_table = aperture_photometry(data, apertures)
    >>> print(phot_table)    # doctest: +SKIP
     id xcenter ycenter  aperture_sum
          pix     pix
    --- ------- ------- -------------
      1    30.0    30.0 28.2743338823
      2    40.0    40.0 28.2743338823

This function returns the results of the photometry in an Astropy
`~astropy.table.QTable`.  In this example, the table has four columns,
named ``'id'``, ``'xcenter'``, ``'ycenter'``, and ``'aperture_sum'``.

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
the aperture based on their centers.  For this method, the number of
subpixels needs to be set with the ``subpixels`` keyword.

This example uses the ``'subpixel'`` method where pixels are resampled
by a factor of 5 (``subpixels=5``) in each dimension::

    >>> phot_table = aperture_photometry(data, apertures, method='subpixel',
    ...                                  subpixels=5)
    >>> print(phot_table)    # doctest: +SKIP
     id xcenter ycenter aperture_sum
          pix     pix
    --- ------- ------- ------------
      1    30.0    30.0        27.96
      2    40.0    40.0        27.96

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
they must have a fixed shape, e.g. radius, size, and orientation.

To perform photometry in multiple apertures at each position, one may
input a list of aperture objects to the
:func:`~photutils.aperture_photometry` function.

Suppose that we wish to use three circular apertures, with radii of 3,
4, and 5 pixels, on each source::

    >>> radii = [3., 4., 5.]
    >>> apertures = [CircularAperture(positions, r=r) for r in radii]
    >>> phot_table = aperture_photometry(data, apertures)
    >>> print(phot_table)    # doctest: +SKIP
     id xcenter ycenter aperture_sum_0 aperture_sum_1 aperture_sum_2
          pix     pix
    --- ------- ------- -------------- -------------- --------------
      1    30.0    30.0  28.2743338823  50.2654824574  78.5398163397
      2    40.0    40.0  28.2743338823  50.2654824574  78.5398163397

For multiple apertures, the output table column names are appended
with the ``positions`` index.

Other apertures have multiple parameters specifying the aperture size
and orientation.  For example, for elliptical apertures, one must
specify ``a``, ``b``, and ``theta``::

    >>> from photutils import EllipticalAperture
    >>> a = 5.
    >>> b = 3.
    >>> theta = np.pi / 4.
    >>> apertures = EllipticalAperture(positions, a, b, theta)
    >>> phot_table = aperture_photometry(data, apertures)
    >>> print(phot_table)    # doctest: +SKIP
     id xcenter ycenter  aperture_sum
          pix     pix
    --- ------- ------- -------------
      1    30.0    30.0 47.1238898038
      2    40.0    40.0 47.1238898038

Again, for multiple apertures one should input a list of aperture
objects, each with identical positions::

    >>> a = [5., 6., 7.]
    >>> b = [3., 4., 5.]
    >>> theta = np.pi / 4.
    >>> apertures = [EllipticalAperture(positions, a=ai, b=bi, theta=theta)
    ...              for (ai, bi) in zip(a, b)]
    >>> phot_table = aperture_photometry(data, apertures)
    >>> print(phot_table)    # doctest: +SKIP
     id xcenter ycenter aperture_sum_0 aperture_sum_1 aperture_sum_2
          pix     pix
    --- ------- ------- -------------- -------------- --------------
      1    30.0    30.0  47.1238898038  75.3982236862  109.955742876
      2    40.0    40.0  47.1238898038  75.3982236862  109.955742876


Background Subtraction
----------------------

Global Background Subtraction
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:func:`~photutils.aperture_photometry` assumes that the data have been
background-subtracted.  If ``bkg`` is an array representing the
background of the data (determined by
`~photutils.background.Background2D` or an external function), simply
do::

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

We then perform the photometry in both apertures::

    >>> apers = [apertures, annulus_apertures]
    >>> phot_table = aperture_photometry(data, apers)
    >>> print(phot_table)    # doctest: +SKIP
     id xcenter ycenter aperture_sum_0 aperture_sum_1
          pix     pix
    --- ------- ------- -------------- --------------
      1    30.0    30.0  28.2743338823  87.9645943005
      2    40.0    40.0  28.2743338823  87.9645943005

Note that we cannot simply subtract the aperture sums because the
apertures have different areas.

To calculate the mean local background within the circular annulus
aperture, we need to divide its sum by its area, which can be
calculated using the :meth:`~photutils.CircularAnnulus.area` method::

    >>> bkg_mean = phot_table['aperture_sum_1'] / annulus_apertures.area()

The background sum within the circular aperture is then the mean local
background times the circular aperture area::

    >>> bkg_sum = bkg_mean * apertures.area()
    >>> final_sum = phot_table['aperture_sum_0'] - bkg_sum
    >>> phot_table['residual_aperture_sum'] = final_sum
    >>> print(phot_table['residual_aperture_sum'])    # doctest: +FLOAT_CMP
    residual_aperture_sum
    ---------------------
        -7.1054273576e-15
        -7.1054273576e-15

The result here should be zero because all of the data values are 1.0
(the tiny difference from 0.0 is due to numerical precision).


.. _error_estimation:


Error Estimation
----------------

If and only if the ``error`` keyword is input to
:func:`~photutils.aperture_photometry`, the returned table will
include a ``'aperture_sum_err'`` column in addition to
``'aperture_sum'``.  ``'aperture_sum_err'`` provides the propagated
uncertainty associated with ``'aperture_sum'``.

For example, suppose we have previously calculated the error on each
pixel's value and saved it in the array ``error``::

    >>> error = 0.1 * data
    >>> phot_table = aperture_photometry(data, apertures, error=error)
    >>> print(phot_table)    # doctest: +SKIP
     id xcenter ycenter  aperture_sum aperture_sum_err
          pix     pix
    --- ------- ------- ------------- ----------------
      1    30.0    30.0 28.2743338823   0.531736155272
      2    40.0    40.0 28.2743338823   0.531736155272

``'aperture_sum_err'`` values are given by:

    .. math:: \Delta F = \sqrt{\sum_{i \in A}
              \sigma_{\mathrm{tot}, i}^2}

where :math:`\Delta F` is
`~photutils.SourceProperties.source_sum_err`, :math:`A` are the
non-masked pixels in the aperture, and :math:`\sigma_{\mathrm{tot},
i}` is the input ``error`` array.

In the example above, it is assumed that the ``error`` keyword
specifies the *total* error -- either it includes Poisson noise due to
individual sources or such noise is irrelevant.  However, it is often
the case that one has calculated a smooth "background-only error"
array, which by design doesn't include increased noise on bright
pixels.  To include Poisson noise from the sources, we can use the
:func:`~photutils.utils.calc_total_error` function.

Let's assume we we have a background-only image called ``bkg_error``.
If our data are in units of electrons/s, we would use the exposure
time as the effective gain::

    >>> from photutils.utils import calc_total_error
    >>> effective_gain = 500   # seconds
    >>> error = calc_total_error(data, bkg_error, effective_gain)    # doctest: +SKIP
    >>> phot_table = aperture_photometry(data - bkg, apertures, error=error)    # doctest: +SKIP

.. note::

    In cases where the ``error`` array is slowly varying across the
    image, it is not necessary to sum the error from every pixel in
    the aperture individually.  Instead, we can approximate the error
    as being roughly constant across the aperture and simply take the
    value of :math:`\sigma` at the center of the aperture.  This can
    be done by setting the keyword ``pixelwise_errors=False``.  In
    this case the flux error is

    .. math:: \Delta F = \sigma \sqrt{A}

    where :math:`\sigma` is the ``error`` at the center of the
    aperture and :math:`A` is the area of the aperture.


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
    >>> print(t1['aperture_sum'])    # doctest: +FLOAT_CMP
     aperture_sum
     -------------
     11.5663706144

The result is very different if a ``mask`` image is not provided::

    >>> t2 = aperture_photometry(data, aperture)
    >>> print(t2['aperture_sum'])    # doctest: +FLOAT_CMP
    aperture_sum
    -------------
    111.566370614


Aperture Photometry Using Sky Coordinates
-----------------------------------------

As mentioned in :ref:`creating-aperture-objects`, performing
photometry using apertures defined in celestial coordinates simply
requires defining a "sky" aperture at positions defined by a
:class:`~astropy.coordinates.SkyCoord` object.  Here we show an
example of photometry on real data using a
`~photutils.SkyCircularAperture`.

We start by loading a Spitzer 4.5 micron image of a region of the
Galactic plane::

    >>> from photutils import datasets
    >>> hdu = datasets.load_spitzer_image()   # doctest: +REMOTE_DATA
    Downloading http://data.astropy.org/photometry/spitzer_example_image.fits [Done]
    >>> catalog = datasets.load_spitzer_catalog()   # doctest: +REMOTE_DATA
    Downloading http://data.astropy.org/photometry/spitzer_example_catalog.xml [Done]

The catalog contains (among other things) the Galactic coordinates of
the sources in the image as well as the PSF-fitted fluxes from the
official Spitzer data reduction.  We define the apertures positions
based on the existing catalog positions::

    >>> positions = SkyCoord(catalog['l'], catalog['b'], frame='galactic')   # doctest: +REMOTE_DATA
    >>> apertures = SkyCircularAperture(positions, r=4.8 * u.arcsec)   # doctest: +REMOTE_DATA

Now perform the photometry in these apertures using the ``hdu``.  The
``hdu`` object is a FITS HDU that contains the data and a header
describing the WCS transformation of the image.  The WCS includes the
coordinate frame of the image and the projection from celestial to
pixel coordinates.  The `~photutils.aperture_photometry` function uses
the WCS information to automatically convert the apertures defined in
celestial coordinates into pixel coordinates::

    >>> phot_table = aperture_photometry(hdu, apertures)    # doctest: +REMOTE_DATA

The Spitzer catalog also contains the official fluxes for the sources,
so we can compare to our fluxes.  Because the Spitzer catalog fluxes
are in units of mJy and the data are in units of MJy/sr, we need to
convert units before comparing the results.  The image data have a
pixel scale of 1.2 arcsec/pixel.

    >>> import astropy.units as u
    >>> factor = (1.2 * u.arcsec) ** 2 / u.pixel
    >>> fluxes_catalog = catalog['f4_5']   # doctest: +REMOTE_DATA
    >>> converted_aperture_sum = (phot_table['aperture_sum'] *
    ...                           factor).to(u.mJy / u.pixel)   # doctest: +REMOTE_DATA

Finally, we can plot the comparison of the photometry:

.. doctest-skip::

    >>> import matplotlib.pyplot as plt
    >>> plt.scatter(fluxes_catalog, converted_aperture_sum.value)
    >>> plt.xlabel('Spitzer catalog PSF-fit fluxes ')
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
  import matplotlib.pyplot as plt
  plt.scatter(fluxes_catalog, converted_aperture_sum.value)
  plt.xlabel('Spitzer catalog PSF-fit fluxes ')
  plt.ylabel('Aperture photometry fluxes')
  plt.plot([40, 100, 450],[40, 100, 450], color='black', lw=2)

Despite using different methods, the two catalogs are in good
agreement.  The aperture photometry fluxes are based on a circular
aperture with a radius of 4.8 arcsec.  The Spitzer catalog fluxes were
computed using PSF photometry.  Therefore, differences are expected
between the two measurements.


Aperture Masks
--------------

All `~photutils.PixelAperture` objects have a
:meth:`~photutils.PixelAperture.to_mask` method that returns a list of
`~photutils.ApertureMask` objects, one for each aperture position.
The `~photutils.ApertureMask` object contains a cutout of the aperture
mask and a `~photutils.BoundingBox` object that provides the bounding
box where the mask is to be applied.  It also provides a
:meth:`~photutils.ApertureMask.to_image` method to obtain an image of
the mask in a 2D array of the given shape, a
:meth:`~photutils.ApertureMask.cutout` method to create a cutout from
the input data over the mask bounding box, and an
:meth:`~photutils.ApertureMask.multiply` method to multiply the
aperture mask with the input data to create a mask-weighted data
cutout.   All of these methods properly handle the cases of partial or
no overlap of the aperture mask with the data.

Let's start by creating an aperture object::

    >>> from photutils import CircularAperture
    >>> positions = [(30., 30.), (40., 40.)]
    >>> apertures = CircularAperture(positions, r=3.)

Now let's create a list of `~photutils.ApertureMask` objects using the
:meth:`~photutils.PixelAperture.to_mask` method::

    >>> masks = aperture.to_mask(method='center')

We can now create an image with of the first aperture mask at its
position::

    >>> mask = masks[0]
    >>> image = mask.to_image(shape=((200, 200)))

We can also create a cutout from a data image over the mask domain::

    >>> data_cutout = mask.cutout(data)

We can also create a mask-weighted cutout from the data.  Here the
circular aperture mask has been multiplied with the data::

    >>> data_cutout_aper = mask.multiply(data)


.. _custom-apertures:

Defining Your Own Custom Apertures
----------------------------------

The :func:`~photutils.aperture_photometry` function can perform
aperture photometry in arbitrary apertures.  This function accepts any
`~photutils.Aperture`-derived objects, such as
`~photutils.CircularAperture`.  This makes it simple to extend
functionality: a new type of aperture photometry simply requires the
definition of a new `~photutils.Aperture` subclass.

All `~photutils.PixelAperture` subclasses must define a ``_slices``
property, ``to_mask()`` and ``plot()`` methods, and optionally an
``area()`` method.  All `~photutils.SkyAperture` subclasses must
implement only a ``to_pixel()`` method.

    * ``_slices``:  A property defining the minimal bounding box
      slices for the aperture at each position.

    * ``to_mask()``: A method to return a list of
      `~photutils.ApertureMask` objects, one for each aperture position.

    * ``area()``: A method to return the exact analytical area (in
      pixels**2) of the aperture.

    * ``plot()``: A method to plot the aperture on a
      `matplotlib.axes.Axes` instance.


See Also
--------

1. `IRAF's APPHOT specification [PDF]`_ (Sec. 3.3.5.8 - 3.3.5.9)

2. `SourceExtractor Manual [PDF]`_ (Sec. 9.4 p. 36)

.. _SourceExtractor: http://www.astromatic.net/software/sextractor
.. _SourceExtractor Manual [PDF]: https://www.astromatic.net/pubsvn/software/sextractor/trunk/doc/sextractor.pdf
.. _IRAF's APPHOT specification [PDF]: http://iraf.net/irafdocs/apspec.pdf


Reference/API
-------------

.. automodapi:: photutils.aperture
    :no-heading:
