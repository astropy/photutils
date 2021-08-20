.. _source_detection:

Source Detection (`photutils.detection`)
========================================

Introduction
------------

One generally needs to identify astronomical sources in their data
before they can perform photometry or morphological measurements.
Photutils provides two functions designed specifically to detect
point-like (stellar) sources in an astronomical image.  Photutils also
provides a function to identify local peaks in an image that are above
a specified threshold value.

For general-use source detection and extraction of both point-like
and extended sources, please see :ref:`Image Segmentation
<image_segmentation>`.


Detecting Stars
---------------

Photutils includes two widely-used tools that are used to detect stars
in an image, `DAOFIND`_ and IRAF's `starfind`_.

:class:`~photutils.detection.DAOStarFinder` is a class that provides
an implementation of the `DAOFIND`_ algorithm (`Stetson 1987, PASP 99,
191
<https://ui.adsabs.harvard.edu/abs/1987PASP...99..191S/abstract>`_).
It searches images for local density maxima that have a peak amplitude
greater than a specified threshold (the threshold is applied to a
convolved image) and have a size and shape similar to a defined 2D
Gaussian kernel.  :class:`~photutils.detection.DAOStarFinder` also
provides an estimate of the objects' roundness and sharpness, whose
lower and upper bounds can be specified.

:class:`~photutils.detection.IRAFStarFinder` is a class that
implements IRAF's `starfind`_ algorithm.  It is fundamentally similar
to :class:`~photutils.detection.DAOStarFinder`, but
:class:`~photutils.detection.DAOStarFinder` can use an elliptical
Gaussian kernel. One other difference in
:class:`~photutils.detection.IRAFStarFinder` is that it calculates the
objects' centroid, roundness, and sharpness using image moments.

As an example, let's load an image from the bundled datasets and
select a subset of the image.  We will estimate the background and
background noise using sigma-clipped statistics::

    >>> from astropy.stats import sigma_clipped_stats
    >>> from photutils.datasets import load_star_image
    >>> hdu = load_star_image()  # doctest: +REMOTE_DATA
    >>> data = hdu.data[0:401, 0:401]  # doctest: +REMOTE_DATA
    >>> mean, median, std = sigma_clipped_stats(data, sigma=3.0)  # doctest: +REMOTE_DATA
    >>> print((mean, median, std))  # doctest: +REMOTE_DATA, +FLOAT_CMP
    (3668.09661145823, 3649.0, 204.41388592022315)

Now we will subtract the background and use an instance of
:class:`~photutils.detection.DAOStarFinder` to find the stars in the
image that have FWHMs of around 3 pixels and have peaks approximately
5-sigma above the background. Running this class on the data yields an
astropy `~astropy.table.Table` containing the results of the star
finder:

.. doctest-requires:: scipy

    >>> from photutils.detection import DAOStarFinder
    >>> daofind = DAOStarFinder(fwhm=3.0, threshold=5.*std)  # doctest: +REMOTE_DATA
    >>> sources = daofind(data - median)  # doctest: +REMOTE_DATA
    >>> for col in sources.colnames:  # doctest: +REMOTE_DATA
    ...     sources[col].info.format = '%.8g'  # for consistent table output
    >>> print(sources)  # doctest: +REMOTE_DATA
     id xcentroid ycentroid sharpness  ... sky peak    flux       mag
    --- --------- --------- ---------- ... --- ---- --------- ------------
      1 144.24757 6.3797904 0.58156257 ...   0 6903 5.6976747   -1.8892441
      2 208.66907 6.8205805 0.48348966 ...   0 7896 6.7186388   -2.0682032
      3 216.92614 6.5775933 0.69359525 ...   0 2195 1.6662764  -0.55436758
      4 351.62519 8.5459013 0.48577834 ...   0 6977 5.8970385   -1.9265849
      5 377.51991 12.065501 0.52038488 ...   0 1260 1.1178252  -0.12093477
    ...       ...       ...        ... ... ...  ...       ...          ...
    282 267.90091 398.61991 0.27117231 ...   0 9299 5.4379278   -1.8385836
    283 271.46959 398.91242 0.36738752 ...   0 8028 5.0693475   -1.7623802
    284 299.05003 398.78469 0.25895667 ...   0 9072 5.5584641    -1.862387
    285 299.99359 398.76661 0.29412474 ...   0 9253 5.3233471    -1.815462
    286 360.44533 399.52381 0.37315624 ...   0 8079 6.9203438   -2.1003192
    Length = 286 rows

Let's plot the image and mark the location of detected sources:

.. doctest-skip::

    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from astropy.visualization import SqrtStretch
    >>> from astropy.visualization.mpl_normalize import ImageNormalize
    >>> from photutils.aperture import CircularAperture
    >>> positions = np.transpose((sources['xcentroid'], sources['ycentroid']))
    >>> apertures = CircularAperture(positions, r=4.)
    >>> norm = ImageNormalize(stretch=SqrtStretch())
    >>> plt.imshow(data, cmap='Greys', origin='lower', norm=norm,
    ...            interpolation='nearest')
    >>> apertures.plot(color='blue', lw=1.5, alpha=0.5)

.. plot::

    from astropy.stats import sigma_clipped_stats
    from astropy.visualization import SqrtStretch
    from astropy.visualization.mpl_normalize import ImageNormalize
    import matplotlib.pyplot as plt
    import numpy as np
    from photutils.aperture import CircularAperture
    from photutils.datasets import load_star_image
    from photutils.detection import DAOStarFinder

    hdu = load_star_image()
    data = hdu.data[0:401, 0:401]
    mean, median, std = sigma_clipped_stats(data, sigma=3.0)
    daofind = DAOStarFinder(fwhm=3.0, threshold=5. * std)
    sources = daofind(data - median)
    positions = np.transpose((sources['xcentroid'], sources['ycentroid']))
    apertures = CircularAperture(positions, r=4.)
    norm = ImageNormalize(stretch=SqrtStretch())
    plt.imshow(data, cmap='Greys', origin='lower', norm=norm,
               interpolation='nearest')
    apertures.plot(color='blue', lw=1.5, alpha=0.5)


Masking Regions
^^^^^^^^^^^^^^^

Regions of the input image can be masked by using the ``mask`` keyword
with the :class:`~photutils.detection.DAOStarFinder` or
:class:`~photutils.detection.IRAFStarFinder` instance.  This simple
examples uses :class:`~photutils.detection.DAOStarFinder` and masks
two rectangular regions.  No sources will be detected in the masked
regions:

.. doctest-skip::

   >>> from photutils.detection import DAOStarFinder
   >>> daofind = DAOStarFinder(fwhm=3.0, threshold=5. * std)
   >>> mask = np.zeros(data.shape, dtype=bool)
   >>> mask[50:151, 50:351] = True
   >>> mask[250:351, 150:351] = True
   >>> sources = daofind(data - median, mask=mask)
   >>> for col in sources.colnames:
   >>>     sources[col].info.format = '%.8g'  # for consistent table output
   >>> print(sources)

.. plot::

    from astropy.stats import sigma_clipped_stats
    from astropy.visualization import SqrtStretch
    from astropy.visualization.mpl_normalize import ImageNormalize
    import matplotlib.pyplot as plt
    import numpy as np
    from photutils.aperture import CircularAperture, RectangularAperture
    from photutils.datasets import load_star_image
    from photutils.detection import DAOStarFinder

    hdu = load_star_image()
    data = hdu.data[0:401, 0:401]
    mean, median, std = sigma_clipped_stats(data, sigma=3.0)
    daofind = DAOStarFinder(fwhm=3.0, threshold=5. * std)
    mask = np.zeros(data.shape, dtype=bool)
    mask[50:151, 50:351] = True
    mask[250:351, 150:351] = True
    sources = daofind(data - median, mask=mask)
    positions = np.transpose((sources['xcentroid'], sources['ycentroid']))
    apertures = CircularAperture(positions, r=4.)
    norm = ImageNormalize(stretch=SqrtStretch())
    plt.imshow(data, cmap='Greys', origin='lower', norm=norm,
               interpolation='nearest')
    plt.title('Star finder with a mask to exclude regions')
    apertures.plot(color='blue', lw=1.5, alpha=0.5)
    rect1 = RectangularAperture((200, 100), 300, 100, theta=0.)
    rect2 = RectangularAperture((250, 300), 200, 100, theta=0.)
    rect1.plot(color='salmon', ls='dashed')
    rect2.plot(color='salmon', ls='dashed')


Local Peak Detection
--------------------

Photutils also includes a :func:`~photutils.detection.find_peaks`
function to find local peaks in an image that are above a specified
threshold value. Peaks are the local maxima above a specified
threshold that are separated by a specified minimum number of pixels.

By default, the returned pixel coordinates are always integer-valued
(i.e., no centroiding is performed, only the peak pixel is
identified).  However, a centroiding function can be input via the
``centroid_func`` keyword to :func:`~photutils.detection.find_peaks`
to compute centroid coordinates with subpixel precision.

:func:`~photutils.detection.find_peaks` supports a number of
additional options, including searching for peaks only within a
specified footprint.  Please see the
:func:`~photutils.detection.find_peaks` documentation for more
options.

As a simple example, let's find the local peaks in an image that are 5
sigma above the background and a separated by at least 5 pixels:

.. doctest-requires:: scipy

    >>> from astropy.stats import sigma_clipped_stats
    >>> from photutils.datasets import make_100gaussians_image
    >>> from photutils.detection import find_peaks
    >>> data = make_100gaussians_image()
    >>> mean, median, std = sigma_clipped_stats(data, sigma=3.0)
    >>> threshold = median + (5. * std)
    >>> tbl = find_peaks(data, threshold, box_size=11)
    >>> tbl['peak_value'].info.format = '%.8g'  # for consistent table output
    >>> print(tbl[:10])  # print only the first 10 peaks
    x_peak y_peak peak_value
    ------ ------ ----------
       233      0  27.477852
       493      6  20.404769
       207     11  24.075798
       258     12  17.395025
       366     12  18.729726
       289     22  35.853276
       380     29  19.261986
       442     31  30.239994
       359     36  19.771626
       471     38   25.45583

And let's plot the location of the detected peaks in the image:

.. doctest-skip::

    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from astropy.visualization import simple_norm
    >>> from astropy.visualization.mpl_normalize import ImageNormalize
    >>> from photutils.aperture import CircularAperture
    >>> positions = np.transpose((tbl['x_peak'], tbl['y_peak']))
    >>> apertures = CircularAperture(positions, r=5.)
    >>> norm = simple_norm(data, 'sqrt', percent=99.9)
    >>> plt.imshow(data, cmap='Greys_r', origin='lower', norm=norm,
    ...            interpolation='nearest')
    >>> apertures.plot(color='#0547f9', lw=1.5)
    >>> plt.xlim(0, data.shape[1] - 1)
    >>> plt.ylim(0, data.shape[0] - 1)

.. plot::

    from astropy.stats import sigma_clipped_stats
    from astropy.visualization import simple_norm
    import matplotlib.pyplot as plt
    import numpy as np
    from photutils.aperture import CircularAperture
    from photutils.datasets import make_100gaussians_image
    from photutils.detection import find_peaks

    data = make_100gaussians_image()
    mean, median, std = sigma_clipped_stats(data, sigma=3.0)
    threshold = median + (5.0 * std)
    tbl = find_peaks(data, threshold, box_size=11)

    positions = np.transpose((tbl['x_peak'], tbl['y_peak']))
    apertures = CircularAperture(positions, r=5.)
    norm = simple_norm(data, 'sqrt', percent=99.9)
    plt.imshow(data, cmap='Greys_r', origin='lower', norm=norm,
               interpolation='nearest')
    apertures.plot(color='#0547f9', lw=1.5)
    plt.xlim(0, data.shape[1] - 1)
    plt.ylim(0, data.shape[0] - 1)


Reference/API
-------------

.. automodapi:: photutils.detection
    :no-heading:


.. _DAOFIND: https://iraf.net/irafhelp.php?val=daofind
.. _starfind: https://iraf.net/irafhelp.php?val=starfind
