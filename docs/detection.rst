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
    ...     if col not in ('id', 'npix'):
    ...         sources[col].info.format = '%.2f'  # for consistent table output
    >>> sources.pprint(max_width=76)  # doctest: +REMOTE_DATA, +FLOAT_CMP
     id xcentroid ycentroid sharpness roundness1 ... sky    peak  flux  mag
    --- --------- --------- --------- ---------- ... ---- ------- ---- -----
      1    144.25      6.38      0.58       0.20 ... 0.00 6903.00 5.70 -1.89
      2    208.67      6.82      0.48      -0.13 ... 0.00 7896.00 6.72 -2.07
      3    216.93      6.58      0.69      -0.71 ... 0.00 2195.00 1.67 -0.55
      4    351.63      8.55      0.49      -0.34 ... 0.00 6977.00 5.90 -1.93
      5    377.52     12.07      0.52       0.37 ... 0.00 1260.00 1.12 -0.12
    ...       ...       ...       ...        ... ...  ...     ...  ...   ...
    282    267.90    398.62      0.27      -0.43 ... 0.00 9299.00 5.44 -1.84
    283    271.47    398.91      0.37       0.19 ... 0.00 8028.00 5.07 -1.76
    284    299.05    398.78      0.26      -0.67 ... 0.00 9072.00 5.56 -1.86
    285    299.99    398.77      0.29       0.36 ... 0.00 9253.00 5.32 -1.82
    286    360.45    399.52      0.37      -0.19 ... 0.00 8079.00 6.92 -2.10
    Length = 286 rows

Let's plot the image and mark the location of detected sources:

.. doctest-skip::

    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from astropy.visualization import SqrtStretch
    >>> from astropy.visualization.mpl_normalize import ImageNormalize
    >>> from photutils.aperture import CircularAperture
    >>> positions = np.transpose((sources['xcentroid'], sources['ycentroid']))
    >>> apertures = CircularAperture(positions, r=4.0)
    >>> norm = ImageNormalize(stretch=SqrtStretch())
    >>> plt.imshow(data, cmap='Greys', origin='lower', norm=norm,
    ...            interpolation='nearest')
    >>> apertures.plot(color='blue', lw=1.5, alpha=0.5)

.. plot::

    import matplotlib.pyplot as plt
    import numpy as np
    from astropy.stats import sigma_clipped_stats
    from astropy.visualization import SqrtStretch
    from astropy.visualization.mpl_normalize import ImageNormalize
    from photutils.aperture import CircularAperture
    from photutils.datasets import load_star_image
    from photutils.detection import DAOStarFinder

    hdu = load_star_image()
    data = hdu.data[0:401, 0:401]
    mean, median, std = sigma_clipped_stats(data, sigma=3.0)
    daofind = DAOStarFinder(fwhm=3.0, threshold=5.0 * std)
    sources = daofind(data - median)
    positions = np.transpose((sources['xcentroid'], sources['ycentroid']))
    apertures = CircularAperture(positions, r=4.0)
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
   >>> daofind = DAOStarFinder(fwhm=3.0, threshold=5.0 * std)
   >>> mask = np.zeros(data.shape, dtype=bool)
   >>> mask[50:151, 50:351] = True
   >>> mask[250:351, 150:351] = True
   >>> sources = daofind(data - median, mask=mask)
   >>> for col in sources.colnames:
   >>>     sources[col].info.format = '%.8g'  # for consistent table output
   >>> print(sources)

.. plot::

    import matplotlib.pyplot as plt
    import numpy as np
    from astropy.stats import sigma_clipped_stats
    from astropy.visualization import SqrtStretch
    from astropy.visualization.mpl_normalize import ImageNormalize
    from photutils.aperture import CircularAperture, RectangularAperture
    from photutils.datasets import load_star_image
    from photutils.detection import DAOStarFinder

    hdu = load_star_image()
    data = hdu.data[0:401, 0:401]
    mean, median, std = sigma_clipped_stats(data, sigma=3.0)
    daofind = DAOStarFinder(fwhm=3.0, threshold=5.0 * std)
    mask = np.zeros(data.shape, dtype=bool)
    mask[50:151, 50:351] = True
    mask[250:351, 150:351] = True
    sources = daofind(data - median, mask=mask)
    positions = np.transpose((sources['xcentroid'], sources['ycentroid']))
    apertures = CircularAperture(positions, r=4.0)
    norm = ImageNormalize(stretch=SqrtStretch())
    plt.imshow(data, cmap='Greys', origin='lower', norm=norm,
               interpolation='nearest')
    plt.title('Star finder with a mask to exclude regions')
    apertures.plot(color='blue', lw=1.5, alpha=0.5)
    rect1 = RectangularAperture((200, 100), 300, 100, theta=0)
    rect2 = RectangularAperture((250, 300), 200, 100, theta=0)
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
    >>> threshold = median + (5.0 * std)
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
    >>> apertures = CircularAperture(positions, r=5.0)
    >>> norm = simple_norm(data, 'sqrt', percent=99.9)
    >>> plt.imshow(data, cmap='Greys_r', origin='lower', norm=norm,
    ...            interpolation='nearest')
    >>> apertures.plot(color='#0547f9', lw=1.5)
    >>> plt.xlim(0, data.shape[1] - 1)
    >>> plt.ylim(0, data.shape[0] - 1)

.. plot::

    import matplotlib.pyplot as plt
    import numpy as np
    from astropy.stats import sigma_clipped_stats
    from astropy.visualization import simple_norm
    from photutils.aperture import CircularAperture
    from photutils.datasets import make_100gaussians_image
    from photutils.detection import find_peaks

    data = make_100gaussians_image()
    mean, median, std = sigma_clipped_stats(data, sigma=3.0)
    threshold = median + (5.0 * std)
    tbl = find_peaks(data, threshold, box_size=11)

    positions = np.transpose((tbl['x_peak'], tbl['y_peak']))
    apertures = CircularAperture(positions, r=5.0)
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
    :inherited-members:

.. _DAOFIND: https://iraf.net/irafhelp.php?val=daofind
.. _starfind: https://iraf.net/irafhelp.php?val=starfind
