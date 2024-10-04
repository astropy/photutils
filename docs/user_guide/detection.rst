.. _source_detection:

Source Detection (`photutils.detection`)
========================================

Introduction
------------

One generally needs to identify astronomical sources in their data
before they can perform photometry or morphological measurements.
Photutils provides several tools designed specifically to detect
point-like (stellar) sources in an astronomical image. Photutils also
provides a function to identify local peaks in an image that are above a
specified threshold value.

For general-use source detection and extraction of both point-like
and extended sources, please see :ref:`Image Segmentation
<image_segmentation>`.


Detecting Stars
---------------

Photutils includes two widely-used tools that are used to detect stars
in an image, `DAOFIND`_ and IRAF's `starfind`_, plus a third tool that
allows input of a custom user-defined kernel.

:class:`~photutils.detection.DAOStarFinder` is a class that provides an
implementation of the `DAOFIND`_ algorithm (`Stetson 1987, PASP 99, 191
<https://ui.adsabs.harvard.edu/abs/1987PASP...99..191S/abstract>`_).
It searches images for local density maxima that have a peak amplitude
greater than a specified threshold (the threshold is applied to a
convolved image) and have a size and shape similar to a defined 2D
Gaussian kernel. :class:`~photutils.detection.DAOStarFinder` also
provides an estimate of the objects' roundness and sharpness, whose
lower and upper bounds can be specified.

:class:`~photutils.detection.IRAFStarFinder` is a class that
implements IRAF's `starfind`_ algorithm. It is fundamentally
similar to :class:`~photutils.detection.DAOStarFinder`,
but :class:`~photutils.detection.DAOStarFinder` can use
an elliptical Gaussian kernel. One other difference in
:class:`~photutils.detection.IRAFStarFinder` is that it calculates the
objects' centroid, roundness, and sharpness using image moments.

:class:`~photutils.detection.StarFinder` is a class similar to
:class:`~photutils.detection.IRAFStarFinder`, but which allows input
of a custom user-defined kernel as a 2D array. This allows for more
generalization beyond simple Gaussian kernels.

As an example, let's load an image from the bundled datasets and select
a subset of the image. We will estimate the background and background
noise using sigma-clipped statistics::

    >>> import numpy as np
    >>> from astropy.stats import sigma_clipped_stats
    >>> from photutils.datasets import load_star_image
    >>> hdu = load_star_image()  # doctest: +REMOTE_DATA
    >>> data = hdu.data[0:401, 0:401]  # doctest: +REMOTE_DATA
    >>> mean, median, std = sigma_clipped_stats(data, sigma=3.0)  # doctest: +REMOTE_DATA
    >>> print(np.array((mean, median, std)))  # doctest: +REMOTE_DATA, +FLOAT_CMP
    [3668.09661146 3649.          204.41388592]

Now we will subtract the background and use an instance of
:class:`~photutils.detection.DAOStarFinder` to find the stars in the
image that have FWHMs of around 3 pixels and have peaks approximately
5-sigma above the background. Running this class on the data yields an
astropy `~astropy.table.Table` containing the results of the star
finder::

    >>> from photutils.detection import DAOStarFinder
    >>> daofind = DAOStarFinder(fwhm=3.0, threshold=5.*std)  # doctest: +REMOTE_DATA
    >>> sources = daofind(data - median)  # doctest: +REMOTE_DATA
    >>> for col in sources.colnames:  # doctest: +REMOTE_DATA
    ...     if col not in ('id', 'npix'):
    ...         sources[col].info.format = '%.2f'  # for consistent table output
    >>> sources.pprint(max_width=76)  # doctest: +REMOTE_DATA, +FLOAT_CMP
     id xcentroid ycentroid sharpness ...   peak     flux    mag   daofind_mag
    --- --------- --------- --------- ... ------- --------- ------ -----------
      1    144.25      6.38      0.58 ... 6903.00  45735.00 -11.65       -1.89
      2    208.67      6.82      0.48 ... 7896.00  62118.00 -11.98       -2.07
      3    216.93      6.58      0.69 ... 2195.00  12436.00 -10.24       -0.55
      4    351.63      8.55      0.49 ... 6977.00  55313.00 -11.86       -1.93
      5    377.52     12.07      0.52 ... 1260.00   9078.00  -9.89       -0.12
    ...       ...       ...       ... ...     ...       ...    ...         ...
    282    267.90    398.62      0.27 ... 9299.00 147372.00 -12.92       -1.84
    283    271.47    398.91      0.37 ... 8028.00 115913.00 -12.66       -1.76
    284    299.05    398.78      0.26 ... 9072.00 140781.00 -12.87       -1.86
    285    299.99    398.77      0.29 ... 9253.00 142233.00 -12.88       -1.82
    286    360.45    399.52      0.37 ... 8079.00  81455.00 -12.28       -2.10
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
threshold value. Peaks are the local maxima above a specified threshold
that are separated by a specified minimum number of pixels, defined by a
box size or a local footprint.

The returned pixel coordinates for the peaks are always integer-valued
(i.e., no centroiding is performed, only the peak pixel is identified).
However, a centroiding function can be input via the ``centroid_func``
keyword to :func:`~photutils.detection.find_peaks` to also compute
centroid coordinates with subpixel precision.

As a simple example, let's find the local peaks in an image that are 5
sigma above the background and a separated by at least 5 pixels::

    >>> from astropy.stats import sigma_clipped_stats
    >>> from photutils.datasets import make_100gaussians_image
    >>> from photutils.detection import find_peaks
    >>> data = make_100gaussians_image()
    >>> mean, median, std = sigma_clipped_stats(data, sigma=3.0)
    >>> threshold = median + (5.0 * std)
    >>> tbl = find_peaks(data, threshold, box_size=11)
    >>> tbl['peak_value'].info.format = '%.8g'  # for consistent table output
    >>> print(tbl[:10])  # print only the first 10 peaks
     id x_peak y_peak peak_value
    --- ------ ------ ----------
      1    233      0  27.786048
      2    493      6  18.699406
      3    208      9  22.499317
      4    259     11  16.400909
      5    365     11  17.789691
      6    290     23  34.141532
      7    379     29  16.058562
      8    442     31  32.162038
      9    471     37  24.141928
     10    358     39  18.671565

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


API Reference
-------------

:doc:`../reference/detection_api`


.. _DAOFIND: https://iraf.net/irafhelp.php?val=daofind
.. _starfind: https://iraf.net/irafhelp.php?val=starfind
