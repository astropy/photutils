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

For general-use source detection and extraction of both point-like and
extended sources, please see :ref:`image_segmentation`.


Detecting Stars
---------------

Photutils includes two widely-used tools that are used to detect stars
in an image, `DAOFIND`_ and IRAF's `starfind`_.

:class:`~photutils.DAOStarFinder` is a class that provides an
implementation of the `DAOFIND`_ algorithm (`Stetson 1987, PASP 99,
191 <http://adsabs.harvard.edu/abs/1987PASP...99..191S>`_).  It
searches images for local density maxima that have a peak amplitude
greater than a specified threshold (the threshold is applied to a
convolved image) and have a size and shape similar to a defined 2D
Gaussian kernel.  :class:`~photutils.DAOStarFinder` also provides an
estimate of the objects' roundness and sharpness, whose lower and
upper bounds can be specified.

:class:`~photutils.IRAFStarFinder` is a class that implements IRAF's
`starfind`_ algorithm.  It is fundamentally similar to
:class:`~photutils.DAOStarFinder`, but
:class:`~photutils.DAOStarFinder` can use an elliptical Gaussian
kernel. One other difference in :class:`~photutils.IRAFStarFinder` is
that it calculates the objects' centroid, roundness, and sharpness
using image moments.

As an example, let's load an image from the bundled datasets and
select a subset of the image.  We will estimate the background and
background noise using sigma-clipped statistics::

    >>> from astropy.stats import sigma_clipped_stats
    >>> from photutils import datasets
    >>> hdu = datasets.load_star_image()    # doctest: +REMOTE_DATA
    >>> data = hdu.data[0:400, 0:400]    # doctest: +REMOTE_DATA
    >>> mean, median, std = sigma_clipped_stats(data, sigma=3.0, iters=5)    # doctest: +REMOTE_DATA
    >>> print((mean, median, std))    # doctest: +REMOTE_DATA, +FLOAT_CMP
    (3667.7792400186008, 3649.0, 204.27923665845705)

Now we will subtract the background and use an instance of
:class:`~photutils.DAOStarFinder` to find the stars in the image that
have FWHMs of around 3 pixels and have peaks approximately 5-sigma
above the background. Running this class on the data yields an astropy
`~astropy.table.Table` containing the results of the star finder:

.. doctest-requires:: scipy

    >>> from photutils import DAOStarFinder
    >>> daofind = DAOStarFinder(fwhm=3.0, threshold=5.*std)    # doctest: +REMOTE_DATA
    >>> sources = daofind(data - median)    # doctest: +REMOTE_DATA
    >>> for col in sources.colnames:    # doctest: +REMOTE_DATA
    ...     sources[col].info.format = '%.8g'  # for consistent table output
    >>> print(sources)    # doctest: +REMOTE_DATA
     id xcentroid ycentroid sharpness  ... sky peak    flux       mag
    --- --------- --------- ---------- ... --- ---- --------- ------------
      1 144.24757 6.3797904 0.58156257 ...   0 6903 5.7014303   -1.8899596
      2 208.66907 6.8205805 0.48348966 ...   0 7896 6.7230673   -2.0689186
      3 216.92614 6.5775933 0.69359525 ...   0 2195 1.6673747    -0.555083
      4 351.62519 8.5459013 0.48577834 ...   0 6977 5.9009255   -1.9273003
      5 377.51991 12.065501 0.52038488 ...   0 1260  1.118562  -0.12165019
    ...       ...       ...        ... ... ...  ...       ...          ...
    280 345.59306 395.38222   0.384078 ...   0 9350 5.0559084    -1.759498
    281 268.04924 397.92537 0.29650715 ...   0 9299 6.2202259   -1.9845154
    282 268.47507   398.021 0.28325741 ...   0 8754 6.0507916   -1.9545305
    283 299.80944 398.02791 0.32011339 ...   0 8890 6.1185342   -1.9666185
    284 315.68945 398.70252 0.29502138 ...   0 6485 5.5547111   -1.8616537
    285 360.43724 398.69854 0.81147144 ...   0 8079 5.2654932   -1.8035976
    Length = 285 rows

Let's plot the image and mark the location of detected sources:

.. doctest-skip::

    >>> import matplotlib.pyplot as plt
    >>> from astropy.visualization import SqrtStretch
    >>> from astropy.visualization.mpl_normalize import ImageNormalize
    >>> from photutils import CircularAperture
    >>> positions = (sources['xcentroid'], sources['ycentroid'])
    >>> apertures = CircularAperture(positions, r=4.)
    >>> norm = ImageNormalize(stretch=SqrtStretch())
    >>> plt.imshow(data, cmap='Greys', origin='lower', norm=norm)
    >>> apertures.plot(color='blue', lw=1.5, alpha=0.5)

.. plot::

    import matplotlib.pyplot as plt
    from astropy.stats import sigma_clipped_stats
    from astropy.visualization import SqrtStretch
    from astropy.visualization.mpl_normalize import ImageNormalize
    from photutils import datasets, DAOStarFinder, CircularAperture

    hdu = datasets.load_star_image()
    data = hdu.data[0:400, 0:400]
    mean, median, std = sigma_clipped_stats(data, sigma=3.0)
    daofind = DAOStarFinder(fwhm=3.0, threshold=5.*std)
    sources = daofind(data - median)
    positions = (sources['xcentroid'], sources['ycentroid'])
    apertures = CircularAperture(positions, r=4.)
    norm = ImageNormalize(stretch=SqrtStretch())
    plt.imshow(data, cmap='Greys', origin='lower', norm=norm)
    apertures.plot(color='blue', lw=1.5, alpha=0.5)


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
    >>> from photutils import find_peaks
    >>> data = make_100gaussians_image()
    >>> mean, median, std = sigma_clipped_stats(data, sigma=3.0)
    >>> threshold = median + (5. * std)
    >>> tbl = find_peaks(data, threshold, box_size=11)
    >>> tbl['peak_value'].info.format = '%.8g'  # for consistent table output
    >>> print(tbl[:10])    # print only the first 10 peaks
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

    >>> import matplotlib.pyplot as plt
    >>> from astropy.visualization import simple_norm
    >>> from astropy.visualization.mpl_normalize import ImageNormalize
    >>> from photutils import CircularAperture
    >>> positions = (tbl['x_peak'], tbl['y_peak'])
    >>> apertures = CircularAperture(positions, r=5.)
    >>> norm = simple_norm(data, 'sqrt', percent=99.9)
    >>> plt.imshow(data, cmap='Greys_r', origin='lower', norm=norm)
    >>> apertures.plot(color='#0547f9', lw=1.5)
    >>> plt.xlim(0, data.shape[1]-1)
    >>> plt.ylim(0, data.shape[0]-1)

.. plot::

    from astropy.stats import sigma_clipped_stats
    from photutils import find_peaks, CircularAperture
    from photutils.datasets import make_100gaussians_image
    data = make_100gaussians_image()
    mean, median, std = sigma_clipped_stats(data, sigma=3.0)
    threshold = median + (5.0 * std)
    tbl = find_peaks(data, threshold, box_size=11)

    import matplotlib.pyplot as plt
    from astropy.visualization import simple_norm
    positions = (tbl['x_peak'], tbl['y_peak'])
    apertures = CircularAperture(positions, r=5.)
    norm = simple_norm(data, 'sqrt', percent=99.9)
    plt.imshow(data, cmap='Greys_r', origin='lower', norm=norm)
    apertures.plot(color='#0547f9', lw=1.5)
    plt.xlim(0, data.shape[1]-1)
    plt.ylim(0, data.shape[0]-1)


Reference/API
-------------

.. automodapi:: photutils.detection
    :no-heading:


.. _DAOFIND: http://stsdas.stsci.edu/cgi-bin/gethelp.cgi?daofind
.. _starfind: http://stsdas.stsci.edu/cgi-bin/gethelp.cgi?starfind
