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
    Downloading ...
    >>> data = hdu.data[0:400, 0:400]    # doctest: +REMOTE_DATA
    >>> mean, median, std = sigma_clipped_stats(data, sigma=3.0, iters=5)    # doctest: +REMOTE_DATA
    >>> print(mean, median, std)    # doctest: +REMOTE_DATA
    (3667.7792400186008, 3649.0, 204.27923665845705)

Now we will subtract the background and use an instance of
:class:`~photutils.DAOStarFinder` to find the stars in the image that
have FWHMs of around 3 pixels and have peaks approximately 5-sigma
above the background. Running this class on the data yields an astropy
`~astropy.table.Table` containing the results of the star finder:

.. doctest-requires:: scipy, skimage

    >>> from photutils import DAOStarFinder
    >>> daofind = DAOStarFinder(fwhm=3.0, threshold=5.*std)    # doctest: +REMOTE_DATA
    >>> sources = daofind(data - median)    # doctest: +REMOTE_DATA
    >>> print(sources)    # doctest: +REMOTE_DATA
     id   xcentroid     ycentroid   ...  peak       flux           mag
    --- ------------- ------------- ... ------ ------------- ---------------
      1 144.247567164 6.37979042704 ... 6903.0 5.70143033038  -1.88995955438
      2 208.669068628 6.82058053777 ... 7896.0 6.72306730455  -2.06891864748
      3 216.926136655  6.5775933198 ... 2195.0 1.66737467591 -0.555083002864
      4 351.625190383  8.5459013233 ... 6977.0 5.90092548147  -1.92730032571
      5 377.519909958 12.0655009987 ... 1260.0 1.11856203781 -0.121650189969
    ...           ...           ... ...    ...           ...             ...
    281 268.049236979 397.925371446 ... 9299.0 6.22022587541  -1.98451538884
    282 268.475068392 398.020998272 ... 8754.0 6.05079160593  -1.95453048936
    283  299.80943822 398.027911813 ... 8890.0 6.11853416663  -1.96661847383
    284 315.689448343  398.70251891 ... 6485.0 5.55471107793  -1.86165368631
    285 360.437243037 398.698539555 ... 8079.0 5.26549321379  -1.80359764345
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
identified).  However, :func:`~photutils.detection.find_peaks` may be
used to compute centroid coordinates with subpixel precision whenever
the optional argument ``subpixel`` is set to `True`.

:func:`~photutils.detection.find_peaks` supports a number of
additional options, including searching for peaks only within a
segmentation image or a specified footprint.  Please see the
:func:`~photutils.detection.find_peaks` documentation for more
options.

As simple example, let's find the local peaks in an image that are 10
sigma above the background and a separated by a least 2 pixels:

.. doctest-requires:: skimage

    >>> from astropy.stats import sigma_clipped_stats
    >>> from photutils.datasets import make_100gaussians_image
    >>> from photutils import find_peaks
    >>> data = make_100gaussians_image()
    >>> mean, median, std = sigma_clipped_stats(data, sigma=3.0)
    >>> threshold = median + (10.0 * std)
    >>> tbl = find_peaks(data, threshold, box_size=5)
    >>> print(tbl[:10])    # print only the first 10 peaks
    x_peak y_peak   peak_value
    ------ ------ -------------
       233      0 27.4778521972
       236      1  27.339519624
       289     22 35.8532759965
       442     31 30.2399941373
         1     40 35.5482863002
        89     59 41.2190469279
         7     70 33.2880647048
       258     75 26.5624808518
       463     80 28.7588206692
       182     93 38.0885687202

And let's plot the location of the detected peaks in the image:

.. doctest-skip::

    >>> import matplotlib.pyplot as plt
    >>> from astropy.visualization import SqrtStretch
    >>> from astropy.visualization.mpl_normalize import ImageNormalize
    >>> norm = ImageNormalize(stretch=SqrtStretch())
    >>> plt.imshow(data, cmap='Greys_r', origin='lower', norm=norm)
    >>> plt.plot(tbl['x_peak'], tbl['y_peak'], ls='none', color='cyan',
    ...          marker='+', ms=10, lw=1.5)
    >>> plt.xlim(0, data.shape[1]-1)
    >>> plt.ylim(0, data.shape[0]-1)

.. plot::

    from astropy.stats import sigma_clipped_stats
    from photutils.datasets import make_100gaussians_image
    from photutils import find_peaks
    data = make_100gaussians_image()
    mean, median, std = sigma_clipped_stats(data, sigma=3.0)
    threshold = median + (10.0 * std)
    tbl = find_peaks(data, threshold, box_size=5)

    import matplotlib.pyplot as plt
    from astropy.visualization import SqrtStretch
    from astropy.visualization.mpl_normalize import ImageNormalize
    norm = ImageNormalize(stretch=SqrtStretch())
    plt.imshow(data, cmap='Greys_r', origin='lower', norm=norm)
    plt.plot(tbl['x_peak'], tbl['y_peak'], ls='none', color='cyan',
             marker='+', ms=10, lw=1.5)
    plt.xlim(0, data.shape[1]-1)
    plt.ylim(0, data.shape[0]-1)


Reference/API
-------------

.. automodapi:: photutils.detection
    :no-heading:


.. _DAOFIND: http://stsdas.stsci.edu/cgi-bin/gethelp.cgi?daofind
.. _starfind: http://stsdas.stsci.edu/cgi-bin/gethelp.cgi?starfind
