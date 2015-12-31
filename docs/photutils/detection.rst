Source Detection and Extraction
===============================


Introduction
------------

One generally needs to identify astronomical sources in their data
before they can perform photometry or morphological measurements.
This procedure is referred to as source detection or source
extraction.  Photutils provides two functions designed specifically to
detect point-like (stellar) sources in an astronomical image.
Photutils also provides a general-use function to detect sources (both
point-like and extended) in an image using a process called `image
segmentation`_ in the `computer vision
<http://en.wikipedia.org/wiki/Computer_vision>`_ field.


Detecting Stars
---------------

Photutils includes two widely-used tools that are used to detect stars
in an image, `DAOFIND`_ and IRAF's `starfind`_.

:func:`~photutils.daofind` is an implementation of the `DAOFIND`_
algorithm (`Stetson 1987, PASP 99, 191
<http://adsabs.harvard.edu/abs/1987PASP...99..191S>`_).  It searches
images for local density maxima that have a peak amplitude greater
than a specified threshold (the threshold is applied to a convolved
image) and have a size and shape similar to a defined 2D Gaussian
kernel.  :func:`~photutils.daofind` also provides an estimate of the
objects' roundness and sharpness, whose lower and upper bounds can be
specified.

:func:`~photutils.irafstarfind` is an implementation of IRAF's
`starfind`_ algorithm.  It is similar to :func:`~photutils.daofind`,
but it always uses a 2D circular Gaussian kernel, while
:func:`~photutils.daofind` can use an elliptical Gaussian kernel.
:func:`~photutils.irafstarfind` is also different in that it
calculates the objects' centroid, roundness, and sharpness using image
moments.

As an example, let's load an image from the bundled datasets and
select a subset of the image.  We will estimate the background and
background noise using sigma-clipped statistics::

    >>> from photutils import datasets
    >>> from astropy.stats import sigma_clipped_stats
    >>> hdu = datasets.load_star_image()    # doctest: +REMOTE_DATA
    Downloading ...
    >>> data = hdu.data[0:400, 0:400]    # doctest: +REMOTE_DATA
    >>> mean, median, std = sigma_clipped_stats(data, sigma=3.0, iters=5)    # doctest: +REMOTE_DATA
    >>> print(mean, median, std)    # doctest: +REMOTE_DATA
    (3667.7792400186008, 3649.0, 204.27923665845705)

Now we will subtract the background and use :func:`~photutils.daofind`
to find the stars in the image that have FWHMs of around 3 pixels and
have peaks approximately 5-sigma above the background:

.. doctest-requires:: scipy, skimage

    >>> from photutils import daofind
    >>> sources = daofind(data - median, fwhm=3.0, threshold=5.*std)    # doctest: +REMOTE_DATA
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

    >>> from photutils import CircularAperture
    >>> from astropy.visualization import SqrtStretch
    >>> from astropy.visualization.mpl_normalize import ImageNormalize
    >>> import matplotlib.pylab as plt
    >>> positions = (sources['xcentroid'], sources['ycentroid'])
    >>> apertures = CircularAperture(positions, r=4.)
    >>> norm = ImageNormalize(stretch=SqrtStretch())
    >>> plt.imshow(data, cmap='Greys', origin='lower', norm=norm)
    >>> apertures.plot(color='blue', lw=1.5, alpha=0.5)

.. plot::

    from astropy.stats import sigma_clipped_stats
    from photutils import datasets, daofind, CircularAperture
    from astropy.visualization import SqrtStretch
    from astropy.visualization.mpl_normalize import ImageNormalize
    import matplotlib.pylab as plt
    hdu = datasets.load_star_image()
    data = hdu.data[0:400, 0:400]
    mean, median, std = sigma_clipped_stats(data, sigma=3.0)
    sources = daofind(data - median, fwhm=3.0, threshold=5.*std)
    positions = (sources['xcentroid'], sources['ycentroid'])
    apertures = CircularAperture(positions, r=4.)
    norm = ImageNormalize(stretch=SqrtStretch())
    plt.imshow(data, cmap='Greys', origin='lower', norm=norm)
    apertures.plot(color='blue', lw=1.5, alpha=0.5)


.. _source_extraction:

Source Extraction Using Image Segmentation
------------------------------------------

Photutils also provides tools to detect astronomical sources using
`image segmentation`_, which is a process of assigning a label to
every pixel in an image such that pixels with the same label are part
of the same source.  The segmentation procedure implemented in
Photutils is called the threshold method, where detected sources must
have a minimum number of connected pixels that are each greater than a
specified threshold value in an image.  The threshold level is usually
defined at some multiple of the background standard deviation (sigma)
above the background.  The image can also be filtered before
thresholding to smooth the noise and maximize the detectability of
objects with a shape similar to the filter kernel.

In Photutils, source extraction is performed using the
:func:`~photutils.detection.detect_sources` function.  The
:func:`~photutils.detection.detect_threshold` tool is a convenience
function to generate a 2D detection threshold image using simple
sigma-clipped statistics to estimate the background and background
rms.

For this example, let's detect sources in a synthetic image provided
by the `datasets <datasets.html>`_ module::

    >>> from photutils.datasets import make_100gaussians_image
    >>> data = make_100gaussians_image()

We will use :func:`~photutils.detection.detect_threshold` to produce a
detection threshold image.
:func:`~photutils.detection.detect_threshold` will estimate the
background and background rms using sigma-clipped statistics if they
are not input.  The threshold level is calculated using the ``snr``
input as the sigma level above the background.  Here we generate a
simple pixel-wise threshold at 3 sigma above the background::

    >>> from photutils import detect_threshold
    >>> threshold = detect_threshold(data, snr=3.)

For more sophisticated analyses, one should generate a 2D background
and background-only error image (e.g., from your data reduction or by
using :class:`~photutils.background.Background`).  In that case, a
3-sigma threshold image is simply::

    >>> threshold = bkg + (3.0 * bkg_rms)    # doctest: +SKIP

Note that if the threshold includes the background level (as above),
then the image input into :func:`~photutils.detection.detect_sources`
should *not* be background subtracted.

Let's find sources that have 5 connected pixels that are each greater
than the corresponding pixel-wise ``threshold`` level defined above.
Because the threshold returned by
:func:`~photutils.detection.detect_threshold` includes the background,
we do not subtract the background from the data here.  We will also
input a 2D circular Gaussian kernel with a FWHM of 2 pixels to filter
the image prior to thresholding:

.. doctest-requires:: scipy

    >>> from astropy.convolution import Gaussian2DKernel
    >>> from astropy.stats import gaussian_fwhm_to_sigma
    >>> from photutils import detect_sources
    >>> sigma = 2.0 * gaussian_fwhm_to_sigma    # FWHM = 2.
    >>> kernel = Gaussian2DKernel(sigma, x_size=3, y_size=3)
    >>> kernel.normalize()
    >>> segm = detect_sources(data, threshold, npixels=5, filter_kernel=kernel)

The result is a :class:`~photutils.segmentation.SegmentationImage`
object with the same shape as the data, where sources are labeled by
different positive integer values.  A value of zero is always reserved
for the background.  Let's plot both the image and the segmentation
image showing the detected sources:

.. doctest-skip::

    >>> import numpy as np
    >>> import matplotlib.pylab as plt
    >>> from astropy.visualization import SqrtStretch
    >>> from astropy.visualization.mpl_normalize import ImageNormalize
    >>> from photutils.utils import random_cmap
    >>> rand_cmap = random_cmap(segm.max + 1, random_state=12345)
    >>> norm = ImageNormalize(stretch=SqrtStretch())
    >>> fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))
    >>> ax1.imshow(data, origin='lower', cmap='Greys_r', norm=norm)
    >>> ax2.imshow(segm, origin='lower', cmap=rand_cmap)

.. plot::

    import numpy as np
    import matplotlib.pylab as plt
    from astropy.stats import gaussian_fwhm_to_sigma
    from astropy.convolution import Gaussian2DKernel
    from astropy.visualization import SqrtStretch
    from astropy.visualization.mpl_normalize import ImageNormalize
    from photutils.datasets import make_100gaussians_image
    from photutils import detect_threshold, detect_sources
    from photutils.utils import random_cmap
    data = make_100gaussians_image()
    threshold = detect_threshold(data, snr=3.)
    sigma = 2.0 * gaussian_fwhm_to_sigma    # FWHM = 2.
    kernel = Gaussian2DKernel(sigma, x_size=3, y_size=3)
    kernel.normalize()
    segm = detect_sources(data, threshold, npixels=5, filter_kernel=kernel)
    rand_cmap = random_cmap(segm.max + 1, random_state=12345)
    norm = ImageNormalize(stretch=SqrtStretch())
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))
    ax1.imshow(data, origin='lower', cmap='Greys_r', norm=norm)
    ax2.imshow(segm, origin='lower', cmap=rand_cmap)

When the segmentation image is generated using image thresholding
(e.g., using :func:`~photutils.detect_sources`), the source segments
effectively represent the isophotal footprint of each source.


Source Deblending
^^^^^^^^^^^^^^^^^

Note that overlapping sources are detected as single sources.
Separating those sources requires a deblending procedure, such as a
multi-thresholding technique used by `SExtractor
<http://www.astromatic.net/software/sextractor>`_.  Photutils provides
an experimental :func:`~photutils.deblend_sources` function that
deblends sources uses a combination of multi-thresholding and
`watershed segmentation
<https://en.wikipedia.org/wiki/Watershed_(image_processing)>`_.  In
order to deblend sources, they must be separated enough such that
there is a saddle between them:

.. doctest-requires:: scipy, skimage

    >>> from photutils import deblend_sources
    >>> segm_deblend = deblend_sources(data, segm, npixels=5,
    ...                                filter_kernel=kernel)

where ``segm`` is the
:class:`~photutils.segmentation.SegmentationImage` that was generated
by :func:`~photutils.detect_sources`.  Note that the ``npixels`` and
``filter_kernel`` input values should match those used in
:func:`~photutils.detect_sources`.  The result is a
:class:`~photutils.segmentation.SegmentationImage` object containing
the deblended segmentation image.


Centroids, Photometry, and Morphological Properties
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To calculate the centroids, photometry, and morphological properties
of sources from a segmentation map, please see `Segmentation
Photometry and Properties <segmentation.html>`_.


Local Peak Detection
--------------------

Photutils also includes a :func:`~photutils.detection.find_peaks`
function to find local peaks in an image that are above a specified
threshold value.  Peaks are the local maxima above a specified
threshold that separated by a a specified minimum number of pixels.
The return pixel coordinates are always integer (i.e., no centroiding
is performed, only the peak pixel is identified).
:func:`~photutils.detection.find_peaks` also supports a number of
options, including searching for peaks only within a segmentation
image or a specified footprint.  Please see the
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

    >>> from astropy.visualization import SqrtStretch
    >>> from astropy.visualization.mpl_normalize import ImageNormalize
    >>> import matplotlib.pylab as plt
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

    from astropy.visualization import SqrtStretch
    from astropy.visualization.mpl_normalize import ImageNormalize
    import matplotlib.pylab as plt
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
.. _image segmentation: http://en.wikipedia.org/wiki/Image_segmentation
.. _sep: https://github.com/kbarbary/sep
