.. _image_segmentation:

Image Segmentation (`photutils.segmentation`)
=============================================

Introduction
------------

Photutils includes a general-use function to detect sources (both
point-like and extended) in an image using a process called `image
segmentation <https://en.wikipedia.org/wiki/Image_segmentation>`_ in
the `computer vision <http://en.wikipedia.org/wiki/Computer_vision>`_
field.  After detecting sources using image segmentation, we can then
measure their photometry, centroids, and morphological properties by
using additional tools in Photutils.


Source Extraction Using Image Segmentation
------------------------------------------

Photutils provides tools to detect astronomical sources using image
segmentation, which is a process of assigning a label to every pixel
in an image such that pixels with the same label are part of the same
source.  The segmentation procedure implemented in Photutils is called
the threshold method, where detected sources must have a minimum
number of connected pixels that are each greater than a specified
threshold value in an image.  The threshold level is usually defined
at some multiple of the background standard deviation (sigma) above
the background.  The image can also be filtered before thresholding to
smooth the noise and maximize the detectability of objects with a
shape similar to the filter kernel.

In Photutils, source extraction is performed using the
:func:`~photutils.segmentation.detect_sources` function.  The
:func:`~photutils.detection.detect_threshold` tool is a convenience function
that generates a 2D detection threshold image using simple
sigma-clipped statistics to estimate the background and background
RMS.

For this example, let's detect sources in a synthetic image provided
by the `datasets <datasets.html>`_ module::

    >>> from photutils.datasets import make_100gaussians_image
    >>> data = make_100gaussians_image()

We will use :func:`~photutils.detection.detect_threshold` to produce a
detection threshold image.
:func:`~photutils.detection.detect_threshold` will estimate the
background and background RMS using sigma-clipped statistics, if they
are not input.  The threshold level is calculated using the ``snr``
input as the sigma level above the background.  Here we generate a
simple pixel-wise threshold at 3 sigma above the background::

    >>> from photutils import detect_threshold
    >>> threshold = detect_threshold(data, snr=3.)

For more sophisticated analyses, one should generate a 2D background
and background-only error image (e.g., from your data reduction or by
using :class:`~photutils.background.Background2D`).  In that case, a
3-sigma threshold image is simply::

    >>> threshold = bkg + (3.0 * bkg_rms)    # doctest: +SKIP

Note that if the threshold includes the background level (as above),
then the image input into
:func:`~photutils.segmentation.detect_sources` should *not* be
background subtracted.

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
    >>> import matplotlib.pyplot as plt
    >>> from astropy.visualization import SqrtStretch
    >>> from astropy.visualization.mpl_normalize import ImageNormalize
    >>> norm = ImageNormalize(stretch=SqrtStretch())
    >>> fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))
    >>> ax1.imshow(data, origin='lower', cmap='Greys_r', norm=norm)
    >>> ax2.imshow(segm, origin='lower', cmap=segm.cmap(random_state=12345))

.. plot::

    import numpy as np
    import matplotlib.pyplot as plt
    from astropy.stats import gaussian_fwhm_to_sigma
    from astropy.convolution import Gaussian2DKernel
    from astropy.visualization import SqrtStretch
    from astropy.visualization.mpl_normalize import ImageNormalize
    from photutils.datasets import make_100gaussians_image
    from photutils import detect_threshold, detect_sources
    data = make_100gaussians_image()
    threshold = detect_threshold(data, snr=3.)
    sigma = 2.0 * gaussian_fwhm_to_sigma    # FWHM = 2.
    kernel = Gaussian2DKernel(sigma, x_size=3, y_size=3)
    kernel.normalize()
    segm = detect_sources(data, threshold, npixels=5, filter_kernel=kernel)
    norm = ImageNormalize(stretch=SqrtStretch())
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))
    ax1.imshow(data, origin='lower', cmap='Greys_r', norm=norm)
    ax2.imshow(segm, origin='lower', cmap=segm.cmap(random_state=12345))

When the segmentation image is generated using image thresholding
(e.g., using :func:`~photutils.segmentation.detect_sources`), the
source segments effectively represent the isophotal footprint of each
source.


Source Deblending
^^^^^^^^^^^^^^^^^

In the example above, overlapping sources are detected as single
sources.  Separating those sources requires a deblending procedure,
such as a multi-thresholding technique used by `SExtractor
<http://www.astromatic.net/software/sextractor>`_.  Photutils provides
an experimental :func:`~photutils.segmentation.deblend_sources`
function that deblends sources uses a combination of
multi-thresholding and `watershed segmentation
<https://en.wikipedia.org/wiki/Watershed_(image_processing)>`_.  Note
that in order to deblend sources, they must be separated enough such
that there is a saddle between them.

Here's a simple example of source deblending:

.. doctest-requires:: scipy, skimage

    >>> from photutils import deblend_sources
    >>> segm_deblend = deblend_sources(data, segm, npixels=5,
    ...                                filter_kernel=kernel)

where ``segm`` is the
:class:`~photutils.segmentation.SegmentationImage` that was generated
by :func:`~photutils.segmentation.detect_sources`.  Note that the
``npixels`` and ``filter_kernel`` input values should match those used
in :func:`~photutils.segmentation.detect_sources`.  The result is a
:class:`~photutils.segmentation.SegmentationImage` object containing
the deblended segmentation image.


Modifying a Segmentation Image
------------------------------

The :class:`~photutils.segmentation.SegmentationImage` object provides
several methods that can be used to modify itself (e.g., combining
labels, removing labels, removing border segments) prior to measuring
source photometry and other source properties, including:

  * :meth:`~photutils.segmentation.SegmentationImage.relabel`:
    Relabel one or more label numbers.

  * :meth:`~photutils.segmentation.SegmentationImage.relabel_sequential`:
    Relable the label numbers sequentially.

  * :meth:`~photutils.segmentation.SegmentationImage.keep_labels`:
    Keep only certain label numbers.

  * :meth:`~photutils.segmentation.SegmentationImage.remove_labels`:
    Remove one or more label numbers.

  * :meth:`~photutils.segmentation.SegmentationImage.remove_border_labels`:
    Remove labeled segments near the image border.

  * :meth:`~photutils.segmentation.SegmentationImage.remove_masked_labels`:
    Remove labeled segments located within a masked region.

  * :meth:`~photutils.segmentation.SegmentationImage.outline_segments`:
    Outline the labeled segments for plotting.


Centroids, Photometry, and Morphological Properties
---------------------------------------------------

The :func:`~photutils.segmentation.source_properties` function is the
primary tool for measuring the centroids, photometry, and
morphological properties of sources defined in a segmentation image.
When the segmentation image is generated using image thresholding
(e.g., using :func:`~photutils.segmentation.detect_sources`), the
source segments effectively represent the isophotal footprint of each
source and the resulting photometry is effectively isophotal
photometry.

:func:`~photutils.segmentation.source_properties` returns a
:class:`~photutils.SourceCatalog` object, which effectively acts like
a list of :class:`~photutils.segmentation.SourceProperties` objects,
one for each segmented source (or a specified subset of sources).  An
Astropy `~astropy.table.QTable` of source properties can be generated
using the :meth:`~photutils.SourceCatalog.to_table` method.  Please
see :class:`~photutils.segmentation.SourceProperties` for the list of
the many properties that are calculated for each source.  More
properties are likely to be added in the future.

Let's detect sources and measure their properties in a synthetic
image.  For this example, we will use the
:class:`~photutils.background.Background2D` class to produce a
background and background noise image.  We define a 2D detection
threshold image using the background and background RMS images.  We
set the threshold at 3 sigma above the background:

.. doctest-requires:: scipy

    >>> from astropy.convolution import Gaussian2DKernel
    >>> from photutils.datasets import make_100gaussians_image
    >>> from photutils import Background2D, MedianBackground
    >>> from photutils import detect_threshold, detect_sources
    >>> data = make_100gaussians_image()
    >>> bkg_estimator = MedianBackground()
    >>> bkg = Background2D(data, (50, 50), filter_size=(3, 3),
    ...                    bkg_estimator=bkg_estimator)
    >>> threshold = bkg.background + (3. * bkg.background_rms)

Now we find sources that have 5 connected pixels that are each greater
than the corresponding pixel-wise threshold image defined above.
Because the threshold includes the background, we do not subtract the
background from the data here.  We also input a 2D circular Gaussian
kernel with a FWHM of 2 pixels to filter the image prior to
thresholding:

.. doctest-requires:: scipy, skimage

    >>> from astropy.stats import gaussian_fwhm_to_sigma
    >>> sigma = 2.0 * gaussian_fwhm_to_sigma    # FWHM = 2.
    >>> kernel = Gaussian2DKernel(sigma, x_size=3, y_size=3)
    >>> kernel.normalize()
    >>> segm = detect_sources(data, threshold, npixels=5, filter_kernel=kernel)

The result is a :class:`~photutils.segmentation.SegmentationImage`
where sources are labeled by different positive integer values.  Now
let's measure the properties of the detected sources defined in the
segmentation image with the minimum number of inputs to
:func:`~photutils.segmentation.source_properties`:

.. doctest-requires:: scipy, skimage

    >>> from photutils import source_properties
    >>> cat = source_properties(data, segm)
    >>> tbl = cat.to_table()
    >>> tbl['xcentroid'].info.format = '.10f'  # optional format
    >>> tbl['ycentroid'].info.format = '.10f'
    >>> tbl['cxy'].info.format = '.10f'
    >>> tbl['cyy'].info.format = '.10f'
    >>> print(tbl)
     id   xcentroid      ycentroid    ...      cxy          cyy
             pix            pix       ...    1 / pix2     1 / pix2
    --- -------------- -------------- ... ------------- ------------
      1 235.1877193594   1.0991961528 ... -0.1920746278 1.2174907202
      2 494.1399411137   5.7704424681 ... -0.5417755959 1.0244063365
      3 207.3757266577  10.0753101977 ...  0.7764083298 0.4650609454
      4 364.6895486330  10.8904591886 ... -0.5478887625 0.3040810336
      5 258.1927719916  11.9617673653 ...  0.0443061873 0.3218333804
    ...            ...            ... ...           ...          ...
     82  74.4566900469 259.8333035016 ...  0.4789130934 0.5657327432
     83  82.5392499545 267.7189336671 ...  0.0675912618 0.2448815867
     84 477.6743849969 267.8914460476 ... -0.0214056255 0.3919147600
     85 139.7637841053 275.0413983586 ...  0.2329325365 0.3523911744
     86 434.0406656782 285.6070270358 ... -0.0607421731 0.0555135558
    Length = 86 rows

Let's use the measured morphological properties to define approximate
isophotal ellipses for each source:

.. doctest-requires:: scipy, skimage

    >>> from photutils import source_properties, EllipticalAperture
    >>> cat = source_properties(data, segm)
    >>> r = 3.    # approximate isophotal extent
    >>> apertures = []
    >>> for obj in cat:
    ...     position = (obj.xcentroid.value, obj.ycentroid.value)
    ...     a = obj.semimajor_axis_sigma.value * r
    ...     b = obj.semiminor_axis_sigma.value * r
    ...     theta = obj.orientation.value
    ...     apertures.append(EllipticalAperture(position, a, b, theta=theta))

Now let's plot the results:

.. doctest-skip::

    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from astropy.visualization import SqrtStretch
    >>> from astropy.visualization.mpl_normalize import ImageNormalize
    >>> norm = ImageNormalize(stretch=SqrtStretch())
    >>> fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))
    >>> ax1.imshow(data, origin='lower', cmap='Greys_r', norm=norm)
    >>> ax2.imshow(segm, origin='lower', cmap=segm.cmap(random_state=12345))
    >>> for aperture in apertures:
    ...     aperture.plot(color='blue', lw=1.5, alpha=0.5, ax=ax1)
    ...     aperture.plot(color='white', lw=1.5, alpha=1.0, ax=ax2)

.. plot::

    import numpy as np
    import matplotlib.pyplot as plt
    from astropy.stats import gaussian_fwhm_to_sigma
    from astropy.convolution import Gaussian2DKernel
    from astropy.visualization import SqrtStretch
    from astropy.visualization.mpl_normalize import ImageNormalize
    from photutils.datasets import make_100gaussians_image
    from photutils import Background2D, MedianBackground
    from photutils import detect_threshold, detect_sources
    from photutils import source_properties
    from photutils import EllipticalAperture
    data = make_100gaussians_image()
    bkg_estimator = MedianBackground()
    bkg = Background2D(data, (50, 50), filter_size=(3, 3),
                       bkg_estimator=bkg_estimator)
    threshold = bkg.background + (3. * bkg.background_rms)
    sigma = 2.0 * gaussian_fwhm_to_sigma    # FWHM = 2.
    kernel = Gaussian2DKernel(sigma, x_size=3, y_size=3)
    kernel.normalize()
    segm = detect_sources(data, threshold, npixels=5, filter_kernel=kernel)
    cat = source_properties(data, segm)
    apertures = []
    for obj in cat:
        position = (obj.xcentroid.value, obj.ycentroid.value)
        a = obj.semimajor_axis_sigma.value * 3.
        b = obj.semiminor_axis_sigma.value * 3.
        theta = obj.orientation.value
        apertures.append(EllipticalAperture(position, a, b, theta=theta))
    norm = ImageNormalize(stretch=SqrtStretch())
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))
    ax1.imshow(data, origin='lower', cmap='Greys_r', norm=norm)
    ax2.imshow(segm, origin='lower', cmap=segm.cmap(random_state=12345))
    for aperture in apertures:
        aperture.plot(color='blue', lw=1.5, alpha=0.5, ax=ax1)
        aperture.plot(color='white', lw=1.5, alpha=1.0, ax=ax2)

We can also specify a specific subset of sources, defined by their
labels in the segmentation image:

.. doctest-requires:: scipy, skimage

    >>> labels = [1, 5, 20, 50, 75, 80]
    >>> cat = source_properties(data, segm, labels=labels)
    >>> tbl2 = cat.to_table()
    >>> tbl2['xcentroid'].info.format = '.2f'  # optional format
    >>> tbl2['ycentroid'].info.format = '.2f'
    >>> tbl2['cxx'].info.format = '.2f'
    >>> tbl2['cxy'].info.format = '.2f'
    >>> tbl2['cyy'].info.format = '.2f'
    >>> print(tbl2)
     id xcentroid ycentroid sky_centroid ...   cxx      cxy      cyy
           pix       pix                 ... 1 / pix2 1 / pix2 1 / pix2
    --- --------- --------- ------------ ... -------- -------- --------
      1    235.19      1.10         None ...     0.23    -0.19     1.22
      5    258.19     11.96         None ...     0.73     0.04     0.32
     20    347.18     66.55         None ...     0.47     0.12     0.36
     50    380.80    174.42         None ...     1.36    -1.03     1.28
     75     32.18    241.16         None ...     0.34     0.20     0.60
     80    355.61    252.14         None ...     0.32     0.18     0.40

By default, the :meth:`~photutils.SourceCatalog.to_table` method will
include most scalar-valued properties from
:class:`~photutils.segmentation.SourceProperties`, but a subset of
properties can also be specified (or excluded) in the
`~astropy.table.QTable`:

.. doctest-requires:: scipy, skimage

    >>> labels = [1, 5, 20, 50, 75, 80]
    >>> cat = source_properties(data, segm, labels=labels)
    >>> columns = ['id', 'xcentroid', 'ycentroid', 'source_sum', 'area']
    >>> tbl3 = cat.to_table(columns=columns)
    >>> tbl3['xcentroid'].info.format = '.4f'  # optional format
    >>> tbl3['ycentroid'].info.format = '.4f'
    >>> tbl3['source_sum'].info.format = '.4f'
    >>> print(tbl3)
     id xcentroid ycentroid source_sum area
           pix       pix               pix2
    --- --------- --------- ---------- ----
      1  235.1877    1.0992   496.6356 27.0
      5  258.1928   11.9618   347.6113 25.0
     20  347.1776   66.5510   415.9926 31.0
     50  380.7969  174.4185   145.7264 11.0
     75   32.1762  241.1585   398.4114 29.0
     80  355.6148  252.1423   906.4226 45.0

A `~astropy.wcs.WCS` transformation can also be input to
:func:`~photutils.segmentation.source_properties` via the ``wcs``
keyword, in which case the sky coordinates at the source centroids
will be returned.


Background Properties
^^^^^^^^^^^^^^^^^^^^^

Like with :func:`~photutils.aperture_photometry`, the ``data`` array
that is input to :func:`~photutils.segmentation.source_properties`
should be background subtracted.  If you input your background image
(which should have already been subtracted from the data) into the
``background`` keyword of
:func:`~photutils.segmentation.source_properties`, the background
properties for each source will also be calculated:

.. doctest-requires:: scipy, skimage

    >>> labels = [1, 5, 20, 50, 75, 80]
    >>> cat = source_properties(data, segm, labels=labels,
    ...                         background=bkg.background)
    >>> columns = ['id', 'background_at_centroid', 'background_mean',
    ...            'background_sum']
    >>> tbl4 = cat.to_table(columns=columns)
    >>> tbl4['background_at_centroid'].info.format = '{:.10f}'  # optional format
    >>> tbl4['background_mean'].info.format = '{:.10f}'
    >>> tbl4['background_sum'].info.format = '{:.10f}'
    >>> print(tbl4)
     id background_at_centroid background_mean background_sum
    --- ---------------------- --------------- --------------
      1           5.2020326493    5.2021208257 140.4572622937
      5           5.2137810422    5.2137801450 130.3445036251
     20           5.2788524399    5.2787718244 163.6419265556
     50           5.1986504100    5.1986157424  57.1847731664
     75           5.2106279087    5.2106057357 151.1075663349
     80           5.1249167847    5.1250208080 230.6259363620


Photometric Errors
^^^^^^^^^^^^^^^^^^

:func:`~photutils.segmentation.source_properties` requires inputting a
*total* error array, i.e. the background-only error plus Poisson noise
due to individual sources.  The
:func:`~photutils.utils.calc_total_error` function can be used to
calculate the total error array from a background-only error array and
an effective gain.

The ``effective_gain``, which is the ratio of counts (electrons or
photons) to the units of the data, is used to include the Poisson
noise from the sources.  ``effective_gain`` can either be a scalar
value or a 2D image with the same shape as the ``data``.  A 2D
effective gain image is useful for mosaic images that have variable
depths (i.e., exposure times) across the field. For example, one
should use an exposure-time map as the ``effective_gain`` for a
variable depth mosaic image in count-rate units.

Let's assume our synthetic data is in units of electrons per second.
In that case, the ``effective_gain`` should be the exposure time (here
we set it to 500 seconds):

.. doctest-requires:: scipy, skimage

    >>> from photutils.utils import calc_total_error
    >>> labels = [1, 5, 20, 50, 75, 80]
    >>> effective_gain = 500.
    >>> error = calc_total_error(data, bkg.background_rms, effective_gain)
    >>> cat = source_properties(data, segm, labels=labels, error=error)
    >>> columns = ['id', 'xcentroid', 'ycentroid', 'source_sum',
    ...            'source_sum_err']
    >>> tbl5 = cat.to_table(columns=columns)
    >>> tbl5['xcentroid'].info.format = '{:.4f}'  # optional format
    >>> tbl5['ycentroid'].info.format = '{:.4f}'
    >>> print(tbl5)
     id xcentroid ycentroid   source_sum  source_sum_err
           pix       pix
    --- --------- --------- ------------- --------------
      1  235.1877    1.0992 496.635623206  11.0788667038
      5  258.1928   11.9618 347.611342072   10.723068215
     20  347.1776   66.5510 415.992569678  12.1782078398
     50  380.7969  174.4185 145.726417518  7.29536295106
     75   32.1762  241.1585 398.411403711   11.553412812
     80  355.6148  252.1423 906.422600037  13.7686828317

`~photutils.segmentation.SourceProperties.source_sum` and
`~photutils.segmentation.SourceProperties.source_sum_err` are the
instrumental flux and propagated flux error within the source
segments.


Pixel Masking
^^^^^^^^^^^^^

Pixels can be completely ignored/excluded (e.g. bad pixels) when
measuring the source properties by providing a boolean mask image via
the ``mask`` keyword (`True` pixel values are masked) to the
:func:`~photutils.segmentation.source_properties` function or
:class:`~photutils.segmentation.SourceProperties` class.


Filtering
^^^^^^^^^

`SExtractor`_'s centroid and morphological parameters are always
calculated from a filtered "detection" image.  The usual downside of
the filtering is the sources will be made more circular than they
actually are.  If you wish to reproduce `SExtractor`_ results, then
use the ``filter_kernel`` keyword to
:func:`~photutils.segmentation.source_properties` to filter the
``data`` prior to centroid and morphological measurements.   The input
kernel should be the same one used to define the source segments in
:func:`~photutils.segmentation.detect_sources`.  If ``filter_kernel``
is `None`, then the centroid and morphological measurements will be
performed on the unfiltered ``data``.  Note that photometry is
*always* performed on the unfiltered ``data``.


Reference/API
-------------

.. automodapi:: photutils.segmentation
    :no-heading:


.. _SExtractor:  http://www.astromatic.net/software/sextractor
