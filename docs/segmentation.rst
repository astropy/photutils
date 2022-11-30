.. _image_segmentation:


Image Segmentation (`photutils.segmentation`)
=============================================

Introduction
------------
Photutils includes general-use functions to detect sources (both
point-like and extended) in an image using a process called `image
segmentation <https://en.wikipedia.org/wiki/Image_segmentation>`_. After
detecting sources using image segmentation, we can then measure their
photometry, centroids, and shape properties.


Source Extraction Using Image Segmentation
------------------------------------------
Image segmentation is a process of assigning a label to every pixel
in an image such that pixels with the same label are part of the same
source. Detected sources must have a minimum number of connected pixels
that are each greater than a specified threshold value in an image. The
threshold level is usually defined as some multiple of the background
noise (sigma level) above the background. The image is usually filtered
before thresholding to smooth the noise and maximize the detectability
of objects with a shape similar to the filter kernel.

Let's start by making a synthetic image provided by the
:ref:`photutils.datasets <datasets>` module::

    >>> from photutils.datasets import make_100gaussians_image
    >>> data = make_100gaussians_image()

Next, we need to subtract the background from the image. In this
example, we'll use the :class:`~photutils.background.Background2D` class
to produce a background and background noise image:

.. doctest-requires:: scipy

    >>> from photutils.background import Background2D, MedianBackground
    >>> bkg_estimator = MedianBackground()
    >>> bkg = Background2D(data, (50, 50), filter_size=(3, 3),
    ...                    bkg_estimator=bkg_estimator)
    >>> data -= bkg.background  # subtract the background

After subtracting the background, we need to define the detection
threshold. In this example, we'll define a 2D detection threshold image
using the background RMS image. We set the threshold at the 1.5-sigma (per
pixel) noise level:

.. doctest-requires:: scipy

    >>> threshold = 1.5 * bkg.background_rms

Next, let's convolve the data with a 2D Gaussian kernel with a FWHM of 3
pixels::

    >>> from astropy.convolution import convolve
    >>> from photutils.segmentation import make_2dgaussian_kernel
    >>> kernel = make_2dgaussian_kernel(3.0, size=5)  # FWHM = 3.0
    >>> convolved_data = convolve(data, kernel)

Now we are ready to detect the sources in the background-subtracted
convolved image. Let's find sources that have 10 connected pixels that
are each greater than the corresponding pixel-wise ``threshold`` level
defined above (i.e., 1.5 sigma per pixel above the background noise).

Note that by default "connected pixels" means "8-connected" pixels,
where pixels touch along their edges or corners. One can also use
"4-connected" pixels that touch only along their edges by setting
``connectivity=4``:

.. doctest-requires:: scipy

    >>> from photutils.segmentation import detect_sources
    >>> segment_map = detect_sources(convolved_data, threshold, npixels=10)
    >>> print(segment_map)
    <photutils.segmentation.core.SegmentationImage>
    shape: (300, 500)
    nlabels: 87
    labels: [ 1  2  3  4  5 ... 83 84 85 86 87]

The result is a :class:`~photutils.segmentation.SegmentationImage`
object with the same shape as the data, where detected sources are
labeled by different positive integer values. Background pixels
(non-sources) always have a value of zero. Because the segmentation
image is generated using image thresholding, the source segments
represent the isophotal footprints of each source.

Let's plot both the background-subtracted image and the segmentation
image showing the detected sources:

.. doctest-skip::

    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from astropy.visualization import SqrtStretch
    >>> from astropy.visualization.mpl_normalize import ImageNormalize
    >>> norm = ImageNormalize(stretch=SqrtStretch())
    >>> fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12.5))
    >>> ax1.imshow(data, origin='lower', cmap='Greys_r', norm=norm)
    >>> ax1.set_title('Background-subtracted Data')
    >>> ax2.imshow(segment_map, origin='lower', cmap=segment_map.cmap,
    ...            interpolation='nearest')
    >>> ax2.set_title('Segmentation Image')

.. plot::

    import matplotlib.pyplot as plt
    from astropy.convolution import convolve
    from astropy.visualization import SqrtStretch
    from astropy.visualization.mpl_normalize import ImageNormalize
    from photutils.background import Background2D, MedianBackground
    from photutils.datasets import make_100gaussians_image
    from photutils.segmentation import detect_sources, make_2dgaussian_kernel

    data = make_100gaussians_image()

    bkg_estimator = MedianBackground()
    bkg = Background2D(data, (50, 50), filter_size=(3, 3),
                       bkg_estimator=bkg_estimator)
    data -= bkg.background  # subtract the background

    threshold = 1.5 * bkg.background_rms

    kernel = make_2dgaussian_kernel(3.0, size=5)
    convolved_data = convolve(data, kernel)

    segment_map = detect_sources(convolved_data, threshold, npixels=10)

    norm = ImageNormalize(stretch=SqrtStretch())
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12.5))
    ax1.imshow(data, origin='lower', cmap='Greys_r', norm=norm)
    ax1.set_title('Background-subtracted Data')
    ax2.imshow(segment_map, origin='lower', cmap=segment_map.cmap,
               interpolation='nearest')
    ax2.set_title('Segmentation Image')
    plt.tight_layout()


Source Deblending
-----------------
In the example above, overlapping sources are detected as single
sources. Separating those sources requires a deblending procedure,
such as a multi-thresholding technique used by `SourceExtractor`_.
Photutils provides a :func:`~photutils.segmentation.deblend_sources`
function that deblends sources uses a combination
of multi-thresholding and `watershed segmentation
<https://en.wikipedia.org/wiki/Watershed_(image_processing)>`_. Note
that in order to deblend sources, they must be separated enough such
that there is a saddle point between them.

The amount of deblending can be controlled with the two
:func:`~photutils.segmentation.deblend_sources` keywords ``nlevels`` and
``contrast``. ``nlevels`` is the number of multi-thresholding levels to
use. ``contrast`` is the fraction of the total source flux that a local
peak must have to be considered as a separate object.

Here's a simple example of source deblending:

.. doctest-requires:: scipy, skimage

    >>> from photutils.segmentation import deblend_sources
    >>> segm_deblend = deblend_sources(convolved_data, segment_map,
    ...                                npixels=10, nlevels=32, contrast=0.001,
    ...                                progress_bar=False)

where ``segment_map`` is the
:class:`~photutils.segmentation.SegmentationImage` that was
generated by :func:`~photutils.segmentation.detect_sources`. Note
that the ``convolved_data`` and ``npixels`` input values should
match those used in :func:`~photutils.segmentation.detect_sources`
to generate ``segment_map``. The result is a new
:class:`~photutils.segmentation.SegmentationImage` object containing the
deblended segmentation image:

.. plot::

    import matplotlib.pyplot as plt
    from astropy.convolution import convolve
    from astropy.visualization import SqrtStretch
    from astropy.visualization.mpl_normalize import ImageNormalize
    from photutils.background import Background2D, MedianBackground
    from photutils.datasets import make_100gaussians_image
    from photutils.segmentation import (deblend_sources, detect_sources,
                                        make_2dgaussian_kernel)

    data = make_100gaussians_image()

    bkg_estimator = MedianBackground()
    bkg = Background2D(data, (50, 50), filter_size=(3, 3),
                       bkg_estimator=bkg_estimator)
    data -= bkg.background  # subtract the background

    threshold = 1.5 * bkg.background_rms

    kernel = make_2dgaussian_kernel(3.0, size=5)
    convolved_data = convolve(data, kernel)

    npixels = 10
    segment_map = detect_sources(convolved_data, threshold, npixels=npixels)
    segm_deblend = deblend_sources(convolved_data, segment_map,
                                   npixels=npixels, progress_bar=False)

    norm = ImageNormalize(stretch=SqrtStretch())
    fig, ax = plt.subplots(1, 1, figsize=(10, 6.5))
    ax.imshow(segm_deblend, origin='lower', cmap=segm_deblend.cmap,
              interpolation='nearest')
    ax.set_title('Deblended Segmentation Image')
    plt.tight_layout()

Let's plot one of the deblended sources:

.. plot::

    import matplotlib.pyplot as plt
    from astropy.convolution import convolve
    from astropy.visualization import SqrtStretch
    from astropy.visualization.mpl_normalize import ImageNormalize
    from photutils.background import Background2D, MedianBackground
    from photutils.datasets import make_100gaussians_image
    from photutils.segmentation import (deblend_sources, detect_sources,
                                        make_2dgaussian_kernel)

    data = make_100gaussians_image()

    bkg_estimator = MedianBackground()
    bkg = Background2D(data, (50, 50), filter_size=(3, 3),
                       bkg_estimator=bkg_estimator)
    data -= bkg.background  # subtract the background

    threshold = 1.5 * bkg.background_rms

    kernel = make_2dgaussian_kernel(3.0, size=5)
    convolved_data = convolve(data, kernel)

    npixels = 10
    segment_map = detect_sources(convolved_data, threshold, npixels=npixels)
    segm_deblend = deblend_sources(convolved_data, segment_map,
                                   npixels=npixels, progress_bar=False)

    norm = ImageNormalize(stretch=SqrtStretch())
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 4))
    slc = (slice(273, 297), slice(425, 444))
    ax1.imshow(data[slc], origin='lower')
    ax1.set_title('Background-subtracted Data')
    cmap1 = segment_map.cmap
    ax2.imshow(segment_map.data[slc], origin='lower', cmap=cmap1,
               interpolation='nearest')
    ax2.set_title('Original Segment')
    cmap2 = segm_deblend.cmap
    ax3.imshow(segm_deblend.data[slc], origin='lower', cmap=cmap2,
               interpolation='nearest')
    ax3.set_title('Deblended Segments')
    plt.tight_layout()


SourceFinder
------------
The :class:`~photutils.segmentation.SourceFinder` class
is a convenience class that combines the functionality
of `~photutils.segmentation.detect_sources` and
`~photutils.segmentation.deblend_sources`. After defining the object
with the desired detection and deblending parameters, you call it with
the background-subtracted (convolved) image and threshold:

.. doctest-requires:: scipy, skimage

    >>> from photutils.segmentation import SourceFinder
    >>> finder = SourceFinder(npixels=10, progress_bar=False)
    >>> segment_map = finder(convolved_data, threshold)
    >>> print(segment_map)
    <photutils.segmentation.core.SegmentationImage>
    shape: (300, 500)
    nlabels: 94
    labels: [ 1  2  3  4  5 ... 90 91 92 93 94]


Modifying a Segmentation Image
------------------------------
The :class:`~photutils.segmentation.SegmentationImage` object provides
several methods that can be used to visualize or modify itself (e.g.,
combining labels, removing labels, removing border segments) prior to
measuring source photometry and other source properties, including:

  * :meth:`~photutils.segmentation.SegmentationImage.reassign_label`:
    Reassign one or more label numbers.

  * :meth:`~photutils.segmentation.SegmentationImage.relabel_consecutive`:
    Reassign the label numbers consecutively, such that there are no
    missing label numbers.

  * :meth:`~photutils.segmentation.SegmentationImage.keep_labels`:
    Keep only the specified labels.

  * :meth:`~photutils.segmentation.SegmentationImage.remove_labels`:
    Remove one or more labels.

  * :meth:`~photutils.segmentation.SegmentationImage.remove_border_labels`:
    Remove labeled segments near the image border.

  * :meth:`~photutils.segmentation.SegmentationImage.remove_masked_labels`:
    Remove labeled segments located within a masked region.

  * :meth:`~photutils.segmentation.SegmentationImage.outline_segments`:
    Outline the labeled segments for plotting.


Photometry, Centroids, and Shape Properties
-------------------------------------------
The :class:`~photutils.segmentation.SourceCatalog` class is the primary
tool for measuring the photometry, centroids, and shape/morphological
properties of sources defined in a segmentation image. In its most
basic form, it takes as input the (background-subtracted) image and
the segmentation image. Usually the convolved image is also input,
from which the source centroids and shape/morphological properties are
measured (if not input, the unconvolved image is used instead).

Let's continue our example from above and measure the properties of the
detected sources:

.. doctest-requires:: scipy, skimage

    >>> from photutils.segmentation import SourceCatalog
    >>> cat = SourceCatalog(data, segm_deblend, convolved_data=convolved_data)
    >>> print(cat)
    <photutils.segmentation.catalog.SourceCatalog>
    Length: 94
    labels: [ 1  2  3  4  5 ... 90 91 92 93 94]

The source properties can be accessed using
`~photutils.segmentation.SourceCatalog` attributes or
output to an Astropy `~astropy.table.QTable` using the
:meth:`~photutils.segmentation.SourceCatalog.to_table` method. Please
see :class:`~photutils.segmentation.SourceCatalog` for the the many
properties that can be calculated for each source. More properties are
likely to be added in the future.

Here we'll use the
:meth:`~photutils.segmentation.SourceCatalog.to_table` method to
generate a `~astropy.table.QTable` of source properties. Each row in the
table represents a source. The columns represent the calculated source
properties. The ``label`` column corresponds to the label value in the
input segmentation image. Note that only a small subset of the source
properties are shown below:

.. doctest-requires:: scipy, skimage

    >>> tbl = cat.to_table()
    >>> tbl['xcentroid'].info.format = '.2f'  # optional format
    >>> tbl['ycentroid'].info.format = '.2f'
    >>> tbl['kron_flux'].info.format = '.2f'
    >>> print(tbl)
    label xcentroid ycentroid ... segment_fluxerr kron_flux kron_fluxerr
                              ...
    ----- --------- --------- ... --------------- --------- ------------
        1    235.31      1.45 ...             nan    509.74          nan
        2    493.92      5.79 ...             nan    544.31          nan
        3    207.42      9.81 ...             nan    722.26          nan
        4    364.86     11.11 ...             nan    704.23          nan
        5    258.27     11.94 ...             nan    661.22          nan
      ...       ...       ... ...             ...       ...          ...
       90    419.52    216.55 ...             nan    866.40          nan
       91     74.55    259.86 ...             nan    870.31          nan
       92     82.56    267.55 ...             nan    811.81          nan
       93    433.88    280.75 ...             nan    652.12          nan
       94    434.07    288.90 ...             nan    942.22          nan
    Length = 94 rows

The error columns are NaN because we did not input an error array (see
the :ref:`photutils-segmentation_errors` section below).

Let's plot the calculated elliptical Kron apertures (based on the shapes
of each source) on the data:

.. doctest-skip::

    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from astropy.visualization import simple_norm
    >>> norm = simple_norm(data, 'sqrt')
    >>> fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12.5))
    >>> ax1.imshow(data, origin='lower', cmap='Greys_r', norm=norm)
    >>> ax1.set_title('Data')
    >>> ax2.imshow(segm_deblend, origin='lower', cmap=segm_deblend.cmap,
    ...            interpolation='nearest')
    >>> ax2.set_title('Segmentation Image')
    >>> cat.plot_kron_apertures(ax=ax1, color='white', lw=1.5)
    >>> cat.plot_kron_apertures(ax=ax2, color='white', lw=1.5)

.. plot::

    import matplotlib.pyplot as plt
    from astropy.convolution import convolve
    from astropy.visualization import simple_norm
    from photutils.background import Background2D, MedianBackground
    from photutils.datasets import make_100gaussians_image
    from photutils.segmentation import (SourceCatalog, SourceFinder,
                                        make_2dgaussian_kernel)

    data = make_100gaussians_image()

    bkg_estimator = MedianBackground()
    bkg = Background2D(data, (50, 50), filter_size=(3, 3),
                       bkg_estimator=bkg_estimator)
    data -= bkg.background  # subtract the background

    threshold = 1.5 * bkg.background_rms

    kernel = make_2dgaussian_kernel(3.0, size=5)
    convolved_data = convolve(data, kernel)

    npixels = 10
    finder = SourceFinder(npixels=npixels, progress_bar=False)
    segment_map = finder(convolved_data, threshold)

    cat = SourceCatalog(data, segment_map, convolved_data=convolved_data)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12.5))
    norm = simple_norm(data, 'sqrt')
    ax1.imshow(data, origin='lower', cmap='Greys_r', norm=norm)
    ax1.set_title('Data with Kron apertures')
    ax2.imshow(segment_map, origin='lower', cmap=segment_map.cmap,
               interpolation='nearest')
    ax2.set_title('Segmentation Image with Kron apertures')
    cat.plot_kron_apertures(ax=ax1, color='white', lw=1.5)
    cat.plot_kron_apertures(ax=ax2, color='white', lw=1.5)
    plt.tight_layout()


We can also create a `~photutils.segmentation.SourceCatalog` object
containing only a specific subset of sources, defined by their
label numbers in the segmentation image:

.. doctest-requires:: scipy, skimage

    >>> cat = SourceCatalog(data, segm_deblend, convolved_data=convolved_data)
    >>> labels = [1, 5, 20, 50, 75, 80]
    >>> cat_subset = cat.get_labels(labels)
    >>> tbl2 = cat_subset.to_table()
    >>> tbl2['xcentroid'].info.format = '.2f'  # optional format
    >>> tbl2['ycentroid'].info.format = '.2f'
    >>> tbl2['kron_flux'].info.format = '.2f'
    >>> print(tbl2)
    label xcentroid ycentroid ... segment_fluxerr kron_flux kron_fluxerr
                              ...
    ----- --------- --------- ... --------------- --------- ------------
        1    235.31      1.45 ...             nan    509.74          nan
        5    258.27     11.94 ...             nan    661.22          nan
       20    346.99     66.83 ...             nan    811.70          nan
       50      5.29    178.94 ...             nan    614.46          nan
       75     42.96    249.88 ...             nan    617.18          nan
       80    130.75    297.10 ...             nan    246.91          nan

By default, the :meth:`~photutils.segmentation.SourceCatalog.to_table`
includes only a small subset of source properties. The output table
properties can be customized in the `~astropy.table.QTable` using the
``columns`` keyword:

.. doctest-requires:: scipy, skimage

    >>> cat = SourceCatalog(data, segm_deblend, convolved_data=convolved_data)
    >>> labels = [1, 5, 20, 50, 75, 80]
    >>> cat_subset = cat.get_labels(labels)
    >>> columns = ['label', 'xcentroid', 'ycentroid', 'area', 'segment_flux']
    >>> tbl3 = cat_subset.to_table(columns=columns)
    >>> tbl3['xcentroid'].info.format = '.4f'  # optional format
    >>> tbl3['ycentroid'].info.format = '.4f'
    >>> tbl3['segment_flux'].info.format = '.4f'
    >>> print(tbl3)
    label xcentroid ycentroid area segment_flux
                              pix2
    ----- --------- --------- ---- ------------
        1  235.3128    1.4483 47.0     445.6095
        5  258.2727   11.9418 79.0     472.4109
       20  346.9920   66.8323 98.0     571.3138
       50    5.2910  178.9449 57.0     257.4050
       75   42.9629  249.8825 79.0     428.7122
       80  130.7542  297.1048 25.0     108.7877

A `~astropy.wcs.WCS` transformation can also be input to
:class:`~photutils.segmentation.SourceCatalog` via the ``wcs`` keyword,
in which case the sky coordinates of the source centroids can be
calculated.


Background Properties
^^^^^^^^^^^^^^^^^^^^^
Like with :func:`~photutils.aperture.aperture_photometry`, the ``data``
array that is input to :class:`~photutils.segmentation.SourceCatalog`
should be background subtracted. If you input the background image
that was subtracted from the data into the ``background`` keyword
of :class:`~photutils.segmentation.SourceCatalog`, the background
properties for each source will also be calculated:

.. doctest-requires:: scipy, skimage

    >>> cat = SourceCatalog(data, segm_deblend, background=bkg.background)
    >>> labels = [1, 5, 20, 50, 75, 80]
    >>> cat_subset = cat.get_labels(labels)
    >>> columns = ['label', 'background_centroid', 'background_mean',
    ...            'background_sum']
    >>> tbl4 = cat_subset.to_table(columns=columns)
    >>> tbl4['background_centroid'].info.format = '{:.10f}'  # optional format
    >>> tbl4['background_mean'].info.format = '{:.10f}'
    >>> tbl4['background_sum'].info.format = '{:.10f}'
    >>> print(tbl4)
    label background_centroid background_mean background_sum
    ----- ------------------- --------------- --------------
        1        5.2031964942    5.2026137494 244.5228462210
        5        5.2369908637    5.2229743435 412.6149731353
       20        5.2392999510    5.2702094547 516.4805265603
       50        5.1910069327    5.2466148918 299.0570488310
       75        5.2010736005    5.2249844756 412.7737735748
       80        5.2669562370    5.2642197829 131.6054945730


.. _photutils-segmentation_errors:

Photometric Errors
^^^^^^^^^^^^^^^^^^
:class:`~photutils.segmentation.SourceCatalog` requires inputting a
*total* error array, i.e., the background-only error plus Poisson noise
due to individual sources. The :func:`~photutils.utils.calc_total_error`
function can be used to calculate the total error array from a
background-only error array and an effective gain.

The ``effective_gain``, which is the ratio of counts (electrons or
photons) to the units of the data, is used to include the Poisson noise
from the sources. ``effective_gain`` can either be a scalar value or a
2D image with the same shape as the ``data``. A 2D effective gain image
is useful for mosaic images that have variable depths (i.e., exposure
times) across the field. For example, one should use an exposure-time
map as the ``effective_gain`` for a variable depth mosaic image in
count-rate units.

Let's assume our synthetic data is in units of electrons per
second. In that case, the ``effective_gain`` should be the
exposure time (here we set it to 500 seconds). Here we use
:func:`~photutils.utils.calc_total_error` to calculate the total error
and input it into the :class:`~photutils.segmentation.SourceCatalog`
class. When a total ``error`` is input, the
`~photutils.segmentation.SourceCatalog.segment_fluxerr` and
`~photutils.segmentation.SourceCatalog.kron_fluxerr` properties are
calculated. `~photutils.segmentation.SourceCatalog.segment_flux`
and `~photutils.segmentation.SourceCatalog.segment_fluxerr` are the
instrumental flux and propagated flux error within the source segments:

.. doctest-requires:: scipy, skimage

    >>> from photutils.utils import calc_total_error
    >>> effective_gain = 500.0
    >>> error = calc_total_error(data, bkg.background_rms, effective_gain)
    >>> cat = SourceCatalog(data, segm_deblend, error=error)
    >>> labels = [1, 5, 20, 50, 75, 80]
    >>> cat_subset = cat.get_labels(labels)  # select a subset of objects
    >>> columns = ['label', 'xcentroid', 'ycentroid', 'segment_flux',
    ...            'segment_fluxerr']
    >>> tbl5 = cat_subset.to_table(columns=columns)
    >>> tbl5['xcentroid'].info.format = '{:.4f}'  # optional format
    >>> tbl5['ycentroid'].info.format = '{:.4f}'
    >>> tbl5['segment_flux'].info.format = '{:.4f}'
    >>> tbl5['segment_fluxerr'].info.format = '{:.4f}'
    >>> for col in tbl5.colnames:
    ...     tbl5[col].info.format = '%.8g'  # for consistent table output
    >>> print(tbl5)
    label xcentroid ycentroid segment_flux segment_fluxerr
    ----- --------- --------- ------------ ---------------
        1 235.24243 1.2020749    445.60947       14.634553
        5 258.29728 11.899213    472.41087       19.060458
       20 346.94871 66.802889    571.31384       21.674279
       50 5.2694265  178.9268    257.40504       16.431396
       75 43.030177 249.92847    428.71216       19.175229
       80 130.64209 297.11218    108.78768       10.882015


Pixel Masking
^^^^^^^^^^^^^
Pixels can be completely ignored/excluded (e.g., bad pixels) when
measuring the source properties by providing a boolean mask image
via the ``mask`` keyword (`True` pixel values are masked) to the
:class:`~photutils.segmentation.SourceCatalog` class. Note that
non-finite ``data`` values (NaN and inf) are automatically masked.


Filtering
^^^^^^^^^
`SourceExtractor`_'s centroid and morphological parameters are
always calculated from a convolved, or filtered, "detection" image
(``convolved_data``), i.e., the image used to define the segmentation
image. The usual downside of the filtering is the sources will be
made more circular than they actually are. If you wish to reproduce
`SourceExtractor`_ centroid and morphology results, then input the
``convolved_data`` (or ``kernel``, but not both). If ``convolved_data``
and ``kernel`` are both `None`, then the unfiltered ``data`` will be
used for the source centroid and morphological parameters. Note that
photometry is *always* performed on the unfiltered ``data``.


Reference/API
-------------
.. automodapi:: photutils.segmentation
    :no-heading:


.. _SourceExtractor:  https://sextractor.readthedocs.io/en/latest/
