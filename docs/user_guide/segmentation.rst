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
to produce a background and background noise image::

    >>> from photutils.background import Background2D, MedianBackground
    >>> bkg_estimator = MedianBackground()
    >>> bkg = Background2D(data, (50, 50), filter_size=(3, 3),
    ...                    bkg_estimator=bkg_estimator)
    >>> data -= bkg.background  # subtract the background

After subtracting the background, we need to define the detection
threshold. In this example, we'll define a 2D detection threshold image
using the background RMS image. We set the threshold at the 1.5-sigma (per
pixel) noise level::

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
``connectivity=4``::

    >>> from photutils.segmentation import detect_sources
    >>> segment_map = detect_sources(convolved_data, threshold, n_pixels=10)
    >>> print(segment_map)
    <photutils.segmentation.core.SegmentationImage>
    shape: (300, 500)
    n_labels: 86
    labels: [ 1  2  3  4  5 ... 82 83 84 85 86]

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
    >>> from astropy.visualization import simple_norm
    >>> norm = simple_norm(data, 'sqrt', percent=99.5)
    >>> fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12.5))
    >>> ax1.imshow(data, norm=norm, origin='lower')
    >>> ax1.set_title('Background-subtracted Data')
    >>> segment_map.imshow(ax=ax2)
    >>> ax2.set_title('Segmentation Image')

.. plot::

    import matplotlib.pyplot as plt
    from astropy.convolution import convolve
    from astropy.visualization import simple_norm
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

    segment_map = detect_sources(convolved_data, threshold, n_pixels=10)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12.5))
    norm = simple_norm(data, 'sqrt', percent=99.5)
    ax1.imshow(data, norm=norm, origin='lower')
    ax1.set_title('Background-subtracted Data')
    segment_map.imshow(ax=ax2)
    ax2.set_title('Segmentation Image')
    fig.tight_layout()


Source Deblending
-----------------

In the example above, overlapping sources are detected as single
sources. Separating those sources requires a deblending procedure,
such as a multi-thresholding technique used by `SourceExtractor`_.
Photutils provides a :func:`~photutils.segmentation.deblend_sources`
function that deblends sources using a combination
of multi-thresholding and `watershed segmentation
<https://en.wikipedia.org/wiki/Watershed_(image_processing)>`_. Note
that in order to deblend sources, they must be separated enough that ere
this a saddle point between them.

The amount of deblending can be controlled with the two
:func:`~photutils.segmentation.deblend_sources` keywords ``n_levels``
and ``contrast``. ``n_levels`` is the number of multi-thresholding
levels to use. ``contrast`` is the fraction of the total source flux
that a local peak must have to be considered as a separate object.

Here's a simple example of source deblending:

.. doctest-requires:: skimage

    >>> from photutils.segmentation import deblend_sources
    >>> segment_map2 = deblend_sources(convolved_data, segment_map,
    ...                                n_pixels=10, n_levels=32, contrast=0.001,
    ...                                progress_bar=False)

where ``segment_map`` is the
:class:`~photutils.segmentation.SegmentationImage` that was
generated by :func:`~photutils.segmentation.detect_sources`. Note
that the ``convolved_data`` and ``n_pixels`` input values should
match those used in :func:`~photutils.segmentation.detect_sources`
to generate ``segment_map``. The result is a new
:class:`~photutils.segmentation.SegmentationImage` object containing the
deblended segmentation image:

.. plot::

    import matplotlib.pyplot as plt
    from astropy.convolution import convolve
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

    n_pixels = 10
    segment_map = detect_sources(convolved_data, threshold, n_pixels=n_pixels)
    deblended_segment_map = deblend_sources(convolved_data, segment_map,
                                            n_pixels=n_pixels,
                                            progress_bar=False)

    fig, ax = plt.subplots(1, 1, figsize=(10, 6.5))
    deblended_segment_map.imshow(ax=ax)
    ax.set_title('Deblended Segmentation Image')
    fig.tight_layout()

Let's plot one of the deblended sources:

.. plot::

    import matplotlib.pyplot as plt
    from astropy.convolution import convolve
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

    n_pixels = 10
    segment_map = detect_sources(convolved_data, threshold, n_pixels=n_pixels)
    deblended_segment_map = deblend_sources(convolved_data, segment_map,
                                            n_pixels=n_pixels,
                                            progress_bar=False)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 4))
    slc = (slice(273, 297), slice(425, 444))
    ax1.imshow(data[slc], origin='lower')
    ax1.set_title('Background-subtracted Data')

    segm_cutout = segment_map[slc]
    segm_cutout.imshow(ax=ax2, cmap=segment_map.cmap)
    ax2.set_title('Original Segment')

    deblended_segm_cutout = deblended_segment_map[slc]
    deblended_segm_cutout.imshow(ax=ax3, cmap=deblended_segment_map.cmap)
    ax3.set_title('Deblended Segments')
    fig.tight_layout()


SourceFinder
------------

The :class:`~photutils.segmentation.SourceFinder` class
is a convenience class that combines the functionality
of `~photutils.segmentation.detect_sources` and
`~photutils.segmentation.deblend_sources`. After defining the object
with the desired detection and deblending parameters, you call it with
the background-subtracted (convolved) image and threshold:

.. doctest-requires:: skimage

    >>> from photutils.segmentation import SourceFinder
    >>> finder = SourceFinder(n_pixels=10, progress_bar=False)
    >>> segment_map = finder(convolved_data, threshold)
    >>> print(segment_map)
    <photutils.segmentation.core.SegmentationImage>
    shape: (300, 500)
    n_labels: 93
    labels: [ 1  2  3  4  5 ... 89 90 91 92 93]


Modifying a Segmentation Image
------------------------------

The :class:`~photutils.segmentation.SegmentationImage` object provides
several methods that can be used to modify itself (e.g.,
combining labels, removing labels, removing border segments) prior to
measuring source photometry and other source properties, including:

* :meth:`~photutils.segmentation.SegmentationImage.relabel_consecutive`:
  Reassign the label numbers consecutively, such that there are no
  missing label numbers.

* :meth:`~photutils.segmentation.SegmentationImage.reassign_labels`:
  Reassign one or more label numbers.

* :meth:`~photutils.segmentation.SegmentationImage.keep_labels`:
  Keep only the specified labels.

* :meth:`~photutils.segmentation.SegmentationImage.remove_labels`:
  Remove one or more labels.

* :meth:`~photutils.segmentation.SegmentationImage.remove_border_labels`:
  Remove labeled segments near the image border.

* :meth:`~photutils.segmentation.SegmentationImage.remove_masked_labels`:
  Remove labeled segments located within a masked region.

Here's a simple example of removing border labels and relabeling the
result:

.. doctest-requires:: skimage

    >>> segment_map3 = segment_map.copy()
    >>> segment_map3.remove_border_labels(border_width=10, relabel=True)
    >>> print(segment_map3)
    <photutils.segmentation.core.SegmentationImage>
    shape: (300, 500)
    n_labels: 79
    labels: [ 1  2  3  4  5 ... 75 76 77 78 79]


Source Masks
------------

The :meth:`~photutils.segmentation.SegmentationImage.make_source_mask`
method can be used to create a boolean source mask from a segmentation
image. The source mask can be used, for example, to mask sources
when estimating the background level. The source mask can optionally
be dilated using the ``size`` or ``footprint`` keyword to mask a
larger area around each source. Dilating the source mask is useful for
excluding the faint wings of sources when estimating the background:

.. doctest-requires:: skimage

    >>> mask = segment_map.make_source_mask()
    >>> dilated_mask = segment_map.make_source_mask(size=11)

A circular footprint can also be used to dilate the source mask:

.. doctest-requires:: skimage

    >>> from photutils.utils import circular_footprint
    >>> footprint = circular_footprint(radius=5)
    >>> dilated_mask2 = segment_map.make_source_mask(footprint=footprint)

Note that using a square footprint (via the ``size`` keyword) is much
faster than using other shapes (e.g., a circular footprint).


Polygons and Regions
--------------------

The :class:`~photutils.segmentation.SegmentationImage` class
provides several methods for converting source segments into
polygon representations and `regions`_ objects. These are useful
for visualization and for exporting source segments to other tools.
Note that these methods require the `rasterio`_, `shapely`_, and/or
`regions`_ optional packages.

The :attr:`~photutils.segmentation.SegmentationImage.polygons` property
returns a list of `Shapely`_ polygon objects representing each source
segment:

.. doctest-requires:: rasterio, shapely

    >>> polygons = segment_map.polygons

The :meth:`~photutils.segmentation.SegmentationImage.to_patches` method
returns a list of `~matplotlib.patches.PathPatch` objects for the source
segments, which can be overlaid on plots:

.. doctest-requires:: matplotlib, rasterio, shapely

    >>> patches = segment_map.to_patches(edgecolor='white', lw=1.5)

For convenience, the
:meth:`~photutils.segmentation.SegmentationImage.plot_patches` method
will plot these patches directly on an existing matplotlib axes:

.. doctest-skip::

    >>> patches = segment_map.plot_patches(edgecolor='white', lw=1.5)

For working with individual labels, the
:meth:`~photutils.segmentation.SegmentationImage.get_polygon`,
:meth:`~photutils.segmentation.SegmentationImage.get_polygons`,
:meth:`~photutils.segmentation.SegmentationImage.get_patch`,
:meth:`~photutils.segmentation.SegmentationImage.get_patches`,
:meth:`~photutils.segmentation.SegmentationImage.get_region`, and
:meth:`~photutils.segmentation.SegmentationImage.get_regions` methods
are significantly faster than the bulk properties when only a subset of
labels is needed:

.. doctest-requires:: matplotlib, rasterio, regions, shapely

    >>> polygon = segment_map.get_polygon(1)
    >>> patch = segment_map.get_patch(1, edgecolor='red', lw=2)
    >>> region = segment_map.get_region(1)

Here's an example showing the source polygons overlaid on both the
segmentation image and the science image:

.. plot::

    import matplotlib.pyplot as plt
    from astropy.convolution import convolve
    from astropy.visualization import simple_norm
    from photutils.background import Background2D, MedianBackground
    from photutils.datasets import make_100gaussians_image
    from photutils.segmentation import (SourceFinder,
                                        make_2dgaussian_kernel)

    data = make_100gaussians_image()

    bkg_estimator = MedianBackground()
    bkg = Background2D(data, (50, 50), filter_size=(3, 3),
                       bkg_estimator=bkg_estimator)
    data -= bkg.background

    threshold = 1.5 * bkg.background_rms

    kernel = make_2dgaussian_kernel(3.0, size=5)
    convolved_data = convolve(data, kernel)

    finder = SourceFinder(n_pixels=10, progress_bar=False)
    segment_map = finder(convolved_data, threshold)

    fig, (ax1, ax2) = plt.subplots(ncols=1, nrows=2, figsize=(10, 12.5))
    segment_map.imshow(ax=ax1)
    ax1.set_title('Segmentation Image')
    segment_map.plot_patches(ax=ax1, edgecolor='white', lw=1.5)
    norm = simple_norm(data, 'sqrt', percent=99.5)
    ax2.imshow(data, norm=norm, origin='lower')
    ax2.set_title('Background-subtracted Data')
    segment_map.plot_patches(ax=ax2, edgecolor='white', lw=1.5)
    fig.tight_layout()

To convert the source segments to `regions`_
`~regions.PolygonPixelRegion` objects, use the
:meth:`~photutils.segmentation.SegmentationImage.to_regions` method:

.. doctest-requires:: rasterio, regions, shapely

    >>> regions = segment_map.to_regions()


.. _rasterio: https://rasterio.readthedocs.io/en/stable/
.. _shapely: https://shapely.readthedocs.io/en/stable/
.. _regions: https://astropy-regions.readthedocs.io/en/stable/


Segment Objects
---------------

The :class:`~photutils.segmentation.SegmentationImage` class provides
:class:`~photutils.segmentation.Segment` objects that encapsulate
individual labeled regions. Each `~photutils.segmentation.Segment`
contains the label number, bounding-box slices, bounding box, area, and
(optionally) the Shapely polygon outline.

The :attr:`~photutils.segmentation.SegmentationImage.segments` property
returns a list of `~photutils.segmentation.Segment` objects for all
labels::

    >>> segments = segment_map.segments
    >>> segments[0]
    <photutils.segmentation.core.Segment>
    label: 1
    slices: (slice(0, 5, None), slice(230, 242, None))
    area: 47

For working with individual labels, the
:meth:`~photutils.segmentation.SegmentationImage.get_segment` and
:meth:`~photutils.segmentation.SegmentationImage.get_segments` methods
are significantly faster than the bulk ``segments`` property when only a
subset of labels is needed::

    >>> segment = segment_map.get_segment(1)
    >>> print(segment.label, segment.area)
    1 47

    >>> segments = segment_map.get_segments([1, 5, 10])
    >>> [segment.label for segment in segments]
    [np.int32(1), np.int32(5), np.int32(10)]

A `~photutils.segmentation.Segment` can provide cutout arrays
of the segment data and of arbitrary data arrays via its
:attr:`~photutils.segmentation.Segment.data` property and
:meth:`~photutils.segmentation.Segment.make_cutout` method::

    >>> segment = segment_map.get_segment(1)
    >>> segment_cutout = segment.data  # labeled region, others set to 0
    >>> data_cutout = segment.make_cutout(data)  # science data cutout


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

.. doctest-requires:: skimage

    >>> from photutils.segmentation import SourceCatalog
    >>> cat = SourceCatalog(data, segment_map, convolved_data=convolved_data)
    >>> print(cat)
    <photutils.segmentation.catalog.SourceCatalog>
    Length: 93
    labels: [ 1  2  3  4  5 ... 89 90 91 92 93]

The source properties can be accessed using
`~photutils.segmentation.SourceCatalog` attributes or
output to an Astropy `~astropy.table.QTable` using the
:meth:`~photutils.segmentation.SourceCatalog.to_table` method. Please
see :class:`~photutils.segmentation.SourceCatalog` for the many
properties that can be calculated for each source. More properties are
likely to be added in the future.

Here we'll use the
:meth:`~photutils.segmentation.SourceCatalog.to_table` method to
generate a `~astropy.table.QTable` of source properties. Each row in the
table represents a source. The columns represent the calculated source
properties. The ``label`` column corresponds to the label value in the
input segmentation image. Note that only a small subset of the source
properties are shown below:

.. doctest-requires:: skimage

    >>> tbl = cat.to_table()
    >>> tbl['x_centroid'].info.format = '.2f'  # optional format
    >>> tbl['y_centroid'].info.format = '.2f'
    >>> tbl['kron_flux'].info.format = '.2f'
    >>> print(tbl)
    label x_centroid y_centroid ... segment_flux_err kron_flux kron_flux_err
                                ...
    ----- ---------- ---------- ... ---------------- --------- -------------
        1     235.38       1.44 ...              nan    490.35           nan
        2     493.78       5.84 ...              nan    489.37           nan
        3     207.29      10.26 ...              nan    694.24           nan
        4     364.87      11.13 ...              nan    681.20           nan
        5     257.85      12.18 ...              nan    748.18           nan
      ...        ...        ... ...              ...       ...           ...
       89     292.77     244.93 ...              nan    792.63           nan
       90      32.66     241.24 ...              nan    930.77           nan
       91      42.60     249.43 ...              nan    580.54           nan
       92     433.80     280.74 ...              nan    663.44           nan
       93     434.03     288.88 ...              nan    879.64           nan
    Length = 93 rows

The error columns are NaN because we did not input an error array (see
the :ref:`photutils-segmentation_errors` section below).

Let's plot the calculated elliptical Kron apertures (based on the shapes
of each source) on the data:

.. doctest-skip::

    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from astropy.visualization import simple_norm
    >>> norm = simple_norm(data, 'sqrt', percent=99.5)
    >>> fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12.5))
    >>> ax1.imshow(data, norm=norm, origin='lower')
    >>> ax1.set_title('Data')
    >>> segment_map.imshow(ax=ax2)
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

    n_pixels = 10
    finder = SourceFinder(n_pixels=n_pixels, progress_bar=False)
    segment_map = finder(convolved_data, threshold)

    cat = SourceCatalog(data, segment_map, convolved_data=convolved_data)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12.5))
    norm = simple_norm(data, 'sqrt', percent=99.5)
    ax1.imshow(data, norm=norm, origin='lower')
    ax1.set_title('Data with Kron apertures')
    segment_map.imshow(ax=ax2)
    ax2.set_title('Segmentation Image with Kron apertures')
    cat.plot_kron_apertures(ax=ax1, color='white', lw=1.5)
    cat.plot_kron_apertures(ax=ax2, color='white', lw=1.5)
    fig.tight_layout()


We can also create a `~photutils.segmentation.SourceCatalog` object
containing only a specific subset of sources, defined by their
label numbers in the segmentation image:

.. doctest-requires:: skimage

    >>> cat = SourceCatalog(data, segment_map, convolved_data=convolved_data)
    >>> labels = [1, 5, 20, 50, 75, 80]
    >>> cat_subset = cat.select_labels(labels)
    >>> tbl2 = cat_subset.to_table()
    >>> tbl2['x_centroid'].info.format = '.2f'  # optional format
    >>> tbl2['y_centroid'].info.format = '.2f'
    >>> tbl2['kron_flux'].info.format = '.2f'
    >>> print(tbl2)
    label x_centroid y_centroid ... segment_flux_err kron_flux kron_flux_err
                                ...
    ----- ---------- ---------- ... ---------------- --------- -------------
        1     235.38       1.44 ...              nan    490.35           nan
        5     257.85      12.18 ...              nan    748.18           nan
       20     347.17      66.45 ...              nan    855.34           nan
       50     381.02     174.67 ...              nan    438.55           nan
       75      74.44     259.78 ...              nan    876.02           nan
       80      14.93      60.06 ...              nan    878.52           nan

By default, the :meth:`~photutils.segmentation.SourceCatalog.to_table`
includes only a small subset of source properties. The output table
properties can be customized in the `~astropy.table.QTable` using the
``columns`` keyword:

.. doctest-requires:: skimage

    >>> cat = SourceCatalog(data, segment_map, convolved_data=convolved_data)
    >>> labels = [1, 5, 20, 50, 75, 80]
    >>> cat_subset = cat.select_labels(labels)
    >>> columns = ['label', 'x_centroid', 'y_centroid', 'area', 'segment_flux']
    >>> tbl3 = cat_subset.to_table(columns=columns)
    >>> tbl3['x_centroid'].info.format = '.4f'  # optional format
    >>> tbl3['y_centroid'].info.format = '.4f'
    >>> tbl3['segment_flux'].info.format = '.4f'
    >>> print(tbl3)
    label x_centroid y_centroid  area segment_flux
                                 pix2
    ----- ---------- ---------- ----- ------------
        1   235.3827     1.4439  47.0     433.3546
        5   257.8501    12.1764  84.0     489.9653
       20   347.1743    66.4462 103.0     625.9668
       50   381.0186   174.6745  50.0     249.0170
       75    74.4448   259.7843  66.0     836.4803
       80    14.9296    60.0641  87.0     666.6014

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

.. doctest-requires:: scipy >= 1.8
.. doctest-requires:: skimage

    >>> cat = SourceCatalog(data, segment_map, background=bkg.background)
    >>> labels = [1, 5, 20, 50, 75, 80]
    >>> cat_subset = cat.select_labels(labels)
    >>> columns = ['label', 'background_centroid', 'background_mean',
    ...            'background_sum']
    >>> tbl4 = cat_subset.to_table(columns=columns)
    >>> tbl4['background_centroid'].info.format = '{:.10f}'  # optional format
    >>> tbl4['background_mean'].info.format = '{:.10f}'
    >>> tbl4['background_sum'].info.format = '{:.10f}'
    >>> print(tbl4)
    label background_centroid background_mean background_sum
    ----- ------------------- --------------- --------------
        1        5.1950691156    5.1952758684 244.1779658169
        5        5.2065578767    5.2065437428 437.3496743914
       20        5.2185224938    5.2182859243 537.4834502022
       50        5.2278578177    5.2277566101 261.3878305059
       75        5.2200812077    5.2203644550 344.5440540277
       80        5.2177773524    5.2174773951 453.9205333733

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
`~photutils.segmentation.SourceCatalog.segment_flux_err` and
`~photutils.segmentation.SourceCatalog.kron_flux_err` properties are
calculated. `~photutils.segmentation.SourceCatalog.segment_flux`
and `~photutils.segmentation.SourceCatalog.segment_flux_err` are the
instrumental flux and propagated flux error within the source segments:

.. doctest-requires:: scipy >= 1.8
.. doctest-requires:: skimage

    >>> from photutils.utils import calc_total_error
    >>> effective_gain = 500.0
    >>> error = calc_total_error(data, bkg.background_rms, effective_gain)
    >>> cat = SourceCatalog(data, segment_map, error=error)
    >>> labels = [1, 5, 20, 50, 75, 80]
    >>> cat_subset = cat.select_labels(labels)  # select a subset of objects
    >>> columns = ['label', 'x_centroid', 'y_centroid', 'segment_flux',
    ...            'segment_flux_err']
    >>> tbl5 = cat_subset.to_table(columns=columns)
    >>> tbl5['x_centroid'].info.format = '{:.4f}'  # optional format
    >>> tbl5['y_centroid'].info.format = '{:.4f}'
    >>> tbl5['segment_flux'].info.format = '{:.4f}'
    >>> tbl5['segment_flux_err'].info.format = '{:.4f}'
    >>> for col in tbl5.colnames:
    ...     tbl5[col].info.format = '%.8g'  # for consistent table output
    >>> print(tbl5)
    label x_centroid y_centroid segment_flux segment_flux_err
    ----- --------- --------- ------------ ----------------
        1 235.24302 1.1928271    433.35463        14.167067
        5 257.82267 12.228232    489.96534        18.998371
       20 347.15384 66.417567    625.96683        22.475065
       50 380.94448 174.57181    249.01701        15.261334
       75 74.413068 259.76066     836.4803        17.193721
       80 14.920217 60.024006     666.6014        19.605394


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
``convolved_data``. If ``convolved_data`` is `None`, then the unfiltered
``data`` will be used for the source centroid and morphological
parameters. Note that photometry is *always* performed on the unfiltered
``data``.


Dual-Image Mode (Detection Catalog)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In many astronomical workflows, source detection and deblending
are performed on one image (e.g., a deep detection image or
coadd) while photometry is measured on a different image (e.g.,
a single-band image). The ``detection_catalog`` keyword of
:class:`~photutils.segmentation.SourceCatalog` enables this dual-image
mode.

When ``detection_catalog`` is input, the source centroids and
morphological/shape properties are taken from the detection
catalog, while photometry is measured on the input ``data``. For
circular-aperture and Kron photometry, the aperture centers are based on
the centroids from the detection catalog. For Kron photometry, the Kron
apertures are based on the shape properties from the detection catalog.
The ``wcs``, ``aperture_mask_method``, and ``kron_params`` keywords
are inherited from the ``detection_catalog`` and are therefore ignored
when ``detection_catalog`` is input. Note that the segmentation image
used to create the detection catalog must be the same one input to the
measurement catalog:

.. doctest-requires:: skimage

    >>> det_cat = SourceCatalog(data, segment_map,
    ...                         convolved_data=convolved_data)
    >>> measurement_cat = SourceCatalog(data, segment_map,
    ...                                 detection_catalog=det_cat)

In this example, ``measurement_cat`` uses the centroids and shape
properties (and Kron apertures) from ``det_cat`` while measuring
photometry on ``data``.


API Reference
-------------

:doc:`../reference/segmentation_api`


.. _SourceExtractor:  https://sextractor.readthedocs.io/en/latest/
