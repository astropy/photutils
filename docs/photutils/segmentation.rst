Source Photometry and Properties from Image Segmentation
========================================================

Introduction
------------

After detecting sources using image segmentation (see
:ref:`source_extraction`), we can measure their photometry, centroids,
and morphological properties.  Photutils also provides functions for
modifying segmentation images (e.g., combining labels, removing
labels, removing border segments, etc.) prior to measuring photometry
and other source properties.


Getting Started
---------------

The :func:`~photutils.segmentation.segment_properties` function is the
primary tool for measuring the photometry, centroids, and
morphological properties of sources defined in a segmentation image.
When the segmentation image is generated using image thresholding
(e.g., using `~photutils.detect_sources`), the source segments
effectively represent the isophotal footprint of each source and the
resulting photometry is effectively isophotal photometry.

`~photutils.segmentation.segment_properties` returns a list of
:class:`~photutils.segmentation.SegmentProperties` objects, one for
each segmented source (or a specified subset of sources).  An Astropy
`~astropy.table.Table` of source properties can be generated using the
:func:`~photutils.segmentation.properties_table` function.  Please see
`~photutils.segmentation.SegmentProperties` for the list of the many
properties that are calculated for each source.  Even more properties
are likely to be added in the future.

Let's detect sources and measure their properties in a synthetic
image.  For this example, we will use the
`~photutils.background.Background` class to produce a background and
background noise image.  We define a 2D detection threshold image
using the background and background rms maps.  We set the threshold at
3 sigma above the background:

.. doctest-requires:: scipy

    >>> from photutils.datasets import make_100gaussians_image
    >>> from photutils import Background, detect_threshold, detect_sources
    >>> from astropy.convolution import Gaussian2DKernel
    >>> data = make_100gaussians_image()
    >>> bkg = Background(data, (50, 50), filter_shape=(3, 3), method='median')
    >>> threshold = bkg.background + (3. * bkg.background_rms)

Now we find sources that have 5 connected pixels that are each greater
than the corresponding pixel-wise threshold image defined above.
Because the threshold includes the background, we do not subtract the
background from the data here.  We also input a 2D circular Gaussian
kernel with a FWHM of 2 pixels to filter the image prior to
thresholding:

.. doctest-requires:: scipy, skimage

    >>> from photutils.extern.stats import gaussian_fwhm_to_sigma
    >>> sigma = 2.0 * gaussian_fwhm_to_sigma    # FWHM = 2.
    >>> kernel = Gaussian2DKernel(sigma, x_size=3, y_size=3)
    >>> segm = detect_sources(data, threshold, npixels=5, filter_kernel=kernel)

Now let's measure the properties of the detected sources defined in
the segmentation image with the minimum number of inputs to
`~photutils.segmentation.segment_properties`:

.. doctest-requires:: scipy, skimage

    >>> from photutils import segment_properties, properties_table
    >>> props = segment_properties(data, segm)
    >>> tbl = properties_table(props)
    >>> print(tbl)
     id   xcentroid     ycentroid    ...       cxy             cyy
             pix           pix       ...     1 / pix2        1 / pix2
    --- ------------- -------------- ... ---------------- --------------
      1 234.983156946 0.985867670057 ...  -0.290404010574  1.55891445615
      2 494.832664004  6.09314215108 ...  -0.753441096683  1.80243806441
      3 207.165742206  10.2335476324 ...    0.92059377383 0.563137540285
      4 364.729681583  10.9844940633 ...    -1.2288069393 0.635093403231
      5 258.031283251  11.9649001764 ...  0.0599262879469 0.808873726802
    ...           ...            ... ...              ...            ...
     77  82.553164741  267.507957132 ...  0.0959470832587 0.468289801445
     78 477.742087097  267.723968829 ... -0.0470970087203 0.520702684332
     79  139.78439077  275.134937222 ...   0.239005017209 0.385709864349
     80  434.09234827  280.841150924 ...   -1.45089455818 0.815770166696
     81 434.091385011  288.922811107 ...  -0.258098406023 0.454297620264

Let's use the measured source morphological properties to define
approximate isophotal ellipses for each object:

.. doctest-requires:: scipy, skimage

    >>> from photutils import segment_properties, properties_table
    >>> props = segment_properties(data, segm)
    >>> from photutils import EllipticalAperture
    >>> r = 3.    # approximate isophotal extent
    >>> apertures = []
    >>> for prop in props:
    ...     position = (prop.xcentroid.value, prop.ycentroid.value)
    ...     a = prop.semimajor_axis_sigma.value * r
    ...     b = prop.semiminor_axis_sigma.value * r
    ...     theta = prop.orientation.value
    ...     apertures.append(EllipticalAperture(position, a, b, theta=theta))

Now let's plot the results:

.. doctest-skip::

    >>> from photutils.extern.imageutils.normalization import SqrtStretch, ImageNormalize
    >>> import matplotlib.pylab as plt
    >>> norm = ImageNormalize(stretch=SqrtStretch())
    >>> fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))
    >>> ax1.imshow(data, origin='lower', cmap='Greys_r', norm=norm)
    >>> ax2.imshow(segm, origin='lower', cmap='jet')
    >>> for aperture in apertures:
    ...     aperture.plot(color='blue', lw=1.5, alpha=0.5, ax=ax1)
    ...     aperture.plot(color='white', lw=1.5, alpha=1.0, ax=ax2)

.. plot::

    from photutils.datasets import make_100gaussians_image
    from photutils import Background, detect_threshold, detect_sources
    from photutils import segment_properties, properties_table
    from photutils.extern.stats import gaussian_fwhm_to_sigma
    from astropy.convolution import Gaussian2DKernel
    data = make_100gaussians_image()
    bkg = Background(data, (50, 50), filter_shape=(3, 3), method='median')
    threshold = bkg.background + (3. * bkg.background_rms)
    sigma = 2.0 * gaussian_fwhm_to_sigma    # FWHM = 2.
    kernel = Gaussian2DKernel(sigma, x_size=3, y_size=3)
    segm = detect_sources(data, threshold, npixels=5, filter_kernel=kernel)
    props = segment_properties(data, segm)
    from photutils import EllipticalAperture
    apertures = []
    for prop in props:
        position = (prop.xcentroid.value, prop.ycentroid.value)
        a = prop.semimajor_axis_sigma.value * 3.
        b = prop.semiminor_axis_sigma.value * 3.
        theta = prop.orientation.value
        apertures.append(EllipticalAperture(position, a, b, theta=theta))
    from photutils.extern.imageutils.normalization import SqrtStretch, ImageNormalize
    import matplotlib.pylab as plt
    norm = ImageNormalize(stretch=SqrtStretch())
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))
    ax1.imshow(data, origin='lower', cmap='Greys_r', norm=norm)
    ax2.imshow(segm, origin='lower', cmap='jet')
    for aperture in apertures:
        aperture.plot(color='blue', lw=1.5, alpha=0.5, ax=ax1)
        aperture.plot(color='white', lw=1.5, alpha=1.0, ax=ax2)

We can also specify a specific subset of sources, defined by their labels in
the segmentation image:

.. doctest-requires:: scipy, skimage

    >>> labels = [1, 5, 20, 50, 75, 80]
    >>> props = segment_properties(data, segm, labels=labels)
    >>> tbl = properties_table(props)
    >>> print(tbl)
     id   xcentroid     ycentroid    ...       cxy            cyy
             pix           pix       ...     1 / pix2       1 / pix2
    --- ------------- -------------- ... --------------- --------------
      1 234.983156946 0.985867670057 ... -0.290404010574  1.55891445615
      5 258.031283251  11.9649001764 ... 0.0599262879469 0.808873726802
     20 454.388594497  68.5821197941 ...  -0.33618547754  0.97159028054
     50  479.91617242  187.114386765 ... -0.207030049528 0.350331239992
     75 98.0189554022  254.149269803 ...  0.232109296376 0.794200433096
     80  434.09234827  280.841150924 ...  -1.45089455818 0.815770166696

By default, `~photutils.segmentation.properties_table` will include
all scalar-valued properties from
`~photutils.segmentation.SegmentProperties`, but a subset of
properties can also be specified (or excluded) in the
`~astropy.table.Table`:

.. doctest-requires:: scipy, skimage

    >>> labels = [1, 5, 20, 50, 75, 80]
    >>> props = segment_properties(data, segm, labels=labels)
    >>> columns = ['id', 'xcentroid', 'ycentroid', 'segment_sum', 'area']
    >>> tbl = properties_table(props, columns=columns)
    >>> print(tbl)
     id   xcentroid     ycentroid     segment_sum  area
             pix           pix                     pix2
    --- ------------- -------------- ------------- ----
      1 234.983156946 0.985867670057 431.609320002 22.0
      5 258.031283251  11.9649001764  153.84610582 11.0
     20 454.388594497  68.5821197941 279.649184518 17.0
     50  479.91617242  187.114386765 501.805452349 32.0
     75 98.0189554022  254.149269803 639.459662985 40.0
     80  434.09234827  280.841150924 589.761745573 31.0

A `~astropy.wcs.WCS` transformation can also be input to
`~photutils.segmentation.segment_properties` via the ``wcs`` keyword,
in which case the ICRS Right Ascension and Declination coordinates at
the source centroids will be returned.


Background Properties
^^^^^^^^^^^^^^^^^^^^^

Like `~photutils.aperture_photometry`, the ``data`` array that is
input to `~photutils.segmentation.segment_properties` should be
background subtracted.  If you input the ``background`` keyword to
`~photutils.segmentation.segment_properties`, it will calculate
background properties with each source segment:

.. doctest-requires:: scipy, skimage

    >>> labels = [1, 5, 20, 50, 75, 80]
    >>> props = segment_properties(data, segm, labels=labels,
    ...                            background=bkg.background)
    >>> columns = ['id', 'background_atcentroid', 'background_mean',
    ...            'background_sum']
    >>> tbl = properties_table(props, columns=columns)
    >>> print(tbl)
     id background_atcentroid background_mean background_sum
    --- --------------------- --------------- --------------
      1         5.20165925941   5.20204496029  114.444989126
      5         5.21400789846   5.21360209345   57.349623028
     20         5.14522283081   5.14469942742  87.4598902662
     50         5.14924293236   5.14956699294  164.786143774
     75         5.11314976506    5.1138238531  204.552954124
     80         5.10469713755   5.10450103326  158.239532031


Photometric Errors
^^^^^^^^^^^^^^^^^^

With `~photutils.segmentation.segment_properties` we can use the
background-only error image and an effective gain to estimate the
error in the photometry.  Like the aperture photometry
:ref:`error_estimation`, the
`~photutils.segmentation.segment_properties` ``error`` keyword can
either specify the total error array (i.e., it includes Poisson noise
due to individual sources or such noise is irrelevant) or it can
specify the background-only noise.  In the later case, we can specify
the ``effective_gain``, which is the ratio of counts (electrons or
photons) to the units of the data, to explicitly include Poisson noise
from the sources.  The ``effective_gain`` can be a 2D gain image with
the same shape as the ``data``.  This is useful with mosaic images
that have variable depths (i.e., exposure times) across the field. For
example, one should use an exposure-time map as the ``effective_gain``
for a variable depth mosaic image in count-rate units.

Let's assume our synthetic data is in units of electrons per second.
In that case, the ``effective_gain`` should be the exposure time (here
we set it to 500 seconds):

.. doctest-requires:: scipy, skimage

    >>> labels = [1, 5, 20, 50, 75, 80]
    >>> props = segment_properties(data, segm, labels=labels,
    ...                            error=bkg.background_rms,
    ...                            effective_gain=500.)
    >>> columns = ['id', 'xcentroid', 'ycentroid', 'segment_sum',
    ...            'segment_sum_err']
    >>> tbl = properties_table(props, columns=columns)
    >>> print(tbl)
     id   xcentroid     ycentroid     segment_sum  segment_sum_err
             pix           pix
    --- ------------- -------------- ------------- ---------------
      1 234.983156946 0.985867670057 431.609320002    10.002808709
      5 258.031283251  11.9649001764  153.84610582   7.11241135922
     20 454.388594497  68.5821197941 279.649184518   8.96285703949
     50  479.91617242  187.114386765 501.805452349   11.9863586395
     75 98.0189554022  254.149269803 639.459662985   12.9166164989
     80  434.09234827  280.841150924 589.761745573   11.5907931505

`~photutils.SegmentProperties.segment_sum` and
`~photutils.SegmentProperties.segment_sum_err` are the instrumental
flux and propagated flux error within the source segments.


Pixel Masking
^^^^^^^^^^^^^

Pixels can be completely ignored/excluded when measuring the source
properties by providing a boolean mask image via the ``mask`` keyword
(`True` pixel values are masked).  This is generally used to exclude
bad pixels in the data.


Filtering
^^^^^^^^^

`SExtractor`_'s centroid and morphological parameters are always
calculated from a filtered "detection" image.  The usual downside of
the filtering is the sources will be made more circular than they
actually are.  If you wish to reproduce `SExtractor`_ results, then
use the ``filter_kernel`` input to
`~photutils.segmentation.segment_properties` to filter the ``data``
prior to centroid and morphological measurements.   The kernel should
be the same one used to define the source segments in
`~photutils.detect_sources`.  If ``filter_kernel`` is `None`, then the
centroid and morphological measurements will be performed on the
unfiltered ``data``.  Note that photometry is *always* performed on
the unfiltered ``data``.


Modifying Segmentation Images
-----------------------------

Photutils also provides several functions that can be used to modify
segmentation images prior to performing source photometry and
measurements.  For example, segmented sources can be combined or
relabeled using :func:`~photutils.segmentation.relabel_segments`.
Specified labeled segments (e.g., diffraction spikes, known artifacts)
can be removed from a segmentation image using
:func:`~photutils.segmentation.remove_segments`.  Labeled segments can
also be removed based on a mask image
(:func:`~photutils.segmentation.remove_masked_segments`) or in regions
around the border of an image
(:func:`~photutils.segmentation.remove_border_segments`).  Finally,
:func:`~photutils.segmentation.relabel_sequential` can relabel a
segmentation image sequentially such that there are no missing label
numbers.


Reference/API
-------------

.. automodapi:: photutils.segmentation
    :no-heading:


.. _Sextractor:  http://www.astromatic.net/software/sextractor
