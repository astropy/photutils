Morphological Properties (`photutils.morphology`)
=================================================

Introduction
------------

The :func:`~photutils.morphology.data_properties` function can be used
to calculate the morphological properties of a single source in a
cutout image.  `~photutils.morphology.data_properties` returns a
`~photutils.segmentation.SourceProperties` object.  Please see
`~photutils.segmentation.SourceProperties` for the list of the many
properties that are calculated.  Even more properties are likely to be
added in the future.

If you have a segmentation image, the
:func:`~photutils.segmentation.source_properties` function can be used
to calculate the properties for all (or a specified subset) of the
segmented sources.  Please see `Source Photometry and Properties from
Image Segmentation <segmentation.html>`_ for more details.


Getting Started
---------------

Let's extract a single object from a synthetic dataset and find
calculate its morphological properties.  For this example, we will
subtract the background using simple sigma-clipped statistics.

First, we create the source image and subtract its background::

    >>> from photutils.datasets import make_4gaussians_image
    >>> from astropy.stats import sigma_clipped_stats
    >>> data = make_4gaussians_image()[43:79, 76:104]
    >>> mean, median, std = sigma_clipped_stats(data, sigma=3.0)
    >>> data -= median    # subtract background

Then, calculate its properties:

.. doctest-requires:: scipy

    >>> from photutils import data_properties
    >>> cat = data_properties(data)
    >>> columns = ['id', 'xcentroid', 'ycentroid', 'semimajor_axis_sigma',
    ...            'semiminor_axis_sigma', 'orientation']
    >>> tbl = cat.to_table(columns=columns)
    >>> tbl['xcentroid'].info.format = '.10f'  # optional format
    >>> tbl['ycentroid'].info.format = '.10f'
    >>> tbl['semiminor_axis_sigma'].info.format = '.10f'
    >>> tbl['orientation'].info.format = '.10f'
    >>> print(tbl)
     id   xcentroid     ycentroid   ... semiminor_axis_sigma  orientation
             pix           pix      ...         pix               deg
    --- ------------- ------------- ... -------------------- -------------
      1 14.0225090502 16.9901801466 ...         3.6977761870 60.1283048753

Now let's use the measured morphological properties to define an
approximate isophotal ellipse for the source:

.. doctest-skip::

    >>> from photutils import EllipticalAperture
    >>> position = (cat.xcentroid.value, cat.ycentroid.value)
    >>> r = 3.0    # approximate isophotal extent
    >>> a = cat.semimajor_axis_sigma.value * r
    >>> b = cat.semiminor_axis_sigma.value * r
    >>> theta = cat.orientation.value
    >>> apertures = EllipticalAperture(position, a, b, theta=theta)
    >>> plt.imshow(data, origin='lower', cmap='viridis',
    ...            interpolation='nearest')
    >>> apertures.plot(color='#d62728')

.. plot::

    import matplotlib.pyplot as plt
    from photutils import data_properties, EllipticalAperture
    from photutils.datasets import make_4gaussians_image

    data = make_4gaussians_image()[43:79, 76:104]    # extract single object
    cat = data_properties(data)
    columns = ['id', 'xcentroid', 'ycentroid', 'semimajor_axis_sigma',
               'semiminor_axis_sigma', 'orientation']
    tbl = cat.to_table(columns=columns)
    r = 2.5    # approximate isophotal extent
    position = (cat.xcentroid.value, cat.ycentroid.value)
    a = cat.semimajor_axis_sigma.value * r
    b = cat.semiminor_axis_sigma.value * r
    theta = cat.orientation.value
    apertures = EllipticalAperture(position, a, b, theta=theta)
    plt.imshow(data, origin='lower', cmap='viridis', interpolation='nearest')
    apertures.plot(color='#d62728')


Reference/API
-------------

.. automodapi:: photutils.morphology
    :no-heading:
