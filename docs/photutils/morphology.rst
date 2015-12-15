Source Centroids and Morphological Properties
=============================================

Centroiding a Source
--------------------

`photutils.morphology` provides several functions to calculate the
centroid of a single source.  The centroid methods are:

* :func:`~photutils.morphology.centroid_com`: Calculates the object
  center of mass from 2D image moments.

* :func:`~photutils.morphology.centroid_1dg`: Calculates the centroid
  by fitting 1D Gaussians to the marginal x and y distributions of the
  data.

* :func:`~photutils.morphology.centroid_2dg`: Calculates the centroid
  by fitting a 2D Gaussian to the 2D distribution of the data.

Masks can be input into each of these functions to mask bad pixels.
Error arrays can be input into the two fitting methods to weight the
fits.

Let's extract a single object from a synthetic dataset and find its
centroid with each of these methods.  For this simple example we will
not subtract the background from the data (but in practice, one should
subtract the background)::

    >>> from photutils.datasets import make_4gaussians_image
    >>> from photutils.morphology import (centroid_com, centroid_1dg,
    ...                                   centroid_2dg)
    >>> data = make_4gaussians_image()[43:79, 76:104]

.. doctest-requires:: skimage

    >>> x1, y1 = centroid_com(data)
    >>> print((x1, y1))    # doctest: +FLOAT_CMP
    (13.93157998341213, 17.051234441067088)

.. doctest-requires:: scipy

    >>> x2, y2 = centroid_1dg(data)
    >>> print((x2, y2))    # doctest: +FLOAT_CMP
    (14.040352707371396, 16.962306463644801)

.. doctest-requires:: scipy, skimage

    >>> x3, y3 = centroid_2dg(data)
    >>> print((x3, y3))    # doctest: +FLOAT_CMP
    (14.002212073733611, 16.996134592982017)

Now let's plot the results.  Because the centroids are all very
similar, we also include an inset plot zoomed in near the centroid:

.. doctest-skip::

    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots(1, 1)
    >>> ax.imshow(data, origin='lower', cmap='Greys_r')
    >>> marker = '+'
    >>> ms, mew = 30, 2.
    >>> plt.plot(x1, y1, color='red', marker=marker, ms=ms)
    >>> plt.plot(x2, y2, color='blue', marker=marker, ms=ms)
    >>> plt.plot(x3, y3, color='green', marker=marker, ms=ms)
    >>> # include a zoomed inset plot
    >>> from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
    >>> from mpl_toolkits.axes_grid1.inset_locator import mark_inset
    >>> ax2 = zoomed_inset_axes(ax, zoom=6, loc=9)
    >>> ax2.imshow(data, interpolation='nearest', origin='lower',
    ...            cmap='Greys_r', vmin=190, vmax=220)
    >>> ax2.plot(x1, y1, color='red', marker=marker, ms=ms, mew=mew)
    >>> ax2.plot(x2, y2, color='blue', marker=marker, ms=ms, mew=mew)
    >>> ax2.plot(x3, y3, color='green', marker=marker, ms=ms, mew=mew)
    >>> ax2.set_xlim(13, 15)
    >>> ax2.set_ylim(16, 18)
    >>> mark_inset(ax, ax2, loc1=3, loc2=4, fc='none', ec='0.5')
    >>> ax2.axes.get_xaxis().set_visible(False)
    >>> ax2.axes.get_yaxis().set_visible(False)
    >>> ax.set_xlim(0, data.shape[1]-1)
    >>> ax.set_ylim(0, data.shape[0]-1)

.. plot::

    from photutils.datasets import make_4gaussians_image
    from photutils.morphology import (centroid_com, centroid_1dg,
                                      centroid_2dg)
    import matplotlib.pyplot as plt
    data = make_4gaussians_image()[43:79, 76:104]    # extract single object
    x1, y1 = centroid_com(data)
    x2, y2 = centroid_1dg(data)
    x3, y3 = centroid_2dg(data)
    fig, ax = plt.subplots(1, 1)
    ax.imshow(data, origin='lower', cmap='Greys_r')
    marker = '+'
    ms, mew = 30, 2.
    plt.plot(x1, y1, color='red', marker=marker, ms=ms, mew=mew)
    plt.plot(x2, y2, color='blue', marker=marker, ms=ms, mew=mew)
    plt.plot(x3, y3, color='green', marker=marker, ms=ms, mew=mew)
    from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
    from mpl_toolkits.axes_grid1.inset_locator import mark_inset
    ax2 = zoomed_inset_axes(ax, zoom=6, loc=9)
    ax2.imshow(data, interpolation='nearest', origin='lower',
               cmap='Greys_r', vmin=190, vmax=220)
    ax2.plot(x1, y1, color='red', marker=marker, ms=ms, mew=mew)
    ax2.plot(x2, y2, color='blue', marker=marker, ms=ms, mew=mew)
    ax2.plot(x3, y3, color='green', marker=marker, ms=ms, mew=mew)
    ax2.set_xlim(13, 15)
    ax2.set_ylim(16, 18)
    mark_inset(ax, ax2, loc1=3, loc2=4, fc='none', ec='0.5')
    ax2.axes.get_xaxis().set_visible(False)
    ax2.axes.get_yaxis().set_visible(False)
    ax.set_xlim(0, data.shape[1]-1)
    ax.set_ylim(0, data.shape[0]-1)


Source Morphological Properties
-------------------------------

The :func:`~photutils.morphology.data_properties` function can be used
to calculate the properties of a single source from a cutout image.
`~photutils.morphology.data_properties` returns a
`~photutils.segmentation.SourceProperties` object.  Please see
`~photutils.segmentation.SourceProperties` for the list of the many
properties that are calculated.  Even more properties are likely to be
added in the future.

If you have a segmentation image, the
:func:`~photutils.segmentation.source_properties` function can be used
to calculate the properties for all (or a specified subset) of the
segmented sources.  Please see `Source Photometry and Properties from
Image Segmentation <segmentation.html>`_ for more details.

As an example, let's calculate the properties of the source defined
above.  For this example, we will subtract the background using simple
sigma-clipped statistics:

.. doctest-requires:: scipy, skimage

    >>> from astropy.stats import sigma_clipped_stats
    >>> from photutils.morphology import data_properties
    >>> from photutils import properties_table
    >>> mean, median, std = sigma_clipped_stats(data, sigma=3.0, iters=5)
    >>> data -= median    # subtract background
    >>> props = data_properties(data)
    >>> columns = ['id', 'xcentroid', 'ycentroid', 'semimajor_axis_sigma',
    ...            'semiminor_axis_sigma', 'orientation']
    >>> tbl = properties_table(props, columns=columns)
    >>> print(tbl)
     id   xcentroid     ycentroid   ... semiminor_axis_sigma  orientation
             pix           pix      ...         pix               rad
    --- ------------- ------------- ... -------------------- -------------
      1 14.0225090502 16.9901801466 ...        3.69777618702 1.04943689372

Now let's use the measured morphological properties to define an
approximate isophotal ellipse for the source:

.. doctest-skip::

    >>> from photutils import properties_table, EllipticalAperture
    >>> position = (props.xcentroid.value, props.ycentroid.value)
    >>> r = 3.0    # approximate isophotal extent
    >>> a = props.semimajor_axis_sigma.value * r
    >>> b = props.semiminor_axis_sigma.value * r
    >>> theta = props.orientation.value
    >>> apertures = EllipticalAperture(position, a, b, theta=theta)
    >>> plt.imshow(data, origin='lower', cmap='Greys_r')
    >>> apertures.plot(color='red')

.. plot::

    from photutils.datasets import make_4gaussians_image
    from photutils.morphology import data_properties
    from photutils import properties_table, EllipticalAperture
    import matplotlib.pyplot as plt
    data = make_4gaussians_image()[43:79, 76:104]    # extract single object
    props = data_properties(data)
    columns = ['id', 'xcentroid', 'ycentroid', 'semimajor_axis_sigma',
               'semiminor_axis_sigma', 'orientation']
    tbl = properties_table(props, columns=columns)
    r = 2.5    # approximate isophotal extent
    position = (props.xcentroid.value, props.ycentroid.value)
    a = props.semimajor_axis_sigma.value * r
    b = props.semiminor_axis_sigma.value * r
    theta = props.orientation.value
    apertures = EllipticalAperture(position, a, b, theta=theta)
    plt.imshow(data, origin='lower', cmap='Greys_r')
    apertures.plot(color='red')


Reference/API
-------------

.. automodapi:: photutils.morphology
    :no-heading:
