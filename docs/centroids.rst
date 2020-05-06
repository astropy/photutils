Centroids (`photutils.centroids`)
=================================

Introduction
------------

`photutils.centroids` provides several functions to calculate the
centroid of a single source.  The centroid methods are:

* :func:`~photutils.centroids.centroid_com`: Calculates the object
  "center of mass" from 2D image moments.

* :func:`~photutils.centroids.centroid_1dg`: Calculates the centroid
  by fitting 1D Gaussians to the marginal ``x`` and ``y``
  distributions of the data.

* :func:`~photutils.centroids.centroid_2dg`: Calculates the centroid
  by fitting a 2D Gaussian to the 2D distribution of the data.

Masks can be input into each of these functions to mask bad pixels.
Error arrays can be input into the two fitting methods to weight the
fits.


Getting Started
---------------

Let's extract a single object from a synthetic dataset and find its
centroid with each of these methods.  For this simple example we will
not subtract the background from the data (but in practice, one should
subtract the background)::

    >>> from photutils.datasets import make_4gaussians_image
    >>> from photutils import centroid_com, centroid_1dg, centroid_2dg
    >>> data = make_4gaussians_image()[43:79, 76:104]

    >>> x1, y1 = centroid_com(data)
    >>> print((x1, y1))  # doctest: +FLOAT_CMP
    (13.93157998341213, 17.051234441067088)

.. doctest-requires:: scipy

    >>> x2, y2 = centroid_1dg(data)
    >>> print((x2, y2))  # doctest: +FLOAT_CMP
    (14.040352707371396, 16.962306463644801)

.. doctest-requires:: scipy

    >>> x3, y3 = centroid_2dg(data)
    >>> print((x3, y3))  # doctest: +FLOAT_CMP
    (14.002212073733611, 16.996134592982017)

Now let's plot the results.  Because the centroids are all very
similar, we also include an inset plot zoomed in near the centroid:

.. plot::
    :include-source:

    from photutils.datasets import make_4gaussians_image
    from photutils import centroid_com, centroid_1dg, centroid_2dg
    import matplotlib.pyplot as plt

    data = make_4gaussians_image()[43:79, 76:104]  # extract single object
    x1, y1 = centroid_com(data)
    x2, y2 = centroid_1dg(data)
    x3, y3 = centroid_2dg(data)
    fig, ax = plt.subplots(1, 1)
    ax.imshow(data, origin='lower', interpolation='nearest')
    marker = '+'
    ms, mew = 30, 2.
    plt.plot(x1, y1, color='#1f77b4', marker=marker, ms=ms, mew=mew)
    plt.plot(x2, y2, color='#17becf', marker=marker, ms=ms, mew=mew)
    plt.plot(x3, y3, color='#d62728', marker=marker, ms=ms, mew=mew)

    from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
    from mpl_toolkits.axes_grid1.inset_locator import mark_inset
    ax2 = zoomed_inset_axes(ax, zoom=6, loc=9)
    ax2.imshow(data, vmin=190, vmax=220, origin='lower',
               interpolation='nearest')
    ax2.plot(x1, y1, color='#1f77b4', marker=marker, ms=ms, mew=mew)
    ax2.plot(x2, y2, color='#17becf', marker=marker, ms=ms, mew=mew)
    ax2.plot(x3, y3, color='#d62728', marker=marker, ms=ms, mew=mew)
    ax2.set_xlim(13, 15)
    ax2.set_ylim(16, 18)
    mark_inset(ax, ax2, loc1=3, loc2=4, fc='none', ec='0.5')
    ax2.axes.get_xaxis().set_visible(False)
    ax2.axes.get_yaxis().set_visible(False)
    ax.set_xlim(0, data.shape[1]-1)
    ax.set_ylim(0, data.shape[0]-1)


Reference/API
-------------

.. automodapi:: photutils.centroids
    :no-heading:
