Morphological Properties (`photutils.morphology`)
=================================================

Introduction
------------

The :func:`~photutils.morphology.data_properties` function can
be used to calculate the basic morphological properties (e.g.,
elliptical shape properties) of a single source in a cutout
image. :func:`~photutils.morphology.data_properties` returns a
:class:`~photutils.segmentation.SourceCatalog` object. Please see
:class:`~photutils.segmentation.SourceCatalog` for the list of the many
properties that are calculated.

The `photutils.morphology` subpackage also includes the
:func:`~photutils.morphology.gini` function, which calculates the Gini
coefficient of a source in an image.

If you have a segmentation image, the
:class:`~photutils.segmentation.SourceCatalog` class can be used
to calculate the properties for all (or a specified subset) of the
segmented sources. Please see :ref:`Source Photometry and Properties
from Image Segmentation <image_segmentation>` for more details.


Getting Started
---------------

Let's extract a single object from a synthetic dataset and find
calculate its morphological properties. For this example, we will
subtract the background using simple sigma-clipped statistics.

First, we create the source image and subtract its background::

    >>> from astropy.stats import sigma_clipped_stats
    >>> from photutils.datasets import make_4gaussians_image
    >>> data = make_4gaussians_image()[40:80, 75:105]
    >>> mean, median, std = sigma_clipped_stats(data, sigma=3.0)
    >>> data -= median  # subtract background

Then, calculate its properties::

    >>> from photutils.morphology import data_properties
    >>> mask = data < 50
    >>> cat = data_properties(data, mask=mask)
    >>> columns = ['label', 'xcentroid', 'ycentroid', 'semimajor_sigma',
    ...            'semiminor_sigma', 'orientation']
    >>> tbl = cat.to_table(columns=columns)
    >>> tbl['xcentroid'].info.format = '.10f'  # optional format
    >>> tbl['ycentroid'].info.format = '.10f'
    >>> tbl['semiminor_sigma'].info.format = '.10f'
    >>> tbl['orientation'].info.format = '.10f'
    >>> print(tbl)
    label   xcentroid     ycentroid   ... semiminor_sigma  orientation
                                      ...       pix            deg
    ----- ------------- ------------- ... --------------- -------------
        1 15.0203353055 20.0876025118 ...    3.2260911267 59.6896286141


Now let's use the measured morphological properties to define an
approximate isophotal ellipse for the source:

.. doctest-skip::

    >>> import astropy.units as u
    >>> from photutils.aperture import EllipticalAperture
    >>> xypos = (cat.xcentroid, cat.ycentroid)
    >>> r = 2.5  # approximate isophotal extent
    >>> a = cat.semimajor_sigma.value * r
    >>> b = cat.semiminor_sigma.value * r
    >>> theta = cat.orientation.to(u.rad).value
    >>> apertures = EllipticalAperture(xypos, a, b, theta=theta)
    >>> plt.imshow(data, origin='lower', cmap='viridis',
    ...            interpolation='nearest')
    >>> apertures.plot(color='C3')

.. plot::

    import astropy.units as u
    import matplotlib.pyplot as plt
    import numpy as np
    from photutils.aperture import EllipticalAperture
    from photutils.datasets import make_4gaussians_image
    from photutils.morphology import data_properties

    slc = np.s_[40:80, 75:105]
    data = make_4gaussians_image()[slc]  # extract single object
    mask = data < 50
    cat = data_properties(data, mask=mask)
    columns = ['label', 'xcentroid', 'ycentroid', 'semimajor_sigma',
               'semiminor_sigma', 'orientation']
    tbl = cat.to_table(columns=columns)
    r = 2.5  # approximate isophotal extent
    xypos = (cat.xcentroid, cat.ycentroid)
    a = cat.semimajor_sigma.value * r
    b = cat.semiminor_sigma.value * r
    theta = cat.orientation.to(u.rad).value
    apertures = EllipticalAperture(xypos, a, b, theta=theta)

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(data, origin='lower', interpolation='nearest')
    apertures.plot(ax=ax, color='C3', lw=2)

    dx_major = a * np.cos(theta)
    dy_major = a * np.sin(theta)
    color = 'C1'
    width = 0.2
    ax.arrow(cat.xcentroid, cat.ycentroid, dx_major, dy_major, color=color,
             length_includes_head=True, width=width)
    theta2 = theta + np.pi / 2
    dx_minor = b * np.cos(theta2)
    dy_minor = b * np.sin(theta2)
    ax.arrow(cat.xcentroid, cat.ycentroid, dx_minor, dy_minor, color=color,
             length_includes_head=True, width=width)


Gini Coefficient
----------------

The Gini coefficient is a measure of the inequality in the distribution
of flux values in an image. The Gini coefficient ranges from 0 to 1,
where 0 indicates that the flux is equally distributed among all pixels
and 1 indicates that the flux is concentrated in a single pixel.

The :func:`~photutils.morphology.gini` function calculates the Gini
coefficient of a single source using the values in a cutout image.
An optional boolean mask can be used to exclude pixels from the
calculation.

Let's calculate the Gini coefficient of the source in the above
example::

    >>> from photutils.morphology import gini
    >>> g = gini(data, mask=mask)
    >>> print(g)
    0.21943786993407582


API Reference
-------------

:doc:`../reference/morphology_api`
