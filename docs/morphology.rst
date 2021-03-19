Morphological Properties (`photutils.morphology`)
=================================================

Introduction
------------

The :func:`~photutils.morphology.data_properties` function can
be used to calculate the morphological properties of a single
source in a cutout image. `~photutils.morphology.data_properties`
returns a `~photutils.segmentation.SourceCatalog` object. Please see
`~photutils.segmentation.SourceCatalog` for the list of the many
properties that are calculated. Even more properties are likely to be
added in the future.

If you have a segmentation image, the
:class:`~photutils.segmentation.SourceCatalog` class can be used
to calculate the properties for all (or a specified subset) of the
segmented sources. Please see :ref:`Source Photometry and Properties
from Image Segmentation <image_segmentation>` for more details.


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
    >>> data -= median  # subtract background

Then, calculate its properties:

.. doctest-requires:: scipy

    >>> from photutils.morphology import data_properties
    >>> cat = data_properties(data)
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
        1 14.0225090502 16.9901801466 ...    3.6977761870 60.1283048753


Now let's use the measured morphological properties to define an
approximate isophotal ellipse for the source:

.. doctest-skip::

    >>> import astropy.units as u
    >>> from photutils.aperture import EllipticalAperture
    >>> position = (cat.xcentroid, cat.ycentroid)
    >>> r = 3.0  # approximate isophotal extent
    >>> a = cat.semimajor_sigma.value * r
    >>> b = cat.semiminor_sigma.value * r
    >>> theta = cat.orientation.to(u.rad).value
    >>> apertures = EllipticalAperture(position, a, b, theta=theta)
    >>> plt.imshow(data, origin='lower', cmap='viridis',
    ...            interpolation='nearest')
    >>> apertures.plot(color='#d62728')

.. plot::

    import astropy.units as u
    import matplotlib.pyplot as plt
    from photutils.aperture import EllipticalAperture
    from photutils.morphology import data_properties
    from photutils.datasets import make_4gaussians_image

    data = make_4gaussians_image()[43:79, 76:104]  # extract single object
    cat = data_properties(data)
    columns = ['label', 'xcentroid', 'ycentroid', 'semimajor_sigma',
               'semiminor_sigma', 'orientation']
    tbl = cat.to_table(columns=columns)
    r = 2.5  # approximate isophotal extent
    position = (cat.xcentroid, cat.ycentroid)
    a = cat.semimajor_sigma.value * r
    b = cat.semiminor_sigma.value * r
    theta = cat.orientation.to(u.rad).value
    apertures = EllipticalAperture(position, a, b, theta=theta)
    plt.imshow(data, origin='lower', cmap='viridis', interpolation='nearest')
    apertures.plot(color='#d62728')


Reference/API
-------------

.. automodapi:: photutils.morphology
    :no-heading:
