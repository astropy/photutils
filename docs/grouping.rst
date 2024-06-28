.. _psf-grouping:

Source Grouping Algorithms
==========================

Introduction
------------

In Point Spread Function (PSF) photometry, a grouping algorithm can be
used to combine stars into optimum groups. The stars in each group are
usually defined as those close enough together such that they need to be
fit simultaneously, i.e., their profiles overlap.

Stetson (`1987, PASP 99, 191
<https://ui.adsabs.harvard.edu/abs/1987PASP...99..191S/abstract>`_),
provided a simple and powerful grouping algorithm to decide whether
the profile of a given star extends into the fitting region of any
other star. The paper defines this in terms of a "critical separation"
parameter, which is defined as the minimal distance that any two stars
must be separated by in order to be in different groups. The critical
separation is generally defined as a multiple of the stellar full width
at half maximum (FWHM).


Getting Started
---------------

Photutils provides the :class:`~photutils.psf.SourceGrouper`
class to group stars. The groups are formed using hierarchical
agglomerative clustering with a distance criterion, calling the
`scipy.cluster.hierarchy.fclusterdata` function.

First, let's create a simulated image containing 2D Gaussian sources
using `~photutils.psf.make_psf_model_image`.

.. doctest-requires:: scipy

    >>> from photutils.psf import IntegratedGaussianPRF, make_psf_model_image
    >>> shape = (256, 256)
    >>> sigma = 2.0
    >>> psf_model = IntegratedGaussianPRF(sigma=sigma)
    >>> psf_shape = (11, 11)
    >>> n_sources = 100
    >>> flux = (500, 1000)
    >>> border_size = (7, 7)
    >>> data, stars = make_psf_model_image(shape, psf_model, n_sources,
    ...                                    model_shape=psf_shape,
    ...                                    flux=flux,
    ...                                    border_size=border_size, seed=123)

Let's display the image:

.. doctest-skip::

    >>> import matplotlib.pyplot as plt
    >>> plt.figure(figsize=(8, 8))
    >>> plt.imshow(data, origin='lower', interpolation='nearest')

.. plot::

    import matplotlib.pyplot as plt
    from photutils.psf import IntegratedGaussianPRF, make_psf_model_image

    shape = (256, 256)
    sigma = 2.0
    psf_model = IntegratedGaussianPRF(sigma=sigma)
    psf_shape = (11, 11)
    n_sources = 100
    flux = (500, 1000)
    border_size = (7, 7)
    data, stars = make_psf_model_image(shape, psf_model, n_sources,
                                       flux=flux,
                                       model_shape=psf_shape,
                                       border_size=border_size, seed=123)
    plt.figure(figsize=(8, 8))
    plt.imshow(data, origin='lower', interpolation='nearest')
    plt.show()

The ``make_psf_model_image`` function returns the simulated image
(``data``) and a table of the star positions and fluxes (``stars``). The
star positions are stored in the 'x_0' and 'y_0' columns of the table.

Now, let's find the stellar groups. We start by creating
a `~photutils.psf.SourceGrouper` object. Here we set the
``min_separation`` parameter ``2.5 * fwhm``, where the ``fwhm`` is
calculated from the 2D Gaussian standard deviation used to generate
the stars. In general one will need to measure the FWHM of the stellar
profiles.

.. doctest-requires:: scipy

    >>> from astropy.stats import gaussian_sigma_to_fwhm
    >>> from photutils.psf import SourceGrouper
    >>> fwhm = sigma * gaussian_sigma_to_fwhm
    >>> min_separation = 2.5 * fwhm
    >>> grouper = SourceGrouper(min_separation)

We then call the class instance on arrays of the star (x, y) positions.
Here will use the known positions of the stars when we generated the
image. In general, one can use a star finder (:ref:`source_detection`)
to find the sources.

.. doctest-requires:: scipy

   >>> import numpy as np
   >>> x = np.array(stars['x_0'])
   >>> y = np.array(stars['y_0'])
   >>> groups = grouper(x, y)

The ``groups`` output is an array of integers (ordered the same as the
(x, y) inputs) containing the group indices.  Stars with the same group
index are in the same group.

For example, to find all the stars in group 3:

.. doctest-requires:: scipy

   >>> mask = groups == 3
   >>> x[mask], y[mask]
   (array([60.32708921, 58.73063714]), array([147.24184586, 158.0612346 ]))

Here the grouping algorithm separated the 100 stars into 65 distinct groups:

.. doctest-skip::

    >>> print(max(groups))
    65

Finally, let's plot a circular aperture around each star, where stars in
the same group have the same aperture color:

.. doctest-skip::

    >>> import numpy as np
    >>> from photutils.aperture import CircularAperture
    >>> from photutils.utils import make_random_cmap
    >>> plt.imshow(data, origin='lower', interpolation='nearest',
    ...            cmap='Greys_r')
    >>> cmap = make_random_cmap(seed=123)
    >>> for i in np.arange(1, max(groups)):
    >>>     mask = groups == i
    >>>     xypos = zip(x[mask], y[mask])
    >>>     ap = CircularAperture(xypos, r=fwhm)
    >>>     ap.plot(color=cmap.colors[i], lw=2)
    >>> plt.show()

.. plot::

    import matplotlib.pyplot as plt
    import numpy as np
    from astropy.stats import gaussian_sigma_to_fwhm
    from photutils.aperture import CircularAperture
    from photutils.psf import (IntegratedGaussianPRF, SourceGrouper,
                               make_psf_model_image)
    from photutils.utils import make_random_cmap

    shape = (256, 256)
    psf_shape = (11, 11)
    border_size = (6, 6)
    flux = (500, 1000)
    sigma = 2.0
    psf_model = IntegratedGaussianPRF(sigma=sigma)
    n_sources = 100
    data, stars = make_psf_model_image(shape, psf_model, n_sources,
                                       flux=flux,
                                       model_shape=psf_shape,
                                       border_size=border_size, seed=123)

    fwhm = sigma * gaussian_sigma_to_fwhm
    min_separation = 2.5 * fwhm
    grouper = SourceGrouper(min_separation)

    x = np.array(stars['x_0'])
    y = np.array(stars['y_0'])
    groups = grouper(x, y)

    plt.figure(figsize=(8, 8))
    plt.imshow(data, origin='lower', interpolation='nearest', cmap='Greys_r')
    cmap = make_random_cmap(seed=123)
    for i in np.arange(1, max(groups)):
        mask = groups == i
        xypos = zip(x[mask], y[mask])
        ap = CircularAperture(xypos, r=fwhm)
        ap.plot(color=cmap.colors[i], lw=2)

    plt.show()


Reference/API
-------------

.. automodapi:: photutils.psf.groupers
    :no-heading:
