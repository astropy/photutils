Grouping Algorithms
===================

Introduction
------------

In Point Spread Function (PSF) photometry, a grouping algorithm is
used to separate stars into optimum groups.  The stars in each group
are defined as those close enough together such that they need to be
fit simultaneously, i.e. their profiles overlap.


DAOPHOT GROUP
-------------

Stetson, in his seminal paper (`Stetson 1987, PASP 99, 191
<http://adsabs.harvard.edu/abs/1987PASP...99..191S>`_), provided a
simple and powerful grouping algorithm to decide whether or not the
profile of a given star extends into the fitting region of any other
star. Stetson defines this in terms of a "critical separation"
parameter, which is defined as the minimal distance that any two stars
must be separated by in order to be in different groups.  Stetson
gives intuitive reasoning to suggest that the critical separation may
be defined as a multiple of the stellar full width at half maximum
(FWHM).


Grouping Sources
^^^^^^^^^^^^^^^^

Photutils provides an implementation of the DAOPHOT GROUP algorithm in
the :class:`~photutils.psf.DAOGroup` class. Let's take a look at a
simple example.

First, let's make some Gaussian sources using
`~photutils.datasets.make_random_gaussians` and
`~photutils.datasets.make_gaussian_sources`. The former will return a
`~astropy.table.Table` containing parameters for 2D Gaussian sources and the
latter will make an actual image using that table.

.. plot::
    :include-source:

    import numpy as np
    from photutils.datasets import make_gaussian_sources
    from photutils.datasets import make_random_gaussians
    import matplotlib.pyplot as plt

    n_sources = 350
    min_flux = 500
    max_flux = 5000
    min_xmean = min_ymean = 6
    max_xmean = max_ymean = 250
    sigma_psf = 2.0

    starlist = make_random_gaussians(n_sources, [min_flux, max_flux],\
               [min_xmean, max_xmean], [min_ymean, max_ymean],\
               [sigma_psf, sigma_psf], [sigma_psf, sigma_psf],\
               random_state=1234)
    shape = (256, 256)

    sim_image = make_gaussian_sources(shape, starlist)

    plt.imshow(sim_image, origin='lower', interpolation='nearest',
               cmap='viridis')
    plt.show()


``starlist`` is an astropy `~astropy.table.Table` of parameters
defining the position and shape of the stars.

Next, we need to rename the table columns of the centroid positions so
that they agree with the names that `~photutils.psf.DAOGroup` expect.
Here we rename ``x_mean`` to ``x_0`` and ``y_mean`` to ``y_0``:

.. doctest-skip::

    >>> starlist['x_mean'].name = 'x_0'
    >>> starlist['y_mean'].name = 'y_0'

Let's plot circular apertures around each of the stars using
`~photutils.aperture.CircularAperture`.

.. doctest-skip::

    >>> from photutils import CircularAperture
    >>> from astropy.stats import gaussian_sigma_to_fwhm
    >>> circ_aperture = CircularAperture((starlist['x_0'], starlist['y_0']),
    ...                                  r=sigma_psf*gaussian_sigma_to_fwhm)
    >>> plt.imshow(sim_image, origin='lower', interpolation='nearest',
    ...           cmap='viridis')
    >>> circ_aperture.plot(lw=1.5, alpha=0.5, color='gray')
    >>> plt.show()

.. plot::

    import numpy as np
    from photutils.datasets import make_gaussian_sources
    from photutils.datasets import make_random_gaussians
    import matplotlib.pyplot as plt

    n_sources = 350
    min_flux = 500
    max_flux = 5000
    min_xmean = min_ymean = 6
    max_xmean = max_ymean = 250
    sigma_psf = 2.0
    starlist = make_random_gaussians(n_sources, [min_flux, max_flux],\
               [min_xmean, max_xmean], [min_ymean, max_ymean],\
               [sigma_psf, sigma_psf], [sigma_psf, sigma_psf],\
               random_state=1234)

    shape = (256, 256)
    sim_image = make_gaussian_sources(shape, starlist)
    plt.imshow(sim_image, origin='lower', interpolation='nearest',
               cmap='viridis')
    from photutils import CircularAperture
    from astropy.stats import gaussian_sigma_to_fwhm
    circ_aperture = CircularAperture((starlist['x_mean'], starlist['y_mean']),
                                     r=sigma_psf*gaussian_sigma_to_fwhm)
    plt.imshow(sim_image, origin='lower', interpolation='nearest',
               cmap='viridis')
    circ_aperture.plot(lw=1.5, alpha=0.5, color='gray')
    plt.show()

Now, let's find the stellar groups.  We start by creating
`~photutils.DAOGroup` object.  Here we set its ``crit_separation``
parameter ``1.5 * fwhm``, where the stellar ``fwhm`` was defined above
when we created the stars as 2D Gaussians.  In general one will need
to measure the FWHM of the stellar profiles.

.. doctest-skip::

    >>> from photutils.psf.groupstars import DAOGroup
    >>> fwhm = sigma_psf * gaussian_sigma_to_fwhm
    >>> daogroup = DAOGroup(crit_separation=1.5*fwhm)

``daogroup`` is a `~photutils.DAOGroup` instance that can be used as a
calling function that receives as input a table of stars (e.g.
``starlist``):

.. doctest-skip::

    >>> star_groups = daogroup(starlist)

The ``star_groups`` output is copy of the input ``starlist`` table,
but with an extra column called ``group_id``.  This column contains
integers that represent the group assigned to each source.  Here the
grouping algorithm separated the 350 stars into 249 distinct groups:

.. doctest-skip::

    >>> print(max(star_groups['group_id']))
    249

Finally, one can use the ``group_by`` functionality from
`~astropy.table.Table` to create groups according ``group_id``:

.. doctest-skip::

    >>> star_groups = star_groups.group_by('group_id')
    >>> print(star_groups)

         flux          x_0           y_0      ...     theta       id group_id
    ------------- ------------- ------------- ... -------------- --- --------
    1361.83752671 182.958386152 178.708228379 ...  4.36133269879   1        1
    555.831417775 181.611905957  185.16181342 ... 0.801284325687 222        1
    3299.48946968  243.60449392 85.8926967927 ...  2.24138419824   2        2
    2469.77482553 136.657577889 109.771746713 ...  4.82559763746   3        3
    1650.43978895  131.83343504 110.441871517 ...  5.44328378359 153        3
              ...           ...           ... ...            ... ...      ...
     4789.5840034 47.9900598664 29.4596354785 ...  5.47735588068 341      246
    4831.78338403 49.2618839218  24.821038274 ...  3.84946567257 345      246
    643.136283663 81.2058931512 197.205965254 ...  5.75254014417 344      247
    4437.94013032 20.5310110132 159.825683512 ...  5.23140824935 348      248
    1508.68165551 54.0404934991 232.693833605 ...  1.54042673504 349      249
    Length = 350 rows

Now, let's plot rectangular apertures that cover each group:

.. doctest-skip::

    >>> from photutils import RectangularAperture

    >>> plt.imshow(sim_image, origin='lower', interpolation='nearest',
    ...            cmap='viridis')
    >>> for group in star_groups.groups:
    >>>     group_center = (np.median(group['x_0']), np.median(group['y_0']))
    >>>     xmin = np.min(group['x_0']) - fwhm
    >>>     xmax = np.max(group['x_0']) + fwhm
    >>>     ymin = np.min(group['y_0']) - fwhm
    >>>     ymax = np.max(group['y_0']) + fwhm
    >>>     group_width = xmax - xmin + 1
    >>>     group_height = ymax - ymin + 1
    >>>     rect_aperture = RectangularAperture(group_center, group_width,
    ...                                         group_height, theta=0)
    >>>     rect_aperture.plot(lw=1.5, alpha=0.5, color='gray')
    >>> circ_aperture.plot(lw=1.5, alpha=0.5)
    >>> plt.show()

.. plot::

    import numpy as np
    from photutils.datasets import make_gaussian_sources
    from photutils.datasets import make_random_gaussians
    import matplotlib.pyplot as plt
    from matplotlib import rcParams
    rcParams['image.cmap'] = 'viridis'
    rcParams['image.aspect'] = 1  # to get images with square pixels
    rcParams['figure.figsize'] = (7,7)

    n_sources = 350
    min_flux = 500
    max_flux = 5000
    min_xmean = min_ymean = 6
    max_xmean = max_ymean = 250
    sigma_psf = 2.0
    starlist = make_random_gaussians(n_sources, [min_flux, max_flux],\
               [min_xmean, max_xmean], [min_ymean, max_ymean],\
               [sigma_psf, sigma_psf], [sigma_psf, sigma_psf],\
               random_state=1234)
    shape = (256, 256)
    sim_image = make_gaussian_sources(shape, starlist)
    starlist['x_mean'].name = 'x_0'
    starlist['y_mean'].name = 'y_0'

    from photutils import CircularAperture
    from astropy.stats import gaussian_sigma_to_fwhm
    circ_aperture = CircularAperture((starlist['x_0'], starlist['y_0']),
                                     r=sigma_psf*gaussian_sigma_to_fwhm)

    from photutils.psf.groupstars import DAOGroup
    fwhm = sigma_psf*gaussian_sigma_to_fwhm
    daogroup = DAOGroup(crit_separation=1.5*fwhm)
    star_groups = daogroup(starlist)
    star_groups = star_groups.group_by('group_id')

    from photutils import RectangularAperture
    plt.imshow(sim_image, origin='lower', interpolation='nearest')
    for group in star_groups.groups:
        group_center = (np.median(group['x_0']), np.median(group['y_0']))
        xmin = np.min(group['x_0']) - fwhm
        xmax = np.max(group['x_0']) + fwhm
        ymin = np.min(group['y_0']) - fwhm
        ymax = np.max(group['y_0']) + fwhm
        group_width = xmax - xmin + 1
        group_height = ymax - ymin + 1
        rect_aperture = RectangularAperture(group_center, group_width,
                                            group_height, theta=0)
        rect_aperture.plot(lw=1.5, alpha=0.5)
    circ_aperture.plot(lw=1.5, alpha=0.5)
