Grouping Algorithms
===================

Introduction
------------

In Point Spread Function (PSF) photometry, a grouping algorithm is primarily
used to divide a star list into optimum groups. More precisely, a grouping
algorithm must be able to decide whether two or more stars belong to the same
group, i. e., whether there are any pixels whose counts are due to the linear
combination of counts from two or more sources.

DAOPHOT GROUP
-------------

Stetson, in his seminal paper (`Stetson 1987, PASP 99, 191
<http://adsabs.harvard.edu/abs/1987PASP...99..191S>`_), provided a simple and
powerful grouping algorithm to decide whether or not the profile
of a given star extends into the fitting region around the centroid of any
other star. This goal is achieved by means of a variable called "critical
separation", which is defined as the distance such that any two stars
separated by less than it would be overlapping. Stetson also gives intutive
reasoning to suggest that the critical separation may be defined as the
product of fwhm with some positive real number.

Grouping Sources
^^^^^^^^^^^^^^^^

Photutils provides an implementation of DAOPHOT GROUP in the
:class:`~photutils.psf.DAOGroup` class. Let's take a look at a simple example.

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

Now, we need to rename the columns of the centroid positions so that they
agree with the names that `~photutils.psf.DAOGroup` expect:

.. doctest-skip::

    starlist['x_mean'].name = 'x_0'
    starlist['y_mean'].name = 'y_0'

Before finding groups, let's plot circular apertures using
`~photutils.aperture.CircularAperture` around the sources.

.. doctest-skip::

    from photutils import CircularAperture
    from astropy.stats import gaussian_sigma_to_fwhm
    circ_aperture = CircularAperture((starlist['x_0'], starlist['y_0']),
                                     r=sigma_psf*gaussian_sigma_to_fwhm)
    plt.imshow(sim_image, origin='lower', interpolation='nearest',
               cmap='viridis')
    circ_aperture.plot(lw=1.5, alpha=0.5)
    plt.show()

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
    circ_aperture.plot(lw=1.5, alpha=0.5)
    plt.show()


Let's create a `~photutils.DAOGroup` object and set its ``crit_separation``
attribute to ``1.5*fwhm``:

.. doctest-skip::
    
    from photutils.psf.groupstars import DAOGroup
    
    fwhm = sigma_psf*gaussian_sigma_to_fwhm
    daogroup = DAOGroup(crit_separation=1.5*fwhm)

Now, we can use the instance of `~photutils.DAOGroup` as a function calling
which receives as input our list of stars ``starlist``:

.. doctest-skip::

    star_groups = daogroup(starlist)

This procedure copies ``starlist`` into ``star_groups`` and adds a new column
to ``star_groups`` called ``group_id``. This column contains integer numbers
which represent the group that the sources belong to.

Finally, one can use the ``group_by`` functionality from `~astropy.table.Table`
to create groups according ``group_id``:

.. doctest-skip::

    star_groups = star_groups.group_by('group_id')

Now, let's plot rectangular apertures which cover each group:

.. doctest-skip::

    from photutils import RectangularAperture

    plt.imshow(sim_image, origin='lower', interpolation='nearest',
               cmap='viridis')
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
    plt.show()

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
