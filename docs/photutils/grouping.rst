Grouping Algorithms
===================

Introduction
------------

In Point Spread Function (PSF) photometry, a grouping algorithm is primarily
used to decide whether two or more stars belong to the same group, i. e.,
whether there are any pixels whose counts are due to the linear combination
of counts from two or more sources.

DAOPHOT GROUP
-------------

Stetson, in his seminal paper, provided a simple, but yet powerful,
grouping algorithm which is able to decide whether or not a given star is
influencing the brightness of any other star. This goal is achieved by means
of a variable called "critical separation", which is defined as being the
distance such that any two stars separated by less than it would be
overlapping. Stetson also gives intutive reasoning to suggest that the critical
separation may be defined as the product of fwhm with some positive real
number.

Grouping Sources
^^^^^^^^^^^^^^^^

Photutils provides an implementation of DAOPHOT GROUP in the
:class:`~photutils.psf.DAOGroup` class. Let's take a look at a simple example.

First let's load some important modules::

    >>> from photutils.datasets import make_gaussian_sources
    >>> from photutils.datasets import make_random_gaussians
    >>> from photutils.datasets import make_noise_image
    >>> import matplotlib.pyplot as plt
    >>> from matplotlib import rcParams
    >>> rcParams['image.cmap'] = 'viridis'
    >>> rcParams['image.aspect'] = 1
    >>> rcParams['figure.figsize'] = (20, 10)

Now, let us make some gaussian sources:

.. plot::
    :include-source:
    
    >>> n_sources = 350
    >>> min_flux = 500
    >>> max_flux = 5000
    >>> min_xmean = min_ymean = 6
    >>> max_xmean = max_ymean = 250
    >>> sigma_psf = 2.0
    >>> starlist = make_random_gaussians(n_sources, [min_flux, max_flux],\
    ...            [min_xmean, max_xmean], [min_ymean, max_ymean],\
    ...            [sigma_psf, sigma_psf], [sigma_psf, sigma_psf],\
    ...            random_state=1234)
    >>> shape = (256, 256)
    >>> sim_image = make_gaussian_sources(shape, starlist)
    >>> plt.imshow(sim_image, origin='lower', interpolation='nearest')
    >>> plt.show()

Let's rename the centroids columns name, so that they agree with the names
that `~photutils.psf.DAOGroup` expect::

    >>> starlist['x_mean'].name = 'x_0'
    >>> starlist['y_mean'].name = 'y_0'

Let's plot circular apertures around our sources and plot them:

.. plot::
    :include-source:

    >>> from photutils import CircularAperture
    >>> from astropy.stats import gaussian_sigma_to_fwhm
    >>> fwhm = sigma_psf*gaussian_sigma_to_fwhm
    >>> circ_apert = CircularAperture((starlist['x_0'], starlist['y_0']),
    ...                               r=fwhm)
    >>> plt.imshow(sim_image, origin='lower', interpolation='nearest')
    >>> circ_apert.plot(lw=1.5, alpha=0.5)

Now, let's actually find the groups of overlapping sources::

    >>> from photutils.psf.groupstars import DAOGroup
    >>> daogroup = DAOGroup(crit_separation=1.5*fwhm)
    >>> star_groups = daogroup(starlist)
    >>> star_groups = star_groups.group_by('group_id')

Let's plot rectangular apertures (which is actually the region that is used
to do simultaneous fitting) which cover each group:

.. plot::
    :include-source:

    >>> from photutils import RectangularAperture
    >>> plt.imshow(sim_image, origin='lower', interpolation='nearest')
    >>> for group in star_groups.groups:
    ...     center = (np.median(group['x_0']), np.median(group['y_0']))
    ...     xmin = np.min(group['x_0']) - fwhm
    ...     xmax = np.max(group['x_0']) + fwhm
    ...     ymin = np.min(group['y_0']) - fwhm
    ...     ymax = np.max(group['y_0']) + fwhm
    ...     width = xmax - xmin + 1
    ...     height = ymax - ymin + 1
    ...     rect_apert = RectangularAperture(center, width, height, theta=0)
    ...     rect_apert.plot(lw=1.5, alpha=0.5)
    >>> circ_apert.plot(lw=1.5, alpha=0.5)
