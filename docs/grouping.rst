Grouping Algorithms
===================

Introduction
------------

In Point Spread Function (PSF) photometry, a grouping algorithm is
used to separate stars into optimum groups.  The stars in each group
are defined as those close enough together such that they need to be
fit simultaneously, i.e., their profiles overlap.

Photutils currently provides two classes to group stars:

  * :class:`~photutils.psf.DAOGroup`:  An implementation of the
    `DAOPHOT
    <https://ui.adsabs.harvard.edu/abs/1987PASP...99..191S/abstract>`_
    GROUP algorithm.

  * :class:`~photutils.psf.DBSCANGroup`:  Grouping is based on the
    `Density-Based Spatial Clustering of Applications with Noise
    (DBSCAN) <https://en.wikipedia.org/wiki/DBSCAN>`_ algorithm.


DAOPHOT GROUP
-------------

Stetson, in his seminal paper (`Stetson 1987, PASP 99, 191
<https://ui.adsabs.harvard.edu/abs/1987PASP...99..191S/abstract>`_),
provided a simple and powerful grouping algorithm to decide whether
the profile of a given star extends into the fitting region of any
other star.  Stetson defines this in terms of a "critical separation"
parameter, which is defined as the minimal distance that any two stars
must be separated by in order to be in different groups.  Stetson
gives intuitive reasoning to suggest that the critical separation may
be defined as a multiple of the stellar full width at half maximum
(FWHM).

Photutils provides an implementation of the DAOPHOT GROUP algorithm in
the :class:`~photutils.psf.DAOGroup` class. Let's take a look at a
simple example.

First, let's make some Gaussian sources using
`~photutils.datasets.make_random_gaussians_table` and
`~photutils.datasets.make_gaussian_sources_image`. The former will
return a `~astropy.table.Table` containing parameters for 2D Gaussian
sources and the latter will make an actual image using that table.

.. plot::
    :include-source:

    import matplotlib.pyplot as plt
    import numpy as np
    from photutils.datasets import (make_gaussian_sources_image,
                                    make_random_gaussians_table)

    n_sources = 350
    sigma_psf = 2.0

    params = {'flux': [500, 5000],
              'x_mean': [6, 250],
              'y_mean': [6, 250],
              'x_stddev': [sigma_psf, sigma_psf],
              'y_stddev': [sigma_psf, sigma_psf],
              'theta': [0, np.pi]}
    starlist = make_random_gaussians_table(n_sources, params, seed=123)

    shape = (256, 256)
    sim_image = make_gaussian_sources_image(shape, starlist)

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

Now, let's find the stellar groups.  We start by creating a
`~photutils.psf.DAOGroup` object.  Here we set its ``crit_separation``
parameter ``2.5 * fwhm``, where the stellar ``fwhm`` was defined above
when we created the stars as 2D Gaussians.  In general one will need
to measure the FWHM of the stellar profiles.

.. doctest-skip::

    >>> from astropy.stats import gaussian_sigma_to_fwhm
    >>> from photutils.psf.groupstars import DAOGroup
    >>> fwhm = sigma_psf * gaussian_sigma_to_fwhm
    >>> daogroup = DAOGroup(crit_separation=2.5 * fwhm)

``daogroup`` is a `~photutils.psf.DAOGroup` instance that can be used
as a calling function that receives as input a table of stars (e.g.,
``starlist``):

.. doctest-skip::

    >>> star_groups = daogroup(starlist)

The ``star_groups`` output is copy of the input ``starlist`` table,
but with an extra column called ``group_id``.  This column contains
integers that represent the group assigned to each source.  Here the
grouping algorithm separated the 350 stars into 92 distinct groups:

.. doctest-skip::

    >>> print(max(star_groups['group_id']))
    92

One can use the ``group_by`` functionality from `~astropy.table.Table`
to create groups according to ``group_id``:

.. doctest-skip::

    >>> star_groups = star_groups.group_by('group_id')
    >>> print(star_groups)
         flux          x_0           y_0      ...   amplitude    id group_id
    ------------- ------------- ------------- ... ------------- --- --------
    1361.83752671 182.958386152 178.708228379 ... 54.1857935158   1        1
    4282.41965053 179.998944123 171.437757021 ... 170.392063944 183        1
    555.831417775 181.611905957  185.16181342 ... 22.1158294162 222        1
    3299.48946968  243.60449392 85.8926967927 ... 131.282514695   2        2
    2469.77482553 136.657577889 109.771746713 ... 98.2692179518   3        3
              ...           ...           ... ...           ... ...      ...
    818.132804377 117.787387455 92.4349134636 ... 32.5524699806 313       88
    3979.57421702  154.85279495 18.3148180315 ...  158.34222701 318       89
    3622.30997136 97.0901736699 50.3565997421 ... 144.127134338 323       90
     765.47561385 144.952825542 7.57086675812 ... 30.4573069401 330       91
    1508.68165551 54.0404934991 232.693833605 ... 60.0285357567 349       92
    Length = 350 rows

Finally, let's plot a circular aperture around each star, where stars
in the same group have the same aperture color:

.. doctest-skip::

    >>> import numpy as np
    >>> from photutils.aperture import CircularAperture
    >>> from photutils.utils import make_random_cmap
    >>> plt.imshow(sim_image, origin='lower', interpolation='nearest',
    ...            cmap='Greys_r')
    >>> cmap = make_random_cmap(seed=123)
    >>> for i, group in enumerate(star_groups.groups):
    >>>     xypos = np.transpose([group['x_0'], group['y_0']])
    >>>     ap = CircularAperture(xypos, r=fwhm)
    >>>     ap.plot(color=cmap.colors[i])
    >>> plt.show()

.. plot::

    import matplotlib.pyplot as plt
    import numpy as np
    from astropy.stats import gaussian_sigma_to_fwhm
    from matplotlib import rcParams
    from photutils.aperture import CircularAperture
    from photutils.datasets import (make_gaussian_sources_image,
                                    make_random_gaussians_table)
    from photutils.psf.groupstars import DAOGroup
    from photutils.utils import make_random_cmap

    rcParams['image.aspect'] = 1  # to get images with square pixels
    rcParams['figure.figsize'] = (7, 7)

    n_sources = 350
    sigma_psf = 2.0
    params = {'flux': [500, 5000],
              'x_mean': [6, 250],
              'y_mean': [6, 250],
              'x_stddev': [sigma_psf, sigma_psf],
              'y_stddev': [sigma_psf, sigma_psf],
              'theta': [0, np.pi]}
    starlist = make_random_gaussians_table(n_sources, params, seed=123)

    shape = (256, 256)
    sim_image = make_gaussian_sources_image(shape, starlist)

    starlist['x_mean'].name = 'x_0'
    starlist['y_mean'].name = 'y_0'

    fwhm = sigma_psf * gaussian_sigma_to_fwhm
    daogroup = DAOGroup(crit_separation=2.5 * fwhm)
    star_groups = daogroup(starlist)
    star_groups = star_groups.group_by('group_id')

    plt.imshow(sim_image, origin='lower', interpolation='nearest',
               cmap='Greys_r')

    cmap = make_random_cmap(seed=123)
    for i, group in enumerate(star_groups.groups):
        xypos = np.transpose([group['x_0'], group['y_0']])
        ap = CircularAperture(xypos, r=fwhm)
        ap.plot(color=cmap.colors[i])


DBSCANGroup
-----------

Photutils also provides a :class:`~photutils.psf.DBSCANGroup` class to
group stars based on the `Density-Based Spatial Clustering of
Applications with Noise (DBSCAN)
<https://en.wikipedia.org/wiki/DBSCAN>`_ algorithm.
:class:`~photutils.psf.DBSCANGroup` provides a more general algorithm
than :class:`~photutils.psf.DAOGroup`.

Here's a simple example using :class:`~photutils.psf.DBSCANGroup` with
``min_samples=1`` and ``metric=euclidean``.  With these parameters,
the result is identical to the `~photutils.psf.DAOGroup` algorithm.
Note that `scikit-learn <https://scikit-learn.org/>`_ must be installed
to use :class:`~photutils.psf.DBSCANGroup`.

.. plot::

    import matplotlib.pyplot as plt
    import numpy as np
    from astropy.stats import gaussian_sigma_to_fwhm
    from matplotlib import rcParams
    from photutils.aperture import CircularAperture
    from photutils.datasets import (make_gaussian_sources_image,
                                    make_random_gaussians_table)
    from photutils.psf.groupstars import DBSCANGroup
    from photutils.utils import make_random_cmap

    rcParams['image.aspect'] = 1  # to get images with square pixels
    rcParams['figure.figsize'] = (7, 7)

    n_sources = 350
    sigma_psf = 2.0
    params = {'flux': [500, 5000],
              'x_mean': [6, 250],
              'y_mean': [6, 250],
              'x_stddev': [sigma_psf, sigma_psf],
              'y_stddev': [sigma_psf, sigma_psf],
              'theta': [0, np.pi]}
    starlist = make_random_gaussians_table(n_sources, params, seed=123)

    shape = (256, 256)
    sim_image = make_gaussian_sources_image(shape, starlist)

    starlist['x_mean'].name = 'x_0'
    starlist['y_mean'].name = 'y_0'

    fwhm = sigma_psf * gaussian_sigma_to_fwhm
    group = DBSCANGroup(crit_separation=2.5 * fwhm)
    star_groups = group(starlist)
    star_groups = star_groups.group_by('group_id')

    plt.imshow(sim_image, origin='lower', interpolation='nearest',
               cmap='Greys_r')

    cmap = make_random_cmap(seed=123)
    for i, group in enumerate(star_groups.groups):
        xypos = np.transpose([group['x_0'], group['y_0']])
        ap = CircularAperture(xypos, r=fwhm)
        ap.plot(color=cmap.colors[i])


Reference/API
-------------

.. automodapi:: photutils.psf.groupstars
    :no-heading:
