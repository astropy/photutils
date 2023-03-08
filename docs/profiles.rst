Profiles (`photutils.profiles`)
===============================

Introduction
------------

`photutils.profiles` provides tools to calculate radial profiles and
curves of growth using concentric apertures.


Preliminaries
-------------

Let’s start by making a synthetic image of a single source. Note that
there is no background in this image::

    >>> import numpy as np
    >>> from astropy.modeling.models import Gaussian2D
    >>> from photutils.datasets import make_noise_image

    >>> gmodel = Gaussian2D(42.1, 47.8, 52.4, 4.7, 4.7, 0)
    >>> yy, xx = np.mgrid[0:100, 0:100]
    >>> data = gmodel(xx, yy)
    >>> error = make_noise_image(data.shape, mean=0., stddev=2.4, seed=123)
    >>> data += error

.. plot::

    import matplotlib.pyplot as plt
    import numpy as np
    from astropy.modeling.models import Gaussian2D
    from astropy.visualization import simple_norm
    from photutils.datasets import make_noise_image

    # create an artificial single source
    gmodel = Gaussian2D(42.1, 47.8, 52.4, 4.7, 4.7, 0)
    yy, xx = np.mgrid[0:100, 0:100]
    data = gmodel(xx, yy)
    error = make_noise_image(data.shape, mean=0., stddev=2.4, seed=123)
    data += error

    norm = simple_norm(data, 'sqrt')
    plt.figure(figsize=(5, 5))
    plt.imshow(data, norm=norm)


Creating a Radial Profile
-------------------------

First, we'll use the `~photutils.centroids.centroid_quadratic` function
to find the source centroid::

    >>> from photutils.centroids import centroid_quadratic
    >>> xycen = centroid_quadratic(data, xpeak=48, ypeak=52)
    >>> print(xycen)

Now let's create a radial profile using the
`~photutils.profiles.RadialProfile` class. The radial profile will
be centered at our centroid position. It will be computed over the
radial range from ``min_radius`` to ``max_radius`` with steps of
``radius_step``::

    >>> from photutils.profiles import RadialProfile
    >>> min_radius = 0.0
    >>> max_radius = 25.0
    >>> radius_step = 1.0
    >>> rp = RadialProfile(data, xycen, min_radius, max_radius, radius_step,
    ...                    error=error, mask=None)


The `~photutils.profiles.RadialProfile` instance
has `~photutils.profiles.RadialProfile.radius`,
`~photutils.profiles.RadialProfile.profile`, and
`~photutils.profiles.RadialProfile.profile_error` attributes which
contain 1D `~numpy.ndarray` objects::

    >>> print(rp.radius)
    [ 0.  1.  2.  3.  4.  5.  6.  7.  8.  9. 10. 11. 12. 13. 14. 15. 16. 17.
     18. 19. 20. 21. 22. 23. 24. 25.]

    >>> print(rp.profile)
    [ 4.27430150e+01  4.02150658e+01  3.81601146e+01  3.38116846e+01
      2.89343205e+01  2.34250297e+01  1.84368533e+01  1.44310461e+01
      9.55543388e+00  6.55415896e+00  4.49693014e+00  2.56010523e+00
      1.50362911e+00  7.35389056e-01  6.04663625e-01  8.08820954e-01
      2.31751912e-01 -1.39063329e-01  1.25181410e-01  4.84601845e-01
      1.94567871e-01  4.49109676e-01 -2.00995374e-01 -7.74387397e-02
      5.70302749e-02 -3.27578439e-02]

    >>> print(rp.profile_error)
    [2.95008692 1.17855895 0.6610777  0.51902503 0.47524302 0.43072819
     0.39770113 0.37667594 0.33909996 0.35356048 0.30377721 0.29455808
     0.25670656 0.26599511 0.27354232 0.2430305  0.22910334 0.22204777
     0.22327174 0.23816561 0.2343794  0.2232632  0.19893783 0.17888776
     0.18228289 0.19680823]

The radial profile can be normalized using the
:meth:`~photutils.profiles.RadialProfile.normalize` method::

    >>> rp.normalize(method='max')

There are also convenience methods to plot the radial profile and its
error::

    >>> rp.plot()
    >>> rp.plot_error()

.. plot::

    import matplotlib.pyplot as plt
    import numpy as np
    from astropy.modeling.models import Gaussian2D
    from astropy.visualization import simple_norm

    from photutils.centroids import centroid_quadratic
    from photutils.datasets import make_noise_image
    from photutils.profiles import RadialProfile

    # create an artificial single source
    gmodel = Gaussian2D(42.1, 47.8, 52.4, 4.7, 4.7, 0)
    yy, xx = np.mgrid[0:100, 0:100]
    data = gmodel(xx, yy)
    error = make_noise_image(data.shape, mean=0., stddev=2.4, seed=123)
    data += error

    # find the source centroid
    xycen = centroid_quadratic(data, xpeak=47, ypeak=52)

    # create the radial profile
    min_radius = 0.0
    max_radius = 25.0
    radius_step = 1.0
    rp = RadialProfile(data, xycen, min_radius, max_radius, radius_step,
                       error=error, mask=None)

    # plot the radial profile
    rp.normalize()
    rp.plot()
    rp.plot_error()

The `~photutils.profiles.RadialProfile.apertures` attribute contains a
list of the apertures. Let's plot a couple of the apertures on the data:

.. plot::

    import matplotlib.pyplot as plt
    import numpy as np
    from astropy.modeling.models import Gaussian2D
    from astropy.visualization import simple_norm

    from photutils.centroids import centroid_quadratic
    from photutils.datasets import make_noise_image
    from photutils.profiles import RadialProfile

    # create an artificial single source
    gmodel = Gaussian2D(42.1, 47.8, 52.4, 4.7, 4.7, 0)
    yy, xx = np.mgrid[0:100, 0:100]
    data = gmodel(xx, yy)
    error = make_noise_image(data.shape, mean=0., stddev=2.4, seed=123)
    data += error

    # find the source centroid
    xycen = centroid_quadratic(data, xpeak=47, ypeak=52)

    # create the radial profile
    min_radius = 0.0
    max_radius = 25.0
    radius_step = 1.0
    rp = RadialProfile(data, xycen, min_radius, max_radius, radius_step,
                       error=error, mask=None)

    norm = simple_norm(data, 'sqrt')
    plt.figure(figsize=(5, 5))
    plt.imshow(data, norm=norm)
    rp.apertures[5].plot(color='C0', lw=2)
    rp.apertures[10].plot(color='C1', lw=2)


Creating a Curve of Growth
--------------------------

    >>> from astropy.visualization import simple_norm
    >>> from photutils.centroids import centroid_quadratic
    >>> from photutils.profiles import CurveOfGrowth



Reference/API
-------------

.. automodapi:: photutils.profiles
    :no-heading:
    :inherited-members:
